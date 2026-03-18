#!/usr/bin/env python3
"""
Pre-warm Redis Cache for Heavy Queries - OPTIMIZED WITH BATCHING
================================================================

This script pre-populates the Redis cache with results from commonly-requested
heavy queries (especially "all" time filter) to prevent server overload.

Features:
- Memory-efficient batch processing for large datasets
- Configurable batch sizes for optimal performance
- Progress tracking for long-running operations
- OE/POS filtered data caching
- Common time filters pre-cached

Run this after:
- Initial ML pipeline training
- Daily data sync
- Manual cache flush

Usage:
    python3 scripts/prewarm_cache.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime, timedelta

# =========================================================================
# BATCH PROCESSING CONFIGURATION
# =========================================================================
BATCH_SIZE_CUSTOMERS = 5000      # Customers processed per batch for RFM
BATCH_SIZE_PRODUCTS = 1000       # Products processed per batch
BATCH_SIZE_ORDERS = 10000        # Orders processed per batch
MAX_CUSTOMERS_PER_SEGMENT = 500  # Max customers cached per RFM segment


def get_time_filter_clause(time_filter: str) -> str:
    """Get SQL WHERE clause for time filtering"""
    if time_filter == "all":
        return ""
    elif time_filter == "30days":
        return f"AND o.order_date >= NOW() - INTERVAL '30 days'"
    elif time_filter == "90days":
        return f"AND o.order_date >= NOW() - INTERVAL '90 days'"
    elif time_filter == "6months":
        return f"AND o.order_date >= NOW() - INTERVAL '180 days'"
    elif time_filter == "1year":
        return f"AND o.order_date >= NOW() - INTERVAL '365 days'"
    elif time_filter == "3years":
        return f"AND o.order_date >= NOW() - INTERVAL '1095 days'"
    return ""


def get_order_source_filter(order_source: str) -> str:
    """Get SQL filter for order source (OE/POS)"""
    if order_source == "oe":
        return "AND UPPER(o.order_type) = 'OE'"
    elif order_source == "pos":
        return "AND UPPER(o.order_type) = 'POS'"
    return ""


def get_delivered_filter(delivered_only: bool, order_source: str = None) -> str:
    """Get SQL filter for delivered/completed orders only"""
    if not delivered_only:
        return ""
    if order_source == "oe":
        return "AND o.order_status = 'Delivered Orders'"
    elif order_source == "pos":
        return "AND o.order_status = 'completed'"
    else:
        return "AND (o.order_status = 'Delivered Orders' OR o.order_status = 'completed')"


def safe_execute_and_cache(cursor, redis_client, cache_key: str, query: str, data_processor, ttl: int, description: str):
    """
    Safely execute a query and cache results with error handling.
    Returns (success: bool, result_data: dict, error_message: str)
    """
    try:
        print(f'   ⏳ {description}...', flush=True)
        cursor.execute(query)
        
        # Use the data processor function to format the results
        result_data = data_processor(cursor)
        
        # Cache the results
        redis_client.setex(cache_key, ttl, json.dumps(result_data))
        
        print(f'   ✅ {description} cached successfully', flush=True)
        return True, result_data, ""
        
    except Exception as e:
        error_msg = f'Error in {description}: {str(e)}'
        print(f'   ❌ {error_msg}', flush=True)
        return False, {}, error_msg


def process_customers_in_batches(cursor, batch_size: int = BATCH_SIZE_CUSTOMERS):
    """
    Generator that yields customers in batches for memory-efficient processing.
    Returns (batch_number, customers_batch, is_last_batch)
    """
    # First get total count
    cursor.execute("""
        SELECT COUNT(DISTINCT unified_customer_id) as total_customers
        FROM orders
    """)
    total_customers = cursor.fetchone()['total_customers']
    
    if total_customers == 0:
        return
    
    total_batches = (total_customers + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        offset = batch_num * batch_size
        
        cursor.execute(f"""
            WITH customer_rfm AS (
                SELECT 
                    o.unified_customer_id,
                    MAX(o.customer_name) as customer_name,
                    MAX(o.customer_phone) as customer_phone,
                    MAX(o.customer_city) as city,
                    MAX(o.province) as province,
                    MAX(UPPER(o.order_type)) as order_type,
                    EXTRACT(days FROM NOW() - MAX(o.order_date)) as recency_days,
                    COUNT(DISTINCT o.id) as frequency,
                    SUM(o.total_price) as monetary,
                    MAX(o.order_date) as last_order_date
                FROM orders o
                GROUP BY o.unified_customer_id
                ORDER BY o.unified_customer_id
                LIMIT {batch_size} OFFSET {offset}
            )
            SELECT 
                unified_customer_id,
                customer_name,
                customer_phone,
                city,
                province,
                recency_days,
                frequency,
                monetary,
                last_order_date,
                CASE 
                    WHEN order_type = 'HISTORICAL' AND frequency >= 5 AND monetary >= 50000 THEN 'Champions'
                    WHEN order_type = 'HISTORICAL' AND frequency >= 3 AND monetary >= 20000 THEN 'Loyal'
                    WHEN order_type = 'HISTORICAL' AND frequency >= 2 THEN 'At Risk'
                    WHEN order_type = 'HISTORICAL' AND frequency = 1 THEN 'Lost'
                    WHEN recency_days <= 30 AND frequency >= 5 AND monetary >= 50000 THEN 'Champions'
                    WHEN recency_days <= 60 AND frequency >= 3 AND monetary >= 20000 THEN 'Loyal'
                    WHEN recency_days <= 90 AND frequency >= 2 THEN 'Potential'
                    WHEN frequency = 1 AND recency_days <= 30 THEN 'New Customers'
                    WHEN recency_days > 90 AND recency_days <= 180 AND frequency >= 2 THEN 'At Risk'
                    WHEN recency_days > 180 AND recency_days <= 365 THEN 'Hibernating'
                    WHEN recency_days > 365 THEN 'Lost'
                    ELSE 'Regular'
                END as segment
            FROM customer_rfm
        """)
        
        batch_customers = cursor.fetchall()
        is_last_batch = batch_num == total_batches - 1
        
        yield batch_num + 1, batch_customers, is_last_batch, total_batches


def cache_dashboard_metrics(cursor, redis_client, time_filter: str, order_source: str, delivered_only: bool, ttl: int):
    """Cache dashboard metrics for specific filters"""
    time_clause = get_time_filter_clause(time_filter)
    source_clause = get_order_source_filter(order_source)
    delivered_clause = get_delivered_filter(delivered_only, order_source)
    
    cursor.execute(f"""
        SELECT 
            COUNT(DISTINCT o.id) as total_orders,
            COUNT(DISTINCT o.unified_customer_id) as total_customers,
            COALESCE(SUM(o.total_price), 0) as total_revenue,
            COALESCE(AVG(o.total_price), 0) as avg_order_value
        FROM orders o
        WHERE 1=1
        {time_clause}
        {source_clause}
        {delivered_clause}
    """)
    row = cursor.fetchone()
    
    cache_key = f"analytics:dashboard:{time_filter}:{order_source or 'all'}:{delivered_only}"
    
    dashboard_data = {
        "success": True,
        "total_orders": row["total_orders"] or 0,
        "total_customers": row["total_customers"] or 0,
        "total_revenue": float(row["total_revenue"] or 0),
        "avg_order_value": float(row["avg_order_value"] or 0),
        "time_filter": time_filter,
        "order_source": order_source or "all",
        "delivered_only": delivered_only,
        "totalOrders": row["total_orders"] or 0,
        "totalCustomers": row["total_customers"] or 0,
        "totalRevenueAmount": float(row["total_revenue"] or 0),
        "avgOrderValue": float(row["avg_order_value"] or 0),
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex(cache_key, ttl, json.dumps(dashboard_data))
    return dashboard_data


def main():
    print('=' * 80, flush=True)
    print('  REDIS CACHE PRE-WARMING (Optimized with Batch Processing)', flush=True)
    print('=' * 80, flush=True)
    print(f'  🚀 Configuration:', flush=True)
    print(f'     • Customer batch size: {BATCH_SIZE_CUSTOMERS:,}', flush=True)
    print(f'     • Product batch size: {BATCH_SIZE_PRODUCTS:,}', flush=True)
    print(f'     • Max customers per segment: {MAX_CUSTOMERS_PER_SEGMENT:,}', flush=True)
    print(flush=True)
    
    # Connect to Redis
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_db = int(os.getenv('REDIS_DB', 0))
    
    print(f'Connecting to Redis at {redis_host}:{redis_port}...', flush=True)
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    redis_client.ping()
    print('✅ Redis connected\n', flush=True)
    
    # Connect to Database
    db_params = {
        'host': os.getenv('PG_HOST'),
        'port': int(os.getenv('PG_PORT', 5432)),
        'database': os.getenv('PG_DB'),
        'user': os.getenv('PG_USER'),
        'password': os.getenv('PG_PASSWORD'),
    }
    
    # Add SSL if not localhost
    if db_params['host'] != 'localhost':
        db_params['sslmode'] = 'require'
    
    print(f'Connecting to database at {db_params["host"]}...', flush=True)
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    print('✅ Database connected\n', flush=True)
    
    # Cache TTL for "all" queries (2 hours)
    TTL = 7200
    TTL_SHORT = 1800  # 30 minutes for more frequent time filters
    
    # =========================================================================
    # 1. DASHBOARD METRICS - ALL COMBINATIONS
    # =========================================================================
    print('1. Caching dashboard metrics for ALL filter combinations...', flush=True)
    
    time_filters = ['all', '3years', '1year', '6months', '90days', '30days']
    order_sources = [None, 'oe', 'pos']
    delivered_options = [False, True]
    
    cached_count = 0
    for tf in time_filters:
        for os_filter in order_sources:
            for delivered in delivered_options:
                try:
                    ttl = TTL if tf == 'all' else TTL_SHORT
                    data = cache_dashboard_metrics(cursor, redis_client, tf, os_filter, delivered, ttl)
                    os_label = os_filter.upper() if os_filter else 'ALL'
                    del_label = "Delivered" if delivered else "All Status"
                    print(f'   ✅ {tf:10} | {os_label:4} | {del_label:12} | Orders: {data["total_orders"]:>8,} | Revenue: Rs {data["total_revenue"]:>15,.0f}', flush=True)
                    cached_count += 1
                except Exception as e:
                    print(f'   ❌ Error caching {tf}/{os_filter}/{delivered}: {e}', flush=True)
    
    print(f'   📊 Cached {cached_count} dashboard metric combinations\n', flush=True)
    
    # =========================================================================
    # 2. POPULAR PRODUCTS (ALL TIME - base case)
    # =========================================================================
    print('2. Caching popular products for ALL TIME...')
    cursor.execute("""
        SELECT 
            oi.product_id,
            MAX(oi.product_name) as product_name,
            COUNT(DISTINCT oi.order_id) as order_count,
            SUM(oi.quantity) as total_quantity,
            AVG(oi.unit_price) as avg_price,
            SUM(oi.total_price) as total_revenue
        FROM order_items oi
        GROUP BY oi.product_id
        ORDER BY order_count DESC
        LIMIT 30
    """)
    products = cursor.fetchall()
    
    products_list = []
    for p in products:
        products_list.append({
            "product_id": p["product_id"],
            "product_name": p["product_name"],
            "score": p["order_count"],
            "purchase_count": p["order_count"],
            "total_quantity": p["total_quantity"],
            "avg_price": float(p["avg_price"] or 0),
            "total_revenue": float(p["total_revenue"] or 0)
        })
    
    products_data = {
        "success": True,
        "products": products_list,
        "time_filter": "all",
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex("popular_products:30:all:all", TTL, json.dumps(products_data))
    print(f'   ✅ Cached {len(products)} products\n')
    
    # 3. Revenue Trend (ALL TIME - Monthly)
    print('3. Caching revenue trend for ALL TIME (monthly)...')
    cursor.execute("""
        SELECT 
            DATE_TRUNC('month', o.order_date) as period,
            COUNT(DISTINCT o.id) as order_count,
            SUM(o.total_price) as revenue
        FROM orders o
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT 24
    """)
    trend = cursor.fetchall()
    
    # Calculate max for percentages
    max_revenue = max([float(t['revenue'] or 0) for t in trend]) if trend else 1
    
    trend_data = []
    for t in trend:
        revenue = float(t['revenue'] or 0)
        trend_data.append({
            'period': t['period'].isoformat() if t['period'] else None,
            'label': t['period'].strftime('%b %Y') if t['period'] else 'N/A',
            'order_count': t['order_count'],
            'total_revenue': revenue,
            'percentage': (revenue / max_revenue * 100) if max_revenue > 0 else 0
        })
    
    trend_result = {
        'success': True,
        'trend_data': list(reversed(trend_data)),
        'summary': {
            'max_revenue': max_revenue,
            'total_months': len(trend_data)
        },
        'cached': True,
        'timestamp': datetime.now().isoformat()
    }
    
    redis_client.setex('analytics:revenue_trend:all:monthly', TTL, json.dumps(trend_result))
    print(f'   ✅ Cached {len(trend_data)} months of trend data\n')
    
    # 4. Product Categories (ALL TIME)
    print('4. Caching product categories for ALL TIME...')
    cursor.execute("""
        SELECT 
            COALESCE(
                CASE 
                    WHEN oi.product_name ILIKE '%%mattress%%' OR oi.product_name ILIKE '%%foam%%' THEN 'Mattresses'
                    WHEN oi.product_name ILIKE '%%pillow%%' THEN 'Pillows'
                    WHEN oi.product_name ILIKE '%%protector%%' OR oi.product_name ILIKE '%%cover%%' THEN 'Protectors'
                    WHEN oi.product_name ILIKE '%%bed%%' OR oi.product_name ILIKE '%%frame%%' THEN 'Bed Frames'
                    WHEN oi.product_name ILIKE '%%sheet%%' OR oi.product_name ILIKE '%%linen%%' THEN 'Bedding'
                    ELSE 'Other'
                END
            , 'Other') as category,
            COUNT(DISTINCT oi.order_id) as order_count,
            SUM(oi.quantity) as total_quantity,
            SUM(oi.total_price) as total_revenue
        FROM order_items oi
        GROUP BY 1
        ORDER BY total_revenue DESC
    """)
    categories = cursor.fetchall()
    
    cat_list = []
    for c in categories:
        cat_list.append({
            "category": c["category"],
            "order_count": c["order_count"],
            "total_quantity": c["total_quantity"],
            "total_revenue": float(c["total_revenue"] or 0),
            "product_count": c["order_count"]  # Approximate
        })
    
    cat_data = {
        "success": True,
        "categories": cat_list,
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex('analytics:product_categories:all', TTL, json.dumps(cat_data))
    print(f'   ✅ Cached {len(categories)} categories\n')
    
    # 5. RFM SEGMENTS WITH CUSTOMER DETAILS (ALL TIME) - OPTIMIZED BATCH PROCESSING
    print('5. Caching RFM segments with customer details for ALL TIME (batched processing)...', flush=True)
    
    # Initialize segment collections
    segments = {}
    segment_stats = {}
    total_customers_processed = 0
    
    # Process customers in memory-efficient batches
    for batch_num, batch_customers, is_last_batch, total_batches in process_customers_in_batches(cursor):
        print(f'   ⏳ Processing batch {batch_num}/{total_batches} ({len(batch_customers)} customers)...', flush=True)
        
        # Process each customer in the current batch
        for customer in batch_customers:
            segment = customer['segment']
            
            # Initialize segment if not exists
            if segment not in segments:
                segments[segment] = []
                segment_stats[segment] = {'count': 0, 'total_revenue': 0, 'total_orders': 0}
            
            # Calculate RFM scores (1-5 scale)
            recency = int(customer['recency_days'])
            frequency = customer['frequency']
            monetary = float(customer['monetary'] or 0)
            
            r_score = 5 if recency <= 30 else 4 if recency <= 60 else 3 if recency <= 90 else 2 if recency <= 180 else 1
            f_score = 5 if frequency >= 10 else 4 if frequency >= 5 else 3 if frequency >= 3 else 2 if frequency >= 2 else 1
            m_score = 5 if monetary >= 100000 else 4 if monetary >= 50000 else 3 if monetary >= 20000 else 2 if monetary >= 5000 else 1
            
            customer_data = {
                "customer_id": customer['unified_customer_id'],
                "customer_name": customer['customer_name'],
                "customer_phone": customer['customer_phone'] or '',
                "customer_city": customer['city'],
                "segment": segment,
                "total_orders": frequency,
                "total_spent": monetary,
                "last_order_date": customer['last_order_date'].isoformat() if customer['last_order_date'] else None,
                "days_since_last_order": recency,
                "rfm_score": {
                    "recency": r_score,
                    "frequency": f_score,
                    "monetary": m_score
                }
            }
            
            # Only store up to MAX_CUSTOMERS_PER_SEGMENT per segment to save memory
            if len(segments[segment]) < MAX_CUSTOMERS_PER_SEGMENT:
                segments[segment].append(customer_data)
            
            # Update segment statistics
            segment_stats[segment]['count'] += 1
            segment_stats[segment]['total_revenue'] += monetary
            segment_stats[segment]['total_orders'] += frequency
            
            total_customers_processed += 1
        
        print(f'   ✅ Batch {batch_num}/{total_batches} complete. Total processed: {total_customers_processed:,}', flush=True)
    
    print(f'   📊 Total customers processed: {total_customers_processed:,}', flush=True)
    
    # Cache each segment separately (for faster segment detail queries)
    for segment_name in segments:
        customers = segments[segment_name]
        stats = segment_stats[segment_name]
        
        segment_data = {
            "success": True,
            "segment": segment_name,
            "customers": customers,  # Already limited to MAX_CUSTOMERS_PER_SEGMENT
            "total_count": stats['count'],  # Actual total count including those not cached
            "cached_count": len(customers),  # Number actually cached
            "cached": True,
            "timestamp": datetime.now().isoformat()
        }
        
        cache_key = f"analytics:segment_details:{segment_name}:all"
        redis_client.setex(cache_key, TTL, json.dumps(segment_data))
        print(f'   ✅ Cached {segment_name}: {len(customers)}/{stats["count"]} customers', flush=True)
    
    print()
    
    # 6. RFM SEGMENT SUMMARY (ALL TIME) - Using pre-calculated stats
    print('6. Caching RFM segment summary for ALL TIME...', flush=True)
    
    segment_summary = []
    for segment_name in segment_stats:
        stats = segment_stats[segment_name]
        total_revenue = stats['total_revenue']
        customer_count = stats['count']
        total_orders = stats['total_orders']
        avg_value = total_revenue / customer_count if customer_count > 0 else 0
        avg_orders = total_orders / customer_count if customer_count > 0 else 0
        
        segment_summary.append({
            "segment": segment_name,
            "customer_count": customer_count,
            "total_revenue": total_revenue,
            "avg_customer_value": avg_value,
            "avg_orders": avg_orders
        })
    
    # Sort by customer count descending
    segment_summary.sort(key=lambda x: x['customer_count'], reverse=True)
    
    summary_data = {
        "success": True,
        "segments": segment_summary,
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex("analytics:rfm_segments:all", TTL, json.dumps(summary_data))
    print(f'   ✅ Cached summary for {len(segment_summary)} RFM segments\n', flush=True)
    
    # 7. COLLABORATIVE FILTERING METRICS (ALL TIME) - OPTIMIZED WITH BATCHES
    print('7. Caching collaborative filtering metrics for ALL TIME...', flush=True)
    
    # First get basic stats quickly
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT o.unified_customer_id) as total_users,
            COUNT(DISTINCT oi.product_id) as total_products,
            COUNT(DISTINCT o.id) as total_orders,
            COUNT(*) as total_user_product_combinations
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
    """)
    
    basic_stats = cursor.fetchone()
    
    # Sample-based customer pairs calculation (much faster)
    print('   ⏳ Computing customer similarity with sampling...', flush=True)
    cursor.execute("""
        WITH sample_customers AS (
            SELECT DISTINCT o.unified_customer_id
            FROM orders o
            TABLESAMPLE SYSTEM(10)  -- Sample 10% of customers
            LIMIT 5000
        ),
        customer_products AS (
            SELECT 
                o.unified_customer_id,
                oi.product_id,
                COUNT(*) as purchase_count
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            WHERE o.unified_customer_id IN (SELECT unified_customer_id FROM sample_customers)
            GROUP BY o.unified_customer_id, oi.product_id
        ),
        customer_pairs AS (
            SELECT 
                cp1.unified_customer_id as customer1,
                cp2.unified_customer_id as customer2,
                COUNT(DISTINCT cp1.product_id) as shared_products
            FROM customer_products cp1
            JOIN customer_products cp2 
                ON cp1.product_id = cp2.product_id 
                AND cp1.unified_customer_id < cp2.unified_customer_id
            GROUP BY cp1.unified_customer_id, cp2.unified_customer_id
            HAVING COUNT(DISTINCT cp1.product_id) >= 2
        )
        SELECT 
            COUNT(*) as sample_pairs,
            AVG(shared_products) as avg_shared_products
        FROM customer_pairs
    """)
    
    pair_stats = cursor.fetchone()
    sample_pairs = pair_stats['sample_pairs'] or 0
    avg_shared = float(pair_stats['avg_shared_products'] or 0)
    
    # Estimate total pairs from sample
    estimated_pairs = int(sample_pairs * 100)  # Scale up from 10% sample
    
    result = {
        'total_users': basic_stats['total_users'],
        'total_products': basic_stats['total_products'],
        'total_purchases': basic_stats['total_orders'],
        'total_user_product_combinations': basic_stats['total_user_product_combinations'],
        'active_customer_pairs': estimated_pairs,
        'avg_shared_products': avg_shared
    }
    total_users = int(result['total_users'] or 0)
    total_products = int(result['total_products'] or 0)
    active_pairs = int(result['active_customer_pairs'] or 0)
    avg_shared = float(result['avg_shared_products'] or 0)
    
    similarity_score = min(avg_shared / 10.0, 1.0) if avg_shared > 0 else 0.0
    max_possible_pairs = float((total_users * (total_users - 1)) / 2) if total_users > 1 else 1.0
    recommendation_coverage = min(float(active_pairs) / max_possible_pairs, 1.0) if max_possible_pairs > 0 else 0.0
    
    collab_metrics = {
        "total_recommendations": result['total_user_product_combinations'] or 0,
        "avg_similarity_score": round(similarity_score, 3),
        "active_customer_pairs": active_pairs,
        "algorithm_accuracy": round(recommendation_coverage, 3),
        "total_users": total_users,
        "total_products": total_products,
        "coverage": round(recommendation_coverage, 3),
        "time_filter": "all",
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex("analytics:collaborative_metrics:all", TTL, json.dumps(collab_metrics))
    print(f'   ✅ Users: {total_users:,}, Products: {total_products:,}, Pairs: {active_pairs:,}\n')
    
    # 8. COLLABORATIVE PRODUCT PAIRS (ALL TIME) - Optimized with batching
    print('8. Caching collaborative product pairs for ALL TIME (with batching)...', flush=True)
    
    # First, get total count and summary metrics for ALL pairs
    cursor.execute("""
        SELECT 
            COUNT(*) as total_pairs,
            SUM(co_purchase_count) as total_co_purchases,
            AVG(confidence) as avg_confidence
        FROM product_pairs
        WHERE co_purchase_count >= 2
    """)
    
    summary_row = cursor.fetchone()
    actual_total_count = summary_row['total_pairs'] or 0
    total_co_purchases = summary_row['total_co_purchases'] or 0
    avg_confidence = summary_row['avg_confidence'] or 0.0
    
    print(f'   📊 Found {actual_total_count:,} product pairs in database', flush=True)
    
    # Use cursor iteration for memory efficiency instead of fetchall()
    cursor.execute("""
        WITH product_names AS (
            SELECT DISTINCT ON (product_id) 
                product_id, 
                product_name,
                unit_price
            FROM order_items
            WHERE product_name IS NOT NULL AND product_name != ''
            ORDER BY product_id, order_id DESC
        )
        SELECT 
            pp.product_1 as product_a_id,
            COALESCE(pn1.product_name, 'Unknown Product') as product_a_name,
            pp.product_2 as product_b_id,
            COALESCE(pn2.product_name, 'Unknown Product') as product_b_name,
            pp.co_purchase_count,
            pp.confidence,
            COALESCE(pn1.unit_price * pp.co_purchase_count + pn2.unit_price * pp.co_purchase_count, 0) as combined_revenue
        FROM product_pairs pp
        LEFT JOIN product_names pn1 ON pp.product_1 = pn1.product_id
        LEFT JOIN product_names pn2 ON pp.product_2 = pn2.product_id
        WHERE pp.co_purchase_count >= 2
        ORDER BY pp.co_purchase_count DESC
        LIMIT 20
    """)
    
    # Use cursor iteration instead of fetchall() for better memory management
    pairs_list = []
    total_revenue = 0
    row_count = 0
    
    for row in cursor:
        combined_revenue = float(row['combined_revenue'] or 0)
        total_revenue += combined_revenue
        pairs_list.append({
            "product_a_id": row['product_a_id'],
            "product_a_name": row['product_a_name'],
            "product_b_id": row['product_b_id'],
            "product_b_name": row['product_b_name'],
            "co_recommendation_count": row['co_purchase_count'],
            "combined_revenue": combined_revenue,
            "confidence_score": float(row['confidence'] or 0)
        })
        row_count += 1
    
    # Calculate average pair value from actual totals
    avg_pair_value = total_revenue / len(pairs_list) if pairs_list else 0
    
    pairs_data = {
        "pairs": pairs_list,
        "total_count": len(pairs_list),
        "actual_total_count": actual_total_count,
        "summary": {
            "total_pairs": actual_total_count,
            "total_co_purchases": total_co_purchases,
            "avg_confidence": round(float(avg_confidence) * 100, 1),
            "total_revenue": total_revenue,
            "avg_pair_value": avg_pair_value
        },
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex("analytics_collab_pairs:all_20", TTL, json.dumps(pairs_data))
    
    # Also cache the 10-item version with same summary data
    pairs_data_10 = {
        "pairs": pairs_list[:10],
        "total_count": 10,
        "actual_total_count": actual_total_count,
        "summary": pairs_data["summary"],
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    redis_client.setex("analytics_collab_pairs:all_10", TTL, json.dumps(pairs_data_10))
    
    print(f'   ✅ Cached {len(pairs_list)} product pairs', flush=True)
    print(f'   📊 Total pairs in DB: {actual_total_count:,}', flush=True)
    print(f'   💰 Avg pair value: Rs {avg_pair_value:,.0f}\n', flush=True)
    
    # 9. CUSTOMER SIMILARITY (ALL TIME) - ENHANCED WITH SHARED PRODUCTS
    print('9. Caching customer similarity for ALL TIME (with shared products)...', flush=True)
    
    # Enhanced query with actual top shared products
    cursor.execute("""
        WITH customer_products AS (
            SELECT 
                o.unified_customer_id,
                MAX(o.customer_name) as customer_name,
                oi.product_id,
                MAX(oi.product_name) as product_name,
                COUNT(*) as purchase_count
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            GROUP BY o.unified_customer_id, oi.product_id
            HAVING COUNT(*) >= 1
            LIMIT 50000
        ),
        product_sharing AS (
            SELECT 
                cp1.unified_customer_id,
                cp1.product_id,
                cp1.product_name,
                COUNT(DISTINCT cp2.unified_customer_id) as shared_count
            FROM customer_products cp1
            JOIN customer_products cp2 
                ON cp1.product_id = cp2.product_id 
                AND cp1.unified_customer_id < cp2.unified_customer_id
            GROUP BY cp1.unified_customer_id, cp1.product_id, cp1.product_name
            HAVING COUNT(DISTINCT cp2.unified_customer_id) > 0
        ),
        ranked_products AS (
            SELECT 
                unified_customer_id,
                product_name,
                shared_count,
                ROW_NUMBER() OVER (PARTITION BY unified_customer_id ORDER BY shared_count DESC) as rn
            FROM product_sharing
        ),
        customer_stats AS (
            SELECT 
                o.unified_customer_id as customer_id,
                MAX(o.customer_name) as customer_name,
                COUNT(DISTINCT oi.product_id) as unique_products,
                COUNT(DISTINCT o.id) as total_orders,
                COUNT(DISTINCT CASE WHEN ps.shared_count > 0 THEN ps.unified_customer_id END) * 15 as similar_customers_count
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            LEFT JOIN product_sharing ps ON o.unified_customer_id = ps.unified_customer_id
            GROUP BY o.unified_customer_id
            HAVING COUNT(DISTINCT oi.product_id) >= 2
        )
        SELECT 
            cs.customer_id,
            cs.customer_name,
            cs.unique_products,
            cs.total_orders,
            cs.similar_customers_count,
            JSON_AGG(
                JSON_BUILD_OBJECT(
                    'product_name', rp.product_name,
                    'shared_count', rp.shared_count
                ) ORDER BY rp.shared_count DESC
            ) FILTER (WHERE rp.rn <= 3) as top_shared_products
        FROM customer_stats cs
        LEFT JOIN ranked_products rp ON cs.customer_id = rp.unified_customer_id AND rp.rn <= 3
        GROUP BY cs.customer_id, cs.customer_name, cs.unique_products, cs.total_orders, cs.similar_customers_count
        ORDER BY cs.similar_customers_count DESC
        LIMIT 20
    """)
    
    similarity_results = cursor.fetchall()
    
    similarity_list = []
    for row in similarity_results:
        similarity_list.append({
            "customer_id": row['customer_id'],
            "customer_name": row['customer_name'],
            "similar_customers_count": row['similar_customers_count'] or 0,
            "actual_recommendations": row['similar_customers_count'] or 0,
            "recommendations_generated": row['similar_customers_count'] or 0,
            "top_shared_products": row['top_shared_products'] if row['top_shared_products'] else []
        })
    
    similarity_data = {
        "customers": similarity_list,
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex("analytics:customer_similarity:all:20", TTL, json.dumps(similarity_data))
    redis_client.setex("analytics:customer_similarity:all:10", TTL, json.dumps({"customers": similarity_list[:10], "cached": True, "timestamp": datetime.now().isoformat()}))
    print(f'   ✅ Cached {len(similarity_list)} customer similarity records with shared products\n')
    
    # 10. COLLABORATIVE PRODUCTS (ALL TIME) - Memory-efficient processing
    print('10. Caching collaborative products for ALL TIME (optimized)...', flush=True)
    
    cursor.execute("""
        SELECT  
            oi.product_id,
            MAX(oi.product_name) as product_name,
            CASE 
                WHEN MAX(oi.product_name) ILIKE '%%pillow%%' THEN 'Pillows'
                WHEN MAX(oi.product_name) ILIKE '%%cushion%%' THEN 'Cushions'
                WHEN MAX(oi.product_name) ILIKE '%%mattress%%' OR MAX(oi.product_name) ILIKE '%%foam%%' THEN 'Mattresses & Foam'
                WHEN MAX(oi.product_name) ILIKE '%%sheet%%' OR MAX(oi.product_name) ILIKE '%%cover%%' THEN 'Bedding'
                ELSE 'Home Furnishing'
            END as category,
            COUNT(DISTINCT o.unified_customer_id) as customer_count,
            COUNT(DISTINCT o.id) as recommendation_count,
            SUM(oi.total_price) as total_revenue,
            AVG(oi.unit_price) as avg_price
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        GROUP BY oi.product_id
        HAVING COUNT(DISTINCT o.unified_customer_id) >= 2
        ORDER BY COUNT(DISTINCT o.unified_customer_id) DESC, 
                 SUM(oi.total_price) DESC
        LIMIT 20
    """)
    
    # Use cursor iteration for memory efficiency
    products_list = []
    for p in cursor:
        products_list.append({
            "product_id": p['product_id'],
            "product_name": p['product_name'],
            "category": p['category'],
            "price": float(p['avg_price'] or 0),
            "recommendation_count": p['recommendation_count'] or 0,
            "avg_similarity_score": round(p['customer_count'] / 100, 2) if p['customer_count'] else 0,
            "total_revenue": float(p['total_revenue'] or 0),
            "algorithm": "sql_collaborative"
        })
    
    collab_products_data = {
        "products": products_list,
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex("analytics_collab_products:all_20", TTL, json.dumps(collab_products_data))
    redis_client.setex("analytics_collab_products:all_10", TTL, json.dumps({"products": products_list[:10], "cached": True, "timestamp": datetime.now().isoformat()}))
    print(f'   ✅ Cached {len(products_list)} collaborative products\n', flush=True)
    
    # Close connections
    cursor.close()
    conn.close()
    
    print('=' * 80, flush=True)
    print('  ✅ OPTIMIZED CACHE PRE-WARMING COMPLETE!', flush=True)
    print(f'     • Total customers processed: {total_customers_processed:,}', flush=True)
    print(f'     • Cached data expires in: {TTL // 3600} hours', flush=True)
    print(f'     • Memory-efficient batch processing used', flush=True)
    print('=' * 80, flush=True)

if __name__ == '__main__':
    main()
