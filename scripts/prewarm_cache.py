#!/usr/bin/env python3
"""
Pre-warm Redis Cache for Heavy Queries
=======================================

This script pre-populates the Redis cache with results from commonly-requested
heavy queries (especially "all" time filter) to prevent server overload.

Now also caches OE/POS filtered data and common time filters.

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
    print('=' * 70, flush=True)
    print('  REDIS CACHE PRE-WARMING (with OE/POS filters)', flush=True)
    print('=' * 70, flush=True)
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
    
    # 5. RFM SEGMENTS WITH CUSTOMER DETAILS (ALL TIME)
    print('5. Caching RFM segments with customer details for ALL TIME...')
    
    # Query to get RFM metrics for all customers
    cursor.execute("""
        WITH customer_rfm AS (
            SELECT 
                o.unified_customer_id,
                MAX(o.customer_name) as customer_name,
                MAX(o.customer_city) as city,
                MAX(o.province) as province,
                EXTRACT(days FROM NOW() - MAX(o.order_date)) as recency_days,
                COUNT(DISTINCT o.id) as frequency,
                SUM(o.total_price) as monetary,
                MAX(o.order_date) as last_order_date
            FROM orders o
            GROUP BY o.unified_customer_id
        ),
        segmented AS (
            SELECT 
                unified_customer_id,
                customer_name,
                city,
                province,
                recency_days,
                frequency,
                monetary,
                last_order_date,
                CASE 
                    WHEN recency_days <= 30 AND frequency >= 5 AND monetary >= 50000 THEN 'Champions'
                    WHEN recency_days <= 60 AND frequency >= 3 AND monetary >= 20000 THEN 'Loyal'
                    WHEN recency_days <= 90 AND frequency >= 2 THEN 'Potential'
                    WHEN frequency = 1 AND recency_days <= 30 THEN 'New'
                    WHEN recency_days > 90 AND recency_days <= 180 AND frequency >= 2 THEN 'At Risk'
                    WHEN recency_days > 180 AND recency_days <= 365 THEN 'Hibernating'
                    WHEN recency_days > 365 THEN 'Lost'
                    ELSE 'Regular'
                END as segment
            FROM customer_rfm
        )
        SELECT * FROM segmented
        ORDER BY segment, monetary DESC
    """)
    
    all_customers = cursor.fetchall()
    
    # Group by segment
    segments = {}
    for customer in all_customers:
        segment = customer['segment']
        if segment not in segments:
            segments[segment] = []
        
        # Calculate RFM scores (1-5 scale)
        recency = int(customer['recency_days'])
        frequency = customer['frequency']
        monetary = float(customer['monetary'] or 0)
        
        r_score = 5 if recency <= 30 else 4 if recency <= 60 else 3 if recency <= 90 else 2 if recency <= 180 else 1
        f_score = 5 if frequency >= 10 else 4 if frequency >= 5 else 3 if frequency >= 3 else 2 if frequency >= 2 else 1
        m_score = 5 if monetary >= 100000 else 4 if monetary >= 50000 else 3 if monetary >= 20000 else 2 if monetary >= 5000 else 1
        
        segments[segment].append({
            "customer_id": customer['unified_customer_id'],
            "customer_name": customer['customer_name'],
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
        })
    
    # Cache each segment separately (for faster segment detail queries)
    for segment_name, customers in segments.items():
        segment_data = {
            "success": True,
            "segment": segment_name,
            "customers": customers[:100],  # Cache top 100 per segment
            "total_count": len(customers),
            "cached": True,
            "timestamp": datetime.now().isoformat()
        }
        
        cache_key = f"analytics:segment_details:{segment_name}:all"
        redis_client.setex(cache_key, TTL, json.dumps(segment_data))
        print(f'   ✅ Cached {segment_name}: {len(customers)} customers')
    
    print()
    
    # 6. RFM SEGMENT SUMMARY (ALL TIME)
    print('6. Caching RFM segment summary for ALL TIME...')
    
    segment_summary = []
    for segment_name in segments:
        customers = segments[segment_name]
        total_revenue = sum(c['total_spent'] for c in customers)
        avg_value = total_revenue / len(customers) if customers else 0
        
        segment_summary.append({
            "segment": segment_name,
            "customer_count": len(customers),
            "total_revenue": total_revenue,
            "avg_customer_value": avg_value,
            "avg_orders": sum(c['total_orders'] for c in customers) / len(customers) if customers else 0
        })
    
    summary_data = {
        "success": True,
        "segments": segment_summary,
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex("analytics:rfm_segments:all", TTL, json.dumps(summary_data))
    print(f'   ✅ Cached summary for {len(segment_summary)} RFM segments\n')
    
    # 7. COLLABORATIVE FILTERING METRICS (ALL TIME)
    print('7. Caching collaborative filtering metrics for ALL TIME...')
    
    cursor.execute("""
        WITH customer_products AS (
            SELECT 
                o.unified_customer_id,
                oi.product_id,
                COUNT(*) as purchase_count
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            GROUP BY o.unified_customer_id, oi.product_id
        ),
        customer_pairs AS (
            SELECT DISTINCT
                cp1.unified_customer_id as customer1,
                cp2.unified_customer_id as customer2,
                COUNT(DISTINCT cp1.product_id) as shared_products
            FROM customer_products cp1
            JOIN customer_products cp2 
                ON cp1.product_id = cp2.product_id 
                AND cp1.unified_customer_id < cp2.unified_customer_id
            GROUP BY cp1.unified_customer_id, cp2.unified_customer_id
            HAVING COUNT(DISTINCT cp1.product_id) >= 2
        ),
        stats AS (
            SELECT 
                COUNT(DISTINCT cp.unified_customer_id) as total_users,
                COUNT(DISTINCT cp.product_id) as total_products,
                SUM(cp.purchase_count) as total_purchases,
                COUNT(*) as total_user_product_combinations
            FROM customer_products cp
        ),
        pair_stats AS (
            SELECT 
                COUNT(*) as total_pairs,
                AVG(shared_products) as avg_shared_products
            FROM customer_pairs
        )
        SELECT 
            s.total_users,
            s.total_products,
            s.total_purchases,
            s.total_user_product_combinations,
            COALESCE(ps.total_pairs, 0) as active_customer_pairs,
            COALESCE(ps.avg_shared_products, 0) as avg_shared_products
        FROM stats s
        CROSS JOIN pair_stats ps
    """)
    
    result = cursor.fetchone()
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
    
    # 8. COLLABORATIVE PRODUCT PAIRS (ALL TIME) - Optimized with subqueries
    print('8. Caching collaborative product pairs for ALL TIME...')
    
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
    
    # Use subqueries to efficiently get product names without slow joins
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
    
    pairs_results = cursor.fetchall()
    
    pairs_list = []
    total_revenue = 0
    for row in pairs_results:
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
    
    print(f'   ✅ Cached {len(pairs_list)} product pairs')
    print(f'   📊 Total pairs in DB: {actual_total_count:,}')
    print(f'   💰 Avg pair value: Rs {avg_pair_value:,.0f}\n')
    
    # 9. CUSTOMER SIMILARITY (ALL TIME)
    print('9. Caching customer similarity for ALL TIME...')
    
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
        ),
        customer_stats AS (
            SELECT 
                cp.unified_customer_id,
                MAX(cp.customer_name) as customer_name,
                COUNT(DISTINCT cp.product_id) as unique_products,
                SUM(cp.purchase_count) as total_purchases
            FROM customer_products cp
            GROUP BY cp.unified_customer_id
        ),
        similar_customers AS (
            SELECT 
                cp1.unified_customer_id,
                COUNT(DISTINCT cp2.unified_customer_id) as similar_customers_count
            FROM customer_products cp1
            LEFT JOIN customer_products cp2 
                ON cp1.product_id = cp2.product_id 
                AND cp1.unified_customer_id != cp2.unified_customer_id
            GROUP BY cp1.unified_customer_id
        )
        SELECT 
            cs.unified_customer_id as customer_id,
            cs.customer_name,
            cs.unique_products,
            cs.total_purchases,
            COALESCE(sc.similar_customers_count, 0) as similar_customers_count
        FROM customer_stats cs
        LEFT JOIN similar_customers sc ON cs.unified_customer_id = sc.unified_customer_id
        WHERE cs.unique_products >= 2
        ORDER BY sc.similar_customers_count DESC, cs.total_purchases DESC
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
            "top_shared_products": []
        })
    
    similarity_data = {
        "customers": similarity_list,
        "cached": True,
        "timestamp": datetime.now().isoformat()
    }
    
    redis_client.setex("analytics:customer_similarity:all:20", TTL, json.dumps(similarity_data))
    redis_client.setex("analytics:customer_similarity:all:10", TTL, json.dumps({"customers": similarity_list[:10], "cached": True, "timestamp": datetime.now().isoformat()}))
    print(f'   ✅ Cached {len(similarity_list)} customer similarity records\n')
    
    # 10. COLLABORATIVE PRODUCTS (ALL TIME)
    print('10. Caching collaborative products for ALL TIME...')
    
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
    
    collab_products = cursor.fetchall()
    
    products_list = []
    for p in collab_products:
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
    print(f'   ✅ Cached {len(products_list)} collaborative products\n')
    
    # Close connections
    cursor.close()
    conn.close()
    
    print('=' * 70)
    print('  ✅ CACHE PRE-WARMING COMPLETE!')
    print(f'  All cached data will expire in {TTL // 3600} hours')
    print('=' * 70)

if __name__ == '__main__':
    main()
