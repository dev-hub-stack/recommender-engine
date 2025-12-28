#!/usr/bin/env python3
"""
LOCAL ML RECOMMENDATION PIPELINE
================================

Complete replacement for AWS Personalize using local ML models.

Pipeline Steps (same as AWS Personalize):
1. Fetch fresh data from Master Group APIs (OE + POS)
2. Sync orders to PostgreSQL database
3. Export data to CSV files (interactions, items, users)
4. Train local ML models (SVD, Item Similarity, Popularity)
5. Generate batch recommendations for ALL users
6. Store results in offline cache tables

The existing /api/v1/personalize/* endpoints will serve from cache.

Usage:
    python scripts/local_ml_pipeline.py                    # Full pipeline
    python scripts/local_ml_pipeline.py --skip-sync        # Skip API sync, use existing DB
    python scripts/local_ml_pipeline.py --export-only      # Only export to CSV
    python scripts/local_ml_pipeline.py --train-only       # Only train (skip sync & export)

Schedule (cron):
    # Daily at 2 AM (same as AWS Personalize)
    0 2 * * * cd /opt/mastergroup-api && python3 scripts/local_ml_pipeline.py
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

import requests
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = project_root / 'data' / 'local_ml'
MODELS_DIR = project_root / 'models'

# Master Group API Config
MASTER_API_BASE = os.getenv('MASTER_GROUP_API_BASE', 'https://mes.master.com.pk')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_db_connection():
    """Get PostgreSQL connection"""
    host = os.getenv('PG_HOST', 'localhost')
    sslmode = os.getenv('PG_SSLMODE', 'prefer' if host == 'localhost' else 'require')
    return psycopg2.connect(
        host=host,
        port=int(os.getenv('PG_PORT', '5432')),
        database=os.getenv('PG_DB', 'mastergroup_recommendations'),
        user=os.getenv('PG_USER', 'postgres'),
        password=os.getenv('PG_PASSWORD', ''),
        sslmode=sslmode
    )


# =============================================================================
# STEP 1: FETCH DATA FROM MASTER GROUP APIs
# =============================================================================

def fetch_from_master_apis(days: int = 7) -> Dict:
    """
    Fetch fresh orders from Master Group OE and POS APIs
    """
    logger.info(f"📡 Fetching data from Master Group APIs (last {days} days)...")
    
    results = {
        'oe_orders': 0,
        'pos_orders': 0,
        'errors': []
    }
    
    # Batch fetching configuration
    BATCH_DAYS = 90
    
    total_oe = 0
    total_pos = 0
    
    # Calculate batches
    current_end = datetime.now()
    final_start = current_end - timedelta(days=days)
    
    current_batch_end = current_end
    
    logger.info(f"  🔄 Batch processing: Fetching data in {BATCH_DAYS}-day chunks...")
    
    while current_batch_end > final_start:
        current_batch_start = current_batch_end - timedelta(days=BATCH_DAYS)
        if current_batch_start < final_start:
            current_batch_start = final_start
            
        start_str = current_batch_start.strftime('%Y-%m-%d')
        end_str = current_batch_end.strftime('%Y-%m-%d')
        
        logger.info(f"    Chunk: {start_str} to {end_str}")
        
        end_date = end_str
        start_date = start_str
        
        # Auth headers
        headers = {
            'Authorization': os.getenv('MASTER_GROUP_AUTH_TOKEN', ''),
            'Content-Type': 'application/json'
        }
        
        # Fetch OE Orders
        try:
            url = f"{MASTER_API_BASE}/get_oe_orders"
            response = requests.get(
                url, 
                params={'start_date': start_date, 'end_date': end_date},
                headers=headers,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                orders = data if isinstance(data, list) else data.get('data', [])
                
                # Insert into database
                if orders:
                    insert_orders_to_db(orders, source='OE')
                    total_oe += len(orders)
            else:
                logger.error(f"      ❌ OE API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"      ❌ OE API exception: {e}")

        # Fetch POS Orders
        try:
            url = f"{MASTER_API_BASE}/get_pos_orders"
            response = requests.get(
                url,
                params={'start_date': start_date, 'end_date': end_date},
                headers=headers,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                orders = data if isinstance(data, list) else data.get('data', [])
                
                # Insert into database
                if orders:
                    insert_orders_to_db(orders, source='POS')
                    total_pos += len(orders)
            else:
                logger.error(f"      ❌ POS API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"      ❌ POS API exception: {e}")
            
        # Move to next batch
        current_batch_end = current_batch_start
        
    results['oe_orders'] = total_oe
    results['pos_orders'] = total_pos
    logger.info(f"  ✅ Total OE: {total_oe}, Total POS: {total_pos}")
    
    return results


def insert_orders_to_db(orders: List[Dict], source: str):
    """Insert orders into PostgreSQL (upsert)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    inserted_count = 0
    for order in orders:
        try:
            # Extract order data - handle both OE and POS API formats
            order_id = order.get('order_id') or order.get('id') or order.get('order_name')
            customer_id = order.get('customer_id') or order.get('customer_phone') or order.get('phone')
            customer_name = order.get('customer_name') or order.get('name', '')
            customer_city = order.get('customer_city') or order.get('city', '')
            customer_phone = order.get('customer_phone') or order.get('phone', '')
            order_date = order.get('order_date') or order.get('date')
            total = order.get('total') or order.get('total_price', 0)
            order_status = order.get('order_status', '')
            brand_name = order.get('brand_name', '')
            payment_mode = order.get('payment_mode', '')
            
            # Convert total to float if string
            if isinstance(total, str):
                total = float(total.replace(',', '')) if total else 0
            
            # Skip if missing required fields
            if not order_id or not customer_id:
                logger.debug(f"Skipping order - missing order_id or customer_id: {order.get('id')}")
                continue
            
            # Generate unified customer ID
            first_name = customer_name.split()[0].lower() if customer_name and customer_name.split() else 'customer'
            unified_id = f"{customer_id}_{first_name}"
            
            # Upsert order with all fields
            cursor.execute("""
                INSERT INTO orders (id, unified_customer_id, customer_name, customer_phone, customer_city, 
                                   order_date, total_price, order_status, brand_name, payment_mode, order_type, source_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    unified_customer_id = EXCLUDED.unified_customer_id,
                    customer_name = EXCLUDED.customer_name,
                    customer_phone = EXCLUDED.customer_phone,
                    customer_city = EXCLUDED.customer_city,
                    total_price = EXCLUDED.total_price,
                    order_status = EXCLUDED.order_status,
                    brand_name = EXCLUDED.brand_name,
                    payment_mode = EXCLUDED.payment_mode,
                    order_type = EXCLUDED.order_type,
                    source_type = EXCLUDED.source_type,
                    updated_at = NOW()
            """, (str(order_id), unified_id, customer_name, customer_phone, customer_city,
                  order_date, total, order_status, brand_name, payment_mode, source, source))
            
            inserted_count += 1
            
            # Insert order items - handle different API formats
            items = order.get('items') or order.get('order_items') or order.get('has_items', [])
            for item in items:
                # Get product_id - OE uses 'id' inside has_items, could also be 'product_id'
                product_id = item.get('product_id') or item.get('id') or item.get('sku')
                product_name = item.get('product_name') or item.get('name') or item.get('title', '')
                sku = item.get('sku', '')
                product_type = item.get('product_type', '')
                
                # Use SKU as product_id if available (more unique)
                if sku:
                    product_id = sku
                
                # If SKU exists, append to name if not already there
                if sku and sku not in product_name:
                    product_name = f"{product_name} ({sku})"
                
                quantity = item.get('quantity', 1)
                price = item.get('price') or item.get('unit_price') or item.get('base_price', 0)
                
                # Convert price to float if string
                if isinstance(price, str):
                    price = float(price.replace(',', '')) if price else 0
                
                if product_id:
                    cursor.execute("""
                        INSERT INTO order_items (order_id, product_id, product_name, quantity, unit_price, total_price)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (order_id, product_id) DO UPDATE SET
                            product_name = EXCLUDED.product_name,
                            quantity = EXCLUDED.quantity,
                            unit_price = EXCLUDED.unit_price,
                            total_price = EXCLUDED.total_price
                    """, (str(order_id), str(product_id), product_name, quantity, price, price * quantity))
                    
        except Exception as e:
            logger.warning(f"Order insert error for {order.get('id')}: {e}")
            continue
    
    conn.commit()
    cursor.close()
    conn.close()
    logger.debug(f"  Inserted/updated {inserted_count} orders from {source}")


# =============================================================================
# STEP 2: EXPORT DATA TO CSV (Same format as AWS Personalize)
# =============================================================================

def export_to_csv() -> Dict:
    """
    Export data to CSV files (same format as AWS Personalize export)
    Creates: interactions.csv, items.csv, users.csv
    """
    logger.info("📄 Exporting data to CSV files...")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {}
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # 1. Export Interactions
    logger.info("  Exporting interactions...")
    cursor.execute("""
        SELECT 
            o.unified_customer_id as user_id,
            oi.product_id as item_id,
            EXTRACT(EPOCH FROM o.order_date)::bigint as timestamp,
            'purchase' as event_type,
            oi.quantity as event_value
        FROM orders o
        JOIN order_items oi ON o.id::text = oi.order_id
        WHERE o.unified_customer_id IS NOT NULL 
        AND oi.product_id IS NOT NULL
        AND o.order_date IS NOT NULL
        ORDER BY o.order_date
    """)
    rows = cursor.fetchall()
    
    filepath = DATA_DIR / f'interactions_{timestamp}.csv'
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['USER_ID', 'ITEM_ID', 'TIMESTAMP', 'EVENT_TYPE', 'EVENT_VALUE'])
        for row in rows:
            writer.writerow([row['user_id'], row['item_id'], row['timestamp'], row['event_type'], row['event_value'] or 1])
    
    # Also save as 'latest'
    latest_path = DATA_DIR / 'interactions_latest.csv'
    with open(latest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['USER_ID', 'ITEM_ID', 'TIMESTAMP', 'EVENT_TYPE', 'EVENT_VALUE'])
        for row in rows:
            writer.writerow([row['user_id'], row['item_id'], row['timestamp'], row['event_type'], row['event_value'] or 1])
    
    results['interactions'] = len(rows)
    logger.info(f"  ✅ Exported {len(rows):,} interactions")
    
    # 2. Export Items
    logger.info("  Exporting items...")
    
    # Get all items first
    cursor.execute("""
        SELECT DISTINCT
            oi.product_id as item_id,
            MAX(oi.product_name) as item_name,
            AVG(oi.unit_price) as price,
            COUNT(DISTINCT oi.order_id) as purchase_count,
            MAX(o.order_type) as last_source
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        WHERE oi.product_id IS NOT NULL
        GROUP BY oi.product_id
    """)
    rows = cursor.fetchall()
    
    # Import category extraction logic
    from src.main import extract_smart_category

    filepath = DATA_DIR / f'items_{timestamp}.csv'
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ITEM_ID', 'ITEM_NAME', 'CATEGORY', 'PRICE', 'PURCHASE_COUNT'])
        for row in rows:
            item_name = (row['item_name'] or 'Unknown')
            order_source = row['last_source'] if row['last_source'] in ['pos', 'oe'] else 'pos'
            category = extract_smart_category(item_name, None, order_source)
            
            writer.writerow([
                row['item_id'],
                item_name[:256],
                category,
                round(float(row['price'] or 0), 2),
                row['purchase_count']
            ])
    
    latest_path = DATA_DIR / 'items_latest.csv'
    with open(latest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ITEM_ID', 'ITEM_NAME', 'CATEGORY', 'PRICE', 'PURCHASE_COUNT'])
        for row in rows:
            item_name = (row['item_name'] or 'Unknown')
            order_source = row['last_source'] if row['last_source'] in ['pos', 'oe'] else 'pos'
            category = extract_smart_category(item_name, None, order_source)
            
            writer.writerow([
                row['item_id'],
                item_name[:256],
                category,
                round(float(row['price'] or 0), 2),
                row['purchase_count']
            ])
    
    results['items'] = len(rows)
    logger.info(f"  ✅ Exported {len(rows):,} items")
    
    # 3. Export Users
    logger.info("  Exporting users...")
    cursor.execute("""
        SELECT 
            o.unified_customer_id as user_id,
            MAX(o.customer_city) as city,
            MAX(o.province) as province,
            COUNT(DISTINCT o.id) as order_count,
            COALESCE(SUM(o.total_price), 0) as total_spend
        FROM orders o
        WHERE o.unified_customer_id IS NOT NULL
        AND TRIM(o.unified_customer_id) != ''
        GROUP BY o.unified_customer_id
    """)
    rows = cursor.fetchall()
    
    filepath = DATA_DIR / f'users_{timestamp}.csv'
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['USER_ID', 'CITY', 'PROVINCE', 'ORDER_COUNT', 'TOTAL_SPEND'])
        for row in rows:
            writer.writerow([
                row['user_id'],
                (row['city'] or 'Unknown')[:256],
                (row['province'] or 'Unknown')[:256],
                row['order_count'],
                round(float(row['total_spend'] or 0), 2)
            ])
    
    latest_path = DATA_DIR / 'users_latest.csv'
    with open(latest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['USER_ID', 'CITY', 'PROVINCE', 'ORDER_COUNT', 'TOTAL_SPEND'])
        for row in rows:
            writer.writerow([
                row['user_id'],
                (row['city'] or 'Unknown')[:256],
                (row['province'] or 'Unknown')[:256],
                row['order_count'],
                round(float(row['total_spend'] or 0), 2)
            ])
    
    results['users'] = len(rows)
    logger.info(f"  ✅ Exported {len(rows):,} users")
    
    cursor.close()
    conn.close()
    
    logger.info(f"  📁 Files saved to: {DATA_DIR}")
    return results


# =============================================================================
# STEP 3: TRAIN LOCAL ML MODELS
# =============================================================================

def train_models() -> Dict:
    """
    Train local ML models (SVD, Item Similarity, Popularity)
    Uses the exported CSV data or loads directly from DB
    """
    logger.info("🤖 Training local ML models...")
    
    results = {}
    
    # Load interaction data
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT 
            o.unified_customer_id as user_id,
            oi.product_id as item_id,
            oi.product_name as item_name,
            o.order_date as timestamp,
            oi.quantity
        FROM orders o
        JOIN order_items oi ON o.id::text = oi.order_id
        WHERE o.unified_customer_id IS NOT NULL
        AND oi.product_id IS NOT NULL
        ORDER BY o.order_date
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not rows:
        logger.error("No training data found!")
        return {'error': 'No training data'}
    
    df = pd.DataFrame(rows)
    logger.info(f"  Loaded {len(df):,} interactions ({df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items)")
    
    models = {}
    
    # 1. Build user-item matrix
    logger.info("  Building user-item matrix...")
    user_item = df.groupby(['user_id', 'item_id'])['quantity'].sum().reset_index()
    
    users = user_item['user_id'].unique()
    items = user_item['item_id'].unique()
    
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {i: idx for idx, i in enumerate(items)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_item = {i: item for item, i in item_to_idx.items()}
    item_names = df.groupby('item_id')['item_name'].first().to_dict()
    
    models['user_to_idx'] = user_to_idx
    models['item_to_idx'] = item_to_idx
    models['idx_to_user'] = idx_to_user
    models['idx_to_item'] = idx_to_item
    models['item_names'] = item_names
    
    rows_idx = [user_to_idx[u] for u in user_item['user_id']]
    cols_idx = [item_to_idx[i] for i in user_item['item_id']]
    values = user_item['quantity'].values
    
    user_item_matrix = csr_matrix((values, (rows_idx, cols_idx)), shape=(len(users), len(items)))
    models['user_item_matrix'] = user_item_matrix
    
    # 2. Train Item Similarity
    logger.info("  Training item similarity model...")
    item_similarity = cosine_similarity(user_item_matrix.T)
    models['item_similarity'] = item_similarity
    results['item_similarity'] = {'n_items': len(items), 'matrix_shape': list(item_similarity.shape)}
    
    # 3. Train Popularity model
    logger.info("  Training popularity model...")
    df['days_ago'] = (datetime.now() - pd.to_datetime(df['timestamp'])).dt.days
    df['recency_weight'] = np.exp(-df['days_ago'] / 30)
    popularity = df.groupby('item_id').apply(lambda x: (x['quantity'] * x['recency_weight']).sum()).sort_values(ascending=False)
    models['popularity'] = popularity.to_dict()
    results['popularity'] = {'n_items': len(popularity)}
    
    # 4. User purchase history
    logger.info("  Building user purchase history...")
    user_items = df.groupby('user_id')['item_id'].apply(set).to_dict()
    models['user_items'] = user_items
    
    # 5. Train SVD
    try:
        from surprise import SVD, Dataset, Reader
        
        logger.info("  Training SVD model...")
        reader = Reader(rating_scale=(1, df['quantity'].max()))
        surprise_data = Dataset.load_from_df(user_item[['user_id', 'item_id', 'quantity']], reader)
        trainset = surprise_data.build_full_trainset()
        
        svd = SVD(n_factors=50, n_epochs=20, random_state=42)
        svd.fit(trainset)
        
        models['svd'] = svd
        models['svd_trainset'] = trainset
        results['svd'] = {'n_factors': 50, 'n_users': len(users), 'n_items': len(items)}
        logger.info("  ✅ SVD trained")
        
    except ImportError:
        logger.warning("  ⚠️ Surprise not installed, skipping SVD")
        models['svd'] = None
    
    results['models'] = models
    results['n_users'] = len(users)
    results['n_items'] = len(items)
    
    return results


# =============================================================================
# STEP 4: GENERATE BATCH RECOMMENDATIONS & SAVE TO CACHE
# =============================================================================

def generate_and_cache_recommendations(models: Dict, user_limit: int = 25, item_limit: int = 20) -> Dict:
    """
    Generate recommendations for ALL users and save to PostgreSQL cache tables.
    Same output format as AWS Personalize batch results.
    """
    logger.info("📋 Generating batch recommendations...")
    
    results = {}
    
    # Generate user recommendations
    logger.info(f"  Generating user recommendations (top {user_limit} per user)...")
    user_recommendations = {}
    
    users = list(models['user_to_idx'].keys())
    svd = models.get('svd')
    trainset = models.get('svd_trainset')
    item_similarity = models['item_similarity']
    popularity = models['popularity']
    user_items = models['user_items']
    idx_to_item = models['idx_to_item']
    item_to_idx = models['item_to_idx']
    item_names = models['item_names']
    
    total = len(users)
    for i, user_id in enumerate(users):
        if i % 2000 == 0:
            logger.info(f"    Processing user {i+1:,}/{total:,} ({100*i/total:.1f}%)")
        
        user_recs = []
        purchased = user_items.get(user_id, set())
        
        # SVD predictions
        if svd and trainset:
            try:
                for item_id in item_to_idx.keys():
                    if item_id not in purchased:
                        try:
                            pred = svd.predict(user_id, item_id)
                            user_recs.append({'item_id': item_id, 'score': pred.est, 'algorithm': 'SVD'})
                        except:
                            pass
            except:
                pass
        
        # Item-based collaborative
        if len(user_recs) < user_limit:
            for purchased_item in list(purchased)[:10]:
                if purchased_item not in item_to_idx:
                    continue
                item_idx = item_to_idx[purchased_item]
                similarities = item_similarity[item_idx]
                for sim_idx in np.argsort(similarities)[::-1][:20]:
                    sim_item = idx_to_item[sim_idx]
                    if sim_item not in purchased and similarities[sim_idx] > 0.01:
                        user_recs.append({'item_id': sim_item, 'score': float(similarities[sim_idx]), 'algorithm': 'ItemSimilarity'})
        
        # Popularity fallback
        if len(user_recs) < user_limit:
            max_pop = max(popularity.values()) if popularity else 1
            for item_id, score in sorted(popularity.items(), key=lambda x: -x[1])[:user_limit]:
                if item_id not in purchased:
                    user_recs.append({'item_id': item_id, 'score': score / max_pop, 'algorithm': 'Popularity'})
        
        # Deduplicate and limit
        seen = set()
        unique_recs = []
        for rec in sorted(user_recs, key=lambda x: -x['score']):
            if rec['item_id'] not in seen:
                seen.add(rec['item_id'])
                rec['item_name'] = item_names.get(rec['item_id'], rec['item_id'])
                unique_recs.append(rec)
                if len(unique_recs) >= user_limit:
                    break
        
        user_recommendations[user_id] = unique_recs
    
    logger.info(f"  ✅ Generated recommendations for {len(user_recommendations):,} users")
    
    # Generate similar items
    logger.info(f"  Generating similar items (top {item_limit} per product)...")
    similar_items = {}
    
    total = len(item_to_idx)
    for i, (item_id, item_idx) in enumerate(item_to_idx.items()):
        if i % 500 == 0:
            logger.info(f"    Processing item {i+1:,}/{total:,} ({100*i/total:.1f}%)")
        
        similarities = item_similarity[item_idx]
        item_sims = []
        for sim_idx in np.argsort(similarities)[::-1][1:item_limit+1]:
            sim_item_id = idx_to_item[sim_idx]
            score = float(similarities[sim_idx])
            if score > 0.001:
                item_sims.append({
                    'item_id': sim_item_id,
                    'item_name': item_names.get(sim_item_id, sim_item_id),
                    'score': score
                })
        similar_items[item_id] = item_sims
    
    logger.info(f"  ✅ Generated similar items for {len(similar_items):,} products")
    
    # Save to cache tables
    logger.info("💾 Saving to PostgreSQL cache tables...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    # Save user recommendations
    cursor.execute("TRUNCATE TABLE offline_user_recommendations")
    user_rows = []
    for user_id, recs in user_recommendations.items():
        user_rows.append((user_id, json.dumps(recs, cls=NumpyEncoder), 'local_ml_hybrid', now))
    
    execute_values(cursor, """
        INSERT INTO offline_user_recommendations (user_id, recommendations, recipe_name, updated_at)
        VALUES %s
    """, user_rows, page_size=1000)
    
    logger.info(f"  ✅ Saved {len(user_rows):,} user recommendations")
    
    # Save similar items
    cursor.execute("TRUNCATE TABLE offline_similar_items")
    item_rows = []
    for item_id, sims in similar_items.items():
        item_rows.append((item_id, json.dumps(sims, cls=NumpyEncoder), 'local_ml_similarity', now))
    
    execute_values(cursor, """
        INSERT INTO offline_similar_items (product_id, similar_products, recipe_name, updated_at)
        VALUES %s
    """, item_rows, page_size=1000)
    
    logger.info(f"  ✅ Saved {len(item_rows):,} similar item sets")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    results['users_cached'] = len(user_rows)
    results['items_cached'] = len(item_rows)
    
    return results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(skip_sync: bool = False, export_only: bool = False, train_only: bool = False, sync_days: int = 7):
    """Run the complete local ML pipeline"""
    
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("  LOCAL ML RECOMMENDATION PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\n  This pipeline replaces AWS Personalize with local ML models.")
    print("  Same output format, same cache tables, same API endpoints.\n")
    
    results = {
        'started_at': start_time.isoformat(),
        'steps': {}
    }
    
    # Step 1: Sync from Master APIs
    if not skip_sync and not train_only:
        print("\n" + "-" * 70)
        print("  STEP 1: SYNC FROM MASTER GROUP APIs")
        print("-" * 70)
        sync_result = fetch_from_master_apis(days=sync_days)
        results['steps']['sync'] = sync_result
        print(f"\n  📊 OE Orders: {sync_result['oe_orders']:,}")
        print(f"  📊 POS Orders: {sync_result['pos_orders']:,}")
    else:
        print("\n  ⏭️  Skipping API sync (using existing database)")
    
    # Step 2: Export to CSV
    if not train_only:
        print("\n" + "-" * 70)
        print("  STEP 2: EXPORT DATA TO CSV")
        print("-" * 70)
        export_result = export_to_csv()
        results['steps']['export'] = export_result
        print(f"\n  📄 Interactions: {export_result['interactions']:,}")
        print(f"  📄 Items: {export_result['items']:,}")
        print(f"  📄 Users: {export_result['users']:,}")
    
    if export_only:
        print("\n  ⏭️  Export only mode - stopping here")
        return results
    
    # Step 3: Train Models
    print("\n" + "-" * 70)
    print("  STEP 3: TRAIN LOCAL ML MODELS")
    print("-" * 70)
    train_result = train_models()
    
    if 'error' in train_result:
        print(f"\n  ❌ Training failed: {train_result['error']}")
        return results
    
    results['steps']['train'] = {
        'n_users': train_result['n_users'],
        'n_items': train_result['n_items'],
        'svd': train_result.get('svd'),
        'item_similarity': train_result.get('item_similarity'),
        'popularity': train_result.get('popularity')
    }
    print(f"\n  🤖 Users: {train_result['n_users']:,}")
    print(f"  🤖 Items: {train_result['n_items']:,}")
    print(f"  🤖 SVD: {'✅' if train_result.get('svd') else '❌'}")
    
    # Step 4: Generate & Cache Recommendations
    print("\n" + "-" * 70)
    print("  STEP 4: GENERATE BATCH RECOMMENDATIONS & CACHE")
    print("-" * 70)
    cache_result = generate_and_cache_recommendations(train_result['models'])
    results['steps']['cache'] = cache_result
    print(f"\n  💾 Users with recommendations: {cache_result['users_cached']:,}")
    print(f"  💾 Products with similar items: {cache_result['items_cached']:,}")
    
    # Step 5: Pre-warm Redis cache for heavy queries
    print("\n" + "-" * 70)
    print("  STEP 5: PRE-WARM REDIS CACHE")
    print("-" * 70)
    try:
        import subprocess
        prewarm_script = project_root / 'scripts' / 'prewarm_cache.py'
        subprocess.run([sys.executable, str(prewarm_script)], check=True)
        print("\n  ✅ Cache pre-warming complete")
    except Exception as e:
        logger.warning(f"Cache pre-warming failed: {e}")
        print(f"\n  ⚠️  Cache pre-warming failed (non-critical): {e}")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    results['duration_seconds'] = duration
    results['completed_at'] = datetime.now().isoformat()
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE ✅")
    print("=" * 70)
    print(f"\n  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"  Users: {cache_result['users_cached']:,}")
    print(f"  Items: {cache_result['items_cached']:,}")
    print("\n  The existing /api/v1/personalize/* endpoints now serve")
    print("  recommendations from the local ML cache!")
    print("=" * 70 + "\n")
    
    # Save pipeline log
    log_file = DATA_DIR / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Local ML Recommendation Pipeline')
    parser.add_argument('--skip-sync', action='store_true', help='Skip Master API sync, use existing DB')
    parser.add_argument('--export-only', action='store_true', help='Only export to CSV, skip training')
    parser.add_argument('--train-only', action='store_true', help='Only train models, skip sync & export')
    parser.add_argument('--sync-only', action='store_true', help='Only sync data from APIs, skip export and training (lightweight)')
    parser.add_argument('--sync-days', type=int, default=7, help='Days of data to sync from APIs')
    
    args = parser.parse_args()
    
    # If sync-only, just run the sync step and exit
    if args.sync_only:
        print("\n" + "=" * 70)
        print("  SYNC-ONLY MODE (Lightweight)")
        print("=" * 70)
        start_time = datetime.now()
        sync_result = fetch_from_master_apis(args.sync_days)
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n  ✅ Sync complete in {duration:.1f} seconds")
        print(f"  📊 OE Orders: {sync_result['oe_orders']:,}")
        print(f"  📊 POS Orders: {sync_result['pos_orders']:,}")
        print("=" * 70 + "\n")
        return 0
    
    results = run_pipeline(
        skip_sync=args.skip_sync,
        export_only=args.export_only,
        train_only=args.train_only,
        sync_days=args.sync_days
    )
    
    return 0 if results else 1


if __name__ == "__main__":
    exit(main())
