#!/usr/bin/env python3
"""
Local ML Training & Batch Caching Pipeline

This script replaces AWS Personalize by:
1. Fetching data from Master APIs (or local DB)
2. Training SVD, Item Similarity, and Popularity models
3. Generating batch recommendations for ALL users
4. Storing results in offline_user_recommendations and offline_similar_items tables

The existing /api/v1/personalize/* endpoints will serve from these cache tables.

Usage:
    python scripts/train_and_cache.py
    python scripts/train_and_cache.py --fetch-from-api  # Fetch fresh data from Master APIs first
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import structlog
logger = structlog.get_logger()


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


def fetch_from_master_api():
    """Fetch fresh data from Master Group APIs"""
    import requests
    
    base_url = os.getenv('MASTER_GROUP_API_BASE', 'https://mes.master.com.pk')
    
    print("📡 Fetching data from Master APIs...")
    
    # This would call the sync service
    # For now, we'll use the existing sync_service
    from services.sync_service import MasterGroupSyncService
    
    sync_service = MasterGroupSyncService()
    result = sync_service.sync_all()
    
    print(f"   Synced {result.get('orders_synced', 0)} orders")
    return result


def load_training_data(days: int = 365):
    """Load all interaction data from PostgreSQL"""
    print(f"\n📊 Loading training data (last {days} days)...")
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    start_date = datetime.now() - timedelta(days=days)
    
    cursor.execute("""
        SELECT 
            o.unified_customer_id as user_id,
            oi.product_id as item_id,
            oi.product_name as item_name,
            o.order_date as timestamp,
            oi.quantity,
            COALESCE(oi.unit_price, 0) as price
        FROM orders o
        JOIN order_items oi ON o.id::text = oi.order_id
        WHERE o.order_date >= %s
        AND o.unified_customer_id IS NOT NULL
        AND oi.product_id IS NOT NULL
        ORDER BY o.order_date
    """, (start_date,))
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    df = pd.DataFrame(rows)
    
    print(f"   Loaded {len(df):,} interactions")
    print(f"   Unique users: {df['user_id'].nunique():,}")
    print(f"   Unique items: {df['item_id'].nunique():,}")
    
    return df


def train_models(df: pd.DataFrame):
    """Train SVD, Item Similarity, and Popularity models"""
    print("\n🤖 Training models...")
    
    models = {}
    
    # 1. Build user-item matrix
    print("   Building user-item matrix...")
    user_item = df.groupby(['user_id', 'item_id'])['quantity'].sum().reset_index()
    
    # Create mappings
    users = user_item['user_id'].unique()
    items = user_item['item_id'].unique()
    
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {i: idx for idx, i in enumerate(items)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_item = {i: item for item, i in item_to_idx.items()}
    
    # Get item names
    item_names = df.groupby('item_id')['item_name'].first().to_dict()
    
    models['user_to_idx'] = user_to_idx
    models['item_to_idx'] = item_to_idx
    models['idx_to_user'] = idx_to_user
    models['idx_to_item'] = idx_to_item
    models['item_names'] = item_names
    
    # Build sparse matrix
    rows = [user_to_idx[u] for u in user_item['user_id']]
    cols = [item_to_idx[i] for i in user_item['item_id']]
    values = user_item['quantity'].values
    
    user_item_matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(users), len(items))
    )
    models['user_item_matrix'] = user_item_matrix
    
    # 2. Train Item Similarity (cosine similarity)
    print("   Training item similarity model...")
    item_similarity = cosine_similarity(user_item_matrix.T)
    models['item_similarity'] = item_similarity
    
    # 3. Train Popularity model (weighted by recency)
    print("   Training popularity model...")
    df['days_ago'] = (datetime.now() - pd.to_datetime(df['timestamp'])).dt.days
    df['recency_weight'] = np.exp(-df['days_ago'] / 30)  # Exponential decay
    
    popularity = df.groupby('item_id').apply(
        lambda x: (x['quantity'] * x['recency_weight']).sum()
    ).sort_values(ascending=False)
    
    models['popularity'] = popularity.to_dict()
    
    # 4. Get user purchase history (for excluding already purchased)
    print("   Building user purchase history...")
    user_items = df.groupby('user_id')['item_id'].apply(set).to_dict()
    models['user_items'] = user_items
    
    # 5. Train SVD (using surprise library if available)
    try:
        from surprise import SVD, Dataset, Reader
        from surprise.model_selection import cross_validate
        
        print("   Training SVD model...")
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, df['quantity'].max()))
        surprise_data = Dataset.load_from_df(
            user_item[['user_id', 'item_id', 'quantity']], 
            reader
        )
        
        trainset = surprise_data.build_full_trainset()
        
        svd = SVD(n_factors=50, n_epochs=20, random_state=42)
        svd.fit(trainset)
        
        models['svd'] = svd
        models['svd_trainset'] = trainset
        print("   ✅ SVD trained successfully")
        
    except ImportError:
        print("   ⚠️ Surprise library not available, skipping SVD")
        models['svd'] = None
    
    return models


def generate_user_recommendations(models: dict, limit: int = 25):
    """Generate recommendations for ALL users"""
    print(f"\n📋 Generating recommendations for all users (top {limit} each)...")
    
    recommendations = {}
    users = list(models['user_to_idx'].keys())
    total = len(users)
    
    svd = models.get('svd')
    trainset = models.get('svd_trainset')
    item_similarity = models['item_similarity']
    popularity = models['popularity']
    user_items = models['user_items']
    idx_to_item = models['idx_to_item']
    item_to_idx = models['item_to_idx']
    item_names = models['item_names']
    
    for i, user_id in enumerate(users):
        if i % 1000 == 0:
            print(f"   Processing user {i+1:,}/{total:,} ({100*i/total:.1f}%)")
        
        user_recs = []
        purchased = user_items.get(user_id, set())
        
        # Method 1: SVD predictions
        if svd and trainset:
            try:
                inner_uid = trainset.to_inner_uid(user_id)
                
                for item_id in models['item_to_idx'].keys():
                    if item_id in purchased:
                        continue
                    
                    try:
                        inner_iid = trainset.to_inner_iid(item_id)
                        pred = svd.predict(user_id, item_id)
                        user_recs.append({
                            'item_id': item_id,
                            'score': pred.est,
                            'algorithm': 'SVD'
                        })
                    except:
                        pass
                        
            except:
                pass
        
        # Method 2: Item-based collaborative (for users with few SVD predictions)
        if len(user_recs) < limit:
            for purchased_item in list(purchased)[:10]:  # Top 10 purchased items
                if purchased_item not in item_to_idx:
                    continue
                    
                item_idx = item_to_idx[purchased_item]
                similarities = item_similarity[item_idx]
                
                for sim_idx in np.argsort(similarities)[::-1][:20]:
                    sim_item = idx_to_item[sim_idx]
                    if sim_item not in purchased and similarities[sim_idx] > 0.01:
                        user_recs.append({
                            'item_id': sim_item,
                            'score': float(similarities[sim_idx]),
                            'algorithm': 'ItemSimilarity'
                        })
        
        # Method 3: Popularity fallback
        if len(user_recs) < limit:
            for item_id, score in sorted(popularity.items(), key=lambda x: -x[1])[:limit]:
                if item_id not in purchased:
                    user_recs.append({
                        'item_id': item_id,
                        'score': score / max(popularity.values()),
                        'algorithm': 'Popularity'
                    })
        
        # Deduplicate and sort
        seen = set()
        unique_recs = []
        for rec in sorted(user_recs, key=lambda x: -x['score']):
            if rec['item_id'] not in seen:
                seen.add(rec['item_id'])
                rec['item_name'] = item_names.get(rec['item_id'], rec['item_id'])
                unique_recs.append(rec)
                if len(unique_recs) >= limit:
                    break
        
        recommendations[user_id] = unique_recs
    
    print(f"   ✅ Generated recommendations for {len(recommendations):,} users")
    return recommendations


def generate_similar_items(models: dict, limit: int = 20):
    """Generate similar items for ALL products"""
    print(f"\n🔗 Generating similar items for all products (top {limit} each)...")
    
    similar_items = {}
    item_similarity = models['item_similarity']
    idx_to_item = models['idx_to_item']
    item_to_idx = models['item_to_idx']
    item_names = models['item_names']
    
    total = len(item_to_idx)
    
    for i, (item_id, item_idx) in enumerate(item_to_idx.items()):
        if i % 500 == 0:
            print(f"   Processing item {i+1:,}/{total:,} ({100*i/total:.1f}%)")
        
        similarities = item_similarity[item_idx]
        
        item_sims = []
        for sim_idx in np.argsort(similarities)[::-1][1:limit+1]:  # Exclude self
            sim_item_id = idx_to_item[sim_idx]
            score = float(similarities[sim_idx])
            
            if score > 0.001:  # Only include meaningful similarities
                item_sims.append({
                    'item_id': sim_item_id,
                    'item_name': item_names.get(sim_item_id, sim_item_id),
                    'score': score
                })
        
        similar_items[item_id] = item_sims
    
    print(f"   ✅ Generated similar items for {len(similar_items):,} products")
    return similar_items


def save_to_cache_tables(user_recommendations: dict, similar_items: dict, item_names: dict):
    """Save recommendations to PostgreSQL cache tables (same as AWS Personalize batch results)"""
    print("\n💾 Saving to PostgreSQL cache tables...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Save user recommendations to offline_user_recommendations
    print("   Saving user recommendations...")
    
    # Clear existing data
    cursor.execute("TRUNCATE TABLE offline_user_recommendations")
    
    # Prepare batch insert
    now = datetime.now().isoformat()
    user_rows = []
    for user_id, recs in user_recommendations.items():
        user_rows.append((
            user_id,
            json.dumps(recs, cls=NumpyEncoder),
            'local_ml_hybrid',  # recipe_name
            now
        ))
    
    # Batch insert
    execute_values(
        cursor,
        """
        INSERT INTO offline_user_recommendations 
        (user_id, recommendations, recipe_name, updated_at)
        VALUES %s
        """,
        user_rows,
        page_size=1000
    )
    
    print(f"   ✅ Saved {len(user_rows):,} user recommendations")
    
    # 2. Save similar items to offline_similar_items
    print("   Saving similar items...")
    
    # Clear existing data
    cursor.execute("TRUNCATE TABLE offline_similar_items")
    
    # Prepare batch insert
    item_rows = []
    for item_id, sims in similar_items.items():
        item_rows.append((
            item_id,
            json.dumps(sims, cls=NumpyEncoder),
            'local_ml_similarity',  # recipe_name
            now
        ))
    
    # Batch insert
    execute_values(
        cursor,
        """
        INSERT INTO offline_similar_items 
        (product_id, similar_products, recipe_name, updated_at)
        VALUES %s
        """,
        item_rows,
        page_size=1000
    )
    
    print(f"   ✅ Saved {len(item_rows):,} similar item sets")
    
    conn.commit()
    cursor.close()
    conn.close()


def verify_cache():
    """Verify cache tables have data"""
    print("\n✅ Verifying cache tables...")
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT COUNT(*) as count FROM offline_user_recommendations")
    user_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM offline_similar_items")
    item_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT MAX(updated_at) as last_update FROM offline_user_recommendations")
    last_update = cursor.fetchone()['last_update']
    
    cursor.close()
    conn.close()
    
    print(f"   offline_user_recommendations: {user_count:,} records")
    print(f"   offline_similar_items: {item_count:,} records")
    print(f"   Last updated: {last_update}")
    
    return user_count, item_count


def main():
    parser = argparse.ArgumentParser(description='Train local ML and cache recommendations')
    parser.add_argument('--fetch-from-api', action='store_true', help='Fetch fresh data from Master APIs first')
    parser.add_argument('--days', type=int, default=365, help='Days of training data to use')
    parser.add_argument('--user-limit', type=int, default=25, help='Max recommendations per user')
    parser.add_argument('--item-limit', type=int, default=20, help='Max similar items per product')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("=" * 60)
    print("  LOCAL ML TRAINING & CACHING PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Optionally fetch from APIs
    if args.fetch_from_api:
        try:
            fetch_from_master_api()
        except Exception as e:
            print(f"   ⚠️ API fetch failed: {e}")
            print("   Continuing with existing data...")
    
    # Step 2: Load training data
    df = load_training_data(args.days)
    
    if len(df) == 0:
        print("❌ No training data found!")
        return 1
    
    # Step 3: Train models
    models = train_models(df)
    
    # Step 4: Generate batch recommendations
    user_recs = generate_user_recommendations(models, args.user_limit)
    similar_items = generate_similar_items(models, args.item_limit)
    
    # Step 5: Save to cache tables
    save_to_cache_tables(user_recs, similar_items, models['item_names'])
    
    # Step 6: Verify
    verify_cache()
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Users with recommendations: {len(user_recs):,}")
    print(f"  Products with similar items: {len(similar_items):,}")
    print("\n  The existing /api/v1/personalize/* endpoints will now")
    print("  serve recommendations from the local ML cache!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
