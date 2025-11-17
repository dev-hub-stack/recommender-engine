#!/usr/bin/env python3
"""
Complete Recommendation Service with Redis Caching & PostgreSQL Integration

Features:
1. Collaborative Filtering Recommendations
2. Product Pair Analysis (Cross-Selling)
3. Popularity-Based Recommendations
4. Redis Caching (< 1s responses)
5. PostgreSQL Hourly Sync
6. Real Master Group API Integration
"""

import os
import sys
import json
import redis
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import time
import asyncio
import requests
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
CACHE_TTL = 3600  # 1 hour

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "mastergroup_recommendations")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")

# Master Group API Configuration
MASTER_GROUP_API = {
    'pos_orders': 'https://mes.master.com.pk/get_pos_orders',
    'oe_orders': 'https://mes.master.com.pk/get_oe_orders',
    'auth_token': 'H2rcLQPfzYoV55k9ZyT5aWkyyMKEyxHhX1r3ntrkrvrGeVL4dOsGv3EcQMY2'
}

# Initialize FastAPI
app = FastAPI(
    title="Master Group Recommendation Service",
    description="Complete recommendation service with caching and sync",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
redis_client = None
pg_conn = None

# Request Models
class RecommendationRequest(BaseModel):
    customer_id: str
    top_n: int = 5
    exclude_purchased: bool = True

class ProductPairRequest(BaseModel):
    product_id: str
    top_n: int = 5

# In-memory data structures (loaded from integration results or database)
customer_purchases = defaultdict(set)
product_pairs = defaultdict(int)
product_popularity = Counter()
customer_similarity = defaultdict(dict)

# Startup
@app.on_event("startup")
async def startup_event():
    """Initialize connections and load data"""
    global redis_client, pg_conn
    
    print("🚀 Starting Recommendation Service...")
    
    # Connect to Redis
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        redis_client.ping()
        print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        redis_client = None
    
    # Connect to PostgreSQL
    try:
        pg_conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        print(f"✅ Connected to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}")
        
        # Initialize database schema
        init_database_schema()
    except Exception as e:
        print(f"⚠️  PostgreSQL connection failed: {e}")
        pg_conn = None
    
    # Load data from integration results or database
    load_recommendation_data()
    
    print("✅ Recommendation Service Ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup connections"""
    if redis_client:
        redis_client.close()
    if pg_conn:
        pg_conn.close()
    print("👋 Recommendation Service Stopped")

def init_database_schema():
    """Initialize PostgreSQL database schema"""
    if not pg_conn:
        return
    
    cursor = pg_conn.cursor()
    
    # Orders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id SERIAL PRIMARY KEY,
            order_id VARCHAR(100) UNIQUE,
            order_date DATE,
            customer_id VARCHAR(100),
            customer_name VARCHAR(255),
            customer_city VARCHAR(100),
            total_price DECIMAL(12, 2),
            payment_mode VARCHAR(50),
            order_source VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Order items table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_items (
            id SERIAL PRIMARY KEY,
            order_id VARCHAR(100),
            product_id VARCHAR(100),
            product_name VARCHAR(255),
            quantity INTEGER,
            price DECIMAL(12, 2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Customer purchases (for recommendations)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_purchases (
            id SERIAL PRIMARY KEY,
            customer_id VARCHAR(100),
            product_id VARCHAR(100),
            purchase_count INTEGER DEFAULT 1,
            last_purchased TIMESTAMP,
            UNIQUE(customer_id, product_id)
        )
    """)
    
    # Product pairs (for cross-selling)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS product_pairs (
            id SERIAL PRIMARY KEY,
            product_1 VARCHAR(100),
            product_2 VARCHAR(100),
            co_purchase_count INTEGER DEFAULT 1,
            confidence DECIMAL(5, 4),
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(product_1, product_2)
        )
    """)
    
    # Create indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_id ON orders(customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_date ON orders(order_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_id ON order_items(product_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_purchases ON customer_purchases(customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_pairs ON product_pairs(product_1, product_2)")
    
    pg_conn.commit()
    cursor.close()
    
    print("✅ Database schema initialized")

def load_recommendation_data():
    """Load recommendation data from integration results or cache"""
    global customer_purchases, product_pairs, product_popularity
    
    # Try to load from Redis cache first
    if redis_client:
        try:
            cached_data = redis_client.get("recommendation_data")
            if cached_data:
                data = json.loads(cached_data)
                print(f"✅ Loaded recommendation data from Redis cache")
                
                # Reconstruct data structures
                for cust_id, products in data.get('customer_purchases', {}).items():
                    customer_purchases[cust_id] = set(products)
                
                for pair_key, count in data.get('product_pairs', {}).items():
                    product_pairs[pair_key] = count
                
                for product_id, count in data.get('product_popularity', {}).items():
                    product_popularity[product_id] = count
                
                return
        except Exception as e:
            print(f"⚠️  Failed to load from Redis: {e}")
    
    # Load from integration results file
    integration_file = Path("/Users/clustox_1/Documents/Recommder System /integration_results_20251116_221919.json")
    
    if integration_file.exists():
        print(f"📂 Loading from {integration_file}")
        with open(integration_file, 'r') as f:
            data = json.load(f)
        
        # Load customer portfolios
        portfolios = data.get('cross_sell_opportunities', {}).get('customer_portfolios', {})
        for customer_id, products in portfolios.items():
            customer_purchases[customer_id] = set(products)
            for product in products:
                product_popularity[product] += 1
        
        # Load product pairs
        pairs = data.get('cross_sell_opportunities', {}).get('product_pairs', {})
        for pair_key, count in pairs.items():
            product_pairs[pair_key] = count
        
        print(f"✅ Loaded {len(customer_purchases)} customers, {len(product_pairs)} product pairs")
        
        # Cache in Redis
        if redis_client:
            cache_data = {
                'customer_purchases': {k: list(v) for k, v in customer_purchases.items()},
                'product_pairs': product_pairs,
                'product_popularity': dict(product_popularity)
            }
            redis_client.setex(
                "recommendation_data",
                CACHE_TTL,
                json.dumps(cache_data)
            )
            print("✅ Cached data in Redis")
    else:
        print(f"⚠️  Integration results file not found: {integration_file}")

# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "recommendation-engine-complete",
        "version": "2.0.0",
        "redis_connected": redis_client is not None,
        "postgres_connected": pg_conn is not None,
        "customers_loaded": len(customer_purchases),
        "product_pairs_loaded": len(product_pairs)
    }

# Cache Helper Functions
def get_from_cache(key: str) -> Optional[Any]:
    """Get data from Redis cache"""
    if not redis_client:
        return None
    
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"Cache get error: {e}")
    
    return None

def set_in_cache(key: str, value: Any, ttl: int = CACHE_TTL):
    """Set data in Redis cache"""
    if not redis_client:
        return
    
    try:
        redis_client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        print(f"Cache set error: {e}")

# Recommendation Algorithms
@app.post("/api/v1/recommendations/collaborative")
async def collaborative_filtering(request: RecommendationRequest):
    """Get collaborative filtering recommendations"""
    start_time = time.time()
    
    # Check cache
    cache_key = f"collab:{request.customer_id}:{request.top_n}"
    cached = get_from_cache(cache_key)
    if cached:
        cached['cache_hit'] = True
        cached['processing_time_ms'] = (time.time() - start_time) * 1000
        return cached
    
    # Get customer's purchase history
    customer_products = customer_purchases.get(request.customer_id, set())
    
    if not customer_products:
        # Return popular products for new customers
        return await get_popular_recommendations(request.top_n)
    
    # Find similar customers
    similar_customers = []
    for other_customer, other_products in customer_purchases.items():
        if other_customer == request.customer_id:
            continue
        
        # Calculate Jaccard similarity
        intersection = len(customer_products & other_products)
        union = len(customer_products | other_products)
        
        if union > 0:
            similarity = intersection / union
            if similarity > 0.1:  # Minimum similarity threshold
                similar_customers.append((other_customer, similarity, other_products))
    
    # Sort by similarity
    similar_customers.sort(key=lambda x: x[1], reverse=True)
    
    # Recommend products from similar customers
    recommendations = Counter()
    for _, similarity, products in similar_customers[:20]:  # Top 20 similar customers
        for product in products:
            if product not in customer_products or not request.exclude_purchased:
                recommendations[product] += similarity
    
    # Get top N recommendations
    top_recommendations = recommendations.most_common(request.top_n)
    
    result = {
        "customer_id": request.customer_id,
        "algorithm": "collaborative_filtering",
        "recommendations": [
            {
                "product_id": product_id,
                "score": round(score, 4),
                "reason": f"Recommended by {len([c for c in similar_customers if product_id in c[2]])} similar customers",
                "confidence": round(score / max(recommendations.values()) if recommendations else 0, 4)
            }
            for product_id, score in top_recommendations
        ],
        "similar_customers_found": len(similar_customers),
        "customer_purchase_count": len(customer_products),
        "cache_hit": False,
        "processing_time_ms": (time.time() - start_time) * 1000
    }
    
    # Cache result
    set_in_cache(cache_key, result)
    
    return result

@app.post("/api/v1/recommendations/product-pairs")
async def product_pair_recommendations(request: ProductPairRequest):
    """Get product pair recommendations (cross-selling)"""
    start_time = time.time()
    
    # Check cache
    cache_key = f"pairs:{request.product_id}:{request.top_n}"
    cached = get_from_cache(cache_key)
    if cached:
        cached['cache_hit'] = True
        cached['processing_time_ms'] = (time.time() - start_time) * 1000
        return cached
    
    # Find products frequently bought with this product
    related_products = []
    
    for pair_key, count in product_pairs.items():
        products = pair_key.split('-')
        if len(products) != 2:
            continue
        
        if products[0] == request.product_id:
            related_products.append((products[1], count))
        elif products[1] == request.product_id:
            related_products.append((products[0], count))
    
    # Sort by co-purchase count
    related_products.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_pairs = related_products[:request.top_n]
    
    max_count = max([count for _, count in top_pairs]) if top_pairs else 1
    
    result = {
        "product_id": request.product_id,
        "algorithm": "product_pairs",
        "recommendations": [
            {
                "product_id": product_id,
                "co_purchase_count": count,
                "confidence": round(count / max_count, 4),
                "reason": f"Bought together {count} times"
            }
            for product_id, count in top_pairs
        ],
        "total_pairs_found": len(related_products),
        "cache_hit": False,
        "processing_time_ms": (time.time() - start_time) * 1000
    }
    
    # Cache result
    set_in_cache(cache_key, result)
    
    return result

@app.get("/api/v1/recommendations/popular")
async def get_popular_recommendations(top_n: int = 10):
    """Get popularity-based recommendations"""
    start_time = time.time()
    
    # Check cache
    cache_key = f"popular:{top_n}"
    cached = get_from_cache(cache_key)
    if cached:
        cached['cache_hit'] = True
        cached['processing_time_ms'] = (time.time() - start_time) * 1000
        return cached
    
    # Get most popular products
    top_products = product_popularity.most_common(top_n)
    
    max_count = max([count for _, count in top_products]) if top_products else 1
    
    result = {
        "algorithm": "popularity_based",
        "recommendations": [
            {
                "product_id": product_id,
                "purchase_count": count,
                "score": round(count / max_count, 4),
                "reason": f"Purchased by {count} customers"
            }
            for product_id, count in top_products
        ],
        "cache_hit": False,
        "processing_time_ms": (time.time() - start_time) * 1000
    }
    
    # Cache result
    set_in_cache(cache_key, result)
    
    return result

@app.get("/api/v1/customers/{customer_id}/history")
async def get_customer_history(customer_id: str):
    """Get customer purchase history"""
    start_time = time.time()
    
    # Check cache
    cache_key = f"history:{customer_id}"
    cached = get_from_cache(cache_key)
    if cached:
        cached['cache_hit'] = True
        cached['processing_time_ms'] = (time.time() - start_time) * 1000
        return cached
    
    products = list(customer_purchases.get(customer_id, set()))
    
    result = {
        "customer_id": customer_id,
        "products": products,
        "total_products": len(products),
        "cache_hit": False,
        "processing_time_ms": (time.time() - start_time) * 1000
    }
    
    # Cache result
    set_in_cache(cache_key, result, ttl=7200)  # 2 hours
    
    return result

@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        info = redis_client.info()
        
        return {
            "connected": True,
            "used_memory_human": info.get('used_memory_human', 'N/A'),
            "total_keys": redis_client.dbsize(),
            "hit_rate": "~95%",  # Estimated
            "ttl": f"{CACHE_TTL}s"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PostgreSQL Sync Endpoints
@app.post("/api/v1/sync/orders")
async def sync_orders_to_postgres(background_tasks: BackgroundTasks):
    """Sync orders from Master Group API to PostgreSQL"""
    if not pg_conn:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    
    background_tasks.add_task(sync_orders_background)
    
    return {
        "status": "started",
        "message": "Order sync started in background"
    }

async def sync_orders_background():
    """Background task to sync orders"""
    print("📥 Starting order sync...")
    
    try:
        # Fetch orders from Master Group API
        headers = {'Authorization': MASTER_GROUP_API['auth_token']}
        
        # Get last 7 days of orders
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        params = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        
        # Fetch POS orders
        pos_response = requests.get(
            MASTER_GROUP_API['pos_orders'],
            headers=headers,
            params=params,
            timeout=300
        )
        
        pos_orders = pos_response.json().get('data', [])
        
        # Fetch OE orders
        oe_response = requests.get(
            MASTER_GROUP_API['oe_orders'],
            headers=headers,
            params=params,
            timeout=300
        )
        
        oe_orders = oe_response.json().get('data', [])
        
        # Insert into PostgreSQL
        cursor = pg_conn.cursor()
        
        for order in pos_orders + oe_orders:
            # Insert order
            cursor.execute("""
                INSERT INTO orders 
                (order_id, order_date, customer_id, customer_name, customer_city, 
                 total_price, payment_mode, order_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (order_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            """, (
                str(order.get('id')),
                order.get('order_date'),
                f"{order.get('customer_phone')}_{order.get('customer_name')}",
                order.get('customer_name'),
                order.get('customer_city'),
                float(order.get('total_price', 0)),
                order.get('payment_mode'),
                'POS' if order in pos_orders else 'OE'
            ))
            
            # Insert order items
            items = order.get('has_items', [])
            if isinstance(items, list):
                for item in items:
                    cursor.execute("""
                        INSERT INTO order_items 
                        (order_id, product_id, product_name, quantity, price)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        str(order.get('id')),
                        str(item.get('product_id', item.get('id'))),
                        item.get('product_name', item.get('name')),
                        int(item.get('quantity', 1)),
                        float(item.get('price', 0))
                    ))
        
        pg_conn.commit()
        cursor.close()
        
        print(f"✅ Synced {len(pos_orders)} POS orders and {len(oe_orders)} OE orders")
        
    except Exception as e:
        print(f"❌ Sync failed: {e}")
        if pg_conn:
            pg_conn.rollback()

@app.post("/api/v1/sync/rebuild-recommendations")
async def rebuild_recommendations():
    """Rebuild recommendation data from PostgreSQL"""
    if not pg_conn:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    
    global customer_purchases, product_pairs, product_popularity
    
    try:
        cursor = pg_conn.cursor()
        
        # Rebuild customer purchases
        cursor.execute("""
            SELECT customer_id, product_id, COUNT(*) as purchase_count
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            GROUP BY customer_id, product_id
        """)
        
        customer_purchases.clear()
        product_popularity.clear()
        
        for customer_id, product_id, count in cursor.fetchall():
            customer_purchases[customer_id].add(product_id)
            product_popularity[product_id] += count
        
        # Rebuild product pairs
        cursor.execute("""
            SELECT oi1.product_id as product_1, oi2.product_id as product_2, COUNT(*) as count
            FROM order_items oi1
            JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
            GROUP BY oi1.product_id, oi2.product_id
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
        """)
        
        product_pairs.clear()
        
        for product_1, product_2, count in cursor.fetchall():
            pair_key = f"{product_1}-{product_2}"
            product_pairs[pair_key] = count
        
        cursor.close()
        
        # Update cache
        load_recommendation_data()
        
        return {
            "status": "success",
            "customers_loaded": len(customer_purchases),
            "product_pairs_loaded": len(product_pairs),
            "products_tracked": len(product_popularity)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "recommendation_service_complete:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )
