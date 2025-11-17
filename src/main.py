"""
Recommendation Engine Service Main Application
Core ML algorithms and recommendation inference with Redis caching and PostgreSQL integration
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import start_http_server
import structlog
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import redis
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import os
from collections import defaultdict
import re
import sys

# Add config path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.master_group_api import PG_CONFIG, REDIS_CONFIG, MASTER_GROUP_CONFIG

# Simple settings configuration
class Settings:
    version = "1.0.0"
    debug = False

settings = Settings()

# Use centralized configuration
REDIS_HOST = REDIS_CONFIG.get('host')
REDIS_PORT = REDIS_CONFIG.get('port')
REDIS_PASSWORD = REDIS_CONFIG.get('password')
REDIS_DB = REDIS_CONFIG.get('db')
CACHE_TTL = REDIS_CONFIG.get('ttl')

PG_HOST = PG_CONFIG.get('host')
PG_PORT = PG_CONFIG.get('port')
PG_DB = PG_CONFIG.get('database')
PG_USER = PG_CONFIG.get('user')
PG_PASSWORD = PG_CONFIG.get('password')

# Master Group API configuration from centralized config
MASTER_GROUP_API_BASE = MASTER_GROUP_CONFIG.get('base_url')
AUTH_TOKEN = MASTER_GROUP_CONFIG.get('auth_token')

# Global connections
redis_client = None
pg_conn = None

def get_pg_connection_params():
    """Get PostgreSQL connection parameters with SSL support"""
    params = {
        'host': PG_HOST,
        'port': PG_PORT,
        'database': PG_DB,
        'user': PG_USER,
        'password': PG_PASSWORD
    }
    
    # Add SSL mode for Heroku if present
    if PG_CONFIG.get('sslmode'):
        params['sslmode'] = PG_CONFIG.get('sslmode')
        
    return params

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Pydantic models
class RecommendationRequest(BaseModel):
    customer_id: str
    limit: int = 10

class ProductPairRequest(BaseModel):
    product_id: str
    limit: int = 10

class SyncOrdersRequest(BaseModel):
    start_date: str
    end_date: str
    limit: Optional[int] = None

class Recommendation(BaseModel):
    product_id: str
    score: float
    reason: Optional[str] = None
    purchase_count: Optional[int] = None
    co_purchase_count: Optional[int] = None


def init_redis():
    """Initialize Redis connection"""
    global redis_client
    try:
        # Build Redis connection parameters based on configuration
        redis_params = {
            'host': REDIS_HOST,
            'port': REDIS_PORT,
            'db': REDIS_DB,
            'decode_responses': True
        }
        
        if REDIS_PASSWORD:
            redis_params['password'] = REDIS_PASSWORD
            
        # Add SSL configuration if present (for Heroku)
        if REDIS_CONFIG.get('ssl'):
            redis_params['ssl'] = True
            redis_params['ssl_cert_reqs'] = REDIS_CONFIG.get('ssl_cert_reqs')
            
        redis_client = redis.Redis(**redis_params)
        redis_client.ping()
        logger.info("Redis connection established", host=REDIS_HOST, port=REDIS_PORT)
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        redis_client = None

def init_postgres():
    """Initialize PostgreSQL connection (tables already exist)"""
    global pg_conn
    try:
        pg_conn = psycopg2.connect(**get_pg_connection_params())
        
        logger.info("PostgreSQL connection established", host=PG_HOST, port=PG_PORT)
    except Exception as e:
        logger.error("Failed to connect to PostgreSQL", error=str(e))
        pg_conn = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Recommendation Engine Service", version=settings.version)
    
    # Start Prometheus metrics server on a different port
    try:
        start_http_server(9002)  # Changed from 9001 to 9002
        logger.info("Prometheus metrics server started", port=9002)
    except OSError as e:
        logger.warning("Prometheus metrics server failed to start", error=str(e))
    
    # Initialize Redis cache
    init_redis()
    
    # Initialize PostgreSQL
    init_postgres()
    
    # Initialize and start sync scheduler
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from services.scheduler import get_scheduler
        
        scheduler = get_scheduler()
        scheduler.start()
        logger.info("Auto-sync scheduler started")
    except Exception as e:
        logger.error("Failed to start scheduler", error=str(e))
    
    yield
    
    # Cleanup
    try:
        scheduler.stop()
    except:
        pass
    
    if redis_client:
        redis_client.close()
    if pg_conn:
        pg_conn.close()
    
    logger.info("Shutting down Recommendation Engine Service")


app = FastAPI(
    title="Master Group Recommendation Engine",
    description="Core recommendation algorithms and ML inference",
    version=settings.version,
    lifespan=lifespan
)

# Add CORS middleware to allow dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # Allow dashboard origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Helper functions for caching
def get_cache_key(prefix: str, *args) -> str:
    """Generate cache key"""
    return f"{prefix}:{'_'.join(str(arg) for arg in args)}"

def get_from_cache(key: str):
    """Get data from Redis cache"""
    if not redis_client:
        return None
    try:
        data = redis_client.get(key)
        if data:
            redis_client.incr("cache:hits")
            return json.loads(data)
        redis_client.incr("cache:misses")
        return None
    except Exception as e:
        logger.error("Cache get error", error=str(e))
        return None

def set_to_cache(key: str, data: dict, ttl: int = CACHE_TTL):
    """Set data to Redis cache"""
    if not redis_client:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(data))
    except Exception as e:
        logger.error("Cache set error", error=str(e))

def calculate_date_range(time_filter: str) -> datetime:
    """
    Calculate start date based on time filter
    Returns None for 'all' time filter
    """
    if time_filter == "today":
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_filter == "7days":
        return datetime.now() - timedelta(days=7)
    elif time_filter == "30days":
        return datetime.now() - timedelta(days=30)
    elif time_filter == "mtd":  # Month to Date
        now = datetime.now()
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif time_filter == "90days":
        return datetime.now() - timedelta(days=90)
    elif time_filter == "6months":
        return datetime.now() - timedelta(days=180)
    elif time_filter == "1year":
        return datetime.now() - timedelta(days=365)
    elif ":" in time_filter:  # Custom date range format: start_date:end_date
        try:
            date_parts = time_filter.split(":")
            if len(date_parts) == 2:
                return datetime.strptime(date_parts[0], "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid custom date format: {time_filter}")
            return None
    # For 'all' or any unrecognized filter
    return None

# Smart category extraction function
def extract_smart_category(product_name: str, product_type: str = None, order_source: str = "pos") -> str:
    """
    Smart category extraction function
    Uses OE API categories when available, fallback to product name parsing for POS products
    """
    # If we have a real category from OE API, use it
    if order_source == "oe" and product_type and product_type.lower() not in ["simple", "unknown", ""]:
        return product_type
    
    # For POS products or when category is generic, parse from product name
    if not product_name:
        return "General"
    
    product_name_lower = product_name.lower()
    
    # Mattress categories
    if any(keyword in product_name_lower for keyword in [
        "mattress", "matress", "foam", "sleep", "firm", "soft", "spring", 
        "memory foam", "orthopedic", "pocket spring", "latex"
    ]):
        if any(keyword in product_name_lower for keyword in ["spring", "pocket"]):
            return "Spring Mattresses"
        elif any(keyword in product_name_lower for keyword in ["memory", "ortho"]):
            return "Memory Foam Mattresses"
        elif "firm" in product_name_lower:
            return "Firm Mattresses"
        else:
            return "Mattresses"
    
    # Pillow categories
    if any(keyword in product_name_lower for keyword in [
        "pillow", "cushion", "head", "neck", "lumbar", "support"
    ]):
        if any(keyword in product_name_lower for keyword in ["memory", "foam"]):
            return "Memory Foam Pillows"
        elif any(keyword in product_name_lower for keyword in ["lumbar", "support", "back"]):
            return "Support Cushions"
        else:
            return "Pillows & Accessories"
    
    # Bedding and accessories
    if any(keyword in product_name_lower for keyword in [
        "sheet", "cover", "protector", "topper", "pad", "base", "frame"
    ]):
        return "Bedding & Accessories"
    
    # Furniture categories
    if any(keyword in product_name_lower for keyword in [
        "sofa", "chair", "table", "desk", "cabinet", "wardrobe", "dresser"
    ]):
        return "Furniture"
    
    # Electronics
    if any(keyword in product_name_lower for keyword in [
        "fan", "heater", "air", "conditioner", "remote", "electronic"
    ]):
        return "Electronics"
    
    # Default fallback
    return "General"

# Recommendation algorithms
def collaborative_filtering(customer_id: str, limit: int = 10, time_filter: str = "all") -> List[Dict]:
    """
    Collaborative filtering based on customer purchase patterns
    Recommends products that similar customers have purchased
    """
    if not pg_conn:
        return []
    
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get products purchased by the target customer
            if start_date:
                cur.execute("""
                    SELECT DISTINCT product_id
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.id
                    WHERE o.unified_customer_id = %s
                    AND o.order_date >= %s
                """, (customer_id, start_date))
            else:
                cur.execute("""
                    SELECT DISTINCT product_id
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.id
                    WHERE o.unified_customer_id = %s
                """, (customer_id,))
            customer_products = {row['product_id'] for row in cur.fetchall()}
            
            if not customer_products:
                return []
            
            # Find similar customers (those who bought the same products)
            if start_date:
                cur.execute("""
                    SELECT DISTINCT o.unified_customer_id
                    FROM orders o
                    JOIN order_items oi ON o.id = oi.order_id
                    WHERE oi.product_id = ANY(%s)
                    AND o.unified_customer_id != %s
                    AND o.order_date >= %s
                    LIMIT 50
                """, (list(customer_products), customer_id, start_date))
            else:
                cur.execute("""
                    SELECT DISTINCT o.unified_customer_id
                    FROM orders o
                    JOIN order_items oi ON o.id = oi.order_id
                    WHERE oi.product_id = ANY(%s)
                    AND o.unified_customer_id != %s
                    LIMIT 50
                """, (list(customer_products), customer_id))
            similar_customers = [row['unified_customer_id'] for row in cur.fetchall()]
            
            if not similar_customers:
                return []
            
            # Get products purchased by similar customers
            if start_date:
                cur.execute("""
                    SELECT oi2.product_id, oi2.product_name, o.order_type, COUNT(*) as co_purchase_count
                    FROM order_items oi1
                    JOIN order_items oi2 ON oi1.order_id = oi2.order_id
                    JOIN orders o ON oi1.order_id = o.id
                    WHERE oi1.product_id = %s
                    AND oi2.product_id != %s
                    AND o.order_date >= %s
                    GROUP BY oi2.product_id, oi2.product_name, o.order_type
                    ORDER BY co_purchase_count DESC
                    LIMIT %s
                """, (product_id, product_id, start_date, limit))
            else:
                cur.execute("""
                    SELECT oi2.product_id, oi2.product_name, o.order_type, COUNT(*) as co_purchase_count
                    FROM order_items oi1
                    JOIN order_items oi2 ON oi1.order_id = oi2.order_id
                    JOIN orders o ON oi1.order_id = o.id
                    WHERE oi1.product_id = %s
                    AND oi2.product_id != %s
                    GROUP BY oi2.product_id, oi2.product_name, o.order_type
                    ORDER BY co_purchase_count DESC
                    LIMIT %s
                """, (product_id, product_id, limit))
            
            recommendations = []
            for row in cur.fetchall():
                # Extract smart category using product name and order source
                product_name = row['product_name'] or f"Product {row['product_id']}"
                order_source = row['order_type'] if row['order_type'] in ['pos', 'oe'] else 'pos'
                category = extract_smart_category(product_name, None, order_source)
                
                recommendations.append({
                    "product_id": row['product_id'],
                    "product_name": product_name,
                    "score": float(row['co_purchase_count']),
                    "reason": f"Frequently bought together ({row['co_purchase_count']} times)",
                    "co_purchase_count": row['co_purchase_count'],
                    "category": category
                })
            
            return recommendations
    except Exception as e:
        logger.error("Collaborative filtering error", error=str(e))
        return []

def product_pair_recommendations(product_id: str, limit: int = 10, time_filter: str = "all") -> List[Dict]:
    """
    Product pair recommendations for cross-selling
    Based on co-purchase patterns with time filtering
    """
    if not pg_conn:
        return []
    
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Find products frequently bought together
            if start_date:
                cur.execute("""
                    SELECT oi2.product_id, oi2.product_name, o.order_type, COUNT(*) as co_purchase_count
                    FROM order_items oi1
                    JOIN order_items oi2 ON oi1.order_id = oi2.order_id
                    JOIN orders o ON oi1.order_id = o.id
                    WHERE oi1.product_id = %s
                    AND oi2.product_id != %s
                    AND o.order_date >= %s
                    GROUP BY oi2.product_id, oi2.product_name, o.order_type
                    ORDER BY co_purchase_count DESC
                    LIMIT %s
                """, (product_id, product_id, start_date, limit))
            else:
                cur.execute("""
                    SELECT oi2.product_id, oi2.product_name, o.order_type, COUNT(*) as co_purchase_count
                    FROM order_items oi1
                    JOIN order_items oi2 ON oi1.order_id = oi2.order_id
                    JOIN orders o ON oi1.order_id = o.id
                    WHERE oi1.product_id = %s
                    AND oi2.product_id != %s
                    GROUP BY oi2.product_id, oi2.product_name, o.order_type
                    ORDER BY co_purchase_count DESC
                    LIMIT %s
                """, (product_id, product_id, limit))
            
            recommendations = []
            for row in cur.fetchall():
                # Extract smart category using product name and order source
                product_name = row['product_name'] or f"Product {row['product_id']}"
                order_source = row['order_type'] if row['order_type'] in ['pos', 'oe'] else 'pos'
                category = extract_smart_category(product_name, None, order_source)
                
                recommendations.append({
                    "product_id": row['product_id'],
                    "product_name": product_name,
                    "score": float(row['co_purchase_count']),
                    "reason": f"Frequently bought together ({row['co_purchase_count']} times)",
                    "co_purchase_count": row['co_purchase_count'],
                    "category": category
                })
            
            return recommendations
    except Exception as e:
        logger.error("Product pair recommendations error", error=str(e))
        return []

def popular_products(limit: int = 10, time_filter: str = "7days") -> List[Dict]:
    """Get most popular products based on purchase count with time filtering and caching"""
    # Check cache first
    cache_key = f"popular_products:{limit}:{time_filter}"
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info("Popular products served from cache", time_filter=time_filter, limit=limit)
                return json.loads(cached_result)
        except Exception as e:
            logger.warning("Cache read failed for popular products", error=str(e))
    
    if not pg_conn:
        return []
    
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            if start_date:
                cur.execute("""
                    SELECT oi.product_id, 
                           oi.product_name,
                           o.order_type,
                           COUNT(*) as purchase_count,
                           COUNT(DISTINCT o.unified_customer_id) as unique_customers,
                           AVG(oi.unit_price) as avg_price,
                           SUM(oi.total_price) as total_revenue
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.id
                    WHERE oi.product_name IS NOT NULL
                    AND o.order_date >= %s
                    GROUP BY oi.product_id, oi.product_name, o.order_type
                    ORDER BY purchase_count DESC
                    LIMIT %s
                """, (start_date, limit))
            else:
                cur.execute("""
                    SELECT oi.product_id, 
                           oi.product_name,
                           o.order_type,
                           COUNT(*) as purchase_count,
                           COUNT(DISTINCT o.unified_customer_id) as unique_customers,
                           AVG(oi.unit_price) as avg_price,
                           SUM(oi.total_price) as total_revenue
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.id
                    WHERE oi.product_name IS NOT NULL
                    GROUP BY oi.product_id, oi.product_name, o.order_type
                    ORDER BY purchase_count DESC
                    LIMIT %s
                """, (limit,))
            
            recommendations = []
            for row in cur.fetchall():
                # Extract smart category using product name and order source
                product_name = row['product_name'] or f"Product {row['product_id']}"
                order_source = row['order_type'] if row['order_type'] in ['pos', 'oe'] else 'pos'
                category = extract_smart_category(product_name, None, order_source)
                
                recommendations.append({
                    "product_id": row['product_id'],
                    "product_name": product_name,
                    "score": float(row['purchase_count']),
                    "reason": f"Popular product (purchased {row['purchase_count']} times)",
                    "purchase_count": row['purchase_count'],
                    "unique_customers": row['unique_customers'],
                    "avg_price": float(row['avg_price']) if row['avg_price'] else 0,
                    "total_revenue": float(row['total_revenue']) if row['total_revenue'] else 0,
                    "category": category
                })
            
            # Cache the results
            if redis_client:
                try:
                    cache_ttl = 300 if time_filter in ['today', '7days'] else 1800  # 5 min vs 30 min
                    redis_client.setex(cache_key, cache_ttl, json.dumps(recommendations))
                    logger.info("Popular products cached", time_filter=time_filter, count=len(recommendations), ttl=cache_ttl)
                except Exception as e:
                    logger.warning("Cache write failed for popular products", error=str(e))
            
            return recommendations
    except Exception as e:
        logger.error("Popular products error", error=str(e))
        return []


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "recommendation-engine",
        "version": settings.version,
        "redis_connected": redis_client is not None,
        "postgres_connected": pg_conn is not None
    }

@app.get("/api/v1/recommendations/collaborative")
async def get_collaborative_recommendations(
    customer_id: str = Query(..., description="Customer ID"),
    limit: int = Query(10, ge=1, le=100),
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, all")  # Changed default to 7days
):
    """Get collaborative filtering recommendations with time-based filtering"""
    # Check cache first
    cache_key = get_cache_key("collab", customer_id, limit, time_filter)
    cached = get_from_cache(cache_key)
    if cached:
        logger.info("Returning cached collaborative recommendations", customer_id=customer_id, time_filter=time_filter)
        return cached
    
    # Generate recommendations
    recommendations = collaborative_filtering(customer_id, limit, time_filter)
    
    result = {
        "customer_id": customer_id,
        "recommendations": recommendations,
        "time_filter": time_filter,
        "cached": False,
        "timestamp": datetime.now().isoformat()
    }
    
    # Cache the result (shorter TTL for time-filtered data)
    ttl = 3600 if time_filter == "all" else 300  # 5 minutes for time-filtered data
    set_to_cache(cache_key, result, ttl=ttl)
    
    return result

@app.get("/api/v1/recommendations/product-pairs")
async def get_product_pair_recommendations(
    product_id: str = Query(..., description="Product ID"),
    limit: int = Query(10, ge=1, le=100),
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, all")  # Changed default to 7days
):
    """Get product pair recommendations for cross-selling with time-based filtering"""
    # Check cache first
    cache_key = get_cache_key("pairs", product_id, limit, time_filter)
    cached = get_from_cache(cache_key)
    if cached:
        logger.info("Returning cached product pair recommendations", product_id=product_id, time_filter=time_filter)
        return cached
    
    # Generate recommendations
    recommendations = product_pair_recommendations(product_id, limit, time_filter)
    
    result = {
        "product_id": product_id,
        "recommendations": recommendations,
        "time_filter": time_filter,
        "cached": False,
        "timestamp": datetime.now().isoformat()
    }
    
    # Cache the result (shorter TTL for time-filtered data)
    ttl = 3600 if time_filter == "all" else 300  # 5 minutes for time-filtered data
    set_to_cache(cache_key, result, ttl=ttl)
    
    return result

@app.get("/api/v1/recommendations/popular")
async def get_popular_products_endpoint(
    limit: int = Query(10, ge=1, le=100),
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, all")  # Changed default to 7days
):
    """Get most popular products with time-based filtering"""
    # Check cache first
    cache_key = get_cache_key("popular", limit, time_filter)
    cached = get_from_cache(cache_key)
    if cached:
        logger.info("Returning cached popular products", time_filter=time_filter)
        return cached
    
    # Generate recommendations
    recommendations = popular_products(limit, time_filter)
    
    result = {
        "recommendations": recommendations,
        "time_filter": time_filter,
        "cached": False,
        "timestamp": datetime.now().isoformat()
    }
    
    # Cache the result (shorter TTL for recent data)
    ttl = 300 if time_filter == "all" else 60  # 1 minute for time-filtered data
    set_to_cache(cache_key, result, ttl=ttl)
    
    return result

@app.get("/api/v1/customers/{customer_id}/history")
async def get_customer_history(customer_id: str):
    """Get customer purchase history"""
    if not pg_conn:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get customer statistics
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT oi.product_id) as unique_products,
                    SUM(o.total_price) as total_spent
                FROM orders o
                LEFT JOIN order_items oi ON o.id = oi.order_id
                WHERE o.unified_customer_id = %s
            """, (customer_id,))
            stats = cur.fetchone()
            
            # Get recent orders
            cur.execute("""
                SELECT id as order_id, order_date, total_price as total_amount, order_type as source
                FROM orders
                WHERE unified_customer_id = %s
                ORDER BY order_date DESC
                LIMIT 10
            """, (customer_id,))
            recent_orders = cur.fetchall()
            
            return {
                "customer_id": customer_id,
                "total_orders": stats['total_orders'] or 0,
                "unique_products": stats['unique_products'] or 0,
                "total_spent": float(stats['total_spent']) if stats['total_spent'] else 0.0,
                "recent_orders": recent_orders
            }
    except Exception as e:
        logger.error("Customer history error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    if not redis_client:
        return {"error": "Redis not available"}
    
    try:
        hits = int(redis_client.get("cache:hits") or 0)
        misses = int(redis_client.get("cache:misses") or 0)
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0
        
        return {
            "cache_size": redis_client.dbsize(),
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Cache stats error", error=str(e))
        return {"error": str(e)}

@app.post("/api/v1/sync/trigger")
async def trigger_sync():
    """Manually trigger an incremental sync"""
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from services.sync_service import get_sync_service
        
        sync_service = get_sync_service()
        result = sync_service.sync_incremental()
        
        # Clear relevant caches after successful sync
        if redis_client and result.get('status') == 'success':
            keys_deleted = 0
            for key in redis_client.scan_iter("popular:*"):
                redis_client.delete(key)
                keys_deleted += 1
            for key in redis_client.scan_iter("collab:*"):
                redis_client.delete(key)
                keys_deleted += 1
            logger.info(f"Cleared {keys_deleted} cache keys")
        
        return result
    except Exception as e:
        logger.error("Manual sync error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sync/status")
async def get_sync_status():
    """Get current sync status"""
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from services.sync_service import get_sync_service
        
        sync_service = get_sync_service()
        return sync_service.get_sync_status()
    except Exception as e:
        logger.error("Sync status error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sync/scheduler-status")
async def get_scheduler_status():
    """Get scheduler status"""
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from services.scheduler import get_scheduler
        
        scheduler = get_scheduler()
        return scheduler.get_status()
    except Exception as e:
        logger.error("Scheduler status error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/training/trigger")
async def trigger_manual_training():
    """Manually trigger model training (Auto-Pilot Learning)"""
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from services.scheduler import get_scheduler
        
        scheduler = get_scheduler()
        result = scheduler.trigger_manual_training()
        
        return {
            "status": "success",
            "message": "Model training completed",
            "result": result
        }
    except Exception as e:
        logger.error("Manual training error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/training/status")
async def get_training_status():
    """Get Auto-Pilot Learning status"""
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from services.scheduler import get_scheduler
        
        scheduler = get_scheduler()
        status = scheduler.get_status()
        
        return {
            "auto_pilot_enabled": status.get('auto_pilot_enabled', False),
            "next_training_time": status.get('next_training_time'),
            "scheduler_running": status.get('scheduler_running', False),
            "training_schedule": "Daily at 3:00 AM"
        }
    except Exception as e:
        logger.error("Training status error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sync/history")
async def get_sync_history(limit: int = 10):
    """Get sync history from database with fresh connection"""
    conn = None
    cursor = None
    try:
        # Create fresh database connection to avoid transaction errors
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT sync_type, 
                   last_sync_timestamp as last_sync_time, 
                   sync_status as status, 
                   orders_synced as records_synced, 
                   CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END as errors_count,
                   sync_duration_seconds,
                   created_at
            FROM sync_metadata
            ORDER BY last_sync_timestamp DESC
            LIMIT %s
        """, (limit,))
        history = cursor.fetchall()
        
        # Convert to regular dicts for JSON serialization
        result = {"success": True, "history": [dict(row) for row in history]}
        logger.info(f"Sync history fetched", count=len(history))
        return result
        
    except Exception as e:
        logger.error("Sync history error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, num_recommendations: int = 10):
    """Legacy endpoint - redirect to collaborative filtering"""
    return await get_collaborative_recommendations(customer_id=user_id, limit=num_recommendations)

@app.post("/track-interaction")
async def track_interaction(interaction_data: dict):
    """Track user interaction with recommendations"""
    # Store interaction in database for future model training
    logger.info("Interaction tracked", data=interaction_data)
    return {"status": "tracked", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return {"message": "Metrics available at :9001/metrics"}

@app.get("/api/v1/recommendations/content-based")
async def get_content_based_recommendations(
    customer_id: str = Query(..., description="Customer ID"),
    limit: int = Query(10, ge=1, le=50)
):
    """Get content-based recommendations (similar products based on purchase history)"""
    # Check cache first
    cache_key = get_cache_key(f"content_based:{customer_id}", limit)
    cached = get_from_cache(cache_key)
    if cached:
        logger.info("Returning cached content-based recommendations", customer_id=customer_id)
        return cached
    
    # Generate recommendations
    try:
        # Import here to avoid startup issues
        from algorithms.content_based_filtering import ContentBasedFiltering
        
        cbf = ContentBasedFiltering(pg_conn)
        recommendations = cbf.get_recommendations(customer_id, limit=limit)
        
        result = {
            "customer_id": customer_id,
            "recommendations": recommendations,
            "cached": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the result
        set_to_cache(cache_key, result)
        
        logger.info("Content-based recommendations generated", 
                   customer_id=customer_id, 
                   count=len(recommendations))
        
        return result
        
    except Exception as e:
        logger.error("Content-based filtering failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/recommendations/matrix-factorization")
async def get_matrix_factorization_recommendations(
    customer_id: str = Query(..., description="Customer ID"),
    limit: int = Query(10, ge=1, le=50)
):
    """Get Matrix Factorization (SVD) recommendations - advanced collaborative filtering"""
    # Check cache first
    cache_key = get_cache_key(f"matrix_fact:{customer_id}", limit)
    cached = get_from_cache(cache_key)
    if cached:
        logger.info("Returning cached matrix factorization recommendations", customer_id=customer_id)
        return cached
    
    # Generate recommendations
    try:
        from algorithms.matrix_factorization import MatrixFactorizationSVD
        
        mf = MatrixFactorizationSVD(pg_conn, n_factors=30)
        recommendations = mf.get_recommendations(customer_id, limit=limit)
        
        result = {
            "customer_id": customer_id,
            "recommendations": recommendations,
            "cached": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache for longer since training is expensive
        set_to_cache(cache_key, result, ttl=7200)  # 2 hours
        
        logger.info("Matrix factorization recommendations generated", 
                   customer_id=customer_id, 
                   count=len(recommendations))
        
        return result
        
    except Exception as e:
        logger.error("Matrix factorization failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================
# ANALYTICS ENDPOINTS (REAL-TIME DATA)
# ==============================================

@app.get("/api/v1/analytics/products")
async def get_product_analytics(
    time_filter: str = Query("all", description="Time filter: today, 7days, 30days, all"),
    limit: int = Query(100, description="Maximum number of products to return")
):
    """
    Get real-time product analytics with sales data (100% LIVE DATA)
    Time filters: today, 7days, 30days, all
    """
    conn = None
    cursor = None
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        # Create fresh database connection
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if start_date:
            cursor.execute("""
                SELECT 
                    oi.product_id,
                    oi.product_name,
                    'General' as category,
                    AVG(oi.unit_price) as price,
                    COUNT(DISTINCT oi.order_id) as total_orders,
                    SUM(oi.quantity) as total_quantity_sold,
                    SUM(oi.total_price) as total_revenue,
                    COUNT(DISTINCT o.unified_customer_id) as unique_customers,
                    AVG(oi.total_price) as avg_order_value,
                    MAX(o.order_date) as last_order_date
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                WHERE o.order_date >= %s
                GROUP BY oi.product_id, oi.product_name
                ORDER BY total_revenue DESC
                LIMIT %s
            """, (start_date, limit))
        else:
            cursor.execute("""
                SELECT 
                    oi.product_id,
                    oi.product_name,
                    'General' as category,
                    AVG(oi.unit_price) as price,
                    COUNT(DISTINCT oi.order_id) as total_orders,
                    SUM(oi.quantity) as total_quantity_sold,
                    SUM(oi.total_price) as total_revenue,
                    COUNT(DISTINCT o.unified_customer_id) as unique_customers,
                    AVG(oi.total_price) as avg_order_value,
                    MAX(o.order_date) as last_order_date
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                GROUP BY oi.product_id, oi.product_name
                ORDER BY total_revenue DESC
                LIMIT %s
            """, (limit,))
        
        products = cursor.fetchall()
        
        # Convert to list of dicts and format dates
        product_list = []
        for product in products:
            product_dict = dict(product)
            if product_dict.get('last_order_date'):
                product_dict['last_order_date'] = product_dict['last_order_date'].isoformat()
            product_list.append(product_dict)
        
        logger.info(f"Product analytics fetched", 
                   time_filter=time_filter, 
                   count=len(product_list))
        
        return {
            "success": True,
            "time_filter": time_filter,
            "start_date": start_date.isoformat() if start_date else None,
            "products": product_list,
            "count": len(product_list)
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch product analytics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_metrics(
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, mtd, 90days, 6months, 1year, custom:start:end, all")
):
    """
    Get real-time dashboard metrics (100% LIVE DATA) with Redis caching
    Returns: total revenue, orders, customers, avg order value, products
    """
    # Check cache first for faster response
    cache_key = f"dashboard_metrics:{time_filter}"
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                result['cache_hit'] = True
                logger.info("Dashboard metrics served from cache", time_filter=time_filter)
                return result
        except Exception as e:
            logger.warning("Cache read failed", error=str(e))
    
    conn = None
    cursor = None
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        # Create fresh database connection
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query dashboard metrics - Optimized with separate queries for better performance
        if start_date:
            # Basic order metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    COUNT(DISTINCT unified_customer_id) as total_customers,
                    SUM(total_price) as total_revenue_orders
                FROM orders
                WHERE order_date >= %s
            """, (start_date,))
            basic_metrics = cursor.fetchone()
            
            # Order items metrics
            cursor.execute("""
                SELECT 
                    SUM(oi.total_price) as total_revenue,
                    AVG(oi.total_price) as avg_order_value,
                    COUNT(DISTINCT oi.product_id) as total_products
                FROM order_items oi
                JOIN orders o ON o.id = oi.order_id
                WHERE o.order_date >= %s
            """, (start_date,))
            items_metrics = cursor.fetchone()
            
            # Combine metrics
            metrics = {
                'total_orders': basic_metrics['total_orders'],
                'total_customers': basic_metrics['total_customers'], 
                'total_revenue': items_metrics['total_revenue'] or 0,
                'avg_order_value': items_metrics['avg_order_value'] or 0,
                'total_products': items_metrics['total_products'] or 0
            }
        else:
            # Use pre-computed statistics for better performance on "all" filter
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    COUNT(DISTINCT unified_customer_id) as total_customers
                FROM orders
            """)
            basic_metrics = cursor.fetchone()
            
            cursor.execute("""
                SELECT 
                    SUM(total_price) as total_revenue,
                    AVG(total_price) as avg_order_value,
                    COUNT(DISTINCT product_id) as total_products
                FROM order_items
            """)
            items_metrics = cursor.fetchone()
            
            # Combine metrics
            metrics = {
                'total_orders': basic_metrics['total_orders'],
                'total_customers': basic_metrics['total_customers'],
                'total_revenue': items_metrics['total_revenue'] or 0,
                'avg_order_value': items_metrics['avg_order_value'] or 0,
                'total_products': items_metrics['total_products'] or 0
            }
        
        # Get top selling product - Optimized query
        if start_date:
            cursor.execute("""
                SELECT oi.product_name
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                WHERE o.order_date >= %s AND oi.product_name IS NOT NULL
                GROUP BY oi.product_name
                ORDER BY SUM(oi.total_price) DESC
                LIMIT 1
            """, (start_date,))
        else:
            cursor.execute("""
                SELECT product_name
                FROM order_items
                WHERE product_name IS NOT NULL
                GROUP BY product_name
                ORDER BY SUM(total_price) DESC
                LIMIT 1
            """)
        
        top_product = cursor.fetchone()
        
        result = {
            "success": True,
            "time_filter": time_filter,
            "start_date": start_date.isoformat() if start_date else None,
            "total_orders": metrics['total_orders'],
            "total_revenue": float(metrics['total_revenue']),
            "total_customers": metrics['total_customers'],
            "avg_order_value": float(metrics['avg_order_value']),
            "total_products": metrics['total_products'],
            "top_selling_product": top_product['product_name'] if top_product else 'N/A',
            "time_period": time_filter,
            "cache_hit": False
        }
        
        # Cache the result for faster subsequent requests
        if redis_client:
            try:
                # Use shorter TTL for recent data, longer for older data
                cache_ttl = 300 if time_filter in ['today', '7days'] else 1800  # 5 min vs 30 min
                redis_client.setex(cache_key, cache_ttl, json.dumps(result))
                logger.info("Dashboard metrics cached", time_filter=time_filter, ttl=cache_ttl)
            except Exception as e:
                logger.warning("Cache write failed", error=str(e))
        
        logger.info("Dashboard metrics fetched", time_filter=time_filter, metrics=result)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch dashboard metrics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/api/v1/analytics/pos-vs-oe-revenue")
async def get_pos_vs_oe_revenue(
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, all")  # Changed default to 7days
):
    """
    Get POS vs OE revenue comparison analytics (100% LIVE DATA)
    Returns: breakdown of revenue by order type with detailed metrics
    """
    conn = None
    cursor = None
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        # Create fresh database connection
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query POS vs OE revenue breakdown - Optimized with separate queries
        if start_date:
            # First get order-level metrics
            cursor.execute("""
                SELECT 
                    order_type,
                    COUNT(*) as total_orders,
                    COUNT(DISTINCT unified_customer_id) as unique_customers,
                    MIN(order_date) as earliest_order,
                    MAX(order_date) as latest_order
                FROM orders
                WHERE order_date >= %s
                GROUP BY order_type
            """, (start_date,))
            order_metrics = {row['order_type']: row for row in cursor.fetchall()}
            
            # Then get order items metrics
            cursor.execute("""
                SELECT 
                    o.order_type,
                    SUM(oi.total_price) as total_revenue,
                    AVG(oi.total_price) as avg_order_value,
                    COUNT(DISTINCT oi.product_id) as unique_products
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                WHERE o.order_date >= %s
                GROUP BY o.order_type
            """, (start_date,))
            items_metrics = {row['order_type']: row for row in cursor.fetchall()}
        else:
            # Optimized for "all" time filter
            cursor.execute("""
                SELECT 
                    order_type,
                    COUNT(*) as total_orders,
                    COUNT(DISTINCT unified_customer_id) as unique_customers,
                    MIN(order_date) as earliest_order,
                    MAX(order_date) as latest_order
                FROM orders
                GROUP BY order_type
            """)
            order_metrics = {row['order_type']: row for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT 
                    o.order_type,
                    SUM(oi.total_price) as total_revenue,
                    AVG(oi.total_price) as avg_order_value,
                    COUNT(DISTINCT oi.product_id) as unique_products
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                GROUP BY o.order_type
            """)
            items_metrics = {row['order_type']: row for row in cursor.fetchall()}
        
        # Combine the metrics
        order_type_metrics = []
        for order_type in order_metrics.keys():
            order_data = order_metrics[order_type]
            items_data = items_metrics.get(order_type, {})
            
            order_type_metrics.append({
                'order_type': order_type,
                'total_orders': order_data['total_orders'],
                'total_revenue': float(items_data.get('total_revenue', 0) or 0),
                'unique_customers': order_data['unique_customers'],
                'avg_order_value': float(items_data.get('avg_order_value', 0) or 0),
                'unique_products': items_data.get('unique_products', 0) or 0,
                'earliest_order': order_data['earliest_order'],
                'latest_order': order_data['latest_order']
            })
        
        # Sort by total revenue
        order_type_metrics.sort(key=lambda x: x['total_revenue'], reverse=True)
        
        # Calculate totals and percentages
        total_revenue = sum(row['total_revenue'] for row in order_type_metrics)
        total_orders = sum(row['total_orders'] for row in order_type_metrics)
        
        # Format results with percentages
        revenue_breakdown = []
        for row in order_type_metrics:
            revenue = row['total_revenue']
            orders = row['total_orders']
            
            revenue_breakdown.append({
                "order_type": row['order_type'],
                "total_orders": orders,
                "total_revenue": revenue,
                "unique_customers": row['unique_customers'],
                "avg_order_value": row['avg_order_value'],
                "unique_products": row['unique_products'],
                "revenue_percentage": round((revenue / total_revenue * 100), 2) if total_revenue > 0 else 0,
                "orders_percentage": round((orders / total_orders * 100), 2) if total_orders > 0 else 0,
                "earliest_order": row['earliest_order'].isoformat() if row['earliest_order'] else None,
                "latest_order": row['latest_order'].isoformat() if row['latest_order'] else None
            })
        
        # Get top products per order type
        top_products_per_type = {}
        for row in order_type_metrics:
            order_type = row.get('order_type')
            if start_date:
                cursor.execute("""
                    SELECT oi.product_name, SUM(oi.total_price) as revenue, COUNT(*) as sales_count
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.id
                    WHERE o.order_type = %s AND o.order_date >= %s
                    GROUP BY oi.product_name
                    ORDER BY revenue DESC
                    LIMIT 3
                """, (order_type, start_date))
            else:
                cursor.execute("""
                    SELECT oi.product_name, SUM(oi.total_price) as revenue, COUNT(*) as sales_count
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.id
                    WHERE o.order_type = %s
                    GROUP BY oi.product_name
                    ORDER BY revenue DESC
                    LIMIT 3
                """, (order_type,))
            
            top_products = cursor.fetchall()
            top_products_per_type[order_type] = [
                {
                    "product_name": p.get('product_name'),
                    "revenue": float(p.get('revenue', 0) or 0),
                    "sales_count": p.get('sales_count', 0) or 0
                }
                for p in top_products
            ]
        
        result = {
            "success": True,
            "time_filter": time_filter,
            "start_date": start_date.isoformat() if start_date else None,
            "summary": {
                "total_revenue": total_revenue,
                "total_orders": total_orders,
                "order_types_count": len(order_type_metrics)
            },
            "revenue_breakdown": revenue_breakdown,
            "top_products_per_type": top_products_per_type,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("POS vs OE revenue analytics fetched", time_filter=time_filter, types_found=len(order_type_metrics))
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch POS vs OE revenue analytics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/api/v1/analytics/revenue-trend")
async def get_revenue_trend(
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, mtd, 90days, 6months, 1year, all"),
    period: str = Query("daily", description="Period: daily, weekly, monthly")
):
    """
    Get revenue trend data over time (100% LIVE DATA)
    Returns: time series data for revenue trends with configurable periods
    """
    conn = None
    cursor = None
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        # Create fresh database connection
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Determine grouping based on period and time filter
        if period == "daily":
            date_trunc = "day"
            if time_filter == "today":
                date_trunc = "hour"
        elif period == "weekly":
            date_trunc = "week"
        elif period == "monthly":
            date_trunc = "month"
        else:
            date_trunc = "day"
        
        # Query revenue trend data
        if start_date:
            cursor.execute(f"""
                SELECT 
                    DATE_TRUNC('{date_trunc}', o.order_date) as period_start,
                    SUM(oi.total_price) as total_revenue,
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT o.unified_customer_id) as unique_customers,
                    AVG(oi.total_price) as avg_order_value
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                WHERE o.order_date >= %s
                GROUP BY DATE_TRUNC('{date_trunc}', o.order_date)
                ORDER BY period_start ASC
                LIMIT 30
            """, (start_date,))
        else:
            # For "all" time, limit to last 30 periods
            cursor.execute(f"""
                SELECT 
                    DATE_TRUNC('{date_trunc}', o.order_date) as period_start,
                    SUM(oi.total_price) as total_revenue,
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT o.unified_customer_id) as unique_customers,
                    AVG(oi.total_price) as avg_order_value
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                WHERE o.order_date >= NOW() - INTERVAL '6 months'
                GROUP BY DATE_TRUNC('{date_trunc}', o.order_date)
                ORDER BY period_start ASC
                LIMIT 30
            """)
        
        trend_data = cursor.fetchall()
        
        # Format the data
        formatted_data = []
        max_revenue = 0
        total_revenue = 0
        
        for row in trend_data:
            revenue = float(row['total_revenue'] or 0)
            max_revenue = max(max_revenue, revenue)
            total_revenue += revenue
            
            # Format date based on period
            period_start = row['period_start']
            if date_trunc == "hour":
                label = period_start.strftime("%H:%M")
            elif date_trunc == "day":
                label = period_start.strftime("%a")  # Mon, Tue, etc.
            elif date_trunc == "week":
                label = f"W{period_start.isocalendar()[1]}"  # W45, W46, etc.
            elif date_trunc == "month":
                label = period_start.strftime("%b")  # Jan, Feb, etc.
            else:
                label = period_start.strftime("%m/%d")
            
            formatted_data.append({
                "period": period_start.isoformat(),
                "label": label,
                "total_revenue": revenue,
                "total_orders": row['total_orders'],
                "unique_customers": row['unique_customers'],
                "avg_order_value": float(row['avg_order_value'] or 0),
                "percentage": round((revenue / max_revenue * 100), 1) if max_revenue > 0 else 0
            })
        
        result = {
            "success": True,
            "time_filter": time_filter,
            "period": period,
            "start_date": start_date.isoformat() if start_date else None,
            "trend_data": formatted_data,
            "summary": {
                "total_periods": len(formatted_data),
                "total_revenue": total_revenue,
                "max_revenue": max_revenue,
                "avg_revenue_per_period": total_revenue / len(formatted_data) if formatted_data else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Revenue trend data fetched", 
                   time_filter=time_filter, 
                   period=period, 
                   data_points=len(formatted_data))
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch revenue trend data", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Customer Analytics Endpoints
@app.get("/api/v1/analytics/customers")
async def get_customer_analytics(
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, all"),
    limit: int = Query(50, description="Maximum number of customers to return")
):
    """
    Get real-time customer analytics with names and spending data (100% LIVE DATA)
    Shows customer names instead of IDs for better profiling
    """
    conn = None
    cursor = None
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        # Create fresh database connection
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get customer analytics with names
        if start_date:
            cursor.execute("""
                SELECT 
                    o.unified_customer_id,
                    COALESCE(o.customer_name, 'Unknown Customer') as customer_name,
                    COALESCE(o.customer_city, 'Unknown City') as customer_city,
                    COUNT(DISTINCT o.id) as total_orders,
                    SUM(o.total_price) as total_spent,
                    AVG(o.total_price) as avg_order_value,
                    COUNT(DISTINCT oi.product_id) as unique_products_purchased,
                    MAX(o.order_date) as last_order_date,
                    MIN(o.order_date) as first_order_date,
                    CASE 
                        WHEN SUM(o.total_price) > 100000 THEN 'VIP'
                        WHEN SUM(o.total_price) > 50000 THEN 'Premium' 
                        WHEN SUM(o.total_price) > 20000 THEN 'High Value'
                        ELSE 'Regular'
                    END as customer_segment
                FROM orders o
                LEFT JOIN order_items oi ON o.id = oi.order_id
                WHERE o.order_date >= %s 
                AND o.unified_customer_id IS NOT NULL
                GROUP BY o.unified_customer_id, o.customer_name, o.customer_city
                ORDER BY total_spent DESC
                LIMIT %s
            """, (start_date, limit))
        else:
            cursor.execute("""
                SELECT 
                    o.unified_customer_id,
                    COALESCE(o.customer_name, 'Unknown Customer') as customer_name,
                    COALESCE(o.customer_city, 'Unknown City') as customer_city,
                    COUNT(DISTINCT o.id) as total_orders,
                    SUM(o.total_price) as total_spent,
                    AVG(o.total_price) as avg_order_value,
                    COUNT(DISTINCT oi.product_id) as unique_products_purchased,
                    MAX(o.order_date) as last_order_date,
                    MIN(o.order_date) as first_order_date,
                    CASE 
                        WHEN SUM(o.total_price) > 100000 THEN 'VIP'
                        WHEN SUM(o.total_price) > 50000 THEN 'Premium' 
                        WHEN SUM(o.total_price) > 20000 THEN 'High Value'
                        ELSE 'Regular'
                    END as customer_segment
                FROM orders o
                LEFT JOIN order_items oi ON o.id = oi.order_id
                WHERE o.unified_customer_id IS NOT NULL
                GROUP BY o.unified_customer_id, o.customer_name, o.customer_city
                ORDER BY total_spent DESC
                LIMIT %s
            """, (limit,))
        
        customers = cursor.fetchall()
        
        # Convert to list of dicts and format data
        customer_list = []
        for customer in customers:
            customer_dict = dict(customer)
            # Format dates
            if customer_dict.get('last_order_date'):
                customer_dict['last_order_date'] = customer_dict['last_order_date'].isoformat()
            if customer_dict.get('first_order_date'):
                customer_dict['first_order_date'] = customer_dict['first_order_date'].isoformat()
            
            # Format monetary values
            customer_dict['total_spent'] = float(customer_dict['total_spent'] or 0)
            customer_dict['avg_order_value'] = float(customer_dict['avg_order_value'] or 0)
            
            customer_list.append(customer_dict)
        
        # Get summary statistics
        if customers:
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT unified_customer_id) as total_customers,
                    SUM(total_price) as total_revenue,
                    AVG(total_price) as avg_customer_value
                FROM orders 
                WHERE unified_customer_id IS NOT NULL
                """ + ("AND order_date >= %s" if start_date else ""), 
                (start_date,) if start_date else ())
            
            summary = cursor.fetchone()
            summary_stats = {
                'total_customers': summary['total_customers'],
                'total_revenue': float(summary['total_revenue'] or 0),
                'avg_customer_value': float(summary['avg_customer_value'] or 0)
            }
        else:
            summary_stats = {
                'total_customers': 0,
                'total_revenue': 0,
                'avg_customer_value': 0
            }
        
        logger.info("Customer analytics fetched", 
                   time_filter=time_filter, 
                   count=len(customer_list))
        
        return {
            "success": True,
            "time_filter": time_filter,
            "start_date": start_date.isoformat() if start_date else None,
            "customers": customer_list,
            "count": len(customer_list),
            "summary": summary_stats
        }
        
    except Exception as e:
        logger.error("Failed to fetch customer analytics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/api/v1/analytics/customer-segments")
async def get_customer_segments(
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, all")
):
    """
    Get customer segmentation analytics with names and geographic distribution
    """
    conn = None
    cursor = None
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        # Create fresh database connection
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get customer segment distribution
        if start_date:
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN SUM(total_price) > 100000 THEN 'VIP'
                        WHEN SUM(total_price) > 50000 THEN 'Premium' 
                        WHEN SUM(total_price) > 20000 THEN 'High Value'
                        ELSE 'Regular'
                    END as segment,
                    COUNT(DISTINCT unified_customer_id) as customer_count,
                    SUM(total_price) as total_revenue,
                    AVG(total_price) as avg_spending
                FROM orders 
                WHERE order_date >= %s AND unified_customer_id IS NOT NULL
                GROUP BY unified_customer_id
            """, (start_date,))
        else:
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN SUM(total_price) > 100000 THEN 'VIP'
                        WHEN SUM(total_price) > 50000 THEN 'Premium' 
                        WHEN SUM(total_price) > 20000 THEN 'High Value'
                        ELSE 'Regular'
                    END as segment,
                    COUNT(DISTINCT unified_customer_id) as customer_count,
                    SUM(total_price) as total_revenue,
                    AVG(total_price) as avg_spending
                FROM orders 
                WHERE unified_customer_id IS NOT NULL
                GROUP BY unified_customer_id
            """)
        
        # Process segment data
        segment_data = {}
        for row in cursor.fetchall():
            segment = row[0]  # This is the calculated segment
            if segment not in segment_data:
                segment_data[segment] = {
                    'customer_count': 0,
                    'total_revenue': 0,
                    'avg_spending': 0
                }
            segment_data[segment]['customer_count'] += 1
            segment_data[segment]['total_revenue'] += float(row[2])
            
        # Calculate averages
        for segment in segment_data:
            if segment_data[segment]['customer_count'] > 0:
                segment_data[segment]['avg_spending'] = segment_data[segment]['total_revenue'] / segment_data[segment]['customer_count']
        
        # Get geographic distribution
        if start_date:
            cursor.execute("""
                SELECT 
                    COALESCE(customer_city, 'Unknown') as city,
                    COUNT(DISTINCT unified_customer_id) as customer_count,
                    SUM(total_price) as total_revenue
                FROM orders 
                WHERE order_date >= %s AND unified_customer_id IS NOT NULL
                GROUP BY customer_city
                ORDER BY total_revenue DESC
                LIMIT 10
            """, (start_date,))
        else:
            cursor.execute("""
                SELECT 
                    COALESCE(customer_city, 'Unknown') as city,
                    COUNT(DISTINCT unified_customer_id) as customer_count,
                    SUM(total_price) as total_revenue
                FROM orders 
                WHERE unified_customer_id IS NOT NULL
                GROUP BY customer_city
                ORDER BY total_revenue DESC
                LIMIT 10
            """)
        
        geographic_data = []
        for row in cursor.fetchall():
            geographic_data.append({
                'city': row['city'],
                'customer_count': row['customer_count'],
                'total_revenue': float(row['total_revenue'] or 0)
            })
        
        logger.info("Customer segments fetched", 
                   time_filter=time_filter,
                   segments=len(segment_data),
                   cities=len(geographic_data))
        
        return {
            "success": True,
            "time_filter": time_filter,
            "start_date": start_date.isoformat() if start_date else None,
            "segments": segment_data,
            "geographic_distribution": geographic_data
        }
        
    except Exception as e:
        logger.error("Failed to fetch customer segments", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/api/v1/analytics/customers")
async def get_customer_analytics(
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, mtd, 90days, 6months, 1year, all"),
    limit: int = Query(50, description="Maximum number of customers to return")
):
    """
    Get customer analytics with customer names (not just IDs) for dashboard display
    Shows top spending customers with their actual names and spending details
    """
    conn = None
    cursor = None
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        # Create fresh database connection
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get top customers with their names and spending data
        if start_date:
            cursor.execute("""
                SELECT 
                    o.unified_customer_id as customer_id,
                    MAX(o.customer_name) as customer_name,
                    MAX(o.customer_city) as customer_city,
                    COUNT(DISTINCT o.id) as total_orders,
                    SUM(o.total_price) as total_spent,
                    AVG(o.total_price) as avg_order_value,
                    MAX(o.order_date) as last_order_date,
                    MIN(o.order_date) as first_order_date,
                    CASE 
                        WHEN SUM(o.total_price) > 100000 THEN 'VIP'
                        WHEN SUM(o.total_price) > 50000 THEN 'Premium'
                        WHEN SUM(o.total_price) > 20000 THEN 'High Value'
                        ELSE 'Standard'
                    END as customer_status
                FROM orders o
                WHERE o.unified_customer_id IS NOT NULL 
                AND o.order_date >= %s
                GROUP BY o.unified_customer_id
                HAVING SUM(o.total_price) > 0
                ORDER BY total_spent DESC
                LIMIT %s
            """, (start_date, limit))
        else:
            cursor.execute("""
                SELECT 
                    o.unified_customer_id as customer_id,
                    MAX(o.customer_name) as customer_name,
                    MAX(o.customer_city) as customer_city,
                    COUNT(DISTINCT o.id) as total_orders,
                    SUM(o.total_price) as total_spent,
                    AVG(o.total_price) as avg_order_value,
                    MAX(o.order_date) as last_order_date,
                    MIN(o.order_date) as first_order_date,
                    CASE 
                        WHEN SUM(o.total_price) > 100000 THEN 'VIP'
                        WHEN SUM(o.total_price) > 50000 THEN 'Premium'
                        WHEN SUM(o.total_price) > 20000 THEN 'High Value'
                        ELSE 'Standard'
                    END as customer_status
                FROM orders o
                WHERE o.unified_customer_id IS NOT NULL
                GROUP BY o.unified_customer_id
                HAVING SUM(o.total_price) > 0
                ORDER BY total_spent DESC
                LIMIT %s
            """, (limit,))
        
        customers = cursor.fetchall()
        
        # Convert to list and format dates
        customer_list = []
        for customer in customers:
            customer_dict = dict(customer)
            # Format dates for JSON serialization
            if customer_dict.get('last_order_date'):
                customer_dict['last_order_date'] = customer_dict['last_order_date'].isoformat()
            if customer_dict.get('first_order_date'):
                customer_dict['first_order_date'] = customer_dict['first_order_date'].isoformat()
            
            # Ensure customer has a display name
            if not customer_dict.get('customer_name'):
                customer_dict['customer_name'] = f"Customer {customer_dict['customer_id']}"
            
            # Convert decimal values to float for JSON
            if customer_dict.get('total_spent'):
                customer_dict['total_spent'] = float(customer_dict['total_spent'])
            if customer_dict.get('avg_order_value'):
                customer_dict['avg_order_value'] = float(customer_dict['avg_order_value'])
                
            customer_list.append(customer_dict)
        
        logger.info("Customer analytics fetched", 
                   time_filter=time_filter, 
                   count=len(customer_list))
        
        return {
            "success": True,
            "time_filter": time_filter,
            "start_date": start_date.isoformat() if start_date else None,
            "customers": customer_list,
            "count": len(customer_list),
            "summary": {
                "total_customers": len(customer_list),
                "total_revenue": sum(c['total_spent'] for c in customer_list),
                "vip_customers": len([c for c in customer_list if c['customer_status'] == 'VIP']),
                "premium_customers": len([c for c in customer_list if c['customer_status'] == 'Premium'])
            }
        }
        
    except Exception as e:
        logger.error("Failed to fetch customer analytics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/api/v1/analytics/customers")
async def get_customer_analytics(
    time_filter: str = Query("all", description="Time filter: today, 7days, 30days, all"),
    limit: int = Query(50, description="Maximum number of customers to return"),
    sort_by: str = Query("total_spent", description="Sort by: total_spent, total_orders, avg_order_value")
):
    """
    Get customer analytics with names instead of IDs (for customer profiling dashboard)
    Shows top spending customers with their actual names and detailed metrics
    """
    conn = None
    cursor = None
    try:
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        # Create fresh database connection
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build the query based on time filter
        if start_date:
            cursor.execute("""
                SELECT 
                    o.unified_customer_id as customer_id,
                    COALESCE(NULLIF(TRIM(o.customer_name), ''), 
                             CONCAT('Customer ', SUBSTRING(o.unified_customer_id, 1, 8))) as customer_name,
                    COALESCE(o.customer_city, 'N/A') as customer_city,
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT oi.product_id) as unique_products_purchased,
                    SUM(o.total_price) as total_spent,
                    AVG(o.total_price) as avg_order_value,
                    MIN(o.order_date) as first_order_date,
                    MAX(o.order_date) as last_order_date,
                    CASE 
                        WHEN SUM(o.total_price) > 100000 THEN 'VIP'
                        WHEN SUM(o.total_price) > 50000 THEN 'Premium'
                        WHEN SUM(o.total_price) > 20000 THEN 'High Value'
                        WHEN COUNT(DISTINCT o.id) > 10 THEN 'Frequent'
                        ELSE 'Regular'
                    END as status
                FROM orders o
                LEFT JOIN order_items oi ON o.id = oi.order_id
                WHERE o.unified_customer_id IS NOT NULL
                AND o.order_date >= %s
                GROUP BY o.unified_customer_id, o.customer_name, o.customer_city
                ORDER BY {} DESC
                LIMIT %s
            """.format(sort_by), (start_date, limit))
        else:
            cursor.execute("""
                SELECT 
                    o.unified_customer_id as customer_id,
                    COALESCE(NULLIF(TRIM(o.customer_name), ''), 
                             CONCAT('Customer ', SUBSTRING(o.unified_customer_id, 1, 8))) as customer_name,
                    COALESCE(o.customer_city, 'N/A') as customer_city,
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT oi.product_id) as unique_products_purchased,
                    SUM(o.total_price) as total_spent,
                    AVG(o.total_price) as avg_order_value,
                    MIN(o.order_date) as first_order_date,
                    MAX(o.order_date) as last_order_date,
                    CASE 
                        WHEN SUM(o.total_price) > 100000 THEN 'VIP'
                        WHEN SUM(o.total_price) > 50000 THEN 'Premium'
                        WHEN SUM(o.total_price) > 20000 THEN 'High Value'
                        WHEN COUNT(DISTINCT o.id) > 10 THEN 'Frequent'
                        ELSE 'Regular'
                    END as status
                FROM orders o
                LEFT JOIN order_items oi ON o.id = oi.order_id
                WHERE o.unified_customer_id IS NOT NULL
                GROUP BY o.unified_customer_id, o.customer_name, o.customer_city
                ORDER BY {} DESC
                LIMIT %s
            """.format(sort_by), (limit,))
        
        customers = cursor.fetchall()
        
        # Convert to list of dicts and format dates/numbers
        customer_list = []
        for customer in customers:
            customer_dict = dict(customer)
            
            # Format dates
            if customer_dict.get('first_order_date'):
                customer_dict['first_order_date'] = customer_dict['first_order_date'].isoformat()
            if customer_dict.get('last_order_date'):
                customer_dict['last_order_date'] = customer_dict['last_order_date'].isoformat()
            
            # Format currency values
            if customer_dict.get('total_spent'):
                customer_dict['total_spent'] = float(customer_dict['total_spent'])
            if customer_dict.get('avg_order_value'):
                customer_dict['avg_order_value'] = float(customer_dict['avg_order_value'])
            
            customer_list.append(customer_dict)
        
        logger.info(f"Customer analytics fetched", 
                   time_filter=time_filter, 
                   count=len(customer_list),
                   sort_by=sort_by)
        
        return {
            "success": True,
            "time_filter": time_filter,
            "start_date": start_date.isoformat() if start_date else None,
            "customers": customer_list,
            "count": len(customer_list),
            "sort_by": sort_by
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch customer analytics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))  # Use Heroku's PORT or default to 8001
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=settings.debug
    )