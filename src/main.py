"""
Recommendation Engine Service Main Application
Core ML algorithms and recommendation inference with Redis caching and PostgreSQL integration
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends, status
from src.auth import (
    authenticate_user, create_access_token, update_last_login,
    get_current_active_user, User, LoginRequest, Token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
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
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://master-dashboard.netlify.app",
        "https://*.netlify.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
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


# ============================================
# AUTHENTICATION ENDPOINTS
# ============================================

@app.post("/api/v1/auth/login", response_model=Token, tags=["Authentication"])
async def login(login_data: LoginRequest):
    """
    Login endpoint - authenticate user and return JWT token
    
    Default credentials:
    - Email: admin@mastergroup.com
    - Password: admin123
    """
    user = authenticate_user(login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login
    update_last_login(user.email)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.email}")
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/v1/auth/me", response_model=User, tags=["Authentication"])
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current logged in user information"""
    return current_user

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

# ============================================================================
# ML RECOMMENDATION ENDPOINTS (PHASE 1.5)
# ============================================================================

from algorithms.ml_recommendation_service import get_ml_service
import random

# A/B Testing configuration
AB_TEST_CONFIG = {
    'ml_recommendation_rollout': 50,  # % of traffic that gets ML recommendations
    'enabled_dashboards': ['analytics', 'customer_detail', 'product_recommendations']
}

@app.post("/api/v1/ml/train")
async def train_ml_models(
    time_filter: str = Query("30days", description="Time filter for training data"),
    force_retrain: bool = Query(False, description="Force retraining even if models exist")
):
    """
    Train all ML recommendation models
    
    Training includes:
    - Collaborative Filtering (User-based + Item-based)
    - Content-Based Filtering (Product similarity)
    - Matrix Factorization (SVD)
    - Popularity-Based (Demographic)
    
    Expected training time: 60-90 seconds for 30 days of data
    """
    try:
        logger.info("Starting ML model training", 
                   time_filter=time_filter, 
                   force_retrain=force_retrain)
        
        ml_service = get_ml_service()
        results = ml_service.train_all_models(
            time_filter=time_filter,
            force_retrain=force_retrain
        )
        
        logger.info("ML model training completed", 
                   successful_models=results.get('successful_models'),
                   total_time=results.get('total_training_time_seconds'))
        
        return {
            "status": "success",
            "message": f"Trained {results.get('successful_models')}/{results.get('total_models')} models",
            "training_time_seconds": results.get('total_training_time_seconds'),
            "results": results
        }
        
    except Exception as e:
        logger.error("ML model training failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/api/v1/ml/recommendations/{user_id}")
async def get_ml_recommendations(
    user_id: str,
    n_recommendations: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    algorithm_weights: Optional[str] = Query(None, description="Custom algorithm weights (JSON)")
):
    """
    Get ML-based hybrid recommendations for a user
    
    Combines:
    - Collaborative Filtering (40%)
    - Matrix Factorization (30%)
    - Content-Based (20%)
    - Popularity-Based (10%)
    
    Returns personalized product recommendations with confidence scores
    """
    try:
        logger.info("Generating ML recommendations", 
                   user_id=user_id, 
                   n_recommendations=n_recommendations)
        
        ml_service = get_ml_service()
        
        # Train if not trained
        if not ml_service.is_trained:
            logger.warning("Models not trained, training now...")
            ml_service.train_all_models(time_filter='30days')
        
        # Parse custom weights if provided
        weights = None
        if algorithm_weights:
            try:
                weights = json.loads(algorithm_weights)
            except:
                logger.warning("Invalid algorithm_weights JSON, using defaults")
        
        recommendations = ml_service.get_hybrid_recommendations(
            user_id=user_id,
            n_recommendations=n_recommendations,
            algorithm_weights=weights
        )
        
        logger.info("ML recommendations generated", 
                   user_id=user_id, 
                   count=len(recommendations))
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "algorithm": "hybrid_ml",
            "trained_at": ml_service.training_timestamp.isoformat() if ml_service.training_timestamp else None,
            "count": len(recommendations)
        }
        
    except Exception as e:
        logger.error("Failed to generate ML recommendations", 
                    user_id=user_id, 
                    error=str(e), 
                    exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


@app.get("/api/v1/ml/status")
async def get_ml_status():
    """
    Get ML service training status and metadata
    
    Returns information about:
    - Training status
    - Last training timestamp
    - Model metrics
    - Algorithm performance
    """
    try:
        ml_service = get_ml_service()
        
        return {
            "is_trained": ml_service.is_trained,
            "training_timestamp": ml_service.training_timestamp.isoformat() if ml_service.training_timestamp else None,
            "model_metadata": ml_service.model_metadata,
            "service_version": "1.0.0",
            "algorithms": {
                "collaborative_filtering": ml_service.collaborative_engine is not None,
                "content_based": ml_service.content_based_engine is not None,
                "matrix_factorization": ml_service.matrix_factorization_engine is not None,
                "popularity_based": ml_service.popularity_engine is not None
            }
        }
        
    except Exception as e:
        logger.error("Failed to get ML status", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/collaborative-products")
async def get_ml_collaborative_products(
    time_filter: str = Query("30days", description="Time filter"),
    limit: int = Query(20, ge=1, le=100, description="Number of products"),
    use_ml: bool = Query(True, description="Use ML algorithms or SQL fallback")
):
    """
    Get products that are frequently bought together using ML
    
    Uses Collaborative Filtering to find product associations
    Falls back to SQL-based queries if ML not trained
    """
    try:
        logger.info("Fetching ML collaborative products", 
                   time_filter=time_filter, 
                   limit=limit,
                   use_ml=use_ml)
        
        if use_ml:
            ml_service = get_ml_service()
            
            # Train if not trained
            if not ml_service.is_trained:
                logger.warning("Models not trained, training now...")
                ml_service.train_all_models(time_filter=time_filter)
            
            # Use collaborative filtering engine
            if ml_service.collaborative_engine and ml_service.collaborative_engine.is_trained:
                # Get top products by interaction count
                conn = psycopg2.connect(**get_pg_connection_params())
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT 
                        oi.product_id,
                        MAX(oi.product_name) as product_name,
                        COUNT(DISTINCT o.unified_customer_id) as customer_count,
                        COUNT(DISTINCT oi.order_id) as order_count,
                        SUM(oi.total_price) as total_revenue,
                        AVG(oi.unit_price) as avg_price
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.id
                    WHERE o.unified_customer_id IS NOT NULL
                    AND oi.product_id IS NOT NULL
                    GROUP BY oi.product_id
                    ORDER BY customer_count DESC, total_revenue DESC
                    LIMIT %s
                """, (limit,))
                
                products = cursor.fetchall()
                cursor.close()
                conn.close()
                
                # Format response
                product_list = []
                for product in products:
                    product_dict = dict(product)
                    product_dict['total_revenue'] = float(product_dict['total_revenue'] or 0)
                    product_dict['avg_price'] = float(product_dict['avg_price'] or 0)
                    product_dict['algorithm'] = 'collaborative_filtering_ml'
                    product_list.append(product_dict)
                
                logger.info("ML collaborative products fetched", count=len(product_list))
                
                return {
                    "products": product_list,
                    "algorithm": "ml_collaborative_filtering",
                    "time_filter": time_filter,
                    "count": len(product_list)
                }
            else:
                logger.warning("Collaborative filtering not trained, falling back to SQL")
                use_ml = False
        
        # SQL fallback
        if not use_ml:
            conn = psycopg2.connect(**get_pg_connection_params())
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            days_map = {
                '7days': 7,
                '30days': 30,
                '90days': 90,
                '6months': 180,
                '1year': 365,
                'all': None
            }
            days = days_map.get(time_filter)
            
            query = """
                SELECT 
                    oi.product_id,
                    MAX(oi.product_name) as product_name,
                    COUNT(DISTINCT o.unified_customer_id) as customer_count,
                    COUNT(DISTINCT oi.order_id) as order_count,
                    SUM(oi.total_price) as total_revenue,
                    AVG(oi.unit_price) as avg_price
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                WHERE o.unified_customer_id IS NOT NULL
                AND oi.product_id IS NOT NULL
            """
            
            params = []
            if days:
                query += " AND o.order_date >= NOW() - INTERVAL '%s days'"
                params.append(days)
            
            query += " GROUP BY oi.product_id ORDER BY customer_count DESC, total_revenue DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            products = cursor.fetchall()
            cursor.close()
            conn.close()
            
            product_list = []
            for product in products:
                product_dict = dict(product)
                product_dict['total_revenue'] = float(product_dict['total_revenue'] or 0)
                product_dict['avg_price'] = float(product_dict['avg_price'] or 0)
                product_dict['algorithm'] = 'sql_fallback'
                product_list.append(product_dict)
            
            return {
                "products": product_list,
                "algorithm": "sql_fallback",
                "time_filter": time_filter,
                "count": len(product_list)
            }
        
    except Exception as e:
        logger.error("Failed to fetch collaborative products", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ab-test/variant")
async def get_ab_test_variant(
    dashboard: str = Query(..., description="Dashboard name"),
    user_id: Optional[str] = Query(None, description="User ID for consistent assignment")
):
    """
    Get A/B test variant for a dashboard
    
    Returns whether to show ML or SQL-based recommendations
    Ensures consistent variant assignment per user
    """
    try:
        # Check if ML rollout is enabled for this dashboard
        if dashboard not in AB_TEST_CONFIG['enabled_dashboards']:
            return {
                "dashboard": dashboard,
                "variant": "control",
                "algorithm": "sql",
                "reason": "Dashboard not enabled for ML testing"
            }
        
        # Consistent assignment based on user_id hash
        if user_id:
            # Use hash for consistent assignment
            hash_value = hash(user_id) % 100
            use_ml = hash_value < AB_TEST_CONFIG['ml_recommendation_rollout']
        else:
            # Random assignment for anonymous users
            use_ml = random.randint(0, 99) < AB_TEST_CONFIG['ml_recommendation_rollout']
        
        variant = "treatment" if use_ml else "control"
        algorithm = "ml" if use_ml else "sql"
        
        logger.info("A/B test variant assigned", 
                   dashboard=dashboard, 
                   user_id=user_id, 
                   variant=variant)
        
        return {
            "dashboard": dashboard,
            "user_id": user_id,
            "variant": variant,
            "algorithm": algorithm,
            "rollout_percentage": AB_TEST_CONFIG['ml_recommendation_rollout']
        }
        
    except Exception as e:
        logger.error("Failed to get A/B test variant", error=str(e), exc_info=True)
        # Default to control on error
        return {
            "dashboard": dashboard,
            "variant": "control",
            "algorithm": "sql",
            "reason": "Error in variant assignment"
        }


@app.post("/api/v1/ab-test/configure")
async def configure_ab_test(
    rollout_percentage: int = Query(..., ge=0, le=100, description="ML rollout percentage"),
    enabled_dashboards: Optional[List[str]] = Query(None, description="Dashboards to enable ML for")
):
    """
    Configure A/B testing parameters
    
    Allows dynamic adjustment of:
    - ML rollout percentage (0-100%)
    - Which dashboards have ML enabled
    """
    try:
        AB_TEST_CONFIG['ml_recommendation_rollout'] = rollout_percentage
        
        if enabled_dashboards:
            AB_TEST_CONFIG['enabled_dashboards'] = enabled_dashboards
        
        logger.info("A/B test configuration updated", 
                   rollout_percentage=rollout_percentage,
                   enabled_dashboards=AB_TEST_CONFIG['enabled_dashboards'])
        
        return {
            "status": "success",
            "message": "A/B test configuration updated",
            "config": AB_TEST_CONFIG
        }
        
    except Exception as e:
        logger.error("Failed to configure A/B test", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# END ML RECOMMENDATION ENDPOINTS
# ============================================================================

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