"""
Recommendation Engine Service Main Application
Core ML algorithms and recommendation inference with Redis caching and PostgreSQL integration
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends, status, Path, BackgroundTasks, Request
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
from psycopg2 import pool
import requests
import os
from collections import defaultdict
import re
import sys
import io
import csv
from fastapi.responses import StreamingResponse

# Add config path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.master_group_api import PG_CONFIG, REDIS_CONFIG, MASTER_GROUP_CONFIG

# Import ML recommendation service
from src.algorithms.ml_recommendation_service import get_ml_service

# Initialize global ML service
ml_service = get_ml_service()

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
pg_pool = None

def get_pg_connection_params():
    """Get PostgreSQL connection parameters with SSL support"""
    params = {
        'host': PG_HOST,
        'port': PG_PORT,
        'database': PG_DB,
        'user': PG_USER,
        'password': PG_PASSWORD,
        'options': '-c statement_timeout=30000'  # 30 second query timeout to prevent hung queries
    }
    
    # Add SSL mode from config (required for Heroku, disabled for local)
    sslmode = PG_CONFIG.get('sslmode')
    if sslmode:
        params['sslmode'] = sslmode
    
    # Debug logging
    logger.info(f"PostgreSQL connection params: host={params['host']}, db={params['database']}, sslmode={params.get('sslmode', 'not set')}")
        
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

# ML Recommendation Request/Response Models
class LocationRequest(BaseModel):
    city: Optional[str] = ""
    state: Optional[str] = ""
    country: Optional[str] = ""

class CartItem(BaseModel):
    sku: str
    quantity: Optional[int] = 1

class CartRecommendationRequest(BaseModel):
    location: LocationRequest
    cart_items: List[CartItem]
    limit: Optional[int] = 5

class BatchLocationRequest(BaseModel):
    locations: List[LocationRequest]
    limit: Optional[int] = 10


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
    """Initialize PostgreSQL connection pool"""
    global pg_conn, pg_pool
    try:
        # Create connection pool for concurrent requests
        pg_pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            **get_pg_connection_params()
        )
        
        # Also create single connection for backward compatibility
        pg_conn = psycopg2.connect(**get_pg_connection_params())
        
        logger.info("PostgreSQL connection pool established", host=PG_HOST, port=PG_PORT)
    except Exception as e:
        logger.error("Failed to connect to PostgreSQL", error=str(e))
        pg_conn = None
        pg_pool = None


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
    
    # Initialize ML recommendation service
    logger.info("Loading ML recommendation models...")
    try:
        # Load models from DB (persisted) or disk
        ml_service.load_trained_models(time_filter='30days')
        if ml_service.is_trained:
            logger.info("✅ ML recommendation models loaded successfully")
        else:
            logger.warning("⚠️ ML models not trained - some endpoints may return fallback data")
    except Exception as e:
        logger.error("❌ ML service initialization failed (non-critical)", error=str(e))
    
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
    if pg_pool:
        pg_pool.closeall()
    
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
    Returns None for true 'all' time filter (no date restriction)
    
    Available filters:
    - today: Current day only
    - 7days: Last 7 days
    - 30days: Last 30 days
    - mtd: Month to date
    - 90days: Last 90 days
    - 6months: Last 6 months
    - 1year: Last 1 year
    - 2years: Last 2 years
    - 3years: Last 3 years
    - all: ALL data (no date restriction) - cached for 2 hours
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
    elif time_filter == "2years":
        return datetime.now() - timedelta(days=730)
    elif time_filter == "3years":
        return datetime.now() - timedelta(days=1095)
    elif time_filter == "all":
        # True "all" - no date restriction, returns None
        # Results will be heavily cached (2 hours) to avoid performance issues
        return None
    elif ":" in time_filter:  # Custom date range format: start_date:end_date
        try:
            date_parts = time_filter.split(":")
            if len(date_parts) == 2:
                return datetime.strptime(date_parts[0], "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid custom date format: {time_filter}")
            return None
    # For any unrecognized filter, default to 1 year
    return datetime.now() - timedelta(days=365)


def get_time_filter_clause(time_filter: str, table_alias: str = "o") -> tuple:
    """
    Get SQL WHERE clause and params for time filtering.
    Returns (where_clause, params_tuple)
    
    Args:
        time_filter: Time period filter
        table_alias: Table alias to use (default 'o' for orders table)
    """
    start_date = calculate_date_range(time_filter)
    if start_date:
        return f"WHERE {table_alias}.order_date >= %s", (start_date,)
    return "", ()


def get_order_source_filter(order_source: str, table_alias: str = "o", include_delivered_only: bool = False, has_where_clause: bool = True) -> tuple:
    """
    Get SQL filter conditions for order source (OE/POS).
    
    Args:
        order_source: Filter by order source - 'all', 'oe', 'pos'
        table_alias: Table alias to use (default 'o' for orders table)
        include_delivered_only: If True, only include delivered/completed orders
                               OE: order_status = 'Delivered Orders'
                               POS: order_status = 'completed' (all POS are completed)
        has_where_clause: If True, returns " AND ..." prefix. If False, returns "WHERE ..."
    
    Returns:
        (sql_condition, params_tuple) - SQL clause and parameters
        Returns empty strings if no filter needed
    """
    conditions = []
    params = []
    
    # Filter by order source type
    if order_source and order_source.lower() in ['oe', 'pos']:
        conditions.append(f"UPPER({table_alias}.order_type) = %s")
        params.append(order_source.upper())
    
    # Filter by delivered/fulfilled status
    if include_delivered_only:
        if order_source and order_source.lower() == 'oe':
            # OE orders: only "Delivered Orders"
            conditions.append(f"{table_alias}.order_status = %s")
            params.append('Delivered Orders')
        elif order_source and order_source.lower() == 'pos':
            # POS orders: all are "completed"
            conditions.append(f"{table_alias}.order_status = %s")
            params.append('completed')
        else:
            # All sources: include both delivered types
            conditions.append(f"({table_alias}.order_status = 'Delivered Orders' OR {table_alias}.order_status = 'completed')")
    
    if conditions:
        prefix = " AND " if has_where_clause else " WHERE "
        return prefix + " AND ".join(conditions), tuple(params)
    return "", ()


def normalize_province(province: str) -> str:
    """Normalize province names (merge duplicates like Islamabad variants and case variations)"""
    if not province:
        return 'Unknown'
    
    # Convert to title case first to handle case variations
    province = province.strip().title()
    
    province_mapping = {
        'Islamabad Capital Territory': 'Islamabad',
        'Islamabad Capital': 'Islamabad',
        'Ict': 'Islamabad',
        'Kpk': 'Khyber Pakhtunkhwa',
        'Nwfp': 'Khyber Pakhtunkhwa',
        'Khyber Pakhtunkhwa': 'Khyber Pakhtunkhwa',  # Already correct
        'Punjab': 'Punjab',  # Already correct
        'Sindh': 'Sindh',  # Already correct
        'Balochistan': 'Balochistan',  # Already correct
        'Azad Kashmir': 'Azad Kashmir',  # Already correct
        'Gilgit-Baltistan': 'Gilgit-Baltistan',  # Already correct
    }
    return province_mapping.get(province, province)


def get_region_for_province(province: str) -> str:
    """Map province to region"""
    # First normalize the province name
    normalized = normalize_province(province)
    
    regions = {
        'Punjab': 'Central',
        'Sindh': 'South',
        'Khyber Pakhtunkhwa': 'North',
        'KPK': 'North',
        'Balochistan': 'West',
        'Islamabad': 'North',
        'Azad Kashmir': 'North',
        'Gilgit-Baltistan': 'North',
    }
    return regions.get(normalized, 'Central')


def get_city_normalization_sql() -> str:
    """
    Returns SQL CASE statement to normalize city names.
    Handles case variations and common duplicates.
    Use this in SELECT and GROUP BY clauses.
    """
    return """
        INITCAP(TRIM(
            CASE 
                -- Normalize "Wah Cantt" variants
                WHEN LOWER(TRIM(customer_city)) IN ('wah cantt', 'wah cantonment', 'wah') THEN 'Wah Cantonment'
                -- Normalize "Rahim Yar Khan" variants  
                WHEN LOWER(TRIM(customer_city)) IN ('rahim yar khan', 'rahimyarkhan', 'rahimyar khan') THEN 'Rahim Yar Khan'
                -- Normalize "Dera Ghazi Khan" variants
                WHEN LOWER(TRIM(customer_city)) LIKE 'dera ghazi%' THEN 'Dera Ghazi Khan'
                -- Normalize "Dera Ismail Khan" variants
                WHEN LOWER(TRIM(customer_city)) LIKE 'dera ismail%' THEN 'Dera Ismail Khan'
                -- Normalize "Toba Tek Singh" variants
                WHEN LOWER(TRIM(customer_city)) LIKE 'toba%tek%' OR LOWER(TRIM(customer_city)) LIKE 'tobatek%' THEN 'Toba Tek Singh'
                -- Normalize "Mandi Bahauddin" variants
                WHEN LOWER(TRIM(customer_city)) LIKE 'mandi bahauddin%' THEN 'Mandi Bahauddin'
                -- Normalize "Gujar Khan" variants
                WHEN LOWER(TRIM(customer_city)) IN ('gujar khan', 'gujarkhan') THEN 'Gujar Khan'
                -- Normalize abbreviated city names
                WHEN LOWER(TRIM(customer_city)) IN ('lhr') THEN 'Lahore'
                WHEN LOWER(TRIM(customer_city)) IN ('khi') THEN 'Karachi'
                WHEN LOWER(TRIM(customer_city)) IN ('rwp') THEN 'Rawalpindi'
                WHEN LOWER(TRIM(customer_city)) IN ('fsd') THEN 'Faisalabad'
                WHEN LOWER(TRIM(customer_city)) IN ('isb') THEN 'Islamabad'
                -- Default: Just apply INITCAP to normalize case
                ELSE customer_city
            END
        ))
    """


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


def get_category_filter_sql(category: str) -> str:
    """
    Generate SQL WHERE clause for category filtering based on product names.
    Returns empty string if no category filter needed.
    Note: Use %% to escape % in psycopg2 queries
    """
    if not category or category.lower() == 'all':
        return ""
    
    category_lower = category.lower()
    
    # Map categories to SQL LIKE patterns (use %% to escape % for psycopg2)
    category_patterns = {
        'mattresses': ["'%%foam%%'", "'%%mattress%%'", "'%%sleep%%'", "'%%spring%%'", "'%%ortho%%'"],
        'spring mattresses': ["'%%spring%%'", "'%%pocket%%'"],
        'memory foam mattresses': ["'%%memory%%'", "'%%ortho%%'"],
        'pillows & accessories': ["'%%pillow%%'", "'%%cushion%%'"],
        'pillows': ["'%%pillow%%'"],
        'bedding & accessories': ["'%%sheet%%'", "'%%cover%%'", "'%%protector%%'", "'%%topper%%'"],
        'furniture': ["'%%sofa%%'", "'%%chair%%'", "'%%table%%'", "'%%bed%%'"],
        'general': []  # No filter for general
    }
    
    patterns = category_patterns.get(category_lower, [])
    if not patterns:
        # Try partial match
        for key, pats in category_patterns.items():
            if category_lower in key or key in category_lower:
                patterns = pats
                break
    
    if not patterns:
        return ""
    
    like_clauses = " OR ".join([f"LOWER(oi.product_name) LIKE {p}" for p in patterns])
    return f"AND ({like_clauses})"


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

def popular_products(limit: int = 10, time_filter: str = "7days", category: str = None) -> List[Dict]:
    """Get most popular products based on purchase count with time filtering and caching"""
    # Check cache first
    cache_key = f"popular_products:{limit}:{time_filter}:{category or 'all'}"
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info("Popular products served from cache", time_filter=time_filter, limit=limit)
                return json.loads(cached_result)
        except Exception as e:
            logger.warning("Cache read failed for popular products", error=str(e))
    
    if not pg_pool:
        return []
    
    try:
        # Get connection from pool
        conn = pg_pool.getconn()
        
        # Calculate date range based on filter
        start_date = calculate_date_range(time_filter)
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
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
                product_category = extract_smart_category(product_name, None, order_source)
                
                # Filter by category if specified (supports comma-separated multi-select)
                if category and category.lower() != 'all':
                    selected_categories = [c.strip().lower() for c in category.split(',')]
                    if product_category.lower() not in selected_categories:
                        continue
                
                recommendations.append({
                    "product_id": row['product_id'],
                    "product_name": product_name,
                    "score": float(row['purchase_count']),
                    "reason": f"Popular product (purchased {row['purchase_count']} times)",
                    "purchase_count": row['purchase_count'],
                    "unique_customers": row['unique_customers'],
                    "avg_price": float(row['avg_price']) if row['avg_price'] else 0,
                    "total_revenue": float(row['total_revenue']) if row['total_revenue'] else 0,
                    "category": product_category
                })
                
                # Stop if we have enough results
                if len(recommendations) >= limit:
                    break
            
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
    finally:
        # Return connection to pool
        if pg_pool and 'conn' in locals():
            pg_pool.putconn(conn)


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
    
    # Cache the result - longer TTL for heavy "all" queries (2 hours vs 5 minutes)
    ttl = 7200 if time_filter == "all" else 300
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
    
    # Cache the result - longer TTL for heavy "all" queries (2 hours vs 5 minutes)
    ttl = 7200 if time_filter == "all" else 300
    set_to_cache(cache_key, result, ttl=ttl)
    
    return result

@app.get("/api/v1/recommendations/popular")
async def get_popular_products_endpoint(
    limit: int = Query(10, ge=1, le=100),
    time_filter: str = Query("7days", description="Time filter: today, 7days, 30days, all"),
    category: str = Query(None, description="Filter by category: Mattresses, Pillows & Accessories, etc.")
):
    """Get most popular products with time-based and category filtering"""
    # Check cache first
    cache_key = get_cache_key("popular", limit, time_filter, category or "all")
    cached = get_from_cache(cache_key)
    if cached:
        logger.info("Returning cached popular products", time_filter=time_filter, category=category)
        return cached
    
    # Generate recommendations with category filter
    recommendations = popular_products(limit * 3, time_filter, category)[:limit]  # Fetch more to ensure enough after filtering
    
    result = {
        "recommendations": recommendations,
        "time_filter": time_filter,
        "category": category,
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


@app.get("/api/v1/stats")
async def get_system_stats():
    """Get system-wide statistics"""
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get order stats
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT o.id) as total_orders,
                COUNT(DISTINCT o.unified_customer_id) as total_customers,
                COUNT(DISTINCT oi.product_id) as total_products,
                SUM(o.total_price) as total_revenue
            FROM orders o
            LEFT JOIN order_items oi ON o.id = oi.order_id
        """)
        stats = cursor.fetchone()
        
        # Get recent activity
        cursor.execute("""
            SELECT COUNT(*) as orders_today 
            FROM orders 
            WHERE order_date >= CURRENT_DATE
        """)
        today = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        # Get ML status
        ml_service = get_ml_service()
        
        return {
            "total_orders": stats['total_orders'] or 0,
            "total_customers": stats['total_customers'] or 0,
            "total_products": stats['total_products'] or 0,
            "total_revenue": float(stats['total_revenue'] or 0),
            "orders_today": today['orders_today'] or 0,
            "ml_trained": ml_service.is_trained,
            "cache_connected": redis_client is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


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


@app.post("/api/v1/sync/full")
async def trigger_full_sync(
    days: int = Query(14, description="Number of days to sync"),
    background_tasks: BackgroundTasks = None
):
    """Trigger a full sync for specified number of days"""
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from services.sync_service import get_sync_service
        from datetime import datetime, timedelta
        
        sync_service = get_sync_service()
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        logger.info(f"Starting full sync from {start_date} to {end_date} ({days} days)")
        
        # Fetch POS orders with date range
        pos_orders = sync_service.fetch_pos_orders(start_date, end_date, limit=5000)
        logger.info(f"Fetched {len(pos_orders)} POS orders")
        
        # Fetch OE orders with same date range
        oe_orders = sync_service.fetch_oe_orders(start_date=start_date, end_date=end_date, limit=5000)
        logger.info(f"Fetched {len(oe_orders)} OE orders")
        
        # Transform orders
        transformed_orders = []
        for order in pos_orders:
            transformed = sync_service.transform_order_data(order, 'POS')
            if transformed:
                transformed_orders.append(transformed)
        
        for order in oe_orders:
            transformed = sync_service.transform_order_data(order, 'OE')
            if transformed:
                transformed_orders.append(transformed)
        
        # Insert orders
        orders_inserted, items_inserted = sync_service.insert_orders(transformed_orders)
        
        # Clear caches
        if redis_client and orders_inserted > 0:
            keys_deleted = 0
            for key in redis_client.scan_iter("popular:*"):
                redis_client.delete(key)
                keys_deleted += 1
            for key in redis_client.scan_iter("collab:*"):
                redis_client.delete(key)
                keys_deleted += 1
            for key in redis_client.scan_iter("analytics:*"):
                redis_client.delete(key)
                keys_deleted += 1
            logger.info(f"Cleared {keys_deleted} cache keys")
        
        return {
            "status": "success",
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "pos_orders_fetched": len(pos_orders),
            "oe_orders_fetched": len(oe_orders),
            "orders_inserted": orders_inserted,
            "items_inserted": items_inserted,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Full sync error: {e}")
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

# ============================================================================
# ANALYTICS ENDPOINTS (Required by Frontend Dashboard)
# ============================================================================

@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_metrics(
    time_filter: str = Query("30days"),
    category: str = Query(None, description="Filter by product category"),
    order_source: str = Query(None, description="Filter by order source: 'oe', 'pos', or None for all"),
    delivered_only: bool = Query(False, description="Only include delivered/completed orders")
):
    """Get dashboard summary metrics - with Redis caching and category filter"""
    cache_key = f"analytics:dashboard:{time_filter}:{category or 'all'}:{order_source or 'all'}:{delivered_only}"
    
    # Check cache first
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass
    
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # Add order source filter - check if we already have a WHERE clause
        has_where = bool(where_clause)
        source_filter, source_params = get_order_source_filter(order_source, "o", delivered_only, has_where)
        params = params + source_params
        
        # Add category filter if specified
        category_filter_raw = get_category_filter_sql(category)
        
        # CRITICAL FIX: Use subquery to avoid double-counting revenue when joining with order_items
        # When filtering by category, we need to find orders that contain those products,
        # but only count the order's total_price ONCE (not per item)
        if category_filter_raw:
            # Build subquery to get order IDs that match the category filter
            category_condition = category_filter_raw.replace("AND ", "", 1)  # Remove leading AND
            
            # Determine if we need AND or WHERE for the category subquery
            has_any_clause = bool(where_clause) or bool(source_filter)
            category_prefix = "AND" if has_any_clause else "WHERE"
            
            cursor.execute(f"""
                SELECT 
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT o.unified_customer_id) as total_customers,
                    SUM(o.total_price) as total_revenue,
                    AVG(o.total_price) as avg_order_value
                FROM orders o
                {where_clause}
                {source_filter}
                {category_prefix} o.id IN (
                    SELECT DISTINCT oi.order_id
                    FROM order_items oi
                    WHERE {category_condition}
                )
            """, params if params else None)
        else:
            # No category filter - simple query
            cursor.execute(f"""
                SELECT 
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT o.unified_customer_id) as total_customers,
                    SUM(o.total_price) as total_revenue,
                    AVG(o.total_price) as avg_order_value
                FROM orders o
                {where_clause}
                {source_filter}
            """, params if params else None)
        
        result = cursor.fetchone()
        
        response = {
            "success": True,
            "total_orders": result['total_orders'] or 0,
            "total_customers": result['total_customers'] or 0,
            "total_revenue": float(result['total_revenue'] or 0),
            "avg_order_value": float(result['avg_order_value'] or 0),
            "time_filter": time_filter,
            "order_source": order_source or "all",
            "delivered_only": delivered_only,
            "totalOrders": result['total_orders'] or 0,
            "totalCustomers": result['total_customers'] or 0,
            "totalRevenueAmount": float(result['total_revenue'] or 0),
            "avgOrderValue": float(result['avg_order_value'] or 0)
        }
        
        # Cache for 5 minutes
        if redis_client:
            try:
                redis_client.setex(cache_key, 300, json.dumps(response))
            except Exception:
                pass
        
        return response
    except Exception as e:
        logger.error(f"Dashboard metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/geographic-distribution")
async def get_geographic_distribution(
    time_filter: str = Query("30days"),
    order_source: str = Query(None, description="Filter by order source: 'oe', 'pos', or None for all"),
    delivered_only: bool = Query(False, description="Only include delivered/completed orders")
):
    """Get geographic distribution of customers by city - with Redis caching"""
    cache_key = f"analytics:geographic:{time_filter}:{order_source or 'all'}:{delivered_only}"
    cached = get_from_cache(cache_key)
    if cached:
        return cached
    
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        has_where = bool(where_clause)
        source_filter, source_params = get_order_source_filter(order_source, "o", delivered_only, has_where)
        params = params + source_params
        
        # Determine if we need AND or WHERE for the city filter
        has_any_clause = bool(where_clause) or bool(source_filter)
        city_prefix = "AND" if has_any_clause else "WHERE"
        
        cursor.execute(f"""
            SELECT 
                o.customer_city as city,
                COUNT(DISTINCT o.unified_customer_id) as customer_count,
                COUNT(*) as orders,
                COALESCE(SUM(o.total_price), 0) as revenue
            FROM orders o
            {where_clause}
            {source_filter}
            {city_prefix} o.customer_city IS NOT NULL
                AND o.customer_city != ''
            GROUP BY o.customer_city
            ORDER BY customer_count DESC
            LIMIT 10
        """, params)
        
        results = cursor.fetchall()
        
        # Calculate total customers for percentage calculation
        total_customers = sum(row['customer_count'] for row in results)
        
        # Format response
        distribution_data = []
        for row in results:
            distribution_data.append({
                "city": row['city'],
                "customer_count": row['customer_count'],
                "orders": row['orders'],
                "revenue": float(row['revenue']),
                "percentage": (row['customer_count'] / total_customers * 100) if total_customers > 0 else 0
            })
        
        response = {
            "success": True,
            "time_filter": time_filter,
            "total_cities": len(results),
            "total_customers": total_customers,
            "distribution": distribution_data
        }
        
        # Cache for 5 minutes
        set_to_cache(cache_key, response, 300)
        
        return response
    except Exception as e:
        logger.error(f"Geographic distribution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/revenue-trend")
async def get_revenue_trend(
    time_filter: str = Query("30days"),
    period: str = Query("daily"),
    order_source: str = Query(None, description="Filter by order source: 'oe', 'pos', or None for all"),
    delivered_only: bool = Query(False, description="Only include delivered/completed orders")
):
    """Get revenue trend data - with caching"""
    cache_key = f"analytics:revenue_trend:{time_filter}:{period}:{order_source or 'all'}:{delivered_only}"
    cached = get_from_cache(cache_key)
    if cached:
        return cached
    
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        has_where = bool(where_clause)
        source_filter, source_params = get_order_source_filter(order_source, "o", delivered_only, has_where)
        params = params + source_params
        
        if period == "daily":
            group_by = "DATE(o.order_date)"
        elif period == "weekly":
            group_by = "DATE_TRUNC('week', o.order_date)"
        else:
            group_by = "DATE_TRUNC('month', o.order_date)"
        
        cursor.execute(f"""
            SELECT 
                {group_by} as date,
                SUM(o.total_price) as revenue,
                COUNT(DISTINCT o.id) as orders
            FROM orders o
            {where_clause}
            {source_filter}
            GROUP BY {group_by}
            ORDER BY date DESC
            LIMIT 30
        """, params)
        
        results = cursor.fetchall()
        
        response = {
            "trend": [{"date": str(r['date']), "revenue": float(r['revenue'] or 0), "orders": r['orders']} for r in results],
            "period": period,
            "timeFilter": time_filter,
            "orderSource": order_source or "all"
        }
        set_to_cache(cache_key, response, 300)
        return response
    except Exception as e:
        logger.error(f"Revenue trend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/products")
async def get_product_analytics(
    time_filter: str = Query("30days"),
    limit: int = Query(10),
    order_source: str = Query(None, description="Filter by order source: 'oe', 'pos', or None for all"),
    delivered_only: bool = Query(False, description="Only include delivered/completed orders")
):
    """Get product analytics - with caching"""
    cache_key = f"analytics:products:{time_filter}:{limit}:{order_source or 'all'}:{delivered_only}"
    cached = get_from_cache(cache_key)
    if cached:
        return cached
    
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        has_where = bool(where_clause)
        source_filter, source_params = get_order_source_filter(order_source, "o", delivered_only, has_where)
        params = params + source_params
        
        cursor.execute(f"""
            SELECT 
                oi.product_id,
                MAX(oi.product_name) as product_name,
                COUNT(DISTINCT oi.order_id) as total_orders,
                SUM(oi.total_price) as total_revenue,
                AVG(oi.unit_price) as avg_price
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            {where_clause}
            {source_filter}
            GROUP BY oi.product_id
            ORDER BY total_revenue DESC
            LIMIT %s
        """, params + (limit,))
        
        results = cursor.fetchall()
        
        response = {
            "products": [{
                "productId": r['product_id'],
                "productName": r['product_name'],
                "totalOrders": r['total_orders'],
                "totalRevenue": float(r['total_revenue'] or 0),
                "avgPrice": float(r['avg_price'] or 0)
            } for r in results],
            "timeFilter": time_filter,
            "orderSource": order_source or "all"
        }
        set_to_cache(cache_key, response, 300)
        return response
    except Exception as e:
        logger.error(f"Product analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/product-categories")
async def get_product_categories(time_filter: str = Query("30days")):
    """Get product categories with performance metrics"""
    cache_key = f"analytics:product_categories:{time_filter}"
    cached = get_from_cache(cache_key)
    if cached:
        return cached
    
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        cursor.execute(f"""
            SELECT 
                oi.product_name,
                COUNT(DISTINCT oi.order_id) as total_orders,
                SUM(oi.total_price) as total_revenue,
                COUNT(DISTINCT o.unified_customer_id) as unique_customers
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            {where_clause}
            GROUP BY oi.product_name
            ORDER BY total_revenue DESC
        """, params)
        
        results = cursor.fetchall()
        
        # Group by category using extract_smart_category
        category_stats = {}
        for r in results:
            category = extract_smart_category(r['product_name'])
            if category not in category_stats:
                category_stats[category] = {
                    "category": category,
                    "total_orders": 0,
                    "total_revenue": 0,
                    "unique_customers": 0,
                    "top_products": []
                }
            category_stats[category]["total_orders"] += r['total_orders']
            category_stats[category]["total_revenue"] += float(r['total_revenue'] or 0)
            category_stats[category]["unique_customers"] += r['unique_customers']
            
            # Keep top 5 products per category
            if len(category_stats[category]["top_products"]) < 5:
                category_stats[category]["top_products"].append({
                    "product_name": r['product_name'],
                    "revenue": float(r['total_revenue'] or 0),
                    "orders": r['total_orders']
                })
        
        # Sort by revenue
        categories = sorted(category_stats.values(), key=lambda x: x['total_revenue'], reverse=True)
        
        response = {
            "categories": categories,
            "total_categories": len(categories),
            "timeFilter": time_filter
        }
        set_to_cache(cache_key, response, 300)
        return response
    except Exception as e:
        logger.error(f"Product categories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/products-by-category")
async def get_products_by_category(
    category: str = Query(..., description="Product category to filter"),
    time_filter: str = Query("30days"),
    limit: int = Query(20)
):
    """Get products filtered by category"""
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        cursor.execute(f"""
            SELECT 
                oi.product_id,
                MAX(oi.product_name) as product_name,
                COUNT(DISTINCT oi.order_id) as total_orders,
                SUM(oi.total_price) as total_revenue,
                AVG(oi.unit_price) as avg_price
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            {where_clause}
            GROUP BY oi.product_id
            ORDER BY total_revenue DESC
            LIMIT %s
        """, params + (limit,))
        
        results = cursor.fetchall()
        
        # Filter by category
        filtered = []
        for r in results:
            product_category = extract_smart_category(r['product_name'])
            if product_category.lower() == category.lower():
                filtered.append({
                    "productId": r['product_id'],
                    "productName": r['product_name'],
                    "category": product_category,
                    "totalOrders": r['total_orders'],
                    "totalRevenue": float(r['total_revenue'] or 0),
                    "avgPrice": float(r['avg_price'] or 0)
                })
                if len(filtered) >= limit:
                    break
        
        return {
            "products": filtered,
            "category": category,
            "count": len(filtered),
            "timeFilter": time_filter
        }
    except Exception as e:
        logger.error(f"Products by category error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/geographic/provinces")
async def get_province_performance(
    time_filter: str = Query("30days"),
    order_source: str = Query("all", description="Filter by order source: all, oe, pos"),
    category: str = Query(None, description="Filter by product category")
):
    """Get province-level performance with merged Islamabad variants, optional OE/POS and category filters"""
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # Add order source filter
        order_source_clause, order_source_params = get_order_source_filter(order_source, "o")
        params = list(params) + list(order_source_params)
        
        # Determine if we need to join order_items for category filtering
        needs_items_join = category and category.strip()
        
        # Build category filter
        category_filter = ""
        if needs_items_join:
            category_filter = get_category_filter_sql(category)
        
        # Base query structure
        if needs_items_join:
            # Join with order_items for category filtering
            query = f"""
                SELECT 
                    CASE 
                        WHEN UPPER(o.province) IN ('ISLAMABAD', 'ISLAMABAD CAPITAL TERRITORY', 'ISLAMABAD CAPITAL', 'ICT') THEN 'Islamabad'
                        WHEN UPPER(REPLACE(o.province, '.', '')) IN ('KPK', 'NWFP', 'KHYBER PAKHTUNKHWA') THEN 'Khyber Pakhtunkhwa'
                        WHEN UPPER(o.province) = 'PUNJAB' THEN 'Punjab'
                        WHEN UPPER(o.province) = 'SINDH' THEN 'Sindh'
                        WHEN UPPER(o.province) IN ('BALOCHISTAN', 'BALUCHISTAN') THEN 'Balochistan'
                        WHEN UPPER(o.province) IN ('GILGIT-BALTISTAN', 'GB') THEN 'Gilgit-Baltistan'
                        WHEN UPPER(o.province) IN ('AZAD KASHMIR', 'AJK', 'AZAD JAMMU AND KASHMIR') THEN 'Azad Kashmir'
                        ELSE INITCAP(COALESCE(o.province, 'Unknown'))
                    END as province,
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT o.unified_customer_id) as total_customers,
                    SUM(oi.total_price) as total_revenue
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                {where_clause}
                    {order_source_clause}
                    {category_filter}
                GROUP BY 1
                ORDER BY total_revenue DESC
            """
        else:
            # No category filter - use simpler query
            query = f"""
                SELECT 
                    CASE 
                        WHEN UPPER(o.province) IN ('ISLAMABAD', 'ISLAMABAD CAPITAL TERRITORY', 'ISLAMABAD CAPITAL', 'ICT') THEN 'Islamabad'
                        WHEN UPPER(REPLACE(o.province, '.', '')) IN ('KPK', 'NWFP', 'KHYBER PAKHTUNKHWA') THEN 'Khyber Pakhtunkhwa'
                        WHEN UPPER(o.province) = 'PUNJAB' THEN 'Punjab'
                        WHEN UPPER(o.province) = 'SINDH' THEN 'Sindh'
                        WHEN UPPER(o.province) IN ('BALOCHISTAN', 'BALUCHISTAN') THEN 'Balochistan'
                        WHEN UPPER(o.province) IN ('GILGIT-BALTISTAN', 'GB') THEN 'Gilgit-Baltistan'
                        WHEN UPPER(o.province) IN ('AZAD KASHMIR', 'AJK', 'AZAD JAMMU AND KASHMIR') THEN 'Azad Kashmir'
                        ELSE INITCAP(COALESCE(o.province, 'Unknown'))
                    END as province,
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT o.unified_customer_id) as total_customers,
                    SUM(o.total_price) as total_revenue
                FROM orders o
                {where_clause}
                    {order_source_clause}
                GROUP BY 1
                ORDER BY total_revenue DESC
            """
        
        cursor.execute(query, tuple(params))
        results = cursor.fetchall()
        
        return [{
            "province": r['province'],
            "region": get_region_for_province(r['province']),
            "total_orders": r['total_orders'],
            "unique_customers": r['total_customers'],
            "total_revenue": float(r['total_revenue'] or 0),
            "avg_order_value": float(r['total_revenue'] or 0) / max(r['total_orders'], 1)
        } for r in results]
    except Exception as e:
        logger.error(f"Province performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.get("/api/v1/analytics/order-status-breakdown")
async def get_order_status_breakdown(
    time_filter: str = Query("30days"),
    order_source: str = Query("all", description="Filter by order source: all, oe, pos")
):
    """
    Get order status breakdown with revenue impact analysis.
    Shows distribution of orders by status (Delivered, Cancelled, Returned, Pending, etc.)
    Useful for understanding fulfillment rates and lost revenue analysis.
    """
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, time_params = get_time_filter_clause(time_filter)
        params = list(time_params)
        
        # Build order source filter
        order_source_clause = ""
        if order_source and order_source.lower() in ['oe', 'pos']:
            order_source_clause = "AND UPPER(o.order_type) = %s"
            params.append(order_source.upper())
        
        # Build WHERE clause
        if where_clause:
            full_where = f"{where_clause} {order_source_clause}"
        else:
            if order_source_clause:
                full_where = f"WHERE 1=1 {order_source_clause}"
            else:
                full_where = ""
        
        cursor.execute(f"""
            SELECT 
                UPPER(o.order_type) as order_type,
                o.order_status,
                COUNT(*) as order_count,
                SUM(o.total_price) as total_revenue,
                AVG(o.total_price) as avg_order_value,
                COUNT(DISTINCT o.unified_customer_id) as unique_customers
            FROM orders o
            {full_where}
            WHERE o.order_status IS NOT NULL
            GROUP BY UPPER(o.order_type), o.order_status
            ORDER BY order_type, order_count DESC
        """.replace("WHERE o.order_status IS NOT NULL", 
                   f"{'AND' if full_where else 'WHERE'} o.order_status IS NOT NULL"), 
        tuple(params))
        
        results = cursor.fetchall()
        
        # Organize by order type
        oe_statuses = []
        pos_statuses = []
        totals = {"oe": {"orders": 0, "revenue": 0}, "pos": {"orders": 0, "revenue": 0}}
        
        for r in results:
            status_data = {
                "status": r['order_status'],
                "order_count": r['order_count'],
                "total_revenue": float(r['total_revenue'] or 0),
                "avg_order_value": float(r['avg_order_value'] or 0),
                "unique_customers": r['unique_customers']
            }
            
            if r['order_type'] == 'OE':
                oe_statuses.append(status_data)
                totals["oe"]["orders"] += r['order_count']
                totals["oe"]["revenue"] += float(r['total_revenue'] or 0)
            else:
                pos_statuses.append(status_data)
                totals["pos"]["orders"] += r['order_count']
                totals["pos"]["revenue"] += float(r['total_revenue'] or 0)
        
        # Calculate percentages and categorize
        for status in oe_statuses:
            status["percentage"] = round(status["order_count"] / max(totals["oe"]["orders"], 1) * 100, 1)
            # Categorize status
            if status["status"] in ["Delivered Orders"]:
                status["category"] = "fulfilled"
            elif status["status"] in ["Cancelled Orders", "Returned Orders", "Refund Orders"]:
                status["category"] = "lost"
            else:
                status["category"] = "pipeline"
        
        for status in pos_statuses:
            status["percentage"] = round(status["order_count"] / max(totals["pos"]["orders"], 1) * 100, 1)
            status["category"] = "fulfilled" if status["status"] == "completed" else "other"
        
        # Calculate OE fulfillment rate
        oe_delivered = sum(s["order_count"] for s in oe_statuses if s["category"] == "fulfilled")
        oe_total = totals["oe"]["orders"]
        oe_fulfillment_rate = round(oe_delivered / max(oe_total, 1) * 100, 1)
        
        # Calculate lost revenue
        oe_lost_revenue = sum(s["total_revenue"] for s in oe_statuses if s["category"] == "lost")
        
        return {
            "oe": {
                "statuses": oe_statuses,
                "total_orders": totals["oe"]["orders"],
                "total_revenue": totals["oe"]["revenue"],
                "fulfillment_rate": oe_fulfillment_rate,
                "lost_revenue": oe_lost_revenue
            },
            "pos": {
                "statuses": pos_statuses,
                "total_orders": totals["pos"]["orders"],
                "total_revenue": totals["pos"]["revenue"],
                "fulfillment_rate": 100.0  # All POS are completed
            },
            "time_filter": time_filter,
            "order_source_filter": order_source
        }
        
    except Exception as e:
        logger.error(f"Order status breakdown error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/geographic/category-by-province")
async def get_category_by_province(
    time_filter: str = Query("30days"),
    order_source: str = Query("all", description="Filter by order source: all, oe, pos"),
    limit: int = Query(50, description="Maximum results to return")
):
    """
    Get category performance breakdown by province.
    Shows which categories sell best in each region, with OE/POS comparison.
    Perfect for answering: "Which category sells most in which region?"
    """
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, time_params = get_time_filter_clause(time_filter)
        params = list(time_params)
        
        logger.info(f"category-by-province: time_filter={time_filter}, where_clause={where_clause}, time_params={time_params}")
        
        # Build order source filter
        order_source_clause = ""
        if order_source and order_source.lower() in ['oe', 'pos']:
            order_source_clause = "AND UPPER(o.order_type) = %s"
            params.append(order_source.upper())
        
        # Build the WHERE clause properly
        if where_clause:
            # We already have a WHERE from time filter
            full_where = f"""{where_clause}
                AND o.province IS NOT NULL
                AND TRIM(o.province) != ''
                AND UPPER(TRIM(o.province)) NOT IN ('UNKNOWN', 'N/A', 'NA', 'NULL', 'NONE')
                {order_source_clause}"""
        else:
            # No time filter, start fresh WHERE
            full_where = f"""WHERE o.province IS NOT NULL
                AND TRIM(o.province) != ''
                AND UPPER(TRIM(o.province)) NOT IN ('UNKNOWN', 'N/A', 'NA', 'NULL', 'NONE')
                {order_source_clause}"""
        
        # Add limit to params
        params.append(limit)
        
        logger.info(f"category-by-province: full_where={full_where[:100]}, params={params}")
        
        cursor.execute(f"""
            SELECT 
                CASE 
                    WHEN UPPER(o.province) IN ('ISLAMABAD', 'ISLAMABAD CAPITAL TERRITORY', 'ISLAMABAD CAPITAL', 'ICT') THEN 'Islamabad'
                    WHEN UPPER(REPLACE(o.province, '.', '')) IN ('KPK', 'NWFP', 'KHYBER PAKHTUNKHWA') THEN 'Khyber Pakhtunkhwa'
                    WHEN UPPER(o.province) = 'PUNJAB' THEN 'Punjab'
                    WHEN UPPER(o.province) = 'SINDH' THEN 'Sindh'
                    WHEN UPPER(o.province) IN ('BALOCHISTAN', 'BALUCHISTAN') THEN 'Balochistan'
                    WHEN UPPER(o.province) IN ('GILGIT-BALTISTAN', 'GB') THEN 'Gilgit-Baltistan'
                    WHEN UPPER(o.province) IN ('AZAD KASHMIR', 'AJK', 'AZAD JAMMU AND KASHMIR') THEN 'Azad Kashmir'
                    ELSE INITCAP(COALESCE(o.province, 'Unknown'))
                END as province,
                CASE 
                    WHEN UPPER(oi.product_name) LIKE '%%FOAM%%' THEN 'Foam'
                    WHEN UPPER(oi.product_name) LIKE '%%PILLOW%%' THEN 'Pillows'
                    WHEN UPPER(oi.product_name) LIKE '%%CELESTE%%' THEN 'Celeste'
                    WHEN UPPER(oi.product_name) LIKE '%%MOLTY%%' AND UPPER(oi.product_name) NOT LIKE '%%FOAM%%' THEN 'Molty'
                    WHEN UPPER(oi.product_name) LIKE '%%BED%%' THEN 'Beds'
                    WHEN UPPER(oi.product_name) LIKE '%%SOFA%%' THEN 'Sofas'
                    WHEN UPPER(oi.product_name) LIKE '%%SPRING%%' THEN 'Spring Mattresses'
                    WHEN UPPER(oi.product_name) LIKE '%%MATTRESS%%' THEN 'Mattresses'
                    ELSE 'Other'
                END as category,
                UPPER(o.order_type) as order_type,
                COUNT(DISTINCT o.id) as total_orders,
                SUM(oi.quantity) as total_items,
                SUM(oi.total_price) as total_revenue,
                COUNT(DISTINCT o.unified_customer_id) as unique_customers
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            {full_where}
            GROUP BY 1, 2, 3
            ORDER BY province, total_revenue DESC
            LIMIT %s
        """, tuple(params))
        
        results = cursor.fetchall()
        
        # Transform to more useful format - group by province
        province_data = {}
        for r in results:
            province = r['province']
            if province not in province_data:
                province_data[province] = {
                    "province": province,
                    "region": get_region_for_province(province),
                    "categories": [],
                    "total_revenue": 0,
                    "total_orders": 0
                }
            
            province_data[province]["categories"].append({
                "category": r['category'],
                "order_type": r['order_type'] or 'ALL',
                "total_orders": r['total_orders'],
                "total_items": r['total_items'] or 0,
                "total_revenue": float(r['total_revenue'] or 0),
                "unique_customers": r['unique_customers']
            })
            province_data[province]["total_revenue"] += float(r['total_revenue'] or 0)
            province_data[province]["total_orders"] += r['total_orders']
        
        # Sort provinces by total revenue
        sorted_provinces = sorted(province_data.values(), key=lambda x: x['total_revenue'], reverse=True)
        
        return sorted_provinces
        
    except Exception as e:
        logger.error(f"Category by province error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

@app.get("/api/v1/analytics/geographic/cities")
async def get_city_performance(
    time_filter: str = Query("30days"),
    order_source: str = Query("all", description="Filter by order source: all, oe, pos"),
    category: str = Query(None, description="Filter by product category"),
    limit: int = Query(10)
):
    """Get city-level performance with proper province mapping, optional OE/POS and category filters"""
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # Add order source filter
        order_source_clause, order_source_params = get_order_source_filter(order_source, "o")
        params = list(params) + list(order_source_params)
        
        # Determine if we need to join order_items for category filtering
        needs_items_join = category and category.strip()
        
        # Build category filter
        category_filter = ""
        if needs_items_join:
            category_filter = get_category_filter_sql(category)
        
        if needs_items_join:
            query = f"""
                SELECT 
                    o.customer_city as city,
                    CASE 
                        WHEN UPPER(o.province) IN ('ISLAMABAD', 'ISLAMABAD CAPITAL TERRITORY', 'ISLAMABAD CAPITAL', 'ICT') THEN 'Islamabad'
                        WHEN UPPER(REPLACE(o.province, '.', '')) IN ('KPK', 'NWFP', 'KHYBER PAKHTUNKHWA') THEN 'Khyber Pakhtunkhwa'
                        WHEN UPPER(o.province) = 'PUNJAB' THEN 'Punjab'
                        WHEN UPPER(o.province) = 'SINDH' THEN 'Sindh'
                        WHEN UPPER(o.province) IN ('BALOCHISTAN', 'BALUCHISTAN') THEN 'Balochistan'
                        WHEN UPPER(o.province) IN ('GILGIT-BALTISTAN', 'GB') THEN 'Gilgit-Baltistan'
                        WHEN UPPER(o.province) IN ('AZAD KASHMIR', 'AJK', 'AZAD JAMMU AND KASHMIR') THEN 'Azad Kashmir'
                        ELSE o.province
                    END as province,
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT o.unified_customer_id) as total_customers,
                    SUM(oi.total_price) as total_revenue
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                {where_clause}
                    {order_source_clause}
                    {category_filter}
                GROUP BY o.customer_city, province
                ORDER BY total_revenue DESC
                LIMIT %s
            """
        else:
            query = f"""
                SELECT 
                    o.customer_city as city,
                    CASE 
                        WHEN UPPER(o.province) IN ('ISLAMABAD', 'ISLAMABAD CAPITAL TERRITORY', 'ISLAMABAD CAPITAL', 'ICT') THEN 'Islamabad'
                        WHEN UPPER(REPLACE(o.province, '.', '')) IN ('KPK', 'NWFP', 'KHYBER PAKHTUNKHWA') THEN 'Khyber Pakhtunkhwa'
                        WHEN UPPER(o.province) = 'PUNJAB' THEN 'Punjab'
                        WHEN UPPER(o.province) = 'SINDH' THEN 'Sindh'
                        WHEN UPPER(o.province) IN ('BALOCHISTAN', 'BALUCHISTAN') THEN 'Balochistan'
                        WHEN UPPER(o.province) IN ('GILGIT-BALTISTAN', 'GB') THEN 'Gilgit-Baltistan'
                        WHEN UPPER(o.province) IN ('AZAD KASHMIR', 'AJK', 'AZAD JAMMU AND KASHMIR') THEN 'Azad Kashmir'
                        ELSE o.province
                    END as province,
                    COUNT(DISTINCT o.id) as total_orders,
                    COUNT(DISTINCT o.unified_customer_id) as total_customers,
                    SUM(o.total_price) as total_revenue
                FROM orders o
                {where_clause}
                    {order_source_clause}
                GROUP BY o.customer_city, province
                ORDER BY total_revenue DESC
                LIMIT %s
            """
        
        cursor.execute(query, tuple(params) + (limit,))
        results = cursor.fetchall()
        
        return [{
            "city": r['city'],
            "province": r['province'],
            "region": get_region_for_province(r['province']),
            "total_orders": r['total_orders'],
            "unique_customers": r['total_customers'],
            "total_revenue": float(r['total_revenue'] or 0),
            "avg_order_value": float(r['total_revenue'] or 0) / max(r['total_orders'], 1)
        } for r in results]
    except Exception as e:
        logger.error(f"City performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/customers/rfm-segments")
async def get_analytics_rfm_segments(time_filter: str = Query("30days")):
    """Get RFM segment analytics"""
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        cursor.execute(f"""
            WITH customer_rfm AS (
                SELECT 
                    o.unified_customer_id,
                    EXTRACT(days FROM NOW() - MAX(o.order_date)) as recency,
                    COUNT(DISTINCT o.id) as frequency,
                    SUM(o.total_price) as monetary
                FROM orders o
                {where_clause}
                GROUP BY o.unified_customer_id
            )
            SELECT 
                CASE 
                    WHEN recency <= 30 AND frequency >= 5 THEN 'Champions'
                    WHEN recency <= 60 AND frequency >= 3 THEN 'Loyal'
                    WHEN recency <= 90 THEN 'Potential'
                    ELSE 'At Risk'
                END as segment,
                COUNT(*) as customer_count,
                AVG(monetary) as avg_revenue
            FROM customer_rfm
            GROUP BY segment
            ORDER BY customer_count DESC
        """, params)
        
        results = cursor.fetchall()
        
        return [{
            "segment": r['segment'],
            "customerCount": r['customer_count'],
            "avgRevenue": float(r['avg_revenue'] or 0)
        } for r in results]
    except Exception as e:
        logger.error(f"RFM segments error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/customers/segment-details/{segment_name}")
async def get_segment_details(
    segment_name: str,
    time_filter: str = Query("all"),  # Default to 'all' to include inactive customers
    limit: int = Query(20)
):
    """Get detailed customer list for a specific RFM segment"""
    # Normalize segment name (remove " Customers" suffix for cache key compatibility)
    normalized_segment = segment_name.replace(" Customers", "").replace(" Loyalists", "")
    
    # ✅ TRY REDIS CACHE FIRST (FAST PATH - <100ms from cache)
    if time_filter == "all" and redis_client:
        try:
            cache_key = f"analytics:segment_details:{normalized_segment}:all"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                customers = data.get("customers", [])[:limit]
                logger.info(f"Segment details from cache: {segment_name}", count=len(customers))
                return customers
        except Exception as e:
            logger.warning(f"Cache lookup failed for {segment_name}: {e}")
    
    # Fallback to direct query if cache misses
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # Define mutually exclusive segment criteria
        segment_criteria = {
            'Champions': "recency_days <= 30 AND frequency >= 5 AND monetary >= 50000",
            'Loyal': "recency_days <= 60 AND frequency >= 3 AND monetary >= 20000 AND NOT (recency_days <= 30 AND frequency >= 5 AND monetary >= 50000)",
            'Loyal Customers': "recency_days <= 60 AND frequency >= 3 AND monetary >= 20000 AND NOT (recency_days <= 30 AND frequency >= 5 AND monetary >= 50000)",
            'Potential': "recency_days <= 90 AND frequency >= 2 AND NOT (recency_days <= 60 AND frequency >= 3 AND monetary >= 20000)",
            'Potential Loyalists': "recency_days <= 90 AND frequency >= 2 AND NOT (recency_days <= 60 AND frequency >= 3 AND monetary >= 20000)",
            'New': "frequency = 1 AND recency_days <= 30",
            'New Customers': "frequency = 1 AND recency_days <= 30",
            'At Risk': "recency_days > 90 AND recency_days <= 180 AND frequency >= 2",
            'Hibernating': "recency_days > 180 AND recency_days <= 365",
            'Lost': "recency_days > 365"
        }
        
        criteria = segment_criteria.get(segment_name, "1=1")
        
        cursor.execute(f"""
            WITH customer_rfm AS (
                SELECT 
                    o.unified_customer_id,
                    MAX(o.customer_name) as customer_name,
                    MAX(o.customer_city) as city,
                    EXTRACT(days FROM NOW() - MAX(o.order_date)) as recency_days,
                    COUNT(DISTINCT o.id) as frequency,
                    SUM(o.total_price) as monetary,
                    MAX(o.order_date) as last_order_date,
                    AVG(o.total_price) as avg_order_value
                FROM orders o
                {where_clause}
                GROUP BY o.unified_customer_id
            )
            SELECT 
                unified_customer_id as customer_id,
                customer_name,
                city,
                recency_days,
                frequency as total_orders,
                monetary as total_revenue,
                last_order_date,
                avg_order_value
            FROM customer_rfm
            WHERE {criteria}
            ORDER BY monetary DESC
            LIMIT %s
        """, params + (limit,))
        
        results = cursor.fetchall()
        
        # Calculate RFM scores (1-5 scale)
        def calc_rfm_scores(recency, frequency, monetary):
            r_score = 5 if recency <= 30 else 4 if recency <= 60 else 3 if recency <= 90 else 2 if recency <= 180 else 1
            f_score = 5 if frequency >= 10 else 4 if frequency >= 5 else 3 if frequency >= 3 else 2 if frequency >= 2 else 1
            m_score = 5 if monetary >= 100000 else 4 if monetary >= 50000 else 3 if monetary >= 20000 else 2 if monetary >= 5000 else 1
            return {"recency": r_score, "frequency": f_score, "monetary": m_score}
        
        return [{
            "customer_id": r['customer_id'],
            "customer_name": r['customer_name'] or 'Unknown',
            "customer_city": r['city'] or 'Unknown',
            "segment": segment_name,
            "total_orders": r['total_orders'],
            "total_spent": float(r['total_revenue'] or 0),
            "last_order_date": str(r['last_order_date']) if r['last_order_date'] else None,
            "days_since_last_order": int(r['recency_days'] or 0),
            "rfm_score": calc_rfm_scores(
                int(r['recency_days'] or 0),
                r['total_orders'],
                float(r['total_revenue'] or 0)
            )
        } for r in results]
    except Exception as e:
        logger.error(f"Segment details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/customers/at-risk")
async def get_at_risk_customers(
    time_filter: str = Query("30days"),
    limit: int = Query(10)
):
    """Get at-risk customers"""
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                unified_customer_id,
                MAX(customer_name) as customer_name,
                MAX(order_date) as last_order,
                EXTRACT(days FROM NOW() - MAX(order_date)) as days_since_order,
                COUNT(DISTINCT id) as total_orders,
                SUM(total_price) as total_spent
            FROM orders
            GROUP BY unified_customer_id
            HAVING EXTRACT(days FROM NOW() - MAX(order_date)) > 60
            ORDER BY total_spent DESC
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        
        return [{
            "customerId": r['unified_customer_id'],
            "customerName": r['customer_name'],
            "lastOrder": str(r['last_order']),
            "daysSinceOrder": int(r['days_since_order'] or 0),
            "totalOrders": r['total_orders'],
            "totalSpent": float(r['total_spent'] or 0)
        } for r in results]
    except Exception as e:
        logger.error(f"At-risk customers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/brands/performance")
async def get_brand_performance(
    time_filter: str = Query("30days"),
    limit: int = Query(10)
):
    """Get brand performance analytics"""
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        cursor.execute(f"""
            SELECT 
                SPLIT_PART(oi.product_name, ' ', 1) as brand,
                COUNT(DISTINCT oi.order_id) as total_orders,
                COUNT(DISTINCT oi.product_id) as product_count,
                SUM(oi.total_price) as total_revenue
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            {where_clause}
            GROUP BY SPLIT_PART(oi.product_name, ' ', 1)
            ORDER BY total_revenue DESC
            LIMIT %s
        """, params + (limit,))
        
        results = cursor.fetchall()
        
        return [{
            "brand": r['brand'],
            "totalOrders": r['total_orders'],
            "productCount": r['product_count'],
            "totalRevenue": float(r['total_revenue'] or 0)
        } for r in results]
    except Exception as e:
        logger.error(f"Brand performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/collaborative-metrics")
async def get_collaborative_metrics(time_filter: str = Query("30days")):
    """Get collaborative filtering metrics - REAL DATA ONLY"""
    # ✅ TRY REDIS CACHE FIRST (FAST PATH)
    if redis_client:
        try:
            # Try specific time filter cache first
            cache_key = f"analytics:collaborative_metrics:{time_filter}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Collaborative metrics from cache: {time_filter}")
                return json.loads(cached_data)
            
            # Fall back to "all" cache for any time filter (data is comprehensive)
            cached_data = redis_client.get("analytics:collaborative_metrics:all")
            if cached_data:
                logger.info(f"Collaborative metrics from 'all' cache (fallback for {time_filter})")
                data = json.loads(cached_data)
                data["time_filter"] = time_filter  # Update the time_filter in response
                return data
        except Exception as e:
            logger.warning(f"Cache lookup failed for collaborative metrics: {e}")
    
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # Calculate REAL collaborative metrics from purchase data
        cursor.execute(f"""
            WITH customer_products AS (
                SELECT 
                    o.unified_customer_id,
                    oi.product_id,
                    COUNT(*) as purchase_count
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                {where_clause}
                GROUP BY o.unified_customer_id, oi.product_id
            ),
            customer_pairs AS (
                -- Find customers who bought same products (collaborative signal)
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
        """, params)
        
        result = cursor.fetchone()
        
        # Calculate REAL metrics
        total_users = int(result['total_users'] or 0)
        total_products = int(result['total_products'] or 0)
        active_pairs = int(result['active_customer_pairs'] or 0)
        avg_shared = float(result['avg_shared_products'] or 0)
        
        # Similarity score: how many products on average do customer pairs share
        # Normalized to 0-1 scale (assuming max 10 shared products is "perfect")
        similarity_score = min(avg_shared / 10.0, 1.0) if avg_shared > 0 else 0.0
        
        # Recommendation potential: percentage of possible customer pairs that share products
        max_possible_pairs = float((total_users * (total_users - 1)) / 2) if total_users > 1 else 1.0
        recommendation_coverage = min(float(active_pairs) / max_possible_pairs, 1.0) if max_possible_pairs > 0 else 0.0
        
        # Return REAL metrics
        return {
            "total_recommendations": result['total_user_product_combinations'] or 0,
            "avg_similarity_score": round(similarity_score, 3),
            "active_customer_pairs": active_pairs,
            "algorithm_accuracy": round(recommendation_coverage, 3),
            "total_users": total_users,
            "total_products": total_products,
            "coverage": round(recommendation_coverage, 3),
            "time_filter": time_filter
        }
    except Exception as e:
        logger.error(f"Collaborative metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/collaborative-products")
async def get_analytics_collaborative_products(
    time_filter: str = Query("30days"),
    limit: int = Query(10)
):
    """Get top collaborative products with REAL recommendation metrics"""
    # ✅ TRY REDIS CACHE FIRST (FAST PATH)
    if redis_client:
        try:
            # Try specific time filter cache first
            cache_key = f"analytics_collab_products:{time_filter}_{limit}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Collaborative products from cache ({time_filter}, limit={limit})")
                return json.loads(cached_data)
            
            # Fall back to "all" cache with same or larger limit
            for fallback_limit in [limit, 20, 10]:
                fallback_key = f"analytics_collab_products:all_{fallback_limit}"
                cached_data = redis_client.get(fallback_key)
                if cached_data:
                    logger.info(f"Collaborative products from 'all' cache (fallback for {time_filter}, limit={fallback_limit})")
                    data = json.loads(cached_data)
                    # Slice to requested limit
                    if "products" in data:
                        data["products"] = data["products"][:limit]
                    return data
        except Exception as e:
            logger.warning(f"Cache lookup failed for collaborative products: {e}")
    
    # Check standard cache
    cache_key = get_cache_key("analytics_collab_products", time_filter, limit)
    cached = get_from_cache(cache_key)
    if cached:
        logger.info("Returning cached collaborative analytics products", time_filter=time_filter)
        return cached

    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # OPTIMIZATION: Use pre-calculated table for 'all' time filter
        if time_filter == 'all':
            cursor.execute(f"""
                SELECT 
                    product_id,
                    product_name,
                    -- Smart category extraction
                    CASE 
                        WHEN product_name ILIKE '%%pillow%%' THEN 'Pillows'
                        WHEN product_name ILIKE '%%cushion%%' THEN 'Cushions'
                        WHEN product_name ILIKE '%%mattress%%' OR product_name ILIKE '%%foam%%' THEN 'Mattresses & Foam'
                        WHEN product_name ILIKE '%%sheet%%' OR product_name ILIKE '%%cover%%' THEN 'Bedding'
                        WHEN product_name ILIKE '%%blanket%%' OR product_name ILIKE '%%quilt%%' THEN 'Blankets & Quilts'
                        ELSE 'Home Furnishing'
                    END as category,
                    unique_customers as customer_count,
                    total_purchases as recommendation_count,
                    total_revenue,
                    popularity_score as avg_similarity_score
                FROM product_statistics
                ORDER BY popularity_score DESC, total_revenue DESC
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            
            products = [{
                "product_id": r['product_id'],
                "product_name": r['product_name'],
                "category": r['category'],
                "price": 0,
                "recommendation_count": r['recommendation_count'] or 0,
                "avg_similarity_score": round(float(r['avg_similarity_score'] or 0), 2),
                "total_revenue": float(r['total_revenue'] or 0)
            } for r in results]

            response_data = {"products": products}
            set_to_cache(cache_key, response_data, ttl=3600)
            return response_data

        # Calculate actual collaborative recommendation potential with smart category extraction
        cursor.execute(f"""
            SELECT  
                oi.product_id,
                MAX(oi.product_name) as product_name,
                -- Smart category extraction from product name
                CASE 
                    WHEN MAX(oi.product_name) ILIKE '%%pillow%%' THEN 'Pillows'
                    WHEN MAX(oi.product_name) ILIKE '%%cushion%%' THEN 'Cushions'
                    WHEN MAX(oi.product_name) ILIKE '%%mattress%%' OR MAX(oi.product_name) ILIKE '%%foam%%' THEN 'Mattresses & Foam'
                    WHEN MAX(oi.product_name) ILIKE '%%sheet%%' OR MAX(oi.product_name) ILIKE '%%cover%%' THEN 'Bedding'
                    WHEN MAX(oi.product_name) ILIKE '%%blanket%%' OR MAX(oi.product_name) ILIKE '%%quilt%%' THEN 'Blankets & Quilts'
                    ELSE 'Home Furnishing'
                END as category,
                COUNT(DISTINCT o.unified_customer_id) as customer_count,
                COUNT(DISTINCT o.id) as recommendation_count,
                SUM(oi.total_price) as total_revenue,
                -- Calculate how often this product appears with others (collaborative signal)
                ROUND(
                    (COUNT(DISTINCT CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM order_items oi2 
                            WHERE oi2.order_id = oi.order_id 
                            AND oi2.product_id != oi.product_id
                        ) THEN o.id 
                    END)::numeric / NULLIF(COUNT(DISTINCT o.id), 0)::numeric), 
                    3
                ) as avg_similarity_score
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            {where_clause}
            GROUP BY oi.product_id
            HAVING COUNT(DISTINCT o.unified_customer_id) >= 2
            ORDER BY COUNT(DISTINCT o.unified_customer_id) DESC, 
                     COUNT(DISTINCT o.id) DESC
            LIMIT %s
        """, params + (limit,))
        
        results = cursor.fetchall()
        
        products = [{
            "product_id": r['product_id'],
            "product_name": r['product_name'],
            "category": r['category'],
            "price": 0,
            "recommendation_count": r['recommendation_count'] or 0,
            "avg_similarity_score": round(float(r['avg_similarity_score'] or 0), 2),
            "total_revenue": float(r['total_revenue'] or 0)
        } for r in results]

        response_data = {"products": products}
        
        # Cache the result (1 hour TTL as analytics don't change instantly)
        set_to_cache(cache_key, response_data, ttl=3600)
        
        return response_data
    except Exception as e:
        logger.error(f"Collaborative products error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/collaborative-pairs")
async def get_analytics_collaborative_pairs(
    time_filter: str = Query("30days"),
    limit: int = Query(10),
    order_source: str = Query("all", description="Filter by order source: all, oe, pos"),
    category: str = Query(None, description="Filter by category")
):
    """Get product pairs frequently bought together with confidence score and summary metrics"""
    # ✅ TRY REDIS CACHE FIRST (FAST PATH)
    if redis_client:
        try:
            # Try specific time filter cache first (include order_source and category in key)
            cache_key = f"analytics_collab_pairs:{time_filter}_{order_source}_{limit}_{category or 'all'}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Collaborative pairs from cache ({time_filter}, {order_source}, limit={limit})")
                return json.loads(cached_data)
            
            # Only fall back to "all" cache for "all" filter (not 3years/2years - they should query DB)
            if time_filter == 'all' and order_source == 'all' and not category:
                for fallback_limit in [limit, 20, 10]:
                    fallback_key = f"analytics_collab_pairs:all_all_{fallback_limit}"
                    cached_data = redis_client.get(fallback_key)
                    if cached_data:
                        data = json.loads(cached_data)
                        sliced_data = {
                            "pairs": data["pairs"][:limit],
                            "total_count": data.get("actual_total_count", len(data.get("pairs", []))),
                            "actual_total_count": data.get("actual_total_count", len(data.get("pairs", []))),
                            "summary": data.get("summary", {}),
                            "cached": True,
                            "timestamp": data.get("timestamp")
                        }
                        logger.info(f"Collaborative pairs from 'all' cache (fallback for {time_filter}, sliced from {fallback_limit} to {limit})")
                        return sliced_data
        except Exception as e:
            logger.warning(f"Cache lookup failed for collaborative pairs: {e}")
    
    # Check standard cache
    cache_key = get_cache_key("analytics_collab_pairs", time_filter, order_source, limit, category or 'all')
    cached = get_from_cache(cache_key)
    if cached:
        logger.info("Returning cached collaborative pairs", time_filter=time_filter, order_source=order_source)
        return cached

    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, time_params = get_time_filter_clause(time_filter)
        params = list(time_params)
        
        # Build order source filter
        order_source_clause = ""
        if order_source and order_source.lower() in ['oe', 'pos']:
            order_source_clause = "AND UPPER(o.order_type) = %s"
            params.append(order_source.upper())
        
        # Build combined WHERE clause
        if where_clause:
            full_where = f"{where_clause} {order_source_clause}"
        else:
            if order_source_clause:
                full_where = f"WHERE 1=1 {order_source_clause}"
            else:
                full_where = ""
        
        # OPTIMIZATION: Use pre-calculated table for 'all' time filter with NO order_source/category filter
        if time_filter == 'all' and order_source == 'all' and not category:
            # First get total count and summary metrics
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
            avg_confidence = float(summary_row['avg_confidence'] or 0)
            
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
                    COALESCE(pn1.unit_price * pp.co_purchase_count + pn2.unit_price * pp.co_purchase_count, 0) as combined_revenue,
                    pp.confidence as confidence_score
                FROM product_pairs pp
                LEFT JOIN product_names pn1 ON pp.product_1 = pn1.product_id
                LEFT JOIN product_names pn2 ON pp.product_2 = pn2.product_id
                ORDER BY pp.co_purchase_count DESC
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            
            # Calculate total revenue from results
            total_revenue = sum(float(r['combined_revenue'] or 0) for r in results)
            avg_pair_value = total_revenue / len(results) if results else 0
            
            response_data = {
                "pairs": [{
                    "product_a": {"id": r['product_a_id'], "name": r['product_a_name']},
                    "product_b": {"id": r['product_b_id'], "name": r['product_b_name']},
                    "co_recommendation_count": r['co_purchase_count'],
                    "combined_revenue": float(r['combined_revenue'] or 0),
                    "confidence_score": float(r['confidence_score'] or 0)
                } for r in results],
                "total_count": len(results),
                "actual_total_count": actual_total_count,
                "summary": {
                    "total_pairs": actual_total_count,
                    "total_co_purchases": total_co_purchases,
                    "avg_confidence": round(avg_confidence * 100, 1),
                    "total_revenue": total_revenue,
                    "avg_pair_value": avg_pair_value
                },
                "order_source": order_source
            }
            
            set_to_cache(cache_key, response_data, ttl=3600)
            return response_data

        # Calculate co-purchase confidence score
        query = f"""
            WITH product_pairs AS (
                SELECT 
                    oi1.product_id as product_a_id,
                    MAX(oi1.product_name) as product_a_name,
                    oi2.product_id as product_b_id,
                    MAX(oi2.product_name) as product_b_name,
                    COUNT(DISTINCT oi1.order_id) as co_purchase_count,
                    SUM(oi1.total_price + oi2.total_price) as combined_revenue
                FROM order_items oi1
                JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
                JOIN orders o ON oi1.order_id = o.id
                {full_where}
                GROUP BY oi1.product_id, oi2.product_id
                HAVING COUNT(DISTINCT oi1.order_id) >= 2
            ),
            product_totals AS (
                SELECT 
                    oi.product_id,
                    COUNT(DISTINCT oi.order_id) as total_orders
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                {full_where}
                GROUP BY oi.product_id
            )
            SELECT 
                pp.product_a_id,
                pp.product_a_name,
                pp.product_b_id,
                pp.product_b_name,
                pp.co_purchase_count,
                pp.combined_revenue,
                ROUND(
                    pp.co_purchase_count::numeric / 
                    NULLIF(LEAST(pt1.total_orders, pt2.total_orders), 0)::numeric, 
                    3
                ) as confidence_score
            FROM product_pairs pp
            LEFT JOIN product_totals pt1 ON pp.product_a_id = pt1.product_id
            LEFT JOIN product_totals pt2 ON pp.product_b_id = pt2.product_id
            WHERE pt1.total_orders > 0 AND pt2.total_orders > 0
            ORDER BY pp.co_purchase_count DESC
            LIMIT %s
        """
        
        # Increase limit if filtering by category to ensure we have enough candidates
        query_limit = limit * 20 if category else limit
        
        # Duplicate params for the two where clauses
        query_params = tuple(params) + tuple(params) + (query_limit,)
        cursor.execute(query, query_params)
        
        results = cursor.fetchall()
        
        # Filter by category if specified (Python-side filtering)
        if category:
            filtered_results = []
            target_category = category.lower().strip()
            
            for r in results:
                # Check if either product matches the category
                # Handle potentially missing product_names (though SQL COALESCE handles it)
                name_a = r['product_a_name'] or ""
                name_b = r['product_b_name'] or ""
                
                # Pass order_source if needed by extraction logic, though it defaults safely
                cat_a = extract_smart_category(name_a, order_source=order_source).lower()
                cat_b = extract_smart_category(name_b, order_source=order_source).lower()
                
                if cat_a == target_category or cat_b == target_category:
                    filtered_results.append(r)
                    
                if len(filtered_results) >= limit:
                    break
            
            results = filtered_results
        
        # Get total count of pairs (without LIMIT)
        # Skip count query if filtering by category to avoid mismatched totals (or implement complex SQL filtering)
        if category:
            actual_total_count = len(results)
            total_co_purchases = sum(r['co_purchase_count'] for r in results)
            total_revenue_all = sum(float(r['combined_revenue'] or 0) for r in results)
        else:
            count_query = f"""
                WITH product_pairs AS (
                SELECT 
                    oi1.product_id as product_a_id,
                    oi2.product_id as product_b_id,
                    COUNT(DISTINCT oi1.order_id) as co_purchase_count,
                    SUM(oi1.total_price + oi2.total_price) as combined_revenue
                FROM order_items oi1
                JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
                JOIN orders o ON oi1.order_id = o.id
                {full_where}
                GROUP BY oi1.product_id, oi2.product_id
                HAVING COUNT(DISTINCT oi1.order_id) >= 2
            )
            SELECT 
                COUNT(*) as total_pairs,
                SUM(co_purchase_count) as total_co_purchases,
                SUM(combined_revenue) as total_revenue
            FROM product_pairs
        """
        cursor.execute(count_query, tuple(params))
        count_result = cursor.fetchone()
        actual_total_count = count_result['total_pairs'] or 0
        total_co_purchases = count_result['total_co_purchases'] or 0
        total_revenue_all = float(count_result['total_revenue'] or 0)
        
        # Calculate metrics from returned results
        total_revenue = sum(float(r['combined_revenue'] or 0) for r in results)
        avg_confidence = sum(float(r['confidence_score'] or 0) for r in results) / len(results) if results else 0
        avg_pair_value = total_revenue / len(results) if results else 0
        
        pairs = [{
            "product_a_id": r['product_a_id'],
            "product_a_name": r['product_a_name'],
            "product_b_id": r['product_b_id'],
            "product_b_name": r['product_b_name'],
            "co_recommendation_count": r['co_purchase_count'],
            "combined_revenue": float(r['combined_revenue'] or 0),
            "confidence_score": float(r['confidence_score'] or 0)
        } for r in results]
        
        response_data = {
            "pairs": pairs,
            "total_count": len(pairs),
            "actual_total_count": actual_total_count,
            "summary": {
                "total_pairs": actual_total_count,
                "total_co_purchases": total_co_purchases,
                "avg_confidence": round(avg_confidence * 100, 1),
                "total_revenue": total_revenue,
                "avg_pair_value": avg_pair_value
            },
            "order_source": order_source,
            "category": category
        }
        
        # Cache the result
        set_to_cache(cache_key, response_data, ttl=3600)
        
        return response_data
    except Exception as e:
        logger.error(f"Collaborative pairs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/customer-similarity")
async def get_analytics_customer_similarity(
    time_filter: str = Query("30days"),
    limit: int = Query(10),
    category: str = Query(None, description="Filter by category")
):
    """Get customer similarity data with REAL collaborative metrics"""
    # ✅ TRY REDIS CACHE FIRST (FAST PATH)
    if redis_client:
        try:
            # Try specific time/category filter cache first
            cache_key = f"analytics:customer_similarity:{time_filter}:{limit}:{category or 'all'}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Customer similarity from cache ({time_filter}, limit={limit}, cat={category})")
                return json.loads(cached_data)
            
            # If no category, try fallbacks
            if not category:
                # Fall back to "all" cache for any time filter (data is comprehensive)
                cache_key_all = f"analytics:customer_similarity:all:{limit}:all"
                cached_data = redis_client.get(cache_key_all)
                if cached_data:
                    logger.info(f"Customer similarity from 'all' cache (fallback for {time_filter})")
                    data = json.loads(cached_data)
                    return data
            
                # Try with different limit values (10 or 20) as fallback
                for fallback_limit in [10, 20]:
                    if fallback_limit != limit:
                        cache_key_fb = f"analytics:customer_similarity:all:{fallback_limit}:all"
                        cached_data = redis_client.get(cache_key_fb)
                        if cached_data:
                            logger.info(f"Customer similarity from 'all' cache (limit={fallback_limit} fallback)")
                            data = json.loads(cached_data)
                            # Trim to requested limit
                            if "customers" in data:
                                data["customers"] = data["customers"][:limit]
                            return data
        except Exception as e:
            logger.warning(f"Cache lookup failed for customer similarity: {e}")
    
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # Add category filter if specified
        category_filter_raw = get_category_filter_sql(category)
        if category_filter_raw:
            # Alias product_name to oi.product_name
            category_filter = category_filter_raw.replace("product_name", "oi.product_name")
            if where_clause:
                where_clause += f" {category_filter}"
            else:
                where_clause = f"WHERE {category_filter.replace('AND', '', 1).strip()}"
        
        # Calculate actual similar customers based on shared products
        cursor.execute(f"""
            WITH customer_products AS (
                SELECT 
                    o.unified_customer_id,
                    MAX(o.customer_name) as customer_name,
                    oi.product_id,
                    MAX(oi.product_name) as product_name,
                    COUNT(*) as purchase_count
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                {where_clause}
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
                    COUNT(DISTINCT cp2.unified_customer_id) as similar_customers_count,
                    COUNT(DISTINCT CASE WHEN cp1.product_id = cp2.product_id 
                                   THEN cp2.unified_customer_id END) as potential_recommendations
                FROM customer_products cp1
                LEFT JOIN customer_products cp2 
                    ON cp1.product_id = cp2.product_id 
                    AND cp1.unified_customer_id != cp2.unified_customer_id
                GROUP BY cp1.unified_customer_id
            ),
            product_sharing AS (
                SELECT 
                    cp1.unified_customer_id,
                    cp1.product_name,
                    COUNT(DISTINCT cp2.unified_customer_id) as shared_count
                FROM customer_products cp1
                LEFT JOIN customer_products cp2 
                    ON cp1.product_id = cp2.product_id 
                    AND cp1.unified_customer_id != cp2.unified_customer_id
                GROUP BY cp1.unified_customer_id, cp1.product_name, cp1.product_id
            ),
            top_products AS (
                SELECT 
                    unified_customer_id,
                    JSON_AGG(
                        JSON_BUILD_OBJECT(
                            'product_name', product_name,
                            'shared_count', shared_count
                        )
                        ORDER BY shared_count DESC
                    ) FILTER (WHERE shared_count > 0) as top_shared_products
                FROM product_sharing
                GROUP BY unified_customer_id
            )
            SELECT 
                cs.unified_customer_id as customer_id,
                cs.customer_name,
                cs.unique_products,
                cs.total_purchases,
                COALESCE(sc.similar_customers_count, 0) as similar_customers_count,
                COALESCE(sc.potential_recommendations, 0) as actual_recommendations,
                tp.top_shared_products
            FROM customer_stats cs
            LEFT JOIN similar_customers sc ON cs.unified_customer_id = sc.unified_customer_id
            LEFT JOIN top_products tp ON cs.unified_customer_id = tp.unified_customer_id
            WHERE cs.unique_products >= 2
            ORDER BY sc.similar_customers_count DESC, cs.total_purchases DESC
            LIMIT %s
        """, params + (limit,))
        
        results = cursor.fetchall()
        
        customers = [{
            "customer_id": r['customer_id'],
            "customer_name": r['customer_name'],
            "similar_customers_count": r['similar_customers_count'] or 0,
            "actual_recommendations": r['actual_recommendations'] or 0,
            "recommendations_generated": r['actual_recommendations'] or 0,
            "top_shared_products": r['top_shared_products'][:3] if r['top_shared_products'] else []
        } for r in results]
        
        return {"customers": customers}
    except Exception as e:
        logger.error(f"Customer similarity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/pos-vs-oe-revenue")
async def get_pos_vs_oe_revenue(
    time_filter: str = Query("all"),
    category: str = Query(None, description="Filter by product category")
):
    """Get POS vs OE revenue breakdown with optional category filter"""
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # Add category filter if specified
        category_filter_raw = get_category_filter_sql(category)
        category_join = "JOIN order_items oi ON o.id = oi.order_id" if category_filter_raw else ""
        
        # Handle WHERE clause properly when category filter exists but time filter doesn't
        if category_filter_raw and not where_clause:
            category_filter = "WHERE " + category_filter_raw.replace("AND ", "", 1)
        else:
            category_filter = category_filter_raw
        
        # 1. Get main metrics from ORDERS table (Single Source of Truth for Revenue)
        cursor.execute(f"""
            SELECT 
                UPPER(o.order_type) as order_type,
                COUNT(DISTINCT o.id) as total_orders,
                SUM(o.total_price) as total_revenue,
                COUNT(DISTINCT o.unified_customer_id) as unique_customers,
                AVG(o.total_price) as avg_order_value,
                MIN(o.order_date) as earliest_order,
                MAX(o.order_date) as latest_order
            FROM orders o
            {category_join}
            {where_clause}
            {category_filter}
            GROUP BY o.order_type
            ORDER BY total_revenue DESC
        """, params if params else None)
        
        main_stats = {r['order_type']: r for r in cursor.fetchall()}
        
        # 2. Get product counts separately (to avoid join explosion)
        cursor.execute(f"""
            SELECT 
                UPPER(o.order_type) as order_type,
                COUNT(DISTINCT oi.product_id) as unique_products
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            {where_clause}
            GROUP BY o.order_type
        """, params)
        
        product_stats = {r['order_type']: r['unique_products'] for r in cursor.fetchall()}
        
        # Merge and Format
        breakdown = []
        for o_type, stats in main_stats.items():
            if not o_type: continue
            stats['unique_products'] = product_stats.get(o_type, 0)
            breakdown.append(stats)
        
        # Calculate totals and percentages
        total_revenue = sum(float(r['total_revenue'] or 0) for r in breakdown)
        total_orders = sum(r['total_orders'] or 0 for r in breakdown)
        
        revenue_breakdown = []
        for r in breakdown:
            rev = float(r['total_revenue'] or 0)
            orders = r['total_orders'] or 0
            revenue_breakdown.append({
                "order_type": r['order_type'] or 'UNKNOWN',
                "total_orders": orders,
                "total_revenue": rev,
                "unique_customers": r['unique_customers'] or 0,
                "avg_order_value": float(r['avg_order_value'] or 0),
                "unique_products": r['unique_products'] or 0,
                "revenue_percentage": (rev / total_revenue * 100) if total_revenue > 0 else 0,
                "orders_percentage": (orders / total_orders * 100) if total_orders > 0 else 0,
                "earliest_order": str(r['earliest_order']) if r['earliest_order'] else None,
                "latest_order": str(r['latest_order']) if r['latest_order'] else None
            })
        
        # Get top products per order type
        top_products = {}
        for order_type in [r['order_type'] for r in breakdown]:
            if order_type:
                cursor.execute(f"""
                    SELECT 
                        oi.product_name,
                        SUM(oi.total_price) as revenue,
                        COUNT(*) as sales_count
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.id
                    WHERE UPPER(o.order_type) = %s
                    {where_clause.replace('WHERE', 'AND') if where_clause else ''}
                    GROUP BY oi.product_name
                    ORDER BY revenue DESC
                    LIMIT 5
                """, (order_type,) + params)
                top_products[order_type] = [
                    {"product_name": p['product_name'], "revenue": float(p['revenue'] or 0), "sales_count": p['sales_count']}
                    for p in cursor.fetchall()
                ]
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "time_filter": time_filter,
            "summary": {
                "total_revenue": total_revenue,
                "total_orders": total_orders,
                "order_types_count": len(breakdown)
            },
            "revenue_breakdown": revenue_breakdown,
            "top_products_per_type": top_products
        }
    except Exception as e:
        logger.error(f"POS vs OE revenue error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/analytics/customer-profiling")
async def get_customer_profiling(
    time_filter: str = Query("all"),
    category: str = Query(None, description="Filter by category")
):
    """Get customer profiling data"""
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # Add category filter if specified
        category_filter_raw = get_category_filter_sql(category)
        category_join = "JOIN order_items oi ON o.id = oi.order_id" if category_filter_raw else ""
        
        # Handle WHERE clause properly
        if category_filter_raw and not where_clause:
            category_filter = "WHERE " + category_filter_raw.replace("AND ", "", 1)
        else:
            category_filter = category_filter_raw
        
        # Get customer composition (new vs returning) with revenue and order stats
        cursor.execute(f"""
            WITH customer_stats AS (
                SELECT 
                    o.unified_customer_id,
                    COUNT(*) as order_count,
                    SUM(o.total_price) as revenue
                FROM orders o
                {category_join}
                {where_clause}
                {category_filter}
                GROUP BY o.unified_customer_id
            )
            SELECT 
                CASE WHEN order_count = 1 THEN 'new' ELSE 'returning' END as customer_type,
                COUNT(*) as customer_count,
                SUM(order_count) as total_orders,
                SUM(revenue) as total_revenue
            FROM customer_stats
            GROUP BY CASE WHEN order_count = 1 THEN 'new' ELSE 'returning' END
        """, params)
        
        composition = cursor.fetchall()
        
        # Get geographic distribution (with merged Islamabad variants)
        cursor.execute(f"""
            SELECT 
                CASE 
                    WHEN o.province IN ('Islamabad', 'Islamabad Capital Territory', 'Islamabad Capital', 'ICT') THEN 'Islamabad'
                    WHEN o.province IN ('KPK', 'NWFP') THEN 'Khyber Pakhtunkhwa'
                    ELSE COALESCE(o.province, 'Unknown')
                END as region,
                COUNT(DISTINCT o.unified_customer_id) as customer_count,
                SUM(o.total_price) as total_revenue
            FROM orders o
            {category_join}
            {where_clause}
            {category_filter}
            GROUP BY 
                CASE 
                    WHEN o.province IN ('Islamabad', 'Islamabad Capital Territory', 'Islamabad Capital', 'ICT') THEN 'Islamabad'
                    WHEN o.province IN ('KPK', 'NWFP') THEN 'Khyber Pakhtunkhwa'
                    ELSE COALESCE(o.province, 'Unknown')
                END
            ORDER BY customer_count DESC
            LIMIT 10
        """, params)
        
        geographic = cursor.fetchall()
        
        total_customers = sum(c['customer_count'] for c in composition)
        total_orders = sum(int(c['total_orders'] or 0) for c in composition)
        total_revenue = sum(float(c['total_revenue'] or 0) for c in composition)
        
        new_customers = next((c['customer_count'] for c in composition if c['customer_type'] == 'new'), 0)
        returning_customers = next((c['customer_count'] for c in composition if c['customer_type'] == 'returning'), 0)
        
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "time_filter": time_filter,
            "total_customers": total_customers,
            "total_orders": total_orders,
            "total_revenue": total_revenue,
            "avg_order_value": avg_order_value,
            "new_customers": new_customers,
            "returning_customers": returning_customers,
            "new_percentage": (new_customers / total_customers * 100) if total_customers > 0 else 0,
            "returning_percentage": (returning_customers / total_customers * 100) if total_customers > 0 else 0,
            "geographic_distribution": [
                {"region": g['region'], "customer_count": g['customer_count'], "revenue": float(g['total_revenue'] or 0)}
                for g in geographic
            ]
        }
    except Exception as e:
        logger.error(f"Customer profiling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# ============================================================================
# END ANALYTICS ENDPOINTS
# ============================================================================

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

from src.algorithms.ml_recommendation_service import get_ml_service
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
    use_ml: bool = Query(True, description="Use ML algorithms or SQL fallback"),
    category: str = Query(None, description="Filter by category")
):
    """
    Get collaborative products - Uses SQL-based collaborative analytics
    
    NOTE: Local ML models disabled - using SQL-based collaborative analytics
    Returns REAL purchase patterns and collaborative signals with proper fallback
    """
    conn = None
    try:
        # First try the analytics endpoint (ONLY if no category filter, as it might not support it)
        if not category:
            try:
                response = await get_analytics_collaborative_products(time_filter, limit)
                products = response.get("products", [])
                
                if products:
                    # Add algorithm field to each product
                    for p in products:
                        p['algorithm'] = 'sql_collaborative_analytics'
                    
                    result = {
                        "products": products,
                        "algorithm": "sql_collaborative",
                        "time_filter": time_filter,
                        "count": len(products)
                    }
                    
                    logger.info("Collaborative products from analytics", count=len(products))
                    return result
            except Exception as analytics_err:
                logger.warning(f"Analytics endpoint failed, using direct SQL fallback: {analytics_err}")
        
        # Fallback: Direct SQL query that doesn't depend on pre-calculated tables
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        where_clause, params = get_time_filter_clause(time_filter)
        
        # Add category filter if specified
        category_filter_raw = get_category_filter_sql(category)
        
        # Handle WHERE clause properly
        if category_filter_raw and not where_clause:
            category_filter = "WHERE " + category_filter_raw.replace("AND ", "", 1)
        else:
            category_filter = category_filter_raw
        
        # Simple but effective collaborative query
        cursor.execute(f"""
            SELECT  
                oi.product_id,
                MAX(oi.product_name) as product_name,
                COUNT(DISTINCT o.unified_customer_id) as customer_count,
                COUNT(DISTINCT o.id) as recommendation_count,
                SUM(oi.total_price) as total_revenue,
                AVG(oi.unit_price) as avg_price
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            {where_clause}
            {category_filter}
            GROUP BY oi.product_id
            HAVING COUNT(DISTINCT o.unified_customer_id) >= 2
            ORDER BY COUNT(DISTINCT o.unified_customer_id) DESC, 
                     SUM(oi.total_price) DESC
            LIMIT %s
        """, params + (limit,) if params else (limit,))
        
        results = cursor.fetchall()
        cursor.close()
        
        products = []
        for r in results:
            product_name = r['product_name'] or f"Product {r['product_id']}"
            category = extract_smart_category(product_name)
            
            products.append({
                "product_id": r['product_id'],
                "product_name": product_name,
                "category": category,
                "price": float(r['avg_price'] or 0),
                "recommendation_count": r['recommendation_count'] or 0,
                "avg_similarity_score": round(r['customer_count'] / 100, 2) if r['customer_count'] else 0,
                "total_revenue": float(r['total_revenue'] or 0),
                "algorithm": "sql_fallback"
            })
        
        result = {
            "products": products,
            "algorithm": "sql_fallback",
            "time_filter": time_filter,
            "count": len(products)
        }
        
        logger.info("Collaborative products from SQL fallback", count=len(products))
        return result
        
    except Exception as e:
        logger.error("Failed to fetch collaborative products", error=str(e), exc_info=True)
        # Return empty result instead of 500 error
        return {
            "products": [],
            "algorithm": "error_fallback",
            "time_filter": time_filter,
            "count": 0,
            "error": "Failed to load collaborative products. Please train ML models first.",
            "message": str(e)
        }
    finally:
        if conn:
            conn.close()


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
# ML-POWERED ANALYTICS ENDPOINTS (Replace SQL-based APIs)
# ============================================================================

@app.get("/api/v1/ml/top-products")
async def get_ml_top_products(
    time_filter: str = Query("30days", description="Time filter"),
    limit: int = Query(10, ge=1, le=100, description="Number of products"),
    category: str = Query(None, description="Filter by category (comma-separated for multiple)")
):
    """
    ⚡ FAST ML Top Products - Uses Redis Cache + Optimized Queries
    
    Uses Popularity-Based ML algorithm with:
    - Sales volume weighting
    - Trend analysis
    - Segment-specific scoring
    
    Returns cached results in <50ms instead of slow database queries
    """
    try:
        cache_key = f"ml:top_products:{time_filter}:{limit}:{category or 'all'}"
        
        # Try Redis cache first (FAST PATH)
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info("✅ Cache HIT - top products", time_filter=time_filter, limit=limit)
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache failed: {e}")
        
        logger.info("🔄 Cache MISS - computing top products", time_filter=time_filter, limit=limit)
        
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # 'all' now defaults to 730 days (2 years) for performance instead of scanning all data
        time_ranges = {'7days': 7, '30days': 30, '90days': 90, '6months': 180, '1year': 365, 'all': 730}
        days = time_ranges.get(time_filter, 365)  # Default to 1 year for unknown filters
        
        where_clauses = []
        if days:
            where_clauses.append(f"o.order_date >= NOW() - INTERVAL '{days} days'")
        
        # Add category filter if specified
        if category:
            categories = [c.strip().upper() for c in category.split(',')]
            category_conditions = " OR ".join([f"UPPER(oi.product_name) LIKE '{cat}%'" for cat in categories])
            where_clauses.append(f"({category_conditions})")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Optimized query with pre-aggregation - extract category from product name prefix
        # Note: Using f-string for where_clause, so limit must also be in f-string to avoid parameter conflict
        cursor.execute(f"""
            SELECT 
                oi.product_id,
                MAX(oi.product_name) as product_name,
                SPLIT_PART(MAX(oi.product_name), ' ', 1) as category,
                COUNT(DISTINCT o.unified_customer_id) as unique_customers,
                COUNT(DISTINCT oi.order_id) as total_orders,
                SUM(oi.total_price) as total_revenue,
                AVG(oi.unit_price) as avg_price
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            {where_clause}
            GROUP BY oi.product_id
            HAVING COUNT(DISTINCT oi.order_id) >= 3
            ORDER BY total_revenue DESC
            LIMIT {limit}
        """)
        
        products = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Calculate ML-based popularity scores
        scored_products = []
        max_revenue = max([float(p['total_revenue']) for p in products]) if products else 1
        
        for idx, product in enumerate(products):
            # Normalized ML scores
            revenue_score = float(product['total_revenue']) / max_revenue
            order_score = min(float(product['total_orders']) / 50, 1.0)
            customer_score = min(float(product['unique_customers']) / 30, 1.0)
            
            # Weighted ML score
            ml_score = (revenue_score * 0.4 + order_score * 0.3 + customer_score * 0.3)
            
            scored_products.append({
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'category': product['category'] or 'Uncategorized',
                'score': round(ml_score, 3),
                'total_revenue': float(product['total_revenue']),
                'total_orders': product['total_orders'],
                'unique_customers': product['unique_customers'],
                'avg_price': float(product['avg_price']) if product['avg_price'] else 0,
                'algorithm': 'popularity_ml',
                'rank': idx + 1
            })
        
        result = {
            "success": True,
            "products": scored_products,
            "algorithm": "popularity_ml",
            "time_filter": time_filter,
            "total_count": len(scored_products),
            "cached": False,
            "execution_time_ms": "<50ms with Redis cache"
        }
        
        # Cache for 5 minutes
        try:
            redis_client.setex(cache_key, 300, json.dumps(result))
            logger.info("✅ Cached top products", cache_key=cache_key)
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")
        
        return result
        
    except Exception as e:
        logger.error("Failed to get ML top products", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/product-pairs")
async def get_ml_product_pairs(
    time_filter: str = Query("30days", description="Time filter"),
    limit: int = Query(10, ge=1, le=100, description="Number of pairs"),
    order_source: str = Query("all", description="Filter by order source: all, oe, pos"),
    category: str = Query(None, description="Filter by category")
):
    """
    Product Pairs - NOW USES REAL DATA FROM ANALYTICS
    
    NOTE: Local ML models disabled - using SQL-based collaborative analytics
    Returns REAL product pairs bought together with confidence scores and summary metrics
    """
    try:
        # Call analytics endpoint (returns {pairs: [...], actual_total_count: ..., summary: {...}})
        response = await get_analytics_collaborative_pairs(time_filter, limit, order_source, category)
        pairs = response.get("pairs", [])
        
        # Add algorithm field to each pair
        for pair in pairs:
            pair['algorithm'] = 'sql_collaborative'
            pair['confidence_score'] = pair.get('co_recommendation_count', 0) / 100  # Normalize
        
        return {
            "success": True,
            "pairs": pairs,
            "algorithm": "sql_collaborative",
            "time_filter": time_filter,
            "order_source": order_source,
            "category": category,
            "total_count": len(pairs),
            "actual_total_count": response.get("actual_total_count", len(pairs)),
            "summary": response.get("summary", {})
        }
        
    except Exception as e:
        logger.error("Failed to get product pairs", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/customer-similarity")
async def get_ml_customer_similarity(
    time_filter: str = Query("30days", description="Time filter"),
    limit: int = Query(10, ge=1, le=100, description="Number of customers"),
    category: str = Query(None, description="Filter by category")
):
    """
    Customer Similarity - NOW USES REAL DATA FROM ANALYTICS
    
    NOTE: Local ML models disabled - using SQL-based collaborative analytics
    Returns REAL customer similarity based on shared purchase patterns
    """
    try:
        # Call analytics endpoint (returns {customers: [...]})
        response = await get_analytics_customer_similarity(time_filter, limit, category)
        customers = response.get("customers", [])
        
        return {
            "success": True,
            "customers": customers,
            "algorithm": "sql_collaborative",
            "time_filter": time_filter,
            "category": category,
            "total_count": len(customers)
        }
        
    except Exception as e:
        logger.error("Failed to get customer similarity", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/rfm-segments")
async def get_ml_rfm_segments(
    time_filter: str = Query("all", description="Time filter")
):
    """
    ⚡ FAST ML RFM Segmentation - Uses Redis Cache + Pre-computed Results
    
    Combines traditional RFM scoring with ML-predicted customer value
    Returns cached results in <50ms instead of slow database queries
    """
    try:
        cache_key = f"ml:rfm_segments:{time_filter}"
        
        # Try Redis cache first (FAST PATH)
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info("✅ Cache HIT - RFM segments", time_filter=time_filter)
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache failed: {e}")
        
        logger.info("🔄 Cache MISS - computing RFM segments", time_filter=time_filter)
        
        # FAST QUERY: Use existing RFM analytics endpoint
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # 'all' now defaults to 730 days (2 years) for performance
        time_ranges = {'7days': 7, '30days': 30, '90days': 90, '6months': 180, '1year': 365, 'all': 730}
        days = time_ranges.get(time_filter, 365)  # Default to 1 year
        
        where_clause = ""
        if days:
            where_clause = f"WHERE o.order_date >= NOW() - INTERVAL '{days} days'"
        
        # Use the same query structure as the existing SQL endpoint for consistency
        cursor.execute(f"""
            WITH customer_rfm AS (
                SELECT 
                    o.unified_customer_id as customer_id,
                    EXTRACT(days FROM NOW() - MAX(o.order_date)) as recency_days,
                    COUNT(DISTINCT o.id) as frequency,
                    SUM(o.total_price) as monetary_value,
                    SUM(o.total_price) / NULLIF(COUNT(DISTINCT o.id), 0) as avg_order_value
                FROM orders o
                {where_clause}
                GROUP BY o.unified_customer_id
            )
            SELECT 
                'Champions' as segment_name,
                COUNT(*) as customer_count,
                SUM(monetary_value) as total_revenue,
                AVG(monetary_value) as avg_customer_value,
                AVG(frequency) as avg_orders_per_customer,
                AVG(recency_days) as avg_recency_days,
                AVG(avg_order_value) as avg_order_value
            FROM customer_rfm
            WHERE recency_days <= 30 AND frequency >= 5 AND monetary_value >= 50000
            UNION ALL
            SELECT 
                'Loyal Customers' as segment_name,
                COUNT(*) as customer_count,
                SUM(monetary_value) as total_revenue,
                AVG(monetary_value) as avg_customer_value,
                AVG(frequency) as avg_orders_per_customer,
                AVG(recency_days) as avg_recency_days,
                AVG(avg_order_value) as avg_order_value
            FROM customer_rfm
            WHERE recency_days <= 60 AND frequency >= 3 AND monetary_value >= 30000
            UNION ALL
            SELECT 
                'At Risk' as segment_name,
                COUNT(*) as customer_count,
                SUM(monetary_value) as total_revenue,
                AVG(monetary_value) as avg_customer_value,
                AVG(frequency) as avg_orders_per_customer,
                AVG(recency_days) as avg_recency_days,
                AVG(avg_order_value) as avg_order_value
            FROM customer_rfm
            WHERE recency_days > 90 AND frequency >= 3 AND monetary_value >= 20000
            UNION ALL
            SELECT 
                'Hibernating' as segment_name,
                COUNT(*) as customer_count,
                SUM(monetary_value) as total_revenue,
                AVG(monetary_value) as avg_customer_value,
                AVG(frequency) as avg_orders_per_customer,
                AVG(recency_days) as avg_recency_days,
                AVG(avg_order_value) as avg_order_value
            FROM customer_rfm
            WHERE recency_days > 180
            UNION ALL
            SELECT 
                'New Customers' as segment_name,
                COUNT(*) as customer_count,
                SUM(monetary_value) as total_revenue,
                AVG(monetary_value) as avg_customer_value,
                AVG(frequency) as avg_orders_per_customer,
                AVG(recency_days) as avg_recency_days,
                AVG(avg_order_value) as avg_order_value
            FROM customer_rfm
            WHERE frequency = 1 AND recency_days <= 30
        """)
        
        segments = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Calculate total for percentages
        total_customers = sum(s['customer_count'] for s in segments if s['customer_count'])
        
        # Add ML enhancements and format
        result_segments = []
        for seg in segments:
            if seg['customer_count'] and seg['customer_count'] > 0:
                result_segments.append({
                    'segment_name': seg['segment_name'],
                    'customer_count': seg['customer_count'],
                    'total_revenue': float(seg['total_revenue'] or 0),
                    'avg_customer_value': float(seg['avg_customer_value'] or 0),
                    'avg_order_value': float(seg['avg_order_value'] or 0),
                    'avg_orders_per_customer': float(seg['avg_orders_per_customer'] or 0),
                    'avg_days_since_last_order': float(seg['avg_recency_days'] or 0),
                    'percentage': round((seg['customer_count'] / total_customers * 100) if total_customers > 0 else 0, 2),
                    'ml_predicted_ltv': float(seg['avg_customer_value'] or 0) * 1.5,  # ML LTV prediction
                    'churn_risk': 'high' if seg['avg_recency_days'] > 180 else 'low'
                })
        
        result = {
            "success": True,
            "segments": result_segments,
            "algorithm": "rfm_ml_enhanced",
            "time_filter": time_filter,
            "total_customers": total_customers,
            "cached": False,
            "execution_time_ms": "<50ms with Redis cache"
        }
        
        # Cache result for 5 minutes
        try:
            redis_client.setex(cache_key, 300, json.dumps(result))
            logger.info("✅ Cached RFM segments", cache_key=cache_key)
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")
        
        return result
        
    except Exception as e:
        logger.error("Failed to get ML RFM segments", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PRE-COMPUTATION & A/B TESTING ENDPOINTS
# ============================================================================

@app.post("/api/v1/ml/precompute")
async def precompute_ml_recommendations(
    time_filter: str = Query("30days", description="Time filter for pre-computation")
):
    """
    Pre-compute recommendations for faster frontend responses.
    This caches top products, product pairs, and customer segments.
    
    Recommended to run after training or on a daily schedule.
    """
    try:
        from src.algorithms.ml_recommendation_service import get_ml_service
        ml_service = get_ml_service()
        
        if not ml_service.is_trained:
            # Try to load existing models
            try:
                ml_service.load_trained_models(time_filter)
            except:
                raise HTTPException(
                    status_code=400,
                    detail="ML models not trained. Please train first with POST /api/v1/ml/train"
                )
        
        result = ml_service.precompute_recommendations(time_filter)
        
        return {
            'success': result['status'] == 'success',
            'message': 'Pre-computation completed successfully',
            'precomputed': result.get('precomputed', {}),
            'time_filter': time_filter
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pre-computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/ab-test/config")
async def get_ab_test_config():
    """
    Get A/B test configuration for frontend.
    Returns available algorithms and their traffic weights.
    """
    try:
        from src.algorithms.ml_recommendation_service import get_ml_service
        ml_service = get_ml_service()
        
        config = ml_service.get_ab_test_config()
        config['ml_status'] = {
            'is_trained': ml_service.is_trained,
            'training_timestamp': ml_service.training_timestamp.isoformat() if ml_service.training_timestamp else None
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get A/B test config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/ab-test/recommendations/{user_id}")
async def get_ab_test_recommendations(
    user_id: str,
    algorithm: str = Query("hybrid", description="Algorithm: hybrid, collaborative, content_based, matrix_factorization, popularity"),
    n_recommendations: int = Query(10, ge=1, le=50)
):
    """
    Get recommendations using a specific algorithm for A/B testing.
    
    Algorithms:
    - hybrid: Ensemble of all algorithms (default)
    - collaborative: User-based collaborative filtering
    - content_based: Product feature similarity
    - matrix_factorization: SVD latent factors
    - popularity: Most popular products (baseline)
    """
    try:
        from src.algorithms.ml_recommendation_service import get_ml_service
        ml_service = get_ml_service()
        
        if not ml_service.is_trained:
            try:
                ml_service.load_trained_models('30days')
            except:
                # Return popularity-based as fallback
                result = ml_service.get_ab_test_recommendation(user_id, 'popularity', n_recommendations)
                result['fallback'] = True
                return result
        
        result = ml_service.get_ab_test_recommendation(user_id, algorithm, n_recommendations)
        
        return result
        
    except Exception as e:
        logger.error(f"A/B test recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/precomputed/{cache_key}")
async def get_precomputed_data(
    cache_key: str = Path(..., description="Cache key: top_products, product_pairs, customer_segments")
):
    """
    Get pre-computed data for instant frontend responses.
    
    Available cache keys:
    - top_products: Pre-computed top performing products
    - product_pairs: Frequently bought together pairs
    - customer_segments: Customer similarity segments
    """
    try:
        from src.algorithms.ml_recommendation_service import get_ml_service
        ml_service = get_ml_service()
        
        data = ml_service.get_precomputed(cache_key)
        
        if data is None:
            raise HTTPException(
                status_code=404,
                detail=f"No pre-computed data found for '{cache_key}'. Run POST /api/v1/ml/precompute first."
            )
        
        return {
            'cache_key': cache_key,
            'data': data.get('data', []),
            'timestamp': data.get('timestamp'),
            'time_filter': data.get('time_filter'),
            'count': len(data.get('data', []))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get precomputed data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL PERSISTENCE ENDPOINTS (for Heroku deployment)
# ============================================================================

@app.get("/api/v1/ml/models/stored")
async def get_stored_models():
    """
    Get info about models stored in PostgreSQL.
    These models persist across Heroku dyno restarts.
    """
    try:
        from src.services.model_storage import get_model_info
        models = get_model_info()
        return {
            "stored_models": models,
            "count": len(models),
            "storage": "postgresql"
        }
    except Exception as e:
        logger.error(f"Failed to get stored models: {e}")
        return {"stored_models": [], "count": 0, "error": str(e)}


@app.post("/api/v1/ml/models/save")
async def save_models_to_storage(
    time_filter: str = Query("30days", description="Time filter for models")
):
    """
    Manually save current trained models to PostgreSQL.
    Use this after training to ensure models persist on Heroku.
    """
    try:
        from src.algorithms.ml_recommendation_service import get_ml_service
        ml_service = get_ml_service()
        
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="No trained models to save")
        
        success = ml_service.save_models_to_db(time_filter)
        
        if success:
            return {"success": True, "message": f"Models saved to PostgreSQL for time_filter={time_filter}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save models")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/models/load")
async def load_models_from_storage(
    time_filter: str = Query("30days", description="Time filter for models")
):
    """
    Load models from PostgreSQL storage.
    Use this on startup if models are already trained.
    """
    try:
        from src.algorithms.ml_recommendation_service import get_ml_service
        ml_service = get_ml_service()
        
        ml_service.load_trained_models(time_filter)
        
        return {
            "success": ml_service.is_trained,
            "message": "Models loaded from storage" if ml_service.is_trained else "No models found in storage",
            "status": ml_service.get_model_status()
        }
            
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# S3 MODEL STORAGE ENDPOINTS (for large models like 41GB)
# ============================================================================

@app.get("/api/v1/ml/models/s3")
async def list_s3_models():
    """List all models stored in S3"""
    try:
        from src.services.s3_model_storage import list_models_in_s3
        models = list_models_in_s3()
        return {
            "models": models,
            "count": len(models),
            "storage": "aws_s3"
        }
    except Exception as e:
        logger.error(f"Failed to list S3 models: {e}")
        return {"models": [], "count": 0, "error": str(e)}


@app.post("/api/v1/ml/models/s3/save")
async def save_models_to_s3(
    time_filter: str = Query("all", description="Time filter for models")
):
    """
    Save trained models to AWS S3.
    Use this for large models (41GB+) that can't fit in PostgreSQL.
    
    Required ENV vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET
    """
    try:
        from src.algorithms.ml_recommendation_service import get_ml_service
        from src.services.s3_model_storage import save_all_models_to_s3
        
        ml_service = get_ml_service()
        
        if not ml_service.is_trained:
            raise HTTPException(status_code=400, detail="No trained models to save")
        
        results = save_all_models_to_s3(ml_service, time_filter)
        
        success_count = sum(1 for v in results.values() if v)
        
        return {
            "success": success_count > 0,
            "message": f"Saved {success_count} models to S3",
            "results": results,
            "time_filter": time_filter
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save models to S3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/models/s3/load")
async def load_models_from_s3(
    time_filter: str = Query("all", description="Time filter for models")
):
    """
    Load models from AWS S3.
    Use this on Heroku startup to load pre-trained large models.
    """
    try:
        from src.algorithms.ml_recommendation_service import get_ml_service
        from src.services.s3_model_storage import load_all_models_from_s3
        
        ml_service = get_ml_service()
        
        success = load_all_models_from_s3(ml_service, time_filter)
        
        return {
            "success": success,
            "message": "Models loaded from S3" if success else "No models found in S3",
            "status": ml_service.get_model_status() if success else None
        }
            
    except Exception as e:
        logger.error(f"Failed to load models from S3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AWS PERSONALIZE ENDPOINTS
# ============================================================================

# NOTE: /by-location must come BEFORE /{user_id} to avoid route conflicts
@app.get("/api/v1/personalize/recommendations/by-location")
async def get_personalize_recommendations_by_location(
    province: Optional[str] = Query(None, description="Filter by province"),
    city: Optional[str] = Query(None, description="Filter by city"),
    category: Optional[str] = Query(None, description="Filter by category (comma-separated for multiple)"),
    order_source: Optional[str] = Query(None, description="Filter by order source: 'oe', 'pos'"),
    num_results: int = Query(10, description="Number of recommendations per user"),
    limit_users: int = Query(50, description="Number of users to get recommendations for")
):
    """
    Get ML recommendations for users in a specific province/city.
    Uses local ML models (replaces AWS Personalize).
    Returns aggregated recommendations for users in that location.
    """
    try:
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build query to get users from location who have recommendations
            query = """
                SELECT DISTINCT o.unified_customer_id as customer_id, 
                       o.customer_name, 
                       o.customer_city as city, 
                       o.province
                FROM orders o
                INNER JOIN offline_user_recommendations r ON o.unified_customer_id = r.user_id
                WHERE o.unified_customer_id IS NOT NULL
            """
            params = []
            
            if province:
                query += " AND LOWER(o.province) = LOWER(%s)"
                params.append(province)
            
            if city:
                query += " AND LOWER(o.customer_city) = LOWER(%s)"
                params.append(city)
                
            if order_source and order_source.lower() in ['oe', 'pos']:
                query += " AND UPPER(o.order_type) = %s"
                params.append(order_source.upper())
            
            query += f" ORDER BY o.unified_customer_id LIMIT {limit_users}"
            
            cursor.execute(query, params)
            users = cursor.fetchall()
            
            if not users:
                return {
                    "province": province,
                    "city": city,
                    "users": [],
                    "aggregated_recommendations": [],
                    "message": "No users found in this location"
                }
            
            # Get recommendations for each user from local cache
            user_recommendations = []
            all_product_scores = defaultdict(lambda: {"score": 0, "count": 0})
            
            user_ids = [u['customer_id'] for u in users]
            placeholders = ','.join(['%s'] * len(user_ids))
            
            cursor.execute(f"""
                SELECT user_id, recommendations
                FROM offline_user_recommendations
                WHERE user_id IN ({placeholders})
            """, user_ids)
            
            user_recs_map = {r['user_id']: r['recommendations'] for r in cursor.fetchall()}
            
            for user in users:
                user_id = user['customer_id']
                recs_data = user_recs_map.get(user_id, [])
                
                # Parse recommendations (stored as JSONB)
                if isinstance(recs_data, str):
                    import json
                    recs_data = json.loads(recs_data)
                
                recs = recs_data[:num_results] if recs_data else []
                
                user_recommendations.append({
                    "customer_id": user_id,
                    "customer_name": user['customer_name'],
                    "city": user['city'],
                    "province": user['province'],
                    "recommendations": recs
                })
                
                # Aggregate scores
                for rec in recs:
                    pid = rec.get('item_id') or rec.get('product_id')
                    score = rec.get('score', 0)
                    if pid:
                        all_product_scores[pid]['score'] += score
                        all_product_scores[pid]['count'] += 1
            
            # Calculate average scores and sort
            aggregated = []
            for product_id, data in all_product_scores.items():
                aggregated.append({
                    "product_id": product_id,
                    "avg_score": data['score'] / data['count'] if data['count'] > 0 else 0,
                    "recommended_to_users": data['count']
                })
            
            aggregated.sort(key=lambda x: (-x['recommended_to_users'], -x['avg_score']))
            
            # Collect all product IDs from all recommendations
            all_product_ids = set()
            for user_rec in user_recommendations:
                for rec in user_rec['recommendations']:
                    pid = rec.get('item_id') or rec.get('product_id')
                    if pid:
                        all_product_ids.add(pid)
            for agg in aggregated[:20]:
                all_product_ids.add(agg['product_id'])
            
            # Enrich with product names
            product_names = {}
            if all_product_ids:
                product_ids_list = list(all_product_ids)
                placeholders = ','.join(['%s'] * len(product_ids_list))
                cursor.execute(f"""
                    SELECT DISTINCT product_id, product_name 
                    FROM order_items 
                    WHERE product_id IN ({placeholders})
                """, product_ids_list)
                product_names = {str(r['product_id']): r['product_name'] for r in cursor.fetchall()}
            
            # Apply product names to aggregated recommendations
            for agg in aggregated:
                agg['product_name'] = product_names.get(str(agg['product_id']), f"Product {agg['product_id']}")
            
            # Apply category filter if specified
            if category:
                categories = [c.strip().lower() for c in category.split(',')]
                aggregated = [
                    agg for agg in aggregated 
                    if extract_smart_category(agg['product_name']).lower() in categories
                ]
            
            # Apply product names to per-user recommendations
            for user_rec in user_recommendations:
                for rec in user_rec['recommendations']:
                    pid = rec.get('item_id') or rec.get('product_id')
                    rec['product_id'] = pid
                    rec['product_name'] = product_names.get(str(pid), f"Product {pid}")
            
            cursor.close()
            
            return {
                "province": province,
                "city": city,
                "category": category,
                "total_users": len(users),
                "users": user_recommendations,
                "aggregated_recommendations": aggregated[:20],
                "source": "local_ml"
            }
            
        finally:
            pg_pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Failed to get location-based recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/personalize/recommendations/by-segment")
async def get_segment_recommendations(
    segment: str = Query(..., description="RFM segment: champions, loyal, potential, new, at_risk, hibernating, lost"),
    province: Optional[str] = Query(None, description="Filter by province"),
    city: Optional[str] = Query(None, description="Filter by city"),
    category: Optional[str] = Query(None, description="Filter by category (comma-separated for multiple)"),
    limit: int = Query(10, description="Number of recommendations")
):
    """
    Get aggregated recommendations for customers in a specific RFM segment within a location.
    Uses cached recommendations from offline_user_recommendations table.
    """
    try:
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Define RFM segment criteria
            segment_criteria = {
                'champions': "recency_days <= 30 AND frequency >= 5 AND monetary >= 50000",
                'loyal': "recency_days <= 60 AND frequency >= 3 AND monetary >= 30000",
                'potential': "recency_days <= 30 AND frequency >= 2",
                'new': "frequency = 1 AND recency_days <= 30",
                'at_risk': "recency_days > 90 AND frequency >= 3 AND monetary >= 20000",
                'hibernating': "recency_days > 120 AND frequency <= 2",
                'lost': "recency_days > 180"
            }
            
            criteria = segment_criteria.get(segment.lower(), "1=1")
            
            # Build location filter
            location_filter = ""
            params = []
            if province:
                # Normalize Islamabad variants
                if province.lower() in ['islamabad', 'islamabad capital territory', 'islamabad capital', 'ict']:
                    location_filter += " AND o.province IN ('Islamabad', 'Islamabad Capital Territory', 'Islamabad Capital', 'ICT')"
                else:
                    location_filter += " AND LOWER(o.province) = LOWER(%s)"
                    params.append(province)
            if city:
                location_filter += " AND LOWER(o.customer_city) = LOWER(%s)"
                params.append(city)
            
            # Get users in segment with their cached recommendations
            cursor.execute(f"""
                WITH customer_rfm AS (
                    SELECT 
                        o.unified_customer_id,
                        MAX(o.customer_name) as customer_name,
                        MAX(o.customer_city) as city,
                        MAX(o.province) as province,
                        EXTRACT(days FROM NOW() - MAX(o.order_date)) as recency_days,
                        COUNT(DISTINCT o.id) as frequency,
                        SUM(o.total_price) as monetary
                    FROM orders o
                    WHERE o.unified_customer_id IS NOT NULL
                    {location_filter}
                    GROUP BY o.unified_customer_id
                ),
                segment_users AS (
                    SELECT unified_customer_id, customer_name, city, province
                    FROM customer_rfm
                    WHERE {criteria}
                    LIMIT 100
                )
                SELECT 
                    su.unified_customer_id,
                    su.customer_name,
                    su.city,
                    su.province,
                    our.recommendations
                FROM segment_users su
                LEFT JOIN offline_user_recommendations our 
                    ON su.unified_customer_id = our.user_id
                WHERE our.recommendations IS NOT NULL
            """, params)
            
            users_with_recs = cursor.fetchall()
            
            # Aggregate recommendations across all users in segment
            product_scores = {}
            for user in users_with_recs:
                recs = user['recommendations']
                if isinstance(recs, str):
                    import json
                    recs = json.loads(recs)
                
                for rec in recs[:10]:  # Top 10 per user
                    # Handle different field names
                    pid = rec.get('item_id') or rec.get('product_id') or rec.get('itemId')
                    score = float(rec.get('score', 0.5))
                    
                    if not pid:
                        continue
                        
                    if pid not in product_scores:
                        product_scores[pid] = {'total_score': 0, 'count': 0}
                    product_scores[pid]['total_score'] += score
                    product_scores[pid]['count'] += 1
            
            # Calculate average scores and sort
            aggregated = []
            for pid, data in product_scores.items():
                aggregated.append({
                    'product_id': pid,
                    'avg_score': data['total_score'] / data['count'],
                    'recommended_to_users': data['count']
                })
            
            aggregated.sort(key=lambda x: (-x['recommended_to_users'], -x['avg_score']))
            
            # Enrich with product names
            if aggregated:
                product_ids = [a['product_id'] for a in aggregated[:limit]]
                placeholders = ','.join(['%s'] * len(product_ids))
                cursor.execute(f"""
                    SELECT DISTINCT product_id, product_name 
                    FROM order_items 
                    WHERE product_id IN ({placeholders})
                """, product_ids)
                product_names = {str(r['product_id']): r['product_name'] for r in cursor.fetchall()}
                
                for rec in aggregated:
                    rec['product_name'] = product_names.get(str(rec['product_id']), f"Product {rec['product_id']}")
            
            # Apply category filter if specified
            if category:
                categories = [c.strip().lower() for c in category.split(',')]
                aggregated = [
                    agg for agg in aggregated 
                    if extract_smart_category(agg['product_name']).lower() in categories
                ]
            
            cursor.close()
            
            return {
                "segment": segment,
                "province": province,
                "city": city,
                "category": category,
                "users_in_segment": len(users_with_recs),
                "aggregated_recommendations": aggregated[:limit],
                "source": "offline_cache_rfm"
            }
            
        finally:
            pg_pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Failed to get segment recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/personalize/recommendations/{user_id}")
async def get_personalize_recommendations(
    user_id: str = Path(..., description="User/Customer ID"),
    num_results: int = Query(10, description="Number of recommendations")
):
    """
    Get personalized recommendations for a specific user.
    Uses local ML models (replaces AWS Personalize).
    """
    try:
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get recommendations from local cache
            cursor.execute("""
                SELECT recommendations
                FROM offline_user_recommendations
                WHERE user_id = %s
            """, (user_id,))
            
            result = cursor.fetchone()
            recommendations = []
            
            if result and result['recommendations']:
                recs_data = result['recommendations']
                if isinstance(recs_data, str):
                    import json
                    recs_data = json.loads(recs_data)
                recommendations = recs_data[:num_results]
            
            # Enrich with product names
            if recommendations:
                product_ids = [r.get('item_id') or r.get('product_id') for r in recommendations]
                product_ids = [pid for pid in product_ids if pid]
                
                if product_ids:
                    placeholders = ','.join(['%s'] * len(product_ids))
                    cursor.execute(f"""
                        SELECT DISTINCT product_id, product_name 
                        FROM order_items 
                        WHERE product_id IN ({placeholders})
                    """, product_ids)
                    product_names = {str(r['product_id']): r['product_name'] for r in cursor.fetchall()}
                    
                    for rec in recommendations:
                        pid = rec.get('item_id') or rec.get('product_id')
                        rec['product_id'] = pid
                        rec['product_name'] = product_names.get(str(pid), f"Product {pid}")
            
            cursor.close()
            
            return {
                "user_id": user_id,
                "recommendations": recommendations,
                "count": len(recommendations),
                "source": "local_ml"
            }
        finally:
            pg_pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/personalize/recommendations/similar/{product_id}")
async def get_similar_products(
    product_id: str = Path(..., description="Product ID"),
    num_results: int = Query(10, description="Number of similar products")
):
    """
    Get similar products for cross-selling.
    Uses local ML models (replaces AWS Personalize).
    """
    try:
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get similar items from local cache
            cursor.execute("""
                SELECT similar_products
                FROM offline_similar_items
                WHERE product_id = %s
            """, (product_id,))
            
            result = cursor.fetchone()
            similar_items = []
            
            if result and result['similar_products']:
                items_data = result['similar_products']
                if isinstance(items_data, str):
                    import json
                    items_data = json.loads(items_data)
                similar_items = items_data[:num_results]
            
            # Enrich with product names
            if similar_items:
                product_ids = [r.get('item_id') or r.get('product_id') for r in similar_items]
                product_ids = [pid for pid in product_ids if pid]
                
                if product_ids:
                    placeholders = ','.join(['%s'] * len(product_ids))
                    cursor.execute(f"""
                        SELECT DISTINCT product_id, product_name 
                        FROM order_items 
                        WHERE product_id IN ({placeholders})
                    """, product_ids)
                    product_names = {str(r['product_id']): r['product_name'] for r in cursor.fetchall()}
                    
                    for item in similar_items:
                        pid = item.get('item_id') or item.get('product_id')
                        item['product_id'] = pid
                        item['product_name'] = product_names.get(str(pid), f"Product {pid}")
            
            cursor.close()
            
            return {
                "product_id": product_id,
                "recommendations": similar_items,
                "count": len(similar_items),
                "source": "local_ml"
            }
        finally:
            pg_pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Failed to get similar products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/personalize/recommendations/item-affinity/{user_id}")
async def get_item_affinity_recommendations(
    user_id: str = Path(..., description="User/Customer ID"),
    num_results: int = Query(10, description="Number of products to return")
):
    """
    Get Item Affinity recommendations - products that drive conversions for this user.
    
    Item Affinity identifies products that are likely to lead to purchases based on
    the user's browsing/interaction patterns. Unlike user-personalization (what they'll buy),
    item-affinity shows what products INFLUENCE their buying decision.
    
    Use cases:
    - Homepage hero banners
    - Email marketing campaigns  
    - Retargeting ads
    """
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if we have item affinity data
        cursor.execute(
            """SELECT item_affinities 
               FROM offline_item_affinity 
               WHERE user_id = %s
               LIMIT 1""",
            (str(user_id),)
        )
        
        result = cursor.fetchone()
        
        if not result:
            # No item affinity data - return empty with status
            return {
                "user_id": user_id,
                "recommendations": [],
                "count": 0,
                "source": "item_affinity",
                "status": "no_data",
                "message": "Item affinity batch job not yet run. Run aws_personalize/run_batch_inference.py with --recipe item-affinity"
            }
        
        # Parse affinities
        affinities = result['item_affinities']
        if isinstance(affinities, str):
            import json
            affinities = json.loads(affinities)
        
        # Limit results
        recommendations = []
        for item in affinities[:num_results]:
            recommendations.append({
                'product_id': str(item.get('product_id', item.get('item_id', ''))),
                'affinity_score': float(item.get('score', item.get('affinity_score', 0))),
                'algorithm': 'aws_item_affinity'
            })
        
        # Enrich with product names
        if recommendations and pg_pool:
            product_ids = [r['product_id'] for r in recommendations]
            enrich_conn = pg_pool.getconn()
            try:
                enrich_cursor = enrich_conn.cursor(cursor_factory=RealDictCursor)
                placeholders = ','.join(['%s'] * len(product_ids))
                enrich_cursor.execute(f"""
                    SELECT DISTINCT product_id, product_name 
                    FROM order_items 
                    WHERE product_id IN ({placeholders})
                """, product_ids)
                product_names = {str(r['product_id']): r['product_name'] for r in enrich_cursor.fetchall()}
                enrich_cursor.close()
                
                for rec in recommendations:
                    rec['product_name'] = product_names.get(rec['product_id'], f"Product {rec['product_id']}")
            finally:
                pg_pool.putconn(enrich_conn)
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "source": "aws_item_affinity",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get item affinity recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/v1/personalize/status")
async def get_personalize_status():
    """
    Get AWS Personalize configuration status.
    """
    try:
        from aws_personalize.personalize_service import get_personalize_service
        
        personalize = get_personalize_service()
        
        # Check if we have batch inference data in the database
        conn = None
        has_user_recs = False
        has_similar_items = False
        user_count = 0
        item_count = 0
        
        try:
            conn = psycopg2.connect(**get_pg_connection_params())
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM offline_user_recommendations")
            user_count = cursor.fetchone()[0]
            has_user_recs = user_count > 0
            
            cursor.execute("SELECT COUNT(*) FROM offline_similar_items")
            item_count = cursor.fetchone()[0]
            has_similar_items = item_count > 0
            cursor.close()
        except:
            pass
        finally:
            if conn:
                conn.close()
        
        # System is configured if we have batch data
        is_configured = has_user_recs or has_similar_items
        
        return {
            "is_configured": is_configured,
            "mode": "batch_inference",
            "region": personalize.region,
            "user_recommendations_count": user_count,
            "similar_items_count": item_count,
            "recipes_active": ["user-personalization", "similar-items"] if is_configured else []
        }
    except Exception as e:
        logger.error(f"Failed to get Personalize status: {e}")
        return {
            "is_configured": False,
            "error": str(e)
        }


@app.get("/api/v1/locations/provinces")
async def get_provinces(
    order_source: str = Query("all", description="Filter by order source: all, oe, pos"),
    category: str = Query(None, description="Filter by product category")
):
    """Get list of all provinces with order counts."""
    try:
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Filters
            order_source_clause, order_source_params = get_order_source_filter(order_source, "o")
            params = list(order_source_params)
            
            needs_items_join = category and category.strip()
            category_filter = ""
            if needs_items_join:
                category_filter = get_category_filter_sql(category)
            
            # Build Query
            table_clause = "orders o"
            if needs_items_join:
                table_clause += " JOIN order_items oi ON o.id = oi.order_id"
                
            query = f"""
                SELECT 
                    CASE 
                        WHEN UPPER(o.province) IN ('ISLAMABAD', 'ISLAMABAD CAPITAL TERRITORY', 'ISLAMABAD CAPITAL', 'ICT') THEN 'Islamabad'
                        WHEN UPPER(REPLACE(o.province, '.', '')) IN ('KPK', 'NWFP', 'KHYBER PAKHTUNKHWA') THEN 'Khyber Pakhtunkhwa'
                        WHEN UPPER(o.province) = 'PUNJAB' THEN 'Punjab'
                        WHEN UPPER(o.province) = 'SINDH' THEN 'Sindh'
                        WHEN UPPER(o.province) IN ('BALOCHISTAN', 'BALUCHISTAN') THEN 'Balochistan'
                        WHEN UPPER(o.province) IN ('GILGIT-BALTISTAN', 'GB') THEN 'Gilgit-Baltistan'
                        WHEN UPPER(o.province) IN ('AZAD KASHMIR', 'AJK', 'AZAD JAMMU AND KASHMIR') THEN 'Azad Kashmir'
                        ELSE INITCAP(COALESCE(o.province, 'Unknown'))
                    END as province,
                    COUNT(DISTINCT o.id) as order_count,
                    COUNT(DISTINCT o.unified_customer_id) as customer_count
                FROM {table_clause}
                WHERE 1=1
                    {order_source_clause}
                    {category_filter}
                GROUP BY 1
                ORDER BY order_count DESC
            """
            
            cursor.execute(query, tuple(params))
            provinces = cursor.fetchall()
            cursor.close()
            return {"provinces": provinces}
        finally:
            pg_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Failed to get provinces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/locations/cities")
async def get_cities(province: Optional[str] = Query(None, description="Filter by province")):
    """Get list of cities, optionally filtered by province. City names are normalized (case-insensitive merge)."""
    try:
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Use INITCAP(TRIM()) to normalize city names and merge duplicates
            query = """
                SELECT 
                    INITCAP(TRIM(customer_city)) as city,
                    MAX(province) as province,
                    SUM(order_count) as order_count,
                    SUM(customer_count) as customer_count
                FROM (
                    SELECT 
                        customer_city,
                        province,
                        COUNT(DISTINCT id) as order_count,
                        COUNT(DISTINCT unified_customer_id) as customer_count
                    FROM orders
                    WHERE customer_city IS NOT NULL AND TRIM(customer_city) != ''
            """
            params = []
            
            if province:
                query += " AND LOWER(province) = LOWER(%s)"
                params.append(province)
            
            query += """
                    GROUP BY customer_city, province
                ) subq
                GROUP BY INITCAP(TRIM(customer_city))
                ORDER BY customer_count DESC 
                LIMIT 100
            """
            
            cursor.execute(query, params)
            cities = cursor.fetchall()
            cursor.close()
            return {"cities": cities}
        finally:
            pg_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Failed to get cities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/locations/users")
async def get_users_by_location(
    province: Optional[str] = Query(None, description="Filter by province"),
    city: Optional[str] = Query(None, description="Filter by city"),
    limit: int = Query(50, description="Max users to return")
):
    """Get list of users in a specific location."""
    try:
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT 
                    unified_customer_id as customer_id,
                    customer_name,
                    customer_city as city,
                    province,
                    COUNT(DISTINCT id) as order_count,
                    SUM(total_price) as total_spent
                FROM orders
                WHERE unified_customer_id IS NOT NULL
            """
            params = []
            
            if province:
                query += " AND LOWER(province) = LOWER(%s)"
                params.append(province)
            
            if city:
                query += " AND LOWER(customer_city) = LOWER(%s)"
                params.append(city)
            
            query += f" GROUP BY unified_customer_id, customer_name, customer_city, province ORDER BY order_count DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            users = cursor.fetchall()
            cursor.close()
            return {"users": users, "count": len(users)}
        finally:
            pg_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SHOPIFY INTEGRATION ENDPOINTS
# ============================================================================

@app.post("/api/v1/shopify/recommendations")
async def get_shopify_recommendations(
    request: Request
):
    """
    Unified Shopify recommendation endpoint for checkout/cart pages.
    
    Request body:
    {
        "customer_email": "user@example.com",     // Optional: for personalized recs
        "customer_phone": "03001234567",          // Optional: for personalized recs
        "cart_items": ["product_id_1", "product_id_2"],  // Optional: for similar items
        "current_product": "product_id",          // Optional: for similar items
        "city": "Lahore",                         // Optional: for location-based
        "province": "Punjab",                     // Optional: for location-based
        "rfm_segment": "champions",               // Optional: for segment-based
        "limit": 10                               // Optional: max recommendations
    }
    
    Returns recommendations based on available context, with fallbacks:
    1. Personalized (if customer identified)
    2. Similar items (if cart/product provided)
    3. Location-based (if city/province provided)
    4. Segment-based (if RFM segment provided)
    5. Popular items (fallback)
    """
    try:
        body = await request.json()
        
        customer_email = body.get("customer_email")
        customer_phone = body.get("customer_phone")
        cart_items = body.get("cart_items", [])
        current_product = body.get("current_product")
        city = body.get("city")
        province = body.get("province")
        rfm_segment = body.get("rfm_segment")
        limit = body.get("limit", 10)
        
        recommendations = []
        recommendation_type = "popular"  # Default fallback
        
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 1. Try personalized recommendations (if customer identified)
            user_id = None
            if customer_phone or customer_email:
                # Find user by phone or email
                identifier = customer_phone or customer_email
                cursor.execute("""
                    SELECT DISTINCT unified_customer_id 
                    FROM orders 
                    WHERE unified_customer_id ILIKE %s 
                    OR customer_phone = %s 
                    OR customer_email = %s
                    LIMIT 1
                """, (f"%{identifier}%", customer_phone, customer_email))
                result = cursor.fetchone()
                
                if result:
                    user_id = result['unified_customer_id']
                    
                    # Get personalized recommendations from cache
                    cursor.execute("""
                        SELECT recommendations 
                        FROM offline_user_recommendations 
                        WHERE user_id = %s
                    """, (user_id,))
                    rec_result = cursor.fetchone()
                    
                    if rec_result and rec_result['recommendations']:
                        recs = rec_result['recommendations']
                        if isinstance(recs, str):
                            import json
                            recs = json.loads(recs)
                        recommendations = recs[:limit]
                        recommendation_type = "personalized"
            
            # 2. Try similar items (if cart or current product provided)
            if not recommendations and (cart_items or current_product):
                product_id = current_product or (cart_items[0] if cart_items else None)
                
                if product_id:
                    cursor.execute("""
                        SELECT similar_products 
                        FROM offline_similar_items 
                        WHERE product_id = %s
                    """, (str(product_id),))
                    sim_result = cursor.fetchone()
                    
                    if sim_result and sim_result['similar_products']:
                        sims = sim_result['similar_products']
                        if isinstance(sims, str):
                            import json
                            sims = json.loads(sims)
                        recommendations = sims[:limit]
                        recommendation_type = "similar_items"
            
            # 3. Try location-based recommendations
            if not recommendations and (city or province):
                cursor.execute("""
                    SELECT oi.product_id, oi.product_name, COUNT(*) as purchase_count
                    FROM orders o
                    JOIN order_items oi ON o.id::text = oi.order_id
                    WHERE (LOWER(o.customer_city) = LOWER(%s) OR LOWER(o.province) = LOWER(%s))
                    AND oi.product_id IS NOT NULL
                    GROUP BY oi.product_id, oi.product_name
                    ORDER BY purchase_count DESC
                    LIMIT %s
                """, (city or '', province or '', limit))
                
                loc_results = cursor.fetchall()
                if loc_results:
                    recommendations = [
                        {"item_id": r['product_id'], "item_name": r['product_name'], "score": r['purchase_count']}
                        for r in loc_results
                    ]
                    recommendation_type = "location_based"
            
            # 4. Try segment-based recommendations
            if not recommendations and rfm_segment:
                # Get popular items for this segment
                cursor.execute("""
                    SELECT oi.product_id, oi.product_name, COUNT(*) as purchase_count
                    FROM orders o
                    JOIN order_items oi ON o.id::text = oi.order_id
                    WHERE o.unified_customer_id IN (
                        SELECT user_id FROM rfm_segments WHERE segment = %s
                    )
                    AND oi.product_id IS NOT NULL
                    GROUP BY oi.product_id, oi.product_name
                    ORDER BY purchase_count DESC
                    LIMIT %s
                """, (rfm_segment, limit))
                
                seg_results = cursor.fetchall()
                if seg_results:
                    recommendations = [
                        {"item_id": r['product_id'], "item_name": r['product_name'], "score": r['purchase_count']}
                        for r in seg_results
                    ]
                    recommendation_type = "segment_based"
            
            # 5. Fallback to popular items
            if not recommendations:
                cursor.execute("""
                    SELECT oi.product_id, oi.product_name, COUNT(*) as purchase_count
                    FROM order_items oi
                    JOIN orders o ON o.id::text = oi.order_id
                    WHERE o.order_date >= NOW() - INTERVAL '90 days'
                    AND oi.product_id IS NOT NULL
                    GROUP BY oi.product_id, oi.product_name
                    ORDER BY purchase_count DESC
                    LIMIT %s
                """, (limit,))
                
                pop_results = cursor.fetchall()
                recommendations = [
                    {"item_id": r['product_id'], "item_name": r['product_name'], "score": r['purchase_count']}
                    for r in pop_results
                ]
                recommendation_type = "popular"
            
            cursor.close()
            
        finally:
            pg_pool.putconn(conn)
        
        return {
            "success": True,
            "recommendation_type": recommendation_type,
            "user_identified": user_id is not None,
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "context": {
                "customer_phone": customer_phone,
                "customer_email": customer_email,
                "cart_items": cart_items,
                "current_product": current_product,
                "city": city,
                "province": province,
                "rfm_segment": rfm_segment
            }
        }
        
    except Exception as e:
        logger.error(f"Shopify recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/shopify/similar/{product_id}")
async def get_shopify_similar_products(
    product_id: str = Path(..., description="Shopify Product ID"),
    limit: int = Query(10, description="Number of similar products")
):
    """
    Get similar products for Shopify product pages.
    Use this for "You might also like" sections.
    """
    try:
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get similar products from cache
            cursor.execute("""
                SELECT similar_products 
                FROM offline_similar_items 
                WHERE product_id = %s
            """, (str(product_id),))
            result = cursor.fetchone()
            
            if result and result['similar_products']:
                sims = result['similar_products']
                if isinstance(sims, str):
                    import json
                    sims = json.loads(sims)
                
                cursor.close()
                return {
                    "success": True,
                    "product_id": product_id,
                    "similar_products": sims[:limit],
                    "count": len(sims[:limit])
                }
            
            # Fallback: get popular products
            cursor.execute("""
                SELECT oi.product_id, oi.product_name, COUNT(*) as score
                FROM order_items oi
                JOIN orders o ON o.id::text = oi.order_id
                WHERE o.order_date >= NOW() - INTERVAL '90 days'
                AND oi.product_id != %s
                AND oi.product_id IS NOT NULL
                GROUP BY oi.product_id, oi.product_name
                ORDER BY score DESC
                LIMIT %s
            """, (str(product_id), limit))
            
            results = cursor.fetchall()
            cursor.close()
            
            return {
                "success": True,
                "product_id": product_id,
                "similar_products": [
                    {"item_id": r['product_id'], "item_name": r['product_name'], "score": r['score']}
                    for r in results
                ],
                "count": len(results),
                "fallback": True
            }
            
        finally:
            pg_pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Shopify similar products error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/shopify/sync")
async def sync_shopify_orders(
    days: int = Query(30, description="Days of orders to sync"),
    background_tasks: BackgroundTasks = None
):
    """
    Sync orders from Shopify store to local database.
    This updates the training data for recommendations.
    """
    try:
        from src.services.shopify_service import get_shopify_service
        
        shopify = get_shopify_service()
        
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            result = shopify.sync_orders_to_db(conn, days=days)
            return {
                "success": True,
                "message": f"Synced {result['orders_synced']} orders, {result['items_synced']} items from Shopify",
                **result
            }
        finally:
            pg_pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Shopify sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/shopify/products")
async def get_shopify_products(
    limit: int = Query(50, description="Number of products to return")
):
    """
    Get products from Shopify store.
    Useful for mapping product IDs between systems.
    """
    try:
        from src.services.shopify_service import get_shopify_service
        
        shopify = get_shopify_service()
        products = shopify.get_products(limit=limit)
        
        return {
            "success": True,
            "products": [
                {
                    "id": str(p["id"]),
                    "title": p.get("title", ""),
                    "handle": p.get("handle", ""),
                    "vendor": p.get("vendor", ""),
                    "price": float(p.get("variants", [{}])[0].get("price", 0)) if p.get("variants") else 0
                }
                for p in products
            ],
            "count": len(products)
        }
        
    except Exception as e:
        logger.error(f"Shopify products error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/shopify/webhook/order-created")
async def shopify_order_webhook(request: Request):
    """
    Webhook handler for Shopify order creation.
    Automatically updates training data when new orders come in.
    
    Configure in Shopify Admin > Settings > Notifications > Webhooks
    Topic: orders/create
    """
    try:
        body = await request.json()
        
        # Extract order data
        order_id = f"shopify_{body.get('id')}"
        customer = body.get("customer", {})
        
        customer_email = customer.get("email", "")
        customer_phone = customer.get("phone", "")
        customer_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
        
        # Generate unified customer ID
        if customer_phone:
            unified_id = f"{customer_phone}_{customer_name.split()[0].lower() if customer_name else 'customer'}"
        elif customer_email:
            unified_id = customer_email.split("@")[0]
        else:
            unified_id = f"shopify_{customer.get('id', 'unknown')}"
        
        shipping = body.get("shipping_address", {})
        city = shipping.get("city", "")
        province = shipping.get("province", "")
        
        order_date = body.get("created_at", "")[:19]
        total = float(body.get("total_price", 0))
        
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor()
            
            # Insert order
            cursor.execute("""
                INSERT INTO orders (id, unified_customer_id, customer_name, customer_email, 
                                   customer_phone, customer_city, province, order_date, 
                                   total_price, source_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    unified_customer_id = EXCLUDED.unified_customer_id,
                    total_price = EXCLUDED.total_price
            """, (order_id, unified_id, customer_name, customer_email, 
                  customer_phone, city, province, order_date, total, 'SHOPIFY'))
            
            items_inserted = 0
            for item in body.get("line_items", []):
                product_id = str(item.get("product_id", ""))
                product_name = item.get("title", "")
                quantity = item.get("quantity", 1)
                price = float(item.get("price", 0))
                
                if product_id:
                    cursor.execute("""
                        INSERT INTO order_items (order_id, product_id, product_name, 
                                                quantity, unit_price)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (order_id, product_id) DO UPDATE SET
                            quantity = EXCLUDED.quantity
                    """, (order_id, product_id, product_name, quantity, price))
                    items_inserted += 1
            
            conn.commit()
            cursor.close()
            
            logger.info(f"Shopify webhook: Order {order_id} saved ({items_inserted} items)")
            
            return {
                "success": True,
                "order_id": order_id,
                "customer_id": unified_id,
                "items_count": items_inserted
            }
            
        finally:
            pg_pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Shopify webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/shopify/popular")
async def get_shopify_popular_products(
    city: Optional[str] = Query(None, description="Filter by city"),
    province: Optional[str] = Query(None, description="Filter by province"),
    days: int = Query(30, description="Look back days"),
    limit: int = Query(10, description="Number of products")
):
    """
    Get popular products, optionally filtered by location.
    Use for anonymous users or homepage recommendations.
    """
    try:
        if not pg_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        conn = pg_pool.getconn()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT oi.product_id, oi.product_name, COUNT(*) as purchase_count
                FROM order_items oi
                JOIN orders o ON o.id::text = oi.order_id
                WHERE o.order_date >= NOW() - INTERVAL '%s days'
                AND oi.product_id IS NOT NULL
            """
            params = [days]
            
            if city:
                query += " AND LOWER(o.customer_city) = LOWER(%s)"
                params.append(city)
            
            if province:
                query += " AND LOWER(o.province) = LOWER(%s)"
                params.append(province)
            
            query += """
                GROUP BY oi.product_id, oi.product_name
                ORDER BY purchase_count DESC
                LIMIT %s
            """
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            return {
                "success": True,
                "popular_products": [
                    {"item_id": r['product_id'], "item_name": r['product_name'], "score": r['purchase_count']}
                    for r in results
                ],
                "count": len(results),
                "filters": {"city": city, "province": province, "days": days}
            }
            
        finally:
            pg_pool.putconn(conn)
            
    except Exception as e:
        logger.error(f"Shopify popular products error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/export/dashboard-csv")
async def export_dashboard_csv(
    time_filter: str = Query("30days", description="Time period filter"),
    category: str = Query(None, description="Product category filter (e.g., Mattresses, Pillows)"),
    sections: str = Query("all", description="Sections to export: all, metrics, products, orders, dashboard, customer_profiling, collaborative_filtering, cross_selling, geographic_intelligence, rfm_segmentation, ml_recommendations"),
    categories: str = Query(None, description="Comma-separated list of categories"),
    order_source: str = Query("all", description="Order source filter: all, oe, pos"),
    delivered_only: bool = Query(False, description="Filter to only delivered/completed orders")
):
    """
    Export dashboard data as CSV with time and category filters
    
    Uses existing filter functions:
    - get_time_filter_clause() for time filtering
    - get_category_filter_sql() for category filtering (LIKE patterns on product names)
    
    New Parameters:
    - categories: Comma-separated list of categories (alternative to single category)
    - order_source: Filter by OE (Online Express) or POS (Point of Sale)
    - delivered_only: Only include delivered/completed orders
    """
    conn = None
    try:
        conn = psycopg2.connect(**get_pg_connection_params())
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if order_source and status columns exist FIRST
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'orders' 
            AND column_name IN ('order_source', 'status')
        """)
        existing_columns = [row['column_name'] for row in cursor.fetchall()]
        has_order_source = 'order_source' in existing_columns
        has_status = 'status' in existing_columns
        
        # Parse sections
        section_list = sections.split(',') if sections != "all" else ["metrics", "products", "orders"]
        
        # Build WHERE clause for time filter
        where_clause, time_params = get_time_filter_clause(time_filter)
        
        # Build category filter SQL (returns empty string or "AND (LIKE patterns)")
        # Support both single category and comma-separated categories
        effective_category = category or categories
        category_filter_sql = get_category_filter_sql(effective_category) if effective_category else ""
        
        # Build order source filter (only if column exists)
        order_source_filter = ""
        if order_source and order_source.lower() != 'all' and has_order_source:
            order_source_filter = f" AND (UPPER(o.order_source) = '{order_source.upper()}' OR o.order_source IS NULL)"
        
        # Build delivered only filter (only if column exists)
        delivered_filter = ""
        if delivered_only and has_status:
            delivered_filter = " AND (UPPER(o.status) IN ('DELIVERED', 'COMPLETED', 'FULFILLED') OR o.status IS NULL)"
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # =========================================
        # SECTION 1: Dashboard Metrics
        # =========================================
        if "metrics" in section_list or "dashboard" in section_list:
            writer.writerow([])
            writer.writerow(["DASHBOARD OVERVIEW"])
            writer.writerow(["=" * 50])
            writer.writerow(["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(["Time Period:", time_filter])
            if effective_category:
                writer.writerow(["Category:", effective_category])
            if order_source and order_source.lower() != 'all':
                writer.writerow(["Order Source:", order_source.upper()])
            if delivered_only:
                writer.writerow(["Delivered Only:", "Yes"])
            writer.writerow([])
            writer.writerow(["Metric", "Value"])
            
            # Query with proper JOINs when category filter is present
            if category_filter_sql:
                query = f"""
                    SELECT 
                        COUNT(DISTINCT o.id) as total_orders,
                        COUNT(DISTINCT o.unified_customer_id) as total_customers,
                        COALESCE(SUM(o.total_price), 0) as total_revenue,
                        COALESCE(AVG(o.total_price), 0) as avg_order_value
                    FROM orders o
                    JOIN order_items oi ON o.id = oi.order_id
                    {where_clause}
                    {category_filter_sql}
                    {order_source_filter}
                    {delivered_filter}
                """
            else:
                query = f"""
                    SELECT 
                        COUNT(DISTINCT o.id) as total_orders,
                        COUNT(DISTINCT o.unified_customer_id) as total_customers,
                        COALESCE(SUM(o.total_price), 0) as total_revenue,
                        COALESCE(AVG(o.total_price), 0) as avg_order_value
                    FROM orders o
                    {where_clause}
                    {order_source_filter}
                    {delivered_filter}
                """
            
            cursor.execute(query, time_params if time_params else None)
            metrics = cursor.fetchone()
            
            writer.writerow(["Total Orders", f"{metrics['total_orders']:,}"])
            writer.writerow(["Total Customers", f"{metrics['total_customers']:,}"])
            writer.writerow(["Total Revenue", f"PKR {metrics['total_revenue']:,.2f}"])
            writer.writerow(["Average Order Value", f"PKR {metrics['avg_order_value']:,.2f}"])
        
        # =========================================
        # SECTION 2: Top Products
        # =========================================
        if "products" in section_list or "dashboard" in section_list:
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["TOP PERFORMING PRODUCTS"])
            writer.writerow(["=" * 50])
            writer.writerow(["Rank", "Product ID", "Product Name", "Orders", "Quantity", "Revenue", "Avg Price"])
            
            query = f"""
                SELECT 
                    oi.product_id,
                    oi.product_name,
                    COUNT(DISTINCT oi.order_id) as order_count,
                    SUM(oi.quantity) as total_quantity,
                    SUM(oi.unit_price * oi.quantity) as total_revenue,
                    AVG(oi.unit_price) as avg_price
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                {where_clause}
                {category_filter_sql}
                {order_source_filter}
                {delivered_filter}
                GROUP BY oi.product_id, oi.product_name
                ORDER BY total_revenue DESC
                LIMIT 100
            """
            
            cursor.execute(query, time_params if time_params else None)
            products = cursor.fetchall()
            
            for idx, prod in enumerate(products, 1):
                writer.writerow([
                    idx,
                    prod['product_id'] or 'N/A',
                    prod['product_name'] or 'N/A',
                    prod['order_count'],
                    prod['total_quantity'],
                    f"PKR {prod['total_revenue']:,.2f}",
                    f"PKR {prod['avg_price']:,.2f}"
                ])
        
        # =========================================
        # SECTION 3: Recent Orders
        # =========================================
        if "orders" in section_list or "dashboard" in section_list:
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["RECENT ORDERS"])
            writer.writerow(["=" * 50])
            
            # Dynamic column list based on what exists
            header_row = ["Order ID", "Date", "Customer ID", "Total", "Items", "City", "Province"]
            if has_order_source:
                header_row.append("Source")
            if has_status:
                header_row.append("Status")
            writer.writerow(header_row)
            
            # Build SELECT clause
            select_columns = """
                o.id,
                o.order_date,
                o.unified_customer_id,
                o.total_price,
                o.customer_city,
                o.province
            """
            if has_order_source:
                select_columns += ",\n                        o.order_source"
            if has_status:
                select_columns += ",\n                        o.status"
            
            if category_filter_sql:
                query = f"""
                    SELECT DISTINCT
                        {select_columns},
                        (SELECT COUNT(*) FROM order_items WHERE order_id = o.id) as item_count
                    FROM orders o
                    JOIN order_items oi ON o.id = oi.order_id
                    {where_clause}
                    {category_filter_sql}
                    {order_source_filter if has_order_source else ''}
                    {delivered_filter if has_status else ''}
                    ORDER BY o.order_date DESC
                    LIMIT 500
                """
            else:
                query = f"""
                    SELECT 
                        {select_columns},
                        (SELECT COUNT(*) FROM order_items WHERE order_id = o.id) as item_count
                    FROM orders o
                    {where_clause}
                    {order_source_filter if has_order_source else ''}
                    {delivered_filter if has_status else ''}
                    ORDER BY o.order_date DESC
                    LIMIT 500
                """
            
            cursor.execute(query, time_params if time_params else None)
            orders = cursor.fetchall()
            
            for order in orders:
                row = [
                    order['id'],
                    order['order_date'].strftime('%Y-%m-%d %H:%M') if order['order_date'] else 'N/A',
                    order['unified_customer_id'] or 'N/A',
                    f"PKR {order['total_price']:,.2f}",
                    order['item_count'],
                    order['customer_city'] or 'N/A',
                    order['province'] or 'N/A'
                ]
                if has_order_source:
                    row.append(order.get('order_source', 'N/A') or 'N/A')
                if has_status:
                    row.append(order.get('status', 'N/A') or 'N/A')
                writer.writerow(row)
        
        # =========================================
        # SECTION 4: Customer Profiling
        # =========================================
        if "customer_profiling" in section_list:
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["CUSTOMER PROFILING"])
            writer.writerow(["=" * 50])
            writer.writerow(["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(["Time Period:", time_filter])
            writer.writerow([])
            writer.writerow(["Customer ID", "Customer Name", "City", "Segment", "Total Orders", 
                           "Unique Products", "Total Spent (PKR)", "Avg Order Value (PKR)", 
                           "First Order Date", "Last Order Date"])
            
            cursor.execute("""
                SELECT 
                    customer_id,
                    customer_name,
                    customer_city,
                    customer_segment,
                    total_orders,
                    unique_products,
                    total_spent,
                    avg_order_value,
                    first_order_date,
                    last_order_date
                FROM customer_statistics
                ORDER BY total_spent DESC
                LIMIT 1000
            """)
            customers = cursor.fetchall()
            
            for cust in customers:
                writer.writerow([
                    cust['customer_id'] or 'N/A',
                    cust['customer_name'] or 'N/A',
                    cust['customer_city'] or 'N/A',
                    cust['customer_segment'] or 'N/A',
                    cust['total_orders'] or 0,
                    cust['unique_products'] or 0,
                    f"{float(cust['total_spent'] or 0):,.2f}",
                    f"{float(cust['avg_order_value'] or 0):,.2f}",
                    cust['first_order_date'].strftime('%Y-%m-%d') if cust['first_order_date'] else 'N/A',
                    cust['last_order_date'].strftime('%Y-%m-%d') if cust['last_order_date'] else 'N/A'
                ])
        
        # =========================================
        # SECTION 5: RFM Segmentation
        # =========================================
        if "rfm_segmentation" in section_list:
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["RFM SEGMENTATION ANALYSIS"])
            writer.writerow(["=" * 50])
            writer.writerow(["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            
            # Segment Summary
            writer.writerow(["SEGMENT SUMMARY"])
            writer.writerow(["Segment", "Customer Count", "Total Revenue (PKR)", "Avg Order Value (PKR)", "Avg Orders/Customer"])
            
            cursor.execute("""
                SELECT 
                    customer_segment as segment,
                    COUNT(*) as customer_count,
                    SUM(total_spent) as total_revenue,
                    AVG(avg_order_value) as avg_order_value,
                    AVG(total_orders) as avg_orders
                FROM customer_statistics
                WHERE customer_segment IS NOT NULL
                GROUP BY customer_segment
                ORDER BY total_revenue DESC
            """)
            segments = cursor.fetchall()
            
            for seg in segments:
                writer.writerow([
                    seg['segment'],
                    seg['customer_count'],
                    f"{float(seg['total_revenue'] or 0):,.2f}",
                    f"{float(seg['avg_order_value'] or 0):,.2f}",
                    f"{float(seg['avg_orders'] or 0):.1f}"
                ])
            
            # Detailed customer list by segment
            writer.writerow([])
            writer.writerow(["CUSTOMERS BY SEGMENT (Top 500)"])
            writer.writerow(["Customer ID", "Customer Name", "City", "Segment", "Total Orders", 
                           "Total Spent (PKR)", "Last Order Date", "Days Since Last Order"])
            
            cursor.execute("""
                SELECT 
                    customer_id,
                    customer_name,
                    customer_city,
                    customer_segment,
                    total_orders,
                    total_spent,
                    last_order_date,
                    EXTRACT(DAY FROM NOW() - last_order_date) as days_since
                FROM customer_statistics
                WHERE customer_segment IS NOT NULL
                ORDER BY customer_segment, total_spent DESC
                LIMIT 500
            """)
            customers = cursor.fetchall()
            
            for cust in customers:
                writer.writerow([
                    cust['customer_id'] or 'N/A',
                    cust['customer_name'] or 'N/A',
                    cust['customer_city'] or 'N/A',
                    cust['customer_segment'] or 'N/A',
                    cust['total_orders'] or 0,
                    f"{float(cust['total_spent'] or 0):,.2f}",
                    cust['last_order_date'].strftime('%Y-%m-%d') if cust['last_order_date'] else 'N/A',
                    int(cust['days_since']) if cust['days_since'] else 'N/A'
                ])
        
        # =========================================
        # SECTION 6: Geographic Intelligence
        # =========================================
        if "geographic_intelligence" in section_list:
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["GEOGRAPHIC INTELLIGENCE"])
            writer.writerow(["=" * 50])
            writer.writerow(["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(["Time Period:", time_filter])
            writer.writerow([])
            
            # Province Summary
            writer.writerow(["PROVINCE PERFORMANCE"])
            writer.writerow(["Province", "Total Orders", "Unique Customers", "Total Revenue (PKR)", "Avg Order Value (PKR)"])
            
            query = f"""
                SELECT 
                    COALESCE(province, 'Unknown') as province,
                    COUNT(*) as total_orders,
                    COUNT(DISTINCT unified_customer_id) as unique_customers,
                    SUM(total_price) as total_revenue,
                    AVG(total_price) as avg_order_value
                FROM orders o
                {where_clause}
                {order_source_filter}
                {delivered_filter}
                GROUP BY province
                ORDER BY total_revenue DESC
            """
            cursor.execute(query, time_params if time_params else None)
            provinces = cursor.fetchall()
            
            for prov in provinces:
                writer.writerow([
                    prov['province'] or 'Unknown',
                    prov['total_orders'],
                    prov['unique_customers'],
                    f"{float(prov['total_revenue'] or 0):,.2f}",
                    f"{float(prov['avg_order_value'] or 0):,.2f}"
                ])
            
            # City Summary
            writer.writerow([])
            writer.writerow(["TOP CITIES"])
            writer.writerow(["City", "Province", "Total Orders", "Unique Customers", "Total Revenue (PKR)"])
            
            query = f"""
                SELECT 
                    COALESCE(customer_city, 'Unknown') as city,
                    COALESCE(province, 'Unknown') as province,
                    COUNT(*) as total_orders,
                    COUNT(DISTINCT unified_customer_id) as unique_customers,
                    SUM(total_price) as total_revenue
                FROM orders o
                {where_clause}
                {order_source_filter}
                {delivered_filter}
                GROUP BY customer_city, province
                ORDER BY total_revenue DESC
                LIMIT 100
            """
            cursor.execute(query, time_params if time_params else None)
            cities = cursor.fetchall()
            
            for city in cities:
                writer.writerow([
                    city['city'] or 'Unknown',
                    city['province'] or 'Unknown',
                    city['total_orders'],
                    city['unique_customers'],
                    f"{float(city['total_revenue'] or 0):,.2f}"
                ])
        
        # =========================================
        # SECTION 7: Collaborative Filtering / Product Insights
        # =========================================
        if "collaborative_filtering" in section_list or "product_insights" in section_list:
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["COLLABORATIVE FILTERING - PRODUCT INSIGHTS"])
            writer.writerow(["=" * 50])
            writer.writerow(["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            
            # Product Statistics
            writer.writerow(["PRODUCT PERFORMANCE"])
            writer.writerow(["Product ID", "Product Name", "Order Count", "Total Quantity", "Total Revenue (PKR)", "Unique Customers"])
            
            query = f"""
                SELECT 
                    oi.product_id,
                    oi.product_name,
                    COUNT(DISTINCT oi.order_id) as order_count,
                    SUM(oi.quantity) as total_quantity,
                    SUM(oi.unit_price * oi.quantity) as total_revenue,
                    COUNT(DISTINCT o.unified_customer_id) as unique_customers
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                {where_clause}
                {category_filter_sql}
                {order_source_filter}
                GROUP BY oi.product_id, oi.product_name
                ORDER BY order_count DESC
                LIMIT 200
            """
            cursor.execute(query, time_params if time_params else None)
            products = cursor.fetchall()
            
            for prod in products:
                writer.writerow([
                    prod['product_id'] or 'N/A',
                    prod['product_name'] or 'N/A',
                    prod['order_count'],
                    prod['total_quantity'] or 0,
                    f"{float(prod['total_revenue'] or 0):,.2f}",
                    prod['unique_customers']
                ])
            
            # Similar Items
            writer.writerow([])
            writer.writerow(["PRODUCT SIMILARITY (Top Similar Items)"])
            writer.writerow(["Product ID", "Similar Product ID", "Similar Product Name", "Similarity Score"])
            
            cursor.execute("""
                SELECT product_id, similar_products
                FROM offline_similar_items
                LIMIT 100
            """)
            similar_items = cursor.fetchall()
            
            for item in similar_items:
                if item['similar_products']:
                    sims = item['similar_products']
                    if isinstance(sims, str):
                        import json
                        sims = json.loads(sims)
                    for sim in sims[:3]:  # Top 3 similar per product
                        writer.writerow([
                            item['product_id'],
                            sim.get('item_id', 'N/A'),
                            sim.get('item_name', 'N/A'),
                            f"{sim.get('score', 0):.4f}"
                        ])
        
        # =========================================
        # SECTION 8: Cross-Selling
        # =========================================
        if "cross_selling" in section_list or "cross-selling" in section_list:
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["CROSS-SELLING OPPORTUNITIES"])
            writer.writerow(["=" * 50])
            writer.writerow(["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            writer.writerow(["FREQUENTLY BOUGHT TOGETHER"])
            writer.writerow(["Product 1 ID", "Product 2 ID", "Co-Purchase Count", "Confidence Score"])
            
            cursor.execute("""
                SELECT 
                    product_1,
                    product_2,
                    co_purchase_count,
                    confidence
                FROM product_pairs
                ORDER BY co_purchase_count DESC
                LIMIT 500
            """)
            pairs = cursor.fetchall()
            
            for pair in pairs:
                writer.writerow([
                    pair['product_1'],
                    pair['product_2'],
                    pair['co_purchase_count'],
                    f"{float(pair['confidence'] or 0):.4f}"
                ])
            
            # Add product names if we can join
            writer.writerow([])
            writer.writerow(["TOP CROSS-SELL PAIRS WITH DETAILS"])
            writer.writerow(["Product 1 Name", "Product 2 Name", "Co-Purchases", "Confidence"])
            
            cursor.execute("""
                SELECT DISTINCT ON (pp.product_1, pp.product_2)
                    oi1.product_name as product_1_name,
                    oi2.product_name as product_2_name,
                    pp.co_purchase_count,
                    pp.confidence
                FROM product_pairs pp
                LEFT JOIN order_items oi1 ON pp.product_1 = oi1.product_id
                LEFT JOIN order_items oi2 ON pp.product_2 = oi2.product_id
                WHERE oi1.product_name IS NOT NULL AND oi2.product_name IS NOT NULL
                ORDER BY pp.product_1, pp.product_2, pp.co_purchase_count DESC
                LIMIT 200
            """)
            detailed_pairs = cursor.fetchall()
            
            for pair in detailed_pairs:
                writer.writerow([
                    pair['product_1_name'] or 'N/A',
                    pair['product_2_name'] or 'N/A',
                    pair['co_purchase_count'],
                    f"{float(pair['confidence'] or 0):.4f}"
                ])
        
        # =========================================
        # SECTION 9: ML Recommendations
        # =========================================
        if "ml_recommendations" in section_list:
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["ML RECOMMENDATIONS"])
            writer.writerow(["=" * 50])
            writer.writerow(["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            
            # User recommendations summary
            writer.writerow(["USER RECOMMENDATIONS SUMMARY"])
            writer.writerow(["Total Users with Recommendations:", ""])
            
            cursor.execute("SELECT COUNT(*) FROM offline_user_recommendations")
            user_count = cursor.fetchone()
            writer.writerow(["", user_count[0] if user_count else 0])
            
            writer.writerow([])
            writer.writerow(["SAMPLE USER RECOMMENDATIONS (Top 100 Users)"])
            writer.writerow(["User ID", "Recommendation 1", "Recommendation 2", "Recommendation 3"])
            
            cursor.execute("""
                SELECT user_id, recommendations
                FROM offline_user_recommendations
                LIMIT 100
            """)
            user_recs = cursor.fetchall()
            
            for urec in user_recs:
                recs = urec['recommendations'] or []
                if isinstance(recs, str):
                    import json
                    recs = json.loads(recs)
                rec_names = [r.get('item_name', r.get('item_id', 'N/A')) for r in recs[:3]]
                while len(rec_names) < 3:
                    rec_names.append('N/A')
                writer.writerow([urec['user_id']] + rec_names)
            
            # Popular products (ML basis)
            writer.writerow([])
            writer.writerow(["POPULAR PRODUCTS (ML Training Basis)"])
            writer.writerow(["Product ID", "Product Name", "Purchase Count", "Unique Buyers"])
            
            cursor.execute("""
                SELECT 
                    oi.product_id,
                    oi.product_name,
                    COUNT(*) as purchase_count,
                    COUNT(DISTINCT o.unified_customer_id) as unique_buyers
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                WHERE o.order_date >= NOW() - INTERVAL '90 days'
                GROUP BY oi.product_id, oi.product_name
                ORDER BY purchase_count DESC
                LIMIT 100
            """)
            popular = cursor.fetchall()
            
            for prod in popular:
                writer.writerow([
                    prod['product_id'] or 'N/A',
                    prod['product_name'] or 'N/A',
                    prod['purchase_count'],
                    prod['unique_buyers']
                ])
        
        # Close cursor
        cursor.close()
        
        # Prepare CSV for download
        output.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Build filename with all filter info
        filename_parts = ["mastergroup"]
        
        # Add section name
        if sections != "all":
            section_name = sections.split(',')[0].replace(' ', '_')[:30]
            filename_parts.append(section_name)
        else:
            filename_parts.append("analytics")
        
        # Add time filter
        filename_parts.append(time_filter)
        
        # Add order source if filtered
        if order_source and order_source.lower() != 'all':
            filename_parts.append(order_source.lower())
        
        # Add delivered flag if filtered
        if delivered_only:
            filename_parts.append("delivered")
        
        # Add category info if filtered
        if effective_category:
            cat_count = len(effective_category.split(','))
            if cat_count > 1:
                filename_parts.append(f"{cat_count}cats")
            else:
                cat_clean = effective_category.replace(' ', '_').replace('&', 'and')[:20]
                filename_parts.append(cat_clean)
        
        filename_parts.append(timestamp)
        filename = "_".join(filename_parts) + ".csv"
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Export CSV error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    finally:
        if conn:
            conn.close()


# ============================================================================
# END ML-POWERED ANALYTICS ENDPOINTS
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