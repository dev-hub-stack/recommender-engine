"""
Configuration for Master Group API and database connections.
Reads from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# =============================================================================
# PostgreSQL Configuration
# =============================================================================
PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "database": os.getenv("PG_DB", "mastergroup_recommendations"),
    "dbname": os.getenv("PG_DB", "mastergroup_recommendations"),  # alias for psycopg2
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

# Production PostgreSQL (Lightsail)
PROD_PG_CONFIG = {
    "host": os.getenv("PROD_PG_HOST", ""),
    "port": int(os.getenv("PROD_PG_PORT", 5432)),
    "dbname": os.getenv("PROD_PG_DB", "mastergroup_recommendations"),
    "user": os.getenv("PROD_PG_USER", "postgres"),
    "password": os.getenv("PROD_PG_PASSWORD", ""),
}

# =============================================================================
# Redis Configuration
# =============================================================================
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
}

# =============================================================================
# Master Group API Configuration
# =============================================================================
MASTER_GROUP_CONFIG = {
    "base_url": os.getenv("MASTER_GROUP_API_BASE", "https://mes.master.com.pk"),
    "pos_endpoint": os.getenv("MASTER_GROUP_POS_ENDPOINT", "/get_pos_orders"),
    "oe_endpoint": os.getenv("MASTER_GROUP_OE_ENDPOINT", "/get_oe_orders"),
    "auth_token": os.getenv("MASTER_GROUP_AUTH_TOKEN", ""),
}

# =============================================================================
# Sync Configuration
# =============================================================================
SYNC_CONFIG = {
    "sync_pos_orders": os.getenv("SYNC_POS_ORDERS", "true").lower() == "true",
    "sync_oe_orders": os.getenv("SYNC_OE_ORDERS", "true").lower() == "true",
    "interval_minutes": int(os.getenv("SYNC_INTERVAL_MINUTES", 360)),
    "batch_size": int(os.getenv("SYNC_BATCH_SIZE", 1000)),
    "lookback_minutes": int(os.getenv("SYNC_LOOKBACK_MINUTES", 1440)),
    "enable_auto_sync": os.getenv("ENABLE_AUTO_SYNC", "true").lower() == "true",
}

# =============================================================================
# Shopify Configuration
# =============================================================================
SHOPIFY_CONFIG = {
    "store": os.getenv("SHOPIFY_STORE", "masterverse-project.myshopify.com"),
    "api_key": os.getenv("SHOPIFY_API_KEY", ""),
    "api_secret": os.getenv("SHOPIFY_API_SECRET", ""),
    "access_token": os.getenv("SHOPIFY_ACCESS_TOKEN", ""),
    "api_version": os.getenv("SHOPIFY_API_VERSION", "2024-01"),
}

# =============================================================================
# ML Configuration
# =============================================================================
ML_CONFIG = {
    "use_local_ml": os.getenv("USE_LOCAL_ML", "true").lower() == "true",
    "model_path": os.getenv("ML_MODEL_PATH", "./custom_ml/models"),
    "training_schedule": os.getenv("ML_TRAINING_SCHEDULE", "0 2 * * *"),
    "batch_inference_schedule": os.getenv("ML_BATCH_INFERENCE_SCHEDULE", "0 3 * * *"),
}

# =============================================================================
# Application Settings
# =============================================================================
class Settings:
    debug = os.getenv("DEBUG", "true").lower() == "true"
    environment = os.getenv("ENVIRONMENT", "local")
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", 8001))

settings = Settings()


# =============================================================================
# Helper Functions
# =============================================================================
def get_api_url(endpoint_type: str = "pos") -> str:
    """Get full API URL for Master Group endpoint"""
    base = MASTER_GROUP_CONFIG["base_url"]
    if endpoint_type == "pos":
        return f"{base}{MASTER_GROUP_CONFIG['pos_endpoint']}"
    elif endpoint_type == "oe":
        return f"{base}{MASTER_GROUP_CONFIG['oe_endpoint']}"
    return base


def get_auth_headers() -> dict:
    """Get authentication headers for Master Group API"""
    token = MASTER_GROUP_CONFIG["auth_token"]
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def get_pg_connection_string() -> str:
    """Get PostgreSQL connection string"""
    return (
        f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}"
        f"@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    )
