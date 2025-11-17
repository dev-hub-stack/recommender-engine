"""
Master Group API Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Master Group API Configuration
MASTER_GROUP_CONFIG = {
    'base_url': os.getenv('MASTER_GROUP_API_BASE', 'https://mes.master.com.pk'),
    'endpoints': {
        'pos_orders': os.getenv('MASTER_GROUP_POS_ENDPOINT', '/get_pos_orders'),
        'oe_orders': os.getenv('MASTER_GROUP_OE_ENDPOINT', '/get_oe_orders')
    },
    'auth_token': os.getenv('MASTER_GROUP_AUTH_TOKEN', ''),
    'timeout': 300,  # 5 minutes
    'retry_attempts': 3,
    'retry_delay': 60,  # 1 minute
    'sync_pos': os.getenv('SYNC_POS_ORDERS', 'true').lower() == 'true',
    'sync_oe': os.getenv('SYNC_OE_ORDERS', 'true').lower() == 'true',
    'headers': {
        'Authorization': os.getenv('MASTER_GROUP_AUTH_TOKEN', ''),
        'Content-Type': 'application/json'
    }
}

# Sync Configuration
SYNC_CONFIG = {
    'interval_minutes': int(os.getenv('SYNC_INTERVAL_MINUTES', '15')),
    'batch_size': int(os.getenv('SYNC_BATCH_SIZE', '1000')),
    'lookback_minutes': int(os.getenv('SYNC_LOOKBACK_MINUTES', '30')),
    'enable_auto_sync': os.getenv('ENABLE_AUTO_SYNC', 'true').lower() == 'true',
    'incremental': True  # Only fetch new orders
}

# Database Configuration
PG_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': int(os.getenv('PG_PORT', '5432')),
    'database': os.getenv('PG_DB', 'mastergroup_recommendations'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', 'postgres')
}

# Redis Configuration
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': int(os.getenv('REDIS_DB', '0')),
    'ttl': int(os.getenv('CACHE_TTL', '3600'))
}

def get_api_url(endpoint_name):
    """Get full API URL for an endpoint"""
    base_url = MASTER_GROUP_CONFIG['base_url']
    endpoint = MASTER_GROUP_CONFIG['endpoints'].get(endpoint_name, '')
    return f"{base_url}{endpoint}"

def get_auth_headers():
    """Get authentication headers for API requests"""
    return MASTER_GROUP_CONFIG['headers']
