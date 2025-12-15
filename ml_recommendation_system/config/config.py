"""
Configuration settings for ML Recommendation System
"""
import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# CSV file paths (for development)
POS_ORDERS_CSV = 'all_pos_orders.csv'
OE_ORDERS_CSV = 'all_oe_orders.csv'

# API configuration (for production)
API_CONFIG = {
    'pos_endpoint': 'https://mes.master.com.pk/get_pos_orders',
    'oe_endpoint': 'https://mes.master.com.pk/get_oe_orders',
    'Authorization': 'H2rcLQPfzYoV55k9ZyT5aWkyyMKEyxHhX1r3ntrkrvrGeVL4dOsGv3EcQMY2',
    'headers': {
        'Content-Type': 'application/json'
    }
}

# Pipeline settings
TIME_WINDOW_DAYS = None  # Use all available data (set to number of days to filter)
SOURCE_TYPE = 'database'  # 'csv', 'api', or 'database' - NOW USING DATABASE MODE

# Model training settings
TRAIN_TEST_SPLIT_METHOD = 'time_based'  # 'time_based' or 'random'
TRAIN_TEST_SPLIT_RATIO = 0.9  # 90% train, 10% test
TEST_DAYS = 60  # Last 60 days for testing (if time_based)

MIN_CUSTOMER_INTERACTIONS = 1  # Minimum purchases per customer
MIN_PRODUCT_INTERACTIONS = 1   # Minimum purchases per product (reduced from 2 to include more products)
SIMILARITY_METRIC = 'cosine'   # 'cosine', 'pearson', 'euclidean'

# Similarity matrix settings
TOP_N_SIMILAR_ITEMS = 100     # Store top-100 similar products per product (increased from 50)
TOP_N_SIMILAR_USERS = 100     # Store top-100 similar customers per customer

# Hybrid recommendation settings
USE_POPULARITY_BASELINE = True  # Blend collaborative filtering with popular products
POPULARITY_WEIGHT = 0.3         # Weight for popularity (0.3 = 30% popular, 70% collaborative)

# Recommendation settings
DEFAULT_N_RECOMMENDATIONS = 10

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
