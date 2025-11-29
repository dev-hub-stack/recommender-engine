-- ============================================
-- Master Group Recommendation System
-- Complete Database Setup Script
-- ============================================
-- 
-- This script sets up all required tables for the recommendation system
-- including authentication, orders, recommendations, and analytics
--
-- Usage:
--   psql -U postgres -d mastergroup_recommendations -f deploy_database.sql
--
-- ============================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. ORDERS TABLE (Source Data)
-- ============================================

CREATE TABLE IF NOT EXISTS orders (
    id VARCHAR(100) PRIMARY KEY,
    customer_id VARCHAR(100),
    unified_customer_id VARCHAR(100),
    customer_name VARCHAR(255),
    customer_phone VARCHAR(50),
    customer_city VARCHAR(100),
    customer_address TEXT,
    customer_email VARCHAR(255),
    order_date TIMESTAMP NOT NULL,
    total_amount DECIMAL(12, 2),
    total_price DECIMAL(12, 2),
    order_type VARCHAR(20) DEFAULT 'POS',
    payment_mode VARCHAR(50),
    brand_name VARCHAR(100),
    order_status VARCHAR(50),
    order_name VARCHAR(255),
    items_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    synced_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_unified_customer_id ON orders(unified_customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_order_type ON orders(order_type);

COMMENT ON TABLE orders IS 'Main orders table - source data from POS and OE systems';

-- ============================================
-- 2. ORDER ITEMS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL,
    product_id VARCHAR(100) NOT NULL,
    product_name VARCHAR(255),
    quantity INTEGER DEFAULT 1,
    unit_price DECIMAL(12, 2),
    total_price DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);

COMMENT ON TABLE order_items IS 'Individual items from each order';

-- ============================================
-- 3. CUSTOMER PURCHASES TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS customer_purchases (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(100) NOT NULL,
    product_id VARCHAR(100) NOT NULL,
    purchase_count INTEGER DEFAULT 1,
    last_purchased TIMESTAMP,
    first_purchased TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_spent DECIMAL(12, 2) DEFAULT 0,
    UNIQUE(customer_id, product_id)
);

CREATE INDEX IF NOT EXISTS idx_customer_purchases_customer ON customer_purchases(customer_id);
CREATE INDEX IF NOT EXISTS idx_customer_purchases_product ON customer_purchases(product_id);

COMMENT ON TABLE customer_purchases IS 'Aggregated customer purchase history for collaborative filtering';

-- ============================================
-- 4. PRODUCT PAIRS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS product_pairs (
    id SERIAL PRIMARY KEY,
    product_1 VARCHAR(100) NOT NULL,
    product_2 VARCHAR(100) NOT NULL,
    co_purchase_count INTEGER DEFAULT 1,
    confidence DECIMAL(8, 4),
    support DECIMAL(8, 4),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_1, product_2),
    CHECK (product_1 < product_2)
);

CREATE INDEX IF NOT EXISTS idx_product_pairs_p1 ON product_pairs(product_1);
CREATE INDEX IF NOT EXISTS idx_product_pairs_p2 ON product_pairs(product_2);

COMMENT ON TABLE product_pairs IS 'Frequently bought together products for cross-selling';

-- ============================================
-- 5. PRODUCT STATISTICS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS product_statistics (
    product_id VARCHAR(100) PRIMARY KEY,
    product_name VARCHAR(255),
    total_purchases INTEGER DEFAULT 0,
    unique_customers INTEGER DEFAULT 0,
    total_revenue DECIMAL(15, 2) DEFAULT 0,
    avg_purchase_value DECIMAL(12, 2),
    last_purchased TIMESTAMP,
    popularity_score DECIMAL(5, 4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_product_stats_popularity ON product_statistics(popularity_score DESC);

COMMENT ON TABLE product_statistics IS 'Product popularity metrics for recommendations';

-- ============================================
-- 6. CUSTOMER STATISTICS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS customer_statistics (
    customer_id VARCHAR(100) PRIMARY KEY,
    customer_name VARCHAR(255),
    customer_city VARCHAR(100),
    total_orders INTEGER DEFAULT 0,
    total_products INTEGER DEFAULT 0,
    unique_products INTEGER DEFAULT 0,
    total_spent DECIMAL(15, 2) DEFAULT 0,
    avg_order_value DECIMAL(12, 2),
    first_order_date TIMESTAMP,
    last_order_date TIMESTAMP,
    customer_segment VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_customer_stats_segment ON customer_statistics(customer_segment);
CREATE INDEX IF NOT EXISTS idx_customer_stats_spent ON customer_statistics(total_spent DESC);

COMMENT ON TABLE customer_statistics IS 'Customer behavior analytics and segmentation';

-- ============================================
-- 7. RECOMMENDATION CACHE TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS recommendation_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    recommendation_type VARCHAR(50) NOT NULL,
    customer_id VARCHAR(100),
    product_id VARCHAR(100),
    recommendations JSONB,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rec_cache_key ON recommendation_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_rec_cache_expires ON recommendation_cache(expires_at);

COMMENT ON TABLE recommendation_cache IS 'Cached recommendations for performance';

-- ============================================
-- 8. SYNC METADATA TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS sync_metadata (
    id SERIAL PRIMARY KEY,
    sync_type VARCHAR(100) NOT NULL,
    last_sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sync_started_at TIMESTAMP,
    sync_completed_at TIMESTAMP,
    orders_synced INTEGER DEFAULT 0,
    orders_inserted INTEGER DEFAULT 0,
    orders_updated INTEGER DEFAULT 0,
    sync_duration_seconds NUMERIC,
    sync_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    api_response_time_ms INTEGER,
    data_quality_score NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sync_meta_type ON sync_metadata(sync_type);
CREATE INDEX IF NOT EXISTS idx_sync_meta_timestamp ON sync_metadata(last_sync_timestamp DESC);

COMMENT ON TABLE sync_metadata IS 'Tracks data synchronization from external APIs';

-- ============================================
-- 9. USERS TABLE (Authentication)
-- ============================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

COMMENT ON TABLE users IS 'User authentication for dashboard access';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt hashed password';
COMMENT ON COLUMN users.is_active IS 'Flag to enable/disable user access';

-- ============================================
-- DISPLAY SUMMARY
-- ============================================

DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN (
        'orders', 'order_items', 'customer_purchases', 'product_pairs',
        'product_statistics', 'customer_statistics', 'recommendation_cache',
        'sync_metadata', 'users'
    );
    
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Master Group Recommendation System';
    RAISE NOTICE 'Database Setup Complete!';
    RAISE NOTICE '============================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables Created: %', table_count;
    RAISE NOTICE '';
    RAISE NOTICE 'Core Tables:';
    RAISE NOTICE '  ✓ orders - Source order data';
    RAISE NOTICE '  ✓ order_items - Order line items';
    RAISE NOTICE '  ✓ users - Authentication';
    RAISE NOTICE '';
    RAISE NOTICE 'Recommendation Tables:';
    RAISE NOTICE '  ✓ customer_purchases - Purchase history';
    RAISE NOTICE '  ✓ product_pairs - Cross-selling';
    RAISE NOTICE '  ✓ product_statistics - Product metrics';
    RAISE NOTICE '  ✓ customer_statistics - Customer analytics';
    RAISE NOTICE '  ✓ recommendation_cache - Performance cache';
    RAISE NOTICE '';
    RAISE NOTICE 'System Tables:';
    RAISE NOTICE '  ✓ sync_metadata - Data sync tracking';
    RAISE NOTICE '';
    RAISE NOTICE 'Next Steps:';
    RAISE NOTICE '  1. Run seed_users.sql to create admin user';
    RAISE NOTICE '  2. Configure API sync in .env file';
    RAISE NOTICE '  3. Start the recommendation engine';
    RAISE NOTICE '';
    RAISE NOTICE '============================================';
END $$;
