-- Setup script for Recommendation Service tables
-- Run this to create all required tables for the recommendation engine

-- 1. Order Items table (extracted from orders.items_json)
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

-- 2. Customer Purchases table (for collaborative filtering)
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

-- 3. Product Pairs table (for cross-selling recommendations)
CREATE TABLE IF NOT EXISTS product_pairs (
    id SERIAL PRIMARY KEY,
    product_1 VARCHAR(100) NOT NULL,
    product_2 VARCHAR(100) NOT NULL,
    co_purchase_count INTEGER DEFAULT 1,
    confidence DECIMAL(5, 4),
    support DECIMAL(5, 4),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_1, product_2),
    CHECK (product_1 < product_2)  -- Ensure consistent ordering
);

-- 4. Product Statistics table (for popularity-based recommendations)
CREATE TABLE IF NOT EXISTS product_statistics (
    product_id VARCHAR(100) PRIMARY KEY,
    product_name VARCHAR(255),
    total_purchases INTEGER DEFAULT 0,
    unique_customers INTEGER DEFAULT 0,
    total_revenue DECIMAL(12, 2) DEFAULT 0,
    avg_purchase_value DECIMAL(12, 2),
    last_purchased TIMESTAMP,
    popularity_score DECIMAL(5, 4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Customer Statistics table (for customer segmentation)
CREATE TABLE IF NOT EXISTS customer_statistics (
    customer_id VARCHAR(100) PRIMARY KEY,
    customer_name VARCHAR(255),
    customer_city VARCHAR(100),
    total_orders INTEGER DEFAULT 0,
    total_products INTEGER DEFAULT 0,
    unique_products INTEGER DEFAULT 0,
    total_spent DECIMAL(12, 2) DEFAULT 0,
    avg_order_value DECIMAL(12, 2),
    first_order_date TIMESTAMP,
    last_order_date TIMESTAMP,
    customer_segment VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Recommendation Cache table (for tracking recommendations)
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

-- 7. Sync Metadata table (for tracking synchronization history)
-- Note: This matches the existing table structure from the sync service
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

-- Create indices for performance
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_order_items_order_product ON order_items(order_id, product_id);

CREATE INDEX IF NOT EXISTS idx_customer_purchases_customer ON customer_purchases(customer_id);
CREATE INDEX IF NOT EXISTS idx_customer_purchases_product ON customer_purchases(product_id);
CREATE INDEX IF NOT EXISTS idx_customer_purchases_count ON customer_purchases(purchase_count DESC);

CREATE INDEX IF NOT EXISTS idx_product_pairs_p1 ON product_pairs(product_1);
CREATE INDEX IF NOT EXISTS idx_product_pairs_p2 ON product_pairs(product_2);
CREATE INDEX IF NOT EXISTS idx_product_pairs_count ON product_pairs(co_purchase_count DESC);

CREATE INDEX IF NOT EXISTS idx_product_stats_purchases ON product_statistics(total_purchases DESC);
CREATE INDEX IF NOT EXISTS idx_product_stats_customers ON product_statistics(unique_customers DESC);
CREATE INDEX IF NOT EXISTS idx_product_stats_popularity ON product_statistics(popularity_score DESC);

CREATE INDEX IF NOT EXISTS idx_customer_stats_segment ON customer_statistics(customer_segment);
CREATE INDEX IF NOT EXISTS idx_customer_stats_spent ON customer_statistics(total_spent DESC);
CREATE INDEX IF NOT EXISTS idx_customer_stats_orders ON customer_statistics(total_orders DESC);

CREATE INDEX IF NOT EXISTS idx_rec_cache_type ON recommendation_cache(recommendation_type);
CREATE INDEX IF NOT EXISTS idx_rec_cache_customer ON recommendation_cache(customer_id);
CREATE INDEX IF NOT EXISTS idx_rec_cache_expires ON recommendation_cache(expires_at);

CREATE INDEX IF NOT EXISTS idx_sync_metadata_type ON sync_metadata(sync_type);
CREATE INDEX IF NOT EXISTS idx_sync_metadata_timestamp ON sync_metadata(last_sync_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sync_metadata_status ON sync_metadata(sync_status);

-- Create function to populate order_items from existing orders
CREATE OR REPLACE FUNCTION populate_order_items_from_orders()
RETURNS INTEGER AS $$
DECLARE
    order_record RECORD;
    item_record JSONB;
    items_inserted INTEGER := 0;
BEGIN
    -- Loop through all orders
    FOR order_record IN 
        SELECT id, items_json 
        FROM orders 
        WHERE items_json IS NOT NULL
    LOOP
        -- Loop through items in the JSON array
        FOR item_record IN 
            SELECT * FROM jsonb_array_elements(order_record.items_json)
        LOOP
            -- Insert into order_items
            INSERT INTO order_items (
                order_id,
                product_id,
                product_name,
                quantity,
                unit_price,
                total_price
            ) VALUES (
                order_record.id,
                COALESCE(item_record->>'product_id', item_record->>'id'),
                COALESCE(item_record->>'product_name', item_record->>'name'),
                COALESCE((item_record->>'quantity')::INTEGER, 1),
                COALESCE((item_record->>'unit_price')::DECIMAL, (item_record->>'price')::DECIMAL, 0),
                COALESCE((item_record->>'total_price')::DECIMAL, (item_record->>'price')::DECIMAL, 0)
            )
            ON CONFLICT DO NOTHING;
            
            items_inserted := items_inserted + 1;
        END LOOP;
    END LOOP;
    
    RETURN items_inserted;
END;
$$ LANGUAGE plpgsql;

-- Create function to rebuild customer_purchases
CREATE OR REPLACE FUNCTION rebuild_customer_purchases()
RETURNS INTEGER AS $$
DECLARE
    rows_inserted INTEGER := 0;
BEGIN
    -- Clear existing data
    TRUNCATE customer_purchases;
    
    -- Rebuild from orders and order_items
    INSERT INTO customer_purchases (
        customer_id,
        product_id,
        purchase_count,
        last_purchased,
        first_purchased,
        total_spent
    )
    SELECT 
        o.unified_customer_id,
        oi.product_id,
        COUNT(*) as purchase_count,
        MAX(o.order_date) as last_purchased,
        MIN(o.order_date) as first_purchased,
        SUM(oi.total_price) as total_spent
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.unified_customer_id IS NOT NULL
    GROUP BY o.unified_customer_id, oi.product_id
    ON CONFLICT (customer_id, product_id) DO UPDATE SET
        purchase_count = EXCLUDED.purchase_count,
        last_purchased = EXCLUDED.last_purchased,
        total_spent = EXCLUDED.total_spent;
    
    GET DIAGNOSTICS rows_inserted = ROW_COUNT;
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;

-- Create function to rebuild product_pairs
CREATE OR REPLACE FUNCTION rebuild_product_pairs()
RETURNS INTEGER AS $$
DECLARE
    rows_inserted INTEGER := 0;
BEGIN
    -- Clear existing data
    TRUNCATE product_pairs;
    
    -- Rebuild product pairs from co-purchases
    INSERT INTO product_pairs (
        product_1,
        product_2,
        co_purchase_count,
        confidence,
        last_updated
    )
    SELECT 
        LEAST(oi1.product_id, oi2.product_id) as product_1,
        GREATEST(oi1.product_id, oi2.product_id) as product_2,
        COUNT(*) as co_purchase_count,
        CAST(COUNT(*) AS DECIMAL) / NULLIF(
            (SELECT COUNT(*) FROM order_items WHERE product_id = oi1.product_id), 
            0
        ) as confidence,
        CURRENT_TIMESTAMP as last_updated
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
    GROUP BY oi1.product_id, oi2.product_id
    HAVING COUNT(*) > 1  -- Only pairs bought together at least twice
    ON CONFLICT (product_1, product_2) DO UPDATE SET
        co_purchase_count = EXCLUDED.co_purchase_count,
        confidence = EXCLUDED.confidence,
        last_updated = EXCLUDED.last_updated;
    
    GET DIAGNOSTICS rows_inserted = ROW_COUNT;
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;

-- Create function to rebuild product_statistics
CREATE OR REPLACE FUNCTION rebuild_product_statistics()
RETURNS INTEGER AS $$
DECLARE
    rows_inserted INTEGER := 0;
BEGIN
    -- Clear existing data
    TRUNCATE product_statistics;
    
    -- Rebuild product statistics
    INSERT INTO product_statistics (
        product_id,
        product_name,
        total_purchases,
        unique_customers,
        total_revenue,
        avg_purchase_value,
        last_purchased,
        popularity_score
    )
    SELECT 
        oi.product_id,
        MAX(oi.product_name) as product_name,
        COUNT(*) as total_purchases,
        COUNT(DISTINCT o.unified_customer_id) as unique_customers,
        SUM(oi.total_price) as total_revenue,
        AVG(oi.total_price) as avg_purchase_value,
        MAX(o.order_date) as last_purchased,
        CAST(COUNT(*) AS DECIMAL) / NULLIF(
            (SELECT COUNT(DISTINCT id) FROM orders), 
            0
        ) as popularity_score
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.id
    GROUP BY oi.product_id
    ON CONFLICT (product_id) DO UPDATE SET
        product_name = EXCLUDED.product_name,
        total_purchases = EXCLUDED.total_purchases,
        unique_customers = EXCLUDED.unique_customers,
        total_revenue = EXCLUDED.total_revenue,
        avg_purchase_value = EXCLUDED.avg_purchase_value,
        last_purchased = EXCLUDED.last_purchased,
        popularity_score = EXCLUDED.popularity_score,
        updated_at = CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS rows_inserted = ROW_COUNT;
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;

-- Create function to rebuild customer_statistics
CREATE OR REPLACE FUNCTION rebuild_customer_statistics()
RETURNS INTEGER AS $$
DECLARE
    rows_inserted INTEGER := 0;
BEGIN
    -- Clear existing data
    TRUNCATE customer_statistics;
    
    -- Rebuild customer statistics
    INSERT INTO customer_statistics (
        customer_id,
        customer_name,
        customer_city,
        total_orders,
        total_products,
        unique_products,
        total_spent,
        avg_order_value,
        first_order_date,
        last_order_date
    )
    SELECT 
        o.unified_customer_id,
        MAX(o.customer_name) as customer_name,
        MAX(o.customer_city) as customer_city,
        COUNT(DISTINCT o.id) as total_orders,
        SUM(
            (SELECT COUNT(*) FROM jsonb_array_elements(o.items_json))
        ) as total_products,
        COUNT(DISTINCT oi.product_id) as unique_products,
        SUM(o.total_price) as total_spent,
        AVG(o.total_price) as avg_order_value,
        MIN(o.order_date) as first_order_date,
        MAX(o.order_date) as last_order_date
    FROM orders o
    LEFT JOIN order_items oi ON o.id = oi.order_id
    WHERE o.unified_customer_id IS NOT NULL
    GROUP BY o.unified_customer_id
    ON CONFLICT (customer_id) DO UPDATE SET
        customer_name = EXCLUDED.customer_name,
        customer_city = EXCLUDED.customer_city,
        total_orders = EXCLUDED.total_orders,
        total_products = EXCLUDED.total_products,
        unique_products = EXCLUDED.unique_products,
        total_spent = EXCLUDED.total_spent,
        avg_order_value = EXCLUDED.avg_order_value,
        first_order_date = EXCLUDED.first_order_date,
        last_order_date = EXCLUDED.last_order_date,
        updated_at = CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS rows_inserted = ROW_COUNT;
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;

-- Print success message
DO $$
BEGIN
    RAISE NOTICE 'Recommendation tables created successfully!';
    RAISE NOTICE 'Tables created: 7 total (orders, order_items, customer_purchases, product_pairs, product_statistics, customer_statistics, recommendation_cache, sync_metadata)';
    RAISE NOTICE 'Run the following to populate data:';
    RAISE NOTICE '1. SELECT populate_order_items_from_orders();';
    RAISE NOTICE '2. SELECT rebuild_customer_purchases();';
    RAISE NOTICE '3. SELECT rebuild_product_pairs();';
    RAISE NOTICE '4. SELECT rebuild_product_statistics();';
    RAISE NOTICE '5. SELECT rebuild_customer_statistics();';
END $$;
