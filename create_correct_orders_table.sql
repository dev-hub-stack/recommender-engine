-- Create the orders table with exact same structure as local database
-- This matches the working local schema

CREATE TABLE IF NOT EXISTS orders (
    id character varying(50) NOT NULL PRIMARY KEY,
    order_type character varying(10) NOT NULL CHECK (order_type IN ('POS', 'OE')),
    order_date timestamp without time zone NOT NULL,
    order_name character varying(100),
    order_status character varying(50),
    customer_name character varying(255),
    customer_email character varying(255),
    customer_phone character varying(50),
    customer_city character varying(100),
    customer_address text,
    unified_customer_id character varying(100),
    total_price numeric(12,2) CHECK (total_price >= 0),
    payment_mode character varying(50),
    brand_name character varying(100),
    items_json jsonb,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    synced_at timestamp without time zone DEFAULT now()
);

-- Create all the indexes from the local database
CREATE INDEX IF NOT EXISTS idx_brand_name ON orders(brand_name);
CREATE INDEX IF NOT EXISTS idx_customer_city ON orders(customer_city);
CREATE INDEX IF NOT EXISTS idx_customer_date ON orders(unified_customer_id, order_date DESC);
CREATE INDEX IF NOT EXISTS idx_customer_id ON orders(unified_customer_id);
CREATE INDEX IF NOT EXISTS idx_items_json ON orders USING gin(items_json);
CREATE INDEX IF NOT EXISTS idx_order_date ON orders(order_date DESC);
CREATE INDEX IF NOT EXISTS idx_order_type ON orders(order_type);
CREATE INDEX IF NOT EXISTS idx_order_type_date ON orders(order_type, order_date DESC);
CREATE INDEX IF NOT EXISTS idx_orders_customer_date ON orders(unified_customer_id, order_date DESC);
CREATE INDEX IF NOT EXISTS idx_orders_date_customer ON orders(order_date, unified_customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date_revenue ON orders(order_date, total_price);
CREATE INDEX IF NOT EXISTS idx_orders_date_type ON orders(order_date, order_type);
CREATE INDEX IF NOT EXISTS idx_payment_mode ON orders(payment_mode);
CREATE INDEX IF NOT EXISTS idx_status_date ON orders(order_status, order_date DESC);
CREATE INDEX IF NOT EXISTS idx_synced_at ON orders(synced_at DESC);

-- Create the update trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create the trigger
CREATE TRIGGER update_orders_updated_at 
    BEFORE UPDATE ON orders 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Orders table created with exact local schema!';
    RAISE NOTICE 'Ready for sync operations.';
END $$;
