"""Stored procedures for recommendation engine

Revision ID: 003_stored_procedures
Revises: 002_aws_personalize
Create Date: 2025-11-29

This migration creates all stored procedures/functions for:
- populate_order_items_from_orders: Extract items from orders JSON
- rebuild_customer_purchases: Aggregate customer purchase history
- rebuild_product_pairs: Calculate co-purchase relationships
- rebuild_product_statistics: Update product metrics
- rebuild_customer_statistics: Update customer analytics
- update_updated_at_column: Trigger for auto-updating timestamps
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '003_stored_procedures'
down_revision: Union[str, None] = '002_aws_personalize'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ============================================
    # UPDATE TIMESTAMP TRIGGER
    # ============================================
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;
        CREATE TRIGGER update_orders_updated_at 
            BEFORE UPDATE ON orders 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    # ============================================
    # POPULATE ORDER ITEMS FROM ORDERS JSON
    # ============================================
    op.execute("""
        CREATE OR REPLACE FUNCTION populate_order_items_from_orders()
        RETURNS INTEGER AS $$
        DECLARE
            order_record RECORD;
            item_record JSONB;
            items_inserted INTEGER := 0;
        BEGIN
            FOR order_record IN 
                SELECT id, items_json 
                FROM orders 
                WHERE items_json IS NOT NULL
            LOOP
                FOR item_record IN 
                    SELECT * FROM jsonb_array_elements(order_record.items_json)
                LOOP
                    INSERT INTO order_items (
                        order_id, product_id, product_name, quantity, unit_price, total_price
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
    """)
    
    # ============================================
    # REBUILD CUSTOMER PURCHASES
    # ============================================
    op.execute("""
        CREATE OR REPLACE FUNCTION rebuild_customer_purchases()
        RETURNS INTEGER AS $$
        DECLARE
            rows_inserted INTEGER := 0;
        BEGIN
            TRUNCATE customer_purchases;
            
            INSERT INTO customer_purchases (
                customer_id, product_id, purchase_count, last_purchased, first_purchased, total_spent
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
    """)
    
    # ============================================
    # REBUILD PRODUCT PAIRS (Optimized)
    # ============================================
    op.execute("""
        CREATE OR REPLACE FUNCTION rebuild_product_pairs()
        RETURNS INTEGER AS $$
        DECLARE
            rows_inserted INTEGER := 0;
        BEGIN
            TRUNCATE product_pairs;
            
            CREATE TEMP TABLE temp_product_counts AS
            SELECT product_id, COUNT(*) as total_count
            FROM order_items
            GROUP BY product_id;
            
            CREATE INDEX idx_temp_product_counts ON temp_product_counts(product_id);
            
            INSERT INTO product_pairs (
                product_1, product_2, co_purchase_count, confidence, last_updated
            )
            SELECT 
                LEAST(oi1.product_id, oi2.product_id) as product_1,
                GREATEST(oi1.product_id, oi2.product_id) as product_2,
                COUNT(*) as co_purchase_count,
                CAST(COUNT(*) AS DECIMAL) / NULLIF(pc.total_count, 0) as confidence,
                CURRENT_TIMESTAMP as last_updated
            FROM order_items oi1
            JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
            JOIN temp_product_counts pc ON pc.product_id = oi1.product_id
            GROUP BY oi1.product_id, oi2.product_id, pc.total_count
            HAVING COUNT(*) > 1
            ON CONFLICT (product_1, product_2) DO UPDATE SET
                co_purchase_count = EXCLUDED.co_purchase_count,
                confidence = EXCLUDED.confidence,
                last_updated = EXCLUDED.last_updated;
            
            DROP TABLE temp_product_counts;
            
            GET DIAGNOSTICS rows_inserted = ROW_COUNT;
            RETURN rows_inserted;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # ============================================
    # REBUILD PRODUCT STATISTICS
    # ============================================
    op.execute("""
        CREATE OR REPLACE FUNCTION rebuild_product_statistics()
        RETURNS INTEGER AS $$
        DECLARE
            rows_inserted INTEGER := 0;
        BEGIN
            TRUNCATE product_statistics;
            
            INSERT INTO product_statistics (
                product_id, product_name, total_purchases, unique_customers,
                total_revenue, avg_purchase_value, last_purchased, popularity_score
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
                    (SELECT COUNT(DISTINCT id) FROM orders), 0
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
    """)
    
    # ============================================
    # REBUILD CUSTOMER STATISTICS
    # ============================================
    op.execute("""
        CREATE OR REPLACE FUNCTION rebuild_customer_statistics()
        RETURNS INTEGER AS $$
        DECLARE
            rows_inserted INTEGER := 0;
        BEGIN
            -- Preserve existing customer_segment values
            CREATE TEMP TABLE temp_segments AS
            SELECT customer_id, customer_segment FROM customer_statistics 
            WHERE customer_segment IS NOT NULL;
            
            TRUNCATE customer_statistics;
            
            INSERT INTO customer_statistics (
                customer_id, customer_name, customer_city, total_orders,
                total_products, unique_products, total_spent, avg_order_value,
                first_order_date, last_order_date
            )
            SELECT 
                o.unified_customer_id,
                MAX(o.customer_name) as customer_name,
                MAX(o.customer_city) as customer_city,
                COUNT(DISTINCT o.id) as total_orders,
                COALESCE(SUM(
                    (SELECT COUNT(*) FROM jsonb_array_elements(o.items_json))
                ), 0) as total_products,
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
            
            -- Restore preserved segments
            UPDATE customer_statistics cs
            SET customer_segment = ts.customer_segment
            FROM temp_segments ts
            WHERE cs.customer_id = ts.customer_id;
            
            DROP TABLE temp_segments;
            
            GET DIAGNOSTICS rows_inserted = ROW_COUNT;
            RETURN rows_inserted;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS rebuild_customer_statistics();")
    op.execute("DROP FUNCTION IF EXISTS rebuild_product_statistics();")
    op.execute("DROP FUNCTION IF EXISTS rebuild_product_pairs();")
    op.execute("DROP FUNCTION IF EXISTS rebuild_customer_purchases();")
    op.execute("DROP FUNCTION IF EXISTS populate_order_items_from_orders();")
    op.execute("DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")
