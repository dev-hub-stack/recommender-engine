"""fix rebuild_customer_statistics to preserve segments

Revision ID: a1b2c3d4e5f6
Revises: d32509426ade
Create Date: 2025-11-25 20:55:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'd32509426ade'
branch_labels = None
depends_on = None


def upgrade():
    """
    Update rebuild_customer_statistics function to preserve customer_segment
    This prevents the sync service from wiping out RFM segments
    """
    op.execute("""
        CREATE OR REPLACE FUNCTION rebuild_customer_statistics()
        RETURNS integer
        LANGUAGE plpgsql
        AS $$
        DECLARE
            rows_inserted INTEGER := 0;
        BEGIN
            -- Rebuild customer statistics while preserving customer_segment
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
                last_order_date,
                customer_segment
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
                MAX(o.order_date) as last_order_date,
                NULL as customer_segment
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
                customer_segment = COALESCE(EXCLUDED.customer_segment, customer_statistics.customer_segment),
                updated_at = CURRENT_TIMESTAMP;
            
            GET DIAGNOSTICS rows_inserted = ROW_COUNT;
            
            -- Recalculate segments for customers where segment is NULL
            UPDATE customer_statistics
            SET customer_segment = calculate_rfm_segment(
                GREATEST(COALESCE(EXTRACT(EPOCH FROM (NOW() - last_order_date))::INTEGER / 86400, 0), 0),
                COALESCE(total_orders, 0)::INTEGER,
                COALESCE(total_spent, 0)::NUMERIC
            )
            WHERE customer_segment IS NULL;
            
            RETURN rows_inserted;
        END;
        $$;
    """)


def downgrade():
    """
    Revert to original rebuild_customer_statistics function
    """
    op.execute("""
        CREATE OR REPLACE FUNCTION rebuild_customer_statistics()
        RETURNS integer
        LANGUAGE plpgsql
        AS $$
        DECLARE
            rows_inserted INTEGER := 0;
        BEGIN
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
        $$;
    """)
