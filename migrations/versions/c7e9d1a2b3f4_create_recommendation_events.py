"""create recommendation events table

Revision ID: c7e9d1a2b3f4
Revises: b2d3e4f5g6h7
Create Date: 2026-04-22 18:45:00.000000

This migration creates recommendation_events for Shopify/widget attribution
tracking. It stores impression, click, add_to_cart, and purchase events plus
the core identifiers needed for later attribution and A/B analysis.
"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "c7e9d1a2b3f4"
down_revision = "b2d3e4f5g6h7"
branch_labels = None
depends_on = None


def upgrade():
    """Create recommendation_events table and lookup indexes."""

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_events (
            id BIGSERIAL PRIMARY KEY,
            event_type VARCHAR(32) NOT NULL,
            occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            mg_session_id VARCHAR(128),
            rec_request_id VARCHAR(128),
            storefront VARCHAR(255),
            page_type VARCHAR(64),
            source_widget VARCHAR(64),
            variant VARCHAR(64),
            algorithm VARCHAR(64),
            recommendation_type VARCHAR(64),
            seed_product_id VARCHAR(255),
            seed_shopify_product_id BIGINT,
            recommended_product_id VARCHAR(255),
            recommended_shopify_product_id BIGINT,
            shopify_handle VARCHAR(255),
            user_id VARCHAR(255),
            order_id VARCHAR(255),
            customer_phone VARCHAR(64),
            customer_email VARCHAR(255),
            city VARCHAR(100),
            province VARCHAR(100),
            position INTEGER,
            attributed_event_id BIGINT REFERENCES recommendation_events(id) ON DELETE SET NULL,
            revenue NUMERIC(18, 2),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            CONSTRAINT chk_recommendation_events_type
                CHECK (event_type IN ('impression', 'click', 'add_to_cart', 'purchase'))
        );

        CREATE INDEX IF NOT EXISTS idx_recommendation_events_type_time
            ON recommendation_events(event_type, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_recommendation_events_session_time
            ON recommendation_events(mg_session_id, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_recommendation_events_request_time
            ON recommendation_events(rec_request_id, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_recommendation_events_order_time
            ON recommendation_events(order_id, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_recommendation_events_user_time
            ON recommendation_events(user_id, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_recommendation_events_customer_email_time
            ON recommendation_events(customer_email, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_recommendation_events_customer_phone_time
            ON recommendation_events(customer_phone, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_recommendation_events_product_time
            ON recommendation_events(recommended_product_id, occurred_at DESC);
        CREATE INDEX IF NOT EXISTS idx_recommendation_events_shopify_product_time
            ON recommendation_events(recommended_shopify_product_id, occurred_at DESC);
        """
    )

    print("✅ Created recommendation_events table")


def downgrade():
    """Drop recommendation_events table and its indexes."""

    op.execute(
        """
        DROP TABLE IF EXISTS recommendation_events CASCADE;
        """
    )

    print("✅ Dropped recommendation_events table")
