"""create shopify mapping tables

Revision ID: a1c2e3f4g5h6
Revises: a1b2c3d4e5f6
Create Date: 2026-01-18 23:15:00.000000

This migration creates tables for Shopify integration:
- shopify_product_mapping: Maps Shopify product IDs to MasterGroup product IDs
- shopify_customer_mapping: Maps Shopify customers to MasterGroup customers

This enables the recommendation system to translate between Shopify's
product/customer IDs and the internal MasterGroup IDs used for ML training.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'a1c2e3f4g5h6'
down_revision = 'a1b2c3d4e5f6'  # Links to fix_rebuild_customer_statistics_preserve_segments
branch_labels = None
depends_on = None


def upgrade():
    """Create Shopify integration tables."""
    
    # Use raw SQL with IF NOT EXISTS for idempotency
    op.execute("""
        -- ===========================================
        -- shopify_product_mapping
        -- Maps Shopify products to MasterGroup products
        -- ===========================================
        CREATE TABLE IF NOT EXISTS shopify_product_mapping (
            id SERIAL PRIMARY KEY,
            shopify_product_id BIGINT UNIQUE NOT NULL,
            shopify_title VARCHAR(255),
            shopify_sku VARCHAR(100),
            shopify_handle VARCHAR(255),
            mastergroup_product_id VARCHAR(100),
            mastergroup_product_name VARCHAR(255),
            match_confidence FLOAT DEFAULT 1.0,
            match_method VARCHAR(50),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for fast lookups (use IF NOT EXISTS pattern)
        CREATE INDEX IF NOT EXISTS idx_shopify_product_mapping_shopify_id 
            ON shopify_product_mapping(shopify_product_id);
        CREATE INDEX IF NOT EXISTS idx_shopify_product_mapping_mg_id 
            ON shopify_product_mapping(mastergroup_product_id);
        CREATE INDEX IF NOT EXISTS idx_shopify_product_mapping_sku 
            ON shopify_product_mapping(shopify_sku);
        CREATE INDEX IF NOT EXISTS idx_shopify_product_mapping_active 
            ON shopify_product_mapping(is_active);
    """)
    
    op.execute("""
        -- ===========================================
        -- shopify_customer_mapping
        -- Maps Shopify customers to MasterGroup customers
        -- ===========================================
        CREATE TABLE IF NOT EXISTS shopify_customer_mapping (
            id SERIAL PRIMARY KEY,
            shopify_customer_id BIGINT UNIQUE,
            shopify_email VARCHAR(255),
            shopify_phone VARCHAR(50),
            mastergroup_user_id VARCHAR(100),
            match_method VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for fast lookups
        CREATE INDEX IF NOT EXISTS idx_shopify_customer_mapping_shopify_id 
            ON shopify_customer_mapping(shopify_customer_id);
        CREATE INDEX IF NOT EXISTS idx_shopify_customer_mapping_phone 
            ON shopify_customer_mapping(shopify_phone);
        CREATE INDEX IF NOT EXISTS idx_shopify_customer_mapping_email 
            ON shopify_customer_mapping(shopify_email);
        CREATE INDEX IF NOT EXISTS idx_shopify_customer_mapping_mg_user 
            ON shopify_customer_mapping(mastergroup_user_id);
    """)
    
    print("✅ Created shopify_product_mapping and shopify_customer_mapping tables")


def downgrade():
    """Remove Shopify integration tables."""
    
    op.execute("""
        DROP TABLE IF EXISTS shopify_customer_mapping CASCADE;
        DROP TABLE IF EXISTS shopify_product_mapping CASCADE;
    """)
    
    print("✅ Dropped shopify_product_mapping and shopify_customer_mapping tables")
