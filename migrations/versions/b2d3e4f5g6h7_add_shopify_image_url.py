"""add shopify image url column

Revision ID: b2d3e4f5g6h7
Revises: a1c2e3f4g5h6
Create Date: 2026-01-18 23:50:00.000000

This migration adds the shopify_image_url column to the product mapping table
to store product images for display in the recommendation widgets.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b2d3e4f5g6h7'
down_revision = 'a1c2e3f4g5h6'  # Links to create_shopify_mapping_tables
branch_labels = None
depends_on = None


def upgrade():
    """Add shopify_image_url column to product mapping table."""
    
    op.execute("""
        ALTER TABLE shopify_product_mapping 
        ADD COLUMN IF NOT EXISTS shopify_image_url TEXT;
        
        -- Add index for faster lookups when joining with recommendations
        CREATE INDEX IF NOT EXISTS idx_shopify_product_mapping_handle 
            ON shopify_product_mapping(shopify_handle);
    """)
    
    print("✅ Added shopify_image_url column to shopify_product_mapping")


def downgrade():
    """Remove shopify_image_url column."""
    
    op.execute("""
        DROP INDEX IF EXISTS idx_shopify_product_mapping_handle;
        ALTER TABLE shopify_product_mapping DROP COLUMN IF EXISTS shopify_image_url;
    """)
    
    print("✅ Removed shopify_image_url column from shopify_product_mapping")
