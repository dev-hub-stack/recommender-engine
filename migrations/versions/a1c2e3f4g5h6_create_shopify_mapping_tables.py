"""create shopify mapping tables

Revision ID: a1c2e3f4g5h6
Revises: fix_rebuild_customer_statistics_preserve_segments
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
    
    # ===========================================
    # shopify_product_mapping
    # Maps Shopify products to MasterGroup products
    # ===========================================
    op.create_table('shopify_product_mapping',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('shopify_product_id', sa.BigInteger(), nullable=False),
        sa.Column('shopify_title', sa.String(length=255), nullable=True),
        sa.Column('shopify_sku', sa.String(length=100), nullable=True),
        sa.Column('shopify_handle', sa.String(length=255), nullable=True),
        sa.Column('mastergroup_product_id', sa.String(length=100), nullable=True),
        sa.Column('mastergroup_product_name', sa.String(length=255), nullable=True),
        sa.Column('match_confidence', sa.Float(), server_default='1.0', nullable=True),
        sa.Column('match_method', sa.String(length=50), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('shopify_product_id')
    )
    
    # Indexes for fast lookups
    op.create_index('idx_shopify_product_mapping_shopify_id', 
                    'shopify_product_mapping', ['shopify_product_id'], unique=True)
    op.create_index('idx_shopify_product_mapping_mg_id', 
                    'shopify_product_mapping', ['mastergroup_product_id'], unique=False)
    op.create_index('idx_shopify_product_mapping_sku', 
                    'shopify_product_mapping', ['shopify_sku'], unique=False)
    op.create_index('idx_shopify_product_mapping_active', 
                    'shopify_product_mapping', ['is_active'], unique=False)
    
    # ===========================================
    # shopify_customer_mapping
    # Maps Shopify customers to MasterGroup customers
    # ===========================================
    op.create_table('shopify_customer_mapping',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('shopify_customer_id', sa.BigInteger(), nullable=True),
        sa.Column('shopify_email', sa.String(length=255), nullable=True),
        sa.Column('shopify_phone', sa.String(length=50), nullable=True),
        sa.Column('mastergroup_user_id', sa.String(length=100), nullable=True),
        sa.Column('match_method', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Indexes for fast lookups
    op.create_index('idx_shopify_customer_mapping_shopify_id', 
                    'shopify_customer_mapping', ['shopify_customer_id'], unique=True)
    op.create_index('idx_shopify_customer_mapping_phone', 
                    'shopify_customer_mapping', ['shopify_phone'], unique=False)
    op.create_index('idx_shopify_customer_mapping_email', 
                    'shopify_customer_mapping', ['shopify_email'], unique=False)
    op.create_index('idx_shopify_customer_mapping_mg_user', 
                    'shopify_customer_mapping', ['mastergroup_user_id'], unique=False)
    
    print("✅ Created shopify_product_mapping and shopify_customer_mapping tables")


def downgrade():
    """Remove Shopify integration tables."""
    
    # Drop customer mapping table and indexes
    op.drop_index('idx_shopify_customer_mapping_mg_user', table_name='shopify_customer_mapping')
    op.drop_index('idx_shopify_customer_mapping_email', table_name='shopify_customer_mapping')
    op.drop_index('idx_shopify_customer_mapping_phone', table_name='shopify_customer_mapping')
    op.drop_index('idx_shopify_customer_mapping_shopify_id', table_name='shopify_customer_mapping')
    op.drop_table('shopify_customer_mapping')
    
    # Drop product mapping table and indexes
    op.drop_index('idx_shopify_product_mapping_active', table_name='shopify_product_mapping')
    op.drop_index('idx_shopify_product_mapping_sku', table_name='shopify_product_mapping')
    op.drop_index('idx_shopify_product_mapping_mg_id', table_name='shopify_product_mapping')
    op.drop_index('idx_shopify_product_mapping_shopify_id', table_name='shopify_product_mapping')
    op.drop_table('shopify_product_mapping')
    
    print("✅ Dropped shopify_product_mapping and shopify_customer_mapping tables")
