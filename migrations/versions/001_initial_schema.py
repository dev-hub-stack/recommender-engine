"""Initial schema - Core tables for MasterGroup Recommendation System

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-11-29

This migration creates all core tables needed for the recommendation system:
- orders: Source data from POS/OE systems
- order_items: Individual items from each order
- users: Authentication for dashboard access
- customer_purchases: Purchase history for collaborative filtering
- product_pairs: Cross-selling recommendations
- product_statistics: Product popularity metrics
- customer_statistics: Customer analytics and segmentation
- recommendation_cache: Performance caching
- sync_metadata: Data sync tracking
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # ============================================
    # 1. ORDERS TABLE (Source Data)
    # ============================================
    op.create_table(
        'orders',
        sa.Column('id', sa.String(100), primary_key=True),
        sa.Column('order_type', sa.String(10), nullable=False),
        sa.Column('order_date', sa.DateTime, nullable=False),
        sa.Column('order_name', sa.String(100)),
        sa.Column('order_status', sa.String(50)),
        sa.Column('customer_name', sa.String(255)),
        sa.Column('customer_email', sa.String(255)),
        sa.Column('customer_phone', sa.String(50)),
        sa.Column('customer_city', sa.String(100)),
        sa.Column('customer_address', sa.Text),
        sa.Column('unified_customer_id', sa.String(100)),
        sa.Column('total_price', sa.Numeric(12, 2)),
        sa.Column('payment_mode', sa.String(50)),
        sa.Column('brand_name', sa.String(100)),
        sa.Column('items_json', postgresql.JSONB),
        sa.Column('province', sa.String(100)),
        sa.Column('source_type', sa.String(50)),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('synced_at', sa.DateTime, server_default=sa.func.now()),
        sa.CheckConstraint("order_type IN ('POS', 'OE')", name='check_order_type')
    )
    
    op.create_index('idx_orders_customer_id', 'orders', ['unified_customer_id'])
    op.create_index('idx_orders_order_date', 'orders', ['order_date'])
    op.create_index('idx_orders_order_type', 'orders', ['order_type'])
    op.create_index('idx_orders_brand_name', 'orders', ['brand_name'])
    op.create_index('idx_orders_customer_city', 'orders', ['customer_city'])
    op.create_index('idx_orders_province', 'orders', ['province'])
    op.create_index('idx_orders_items_json', 'orders', ['items_json'], postgresql_using='gin')
    
    # ============================================
    # 2. ORDER ITEMS TABLE
    # ============================================
    op.create_table(
        'order_items',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('order_id', sa.String(100), sa.ForeignKey('orders.id', ondelete='CASCADE'), nullable=False),
        sa.Column('product_id', sa.String(100), nullable=False),
        sa.Column('product_name', sa.String(255)),
        sa.Column('quantity', sa.Integer, default=1),
        sa.Column('unit_price', sa.Numeric(12, 2)),
        sa.Column('total_price', sa.Numeric(12, 2)),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
    )
    
    op.create_index('idx_order_items_order_id', 'order_items', ['order_id'])
    op.create_index('idx_order_items_product_id', 'order_items', ['product_id'])
    
    # ============================================
    # 3. USERS TABLE (Authentication)
    # ============================================
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255)),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('last_login', sa.DateTime)
    )
    
    op.create_index('idx_users_email', 'users', ['email'])
    
    # ============================================
    # 4. CUSTOMER PURCHASES TABLE
    # ============================================
    op.create_table(
        'customer_purchases',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('customer_id', sa.String(100), nullable=False),
        sa.Column('product_id', sa.String(100), nullable=False),
        sa.Column('purchase_count', sa.Integer, default=1),
        sa.Column('last_purchased', sa.DateTime),
        sa.Column('first_purchased', sa.DateTime, server_default=sa.func.now()),
        sa.Column('total_spent', sa.Numeric(12, 2), default=0),
        sa.UniqueConstraint('customer_id', 'product_id', name='uq_customer_product')
    )
    
    op.create_index('idx_customer_purchases_customer', 'customer_purchases', ['customer_id'])
    op.create_index('idx_customer_purchases_product', 'customer_purchases', ['product_id'])
    
    # ============================================
    # 5. PRODUCT PAIRS TABLE
    # ============================================
    op.create_table(
        'product_pairs',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('product_1', sa.String(100), nullable=False),
        sa.Column('product_2', sa.String(100), nullable=False),
        sa.Column('co_purchase_count', sa.Integer, default=1),
        sa.Column('confidence', sa.Numeric(8, 4)),
        sa.Column('support', sa.Numeric(8, 4)),
        sa.Column('last_updated', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('product_1', 'product_2', name='uq_product_pair'),
        sa.CheckConstraint('product_1 < product_2', name='check_product_order')
    )
    
    op.create_index('idx_product_pairs_p1', 'product_pairs', ['product_1'])
    op.create_index('idx_product_pairs_p2', 'product_pairs', ['product_2'])
    
    # ============================================
    # 6. PRODUCT STATISTICS TABLE
    # ============================================
    op.create_table(
        'product_statistics',
        sa.Column('product_id', sa.String(100), primary_key=True),
        sa.Column('product_name', sa.String(255)),
        sa.Column('total_purchases', sa.Integer, default=0),
        sa.Column('unique_customers', sa.Integer, default=0),
        sa.Column('total_revenue', sa.Numeric(15, 2), default=0),
        sa.Column('avg_purchase_value', sa.Numeric(12, 2)),
        sa.Column('last_purchased', sa.DateTime),
        sa.Column('popularity_score', sa.Numeric(5, 4)),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now())
    )
    
    op.create_index('idx_product_stats_popularity', 'product_statistics', ['popularity_score'])
    
    # ============================================
    # 7. CUSTOMER STATISTICS TABLE
    # ============================================
    op.create_table(
        'customer_statistics',
        sa.Column('customer_id', sa.String(100), primary_key=True),
        sa.Column('customer_name', sa.String(255)),
        sa.Column('customer_city', sa.String(100)),
        sa.Column('total_orders', sa.Integer, default=0),
        sa.Column('total_products', sa.Integer, default=0),
        sa.Column('unique_products', sa.Integer, default=0),
        sa.Column('total_spent', sa.Numeric(15, 2), default=0),
        sa.Column('avg_order_value', sa.Numeric(12, 2)),
        sa.Column('first_order_date', sa.DateTime),
        sa.Column('last_order_date', sa.DateTime),
        sa.Column('customer_segment', sa.String(50)),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now())
    )
    
    op.create_index('idx_customer_stats_segment', 'customer_statistics', ['customer_segment'])
    op.create_index('idx_customer_stats_spent', 'customer_statistics', ['total_spent'])
    
    # ============================================
    # 8. RECOMMENDATION CACHE TABLE
    # ============================================
    op.create_table(
        'recommendation_cache',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('cache_key', sa.String(255), unique=True, nullable=False),
        sa.Column('recommendation_type', sa.String(50), nullable=False),
        sa.Column('customer_id', sa.String(100)),
        sa.Column('product_id', sa.String(100)),
        sa.Column('recommendations', postgresql.JSONB),
        sa.Column('hit_count', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime),
        sa.Column('last_accessed', sa.DateTime, server_default=sa.func.now())
    )
    
    op.create_index('idx_rec_cache_key', 'recommendation_cache', ['cache_key'])
    op.create_index('idx_rec_cache_expires', 'recommendation_cache', ['expires_at'])
    
    # ============================================
    # 9. SYNC METADATA TABLE
    # ============================================
    op.create_table(
        'sync_metadata',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('sync_type', sa.String(100), nullable=False),
        sa.Column('last_sync_timestamp', sa.DateTime, server_default=sa.func.now()),
        sa.Column('sync_started_at', sa.DateTime),
        sa.Column('sync_completed_at', sa.DateTime),
        sa.Column('orders_synced', sa.Integer, default=0),
        sa.Column('orders_inserted', sa.Integer, default=0),
        sa.Column('orders_updated', sa.Integer, default=0),
        sa.Column('sync_duration_seconds', sa.Numeric),
        sa.Column('sync_status', sa.String(50), nullable=False, default='pending'),
        sa.Column('error_message', sa.Text),
        sa.Column('api_response_time_ms', sa.Integer),
        sa.Column('data_quality_score', sa.Numeric),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
    )
    
    op.create_index('idx_sync_meta_type', 'sync_metadata', ['sync_type'])
    op.create_index('idx_sync_meta_timestamp', 'sync_metadata', ['last_sync_timestamp'])


def downgrade() -> None:
    op.drop_table('sync_metadata')
    op.drop_table('recommendation_cache')
    op.drop_table('customer_statistics')
    op.drop_table('product_statistics')
    op.drop_table('product_pairs')
    op.drop_table('customer_purchases')
    op.drop_table('users')
    op.drop_table('order_items')
    op.drop_table('orders')
