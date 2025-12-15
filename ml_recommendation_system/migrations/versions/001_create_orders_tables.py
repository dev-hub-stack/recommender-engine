"""create orders tables

Revision ID: 001
Revises: 
Create Date: 2024-12-11

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create pos_orders table
    op.create_table(
        'pos_orders',
        sa.Column('id', sa.String(50), primary_key=True),
        sa.Column('customer_phone', sa.String(50), index=True),
        sa.Column('customer_email', sa.String(255), index=True),
        sa.Column('customer_name', sa.String(255)),
        sa.Column('customer_address', sa.Text()),
        sa.Column('customer_city', sa.String(100)),
        sa.Column('customer_state', sa.String(100)),
        sa.Column('customer_country', sa.String(100)),
        sa.Column('order_date', sa.DateTime(), nullable=False, index=True),
        sa.Column('order_source', sa.String(50)),
        sa.Column('order_status', sa.String(50)),
        sa.Column('order_status_id', sa.Integer()),
        sa.Column('has_items', sa.Text(), nullable=False),
        sa.Column('dealer_id', sa.String(50)),
        sa.Column('dealer_name', sa.String(255)),
        sa.Column('dealership_id', sa.String(50)),
        sa.Column('total_price', sa.Float()),
        sa.Column('discount', sa.Float()),
        sa.Column('dealer_discount', sa.Float()),
        sa.Column('payment_mode', sa.String(50)),
        sa.Column('brand_name', sa.String(100)),
        sa.Column('courier_id', sa.String(50)),
        sa.Column('is_split', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    
    # Create indexes for pos_orders
    op.create_index('idx_pos_customer_phone', 'pos_orders', ['customer_phone'])
    op.create_index('idx_pos_customer_email', 'pos_orders', ['customer_email'])
    op.create_index('idx_pos_order_date', 'pos_orders', ['order_date'])
    op.create_index('idx_pos_customer_date', 'pos_orders', ['customer_phone', 'order_date'])
    
    # Create oe_orders table
    op.create_table(
        'oe_orders',
        sa.Column('id', sa.String(50), primary_key=True),
        sa.Column('customer_phone', sa.String(50), index=True),
        sa.Column('customer_email', sa.String(255), index=True),
        sa.Column('customer_name', sa.String(255)),
        sa.Column('customer_address', sa.Text()),
        sa.Column('customer_city', sa.String(100)),
        sa.Column('customer_state', sa.String(100)),
        sa.Column('customer_country', sa.String(100)),
        sa.Column('order_date', sa.DateTime(), nullable=False, index=True),
        sa.Column('order_name', sa.String(100)),
        sa.Column('order_status', sa.String(50)),
        sa.Column('order_status_id', sa.Integer()),
        sa.Column('order_comments', sa.Text()),
        sa.Column('has_items', sa.Text(), nullable=False),
        sa.Column('total_price', sa.Float()),
        sa.Column('discount', sa.Float()),
        sa.Column('payment_mode', sa.String(50)),
        sa.Column('brand_name', sa.String(100)),
        sa.Column('courier_id', sa.String(50)),
        sa.Column('is_split', sa.Boolean(), default=False),
        sa.Column('assigned_tags', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    
    # Create indexes for oe_orders
    op.create_index('idx_oe_customer_phone', 'oe_orders', ['customer_phone'])
    op.create_index('idx_oe_customer_email', 'oe_orders', ['customer_email'])
    op.create_index('idx_oe_order_date', 'oe_orders', ['order_date'])
    op.create_index('idx_oe_customer_date', 'oe_orders', ['customer_phone', 'order_date'])
    
    # Create sync_logs table
    op.create_table(
        'sync_logs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('sync_type', sa.String(20), nullable=False),
        sa.Column('sync_start_date', sa.DateTime(), nullable=False),
        sa.Column('sync_end_date', sa.DateTime(), nullable=False),
        sa.Column('records_fetched', sa.Integer(), default=0),
        sa.Column('records_inserted', sa.Integer(), default=0),
        sa.Column('records_updated', sa.Integer(), default=0),
        sa.Column('records_failed', sa.Integer(), default=0),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('error_message', sa.Text()),
        sa.Column('duration_seconds', sa.Float()),
        sa.Column('started_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False)
    )
    
    # Create indexes for sync_logs
    op.create_index('idx_sync_type', 'sync_logs', ['sync_type'])
    op.create_index('idx_sync_status', 'sync_logs', ['status'])
    op.create_index('idx_sync_started_at', 'sync_logs', ['started_at'])


def downgrade():
    # Drop tables
    op.drop_table('sync_logs')
    op.drop_table('oe_orders')
    op.drop_table('pos_orders')
