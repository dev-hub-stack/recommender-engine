"""add_geographic_columns_to_orders

Revision ID: 35576334970d
Revises: 
Create Date: 2025-11-25 18:45:27.075776

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '35576334970d'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add province and region columns to orders table
    op.add_column('orders', sa.Column('province', sa.String(50), nullable=True))
    op.add_column('orders', sa.Column('region', sa.String(50), nullable=True))
    
    # Create indexes for geographic queries
    op.create_index('idx_orders_province', 'orders', ['province'])
    op.create_index('idx_orders_region', 'orders', ['region'])
    op.create_index('idx_orders_province_date', 'orders', ['province', 'order_date'])
    op.create_index('idx_orders_region_date', 'orders', ['region', 'order_date'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('idx_orders_region_date', 'orders')
    op.drop_index('idx_orders_province_date', 'orders')
    op.drop_index('idx_orders_region', 'orders')
    op.drop_index('idx_orders_province', 'orders')
    
    # Drop columns
    op.drop_column('orders', 'region')
    op.drop_column('orders', 'province')
