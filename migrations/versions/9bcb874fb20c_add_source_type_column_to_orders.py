"""add_source_type_column_to_orders

Revision ID: 9bcb874fb20c
Revises: a1b2c3d4e5f6
Create Date: 2025-11-26 00:03:22.334009

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9bcb874fb20c'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add source_type column to orders table
    op.add_column('orders', sa.Column('source_type', sa.String(10), nullable=True))
    
    # Create index for source_type queries
    op.create_index('idx_orders_source_type', 'orders', ['source_type'])
    
    # Update existing records - set source_type based on order_type
    # POS orders have order_type='POS', OE orders have order_type='OE'
    op.execute("""
        UPDATE orders 
        SET source_type = CASE 
            WHEN order_type = 'POS' THEN 'POS'
            WHEN order_type = 'OE' THEN 'OE'
            ELSE 'UNKNOWN'
        END
        WHERE source_type IS NULL
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop index
    op.drop_index('idx_orders_source_type', 'orders')
    
    # Drop column
    op.drop_column('orders', 'source_type')
