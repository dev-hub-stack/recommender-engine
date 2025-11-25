"""populate_orders_geographic_data

Revision ID: fcaa65189ae8
Revises: 7002bae4070f
Create Date: 2025-11-25 18:46:53.582306

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fcaa65189ae8'
down_revision: Union[str, Sequence[str], None] = '7002bae4070f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Update orders table with province and region based on customer_city
    # Using city_province_mapping lookup table
    op.execute("""
        UPDATE orders o
        SET 
            province = cpm.province,
            region = cpm.region
        FROM city_province_mapping cpm
        WHERE LOWER(TRIM(o.customer_city)) = LOWER(cpm.city)
    """)
    
    # Log how many orders were updated
    result = op.get_bind().execute(sa.text("""
        SELECT 
            COUNT(*) as total_orders,
            COUNT(province) as orders_with_province,
            COUNT(DISTINCT province) as unique_provinces,
            COUNT(DISTINCT region) as unique_regions
        FROM orders
    """))
    
    for row in result:
        print(f"✓ Geographic data populated:")
        print(f"  Total orders: {row.total_orders:,}")
        print(f"  Orders with province: {row.orders_with_province:,}")
        print(f"  Unique provinces: {row.unique_provinces}")
        print(f"  Unique regions: {row.unique_regions}")


def downgrade() -> None:
    """Downgrade schema."""
    # Clear province and region data
    op.execute("""
        UPDATE orders
        SET province = NULL, region = NULL
    """)
