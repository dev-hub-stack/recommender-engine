"""populate_customer_segments

Revision ID: d32509426ade
Revises: c53c94db0229
Create Date: 2025-11-25 18:48:05.704734

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd32509426ade'
down_revision: Union[str, Sequence[str], None] = 'c53c94db0229'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Update customer_statistics with RFM segments
    # Calculate recency, frequency, and monetary values, then assign segments
    op.execute("""
        UPDATE customer_statistics cs
        SET customer_segment = calculate_rfm_segment(
            EXTRACT(DAY FROM (CURRENT_DATE - cs.last_order_date))::INTEGER,
            cs.total_orders,
            cs.total_spent
        )
        WHERE cs.last_order_date IS NOT NULL
    """)
    
    # Log segment distribution
    result = op.get_bind().execute(sa.text("""
        SELECT 
            customer_segment,
            COUNT(*) as customer_count,
            ROUND(AVG(total_spent), 2) as avg_spent,
            ROUND(AVG(total_orders), 1) as avg_orders
        FROM customer_statistics
        WHERE customer_segment IS NOT NULL
        GROUP BY customer_segment
        ORDER BY customer_count DESC
    """))
    
    print("\n✓ Customer segments populated:")
    print(f"{'Segment':<25} {'Customers':>12} {'Avg Spent (PKR)':>15} {'Avg Orders':>12}")
    print("-" * 70)
    total_customers = 0
    for row in result:
        print(f"{row.customer_segment:<25} {row.customer_count:>12,} {row.avg_spent:>15,.0f} {row.avg_orders:>12}")
        total_customers += row.customer_count
    print("-" * 70)
    print(f"{'TOTAL':<25} {total_customers:>12,}")


def downgrade() -> None:
    """Downgrade schema."""
    # Clear all customer segments
    op.execute("""
        UPDATE customer_statistics
        SET customer_segment = NULL
    """)
