"""create_rfm_segmentation_function

Revision ID: c53c94db0229
Revises: fcaa65189ae8
Create Date: 2025-11-25 18:47:23.124948

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c53c94db0229'
down_revision: Union[str, Sequence[str], None] = 'fcaa65189ae8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create function to calculate RFM segment for a customer
    op.execute("""
        CREATE OR REPLACE FUNCTION calculate_rfm_segment(
            recency_days INTEGER,
            frequency_count INTEGER,
            monetary_value NUMERIC
        ) RETURNS VARCHAR(50) AS $$
        DECLARE
            r_score INTEGER;
            f_score INTEGER;
            m_score INTEGER;
            segment VARCHAR(50);
        BEGIN
            -- Calculate Recency Score (lower days = better)
            IF recency_days <= 30 THEN r_score := 5;
            ELSIF recency_days <= 60 THEN r_score := 4;
            ELSIF recency_days <= 90 THEN r_score := 3;
            ELSIF recency_days <= 180 THEN r_score := 2;
            ELSE r_score := 1;
            END IF;
            
            -- Calculate Frequency Score
            IF frequency_count >= 20 THEN f_score := 5;
            ELSIF frequency_count >= 10 THEN f_score := 4;
            ELSIF frequency_count >= 5 THEN f_score := 3;
            ELSIF frequency_count >= 2 THEN f_score := 2;
            ELSE f_score := 1;
            END IF;
            
            -- Calculate Monetary Score (PKR)
            IF monetary_value >= 500000 THEN m_score := 5;
            ELSIF monetary_value >= 200000 THEN m_score := 4;
            ELSIF monetary_value >= 100000 THEN m_score := 3;
            ELSIF monetary_value >= 50000 THEN m_score := 2;
            ELSE m_score := 1;
            END IF;
            
            -- Determine Segment based on RFM scores
            IF r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN
                segment := 'Champions';
            ELSIF r_score >= 3 AND f_score >= 4 AND m_score >= 4 THEN
                segment := 'Loyal Customers';
            ELSIF r_score >= 4 AND f_score <= 2 AND m_score >= 3 THEN
                segment := 'Potential Loyalists';
            ELSIF r_score >= 4 AND f_score <= 2 AND m_score <= 2 THEN
                segment := 'New Customers';
            ELSIF r_score = 3 AND f_score >= 3 AND m_score >= 3 THEN
                segment := 'Need Attention';
            ELSIF r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN
                segment := 'At Risk';
            ELSIF r_score <= 2 AND f_score >= 4 AND m_score >= 4 THEN
                segment := 'Cannot Lose Them';
            ELSIF r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN
                segment := 'Lost';
            ELSE
                segment := 'Hibernating';
            END IF;
            
            RETURN segment;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    
    print("✓ Created calculate_rfm_segment() function")


def downgrade() -> None:
    """Downgrade schema."""
    # Drop the RFM segmentation function
    op.execute("DROP FUNCTION IF EXISTS calculate_rfm_segment(INTEGER, INTEGER, NUMERIC)")
