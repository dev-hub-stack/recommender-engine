"""AWS Personalize offline recommendation tables

Revision ID: 002_aws_personalize
Revises: 001_initial_schema
Create Date: 2025-11-29

This migration creates tables for storing AWS Personalize batch inference results:
- offline_user_recommendations: User personalization results
- offline_similar_items: Similar items results
- offline_item_affinity: Item affinity results
- city_province_mapping: Geographic data mapping
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_aws_personalize'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ============================================
    # OFFLINE USER RECOMMENDATIONS
    # ============================================
    op.create_table(
        'offline_user_recommendations',
        sa.Column('user_id', sa.String(255), primary_key=True),
        sa.Column('recommendations', postgresql.JSONB),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now())
    )
    
    op.create_index('idx_offline_user_recs_updated', 'offline_user_recommendations', ['updated_at'])
    
    # ============================================
    # OFFLINE SIMILAR ITEMS
    # ============================================
    op.create_table(
        'offline_similar_items',
        sa.Column('product_id', sa.String(255), primary_key=True),
        sa.Column('similar_products', postgresql.JSONB),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now())
    )
    
    op.create_index('idx_offline_similar_items_updated', 'offline_similar_items', ['updated_at'])
    
    # ============================================
    # OFFLINE ITEM AFFINITY
    # ============================================
    op.create_table(
        'offline_item_affinity',
        sa.Column('user_id', sa.String(255), primary_key=True),
        sa.Column('affinity_products', postgresql.JSONB),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now())
    )
    
    op.create_index('idx_offline_item_affinity_updated', 'offline_item_affinity', ['updated_at'])
    
    # ============================================
    # OFFLINE PERSONALIZED RANKING
    # ============================================
    op.create_table(
        'offline_personalized_ranking',
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('category', sa.String(255), nullable=False),
        sa.Column('ranked_products', postgresql.JSONB),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('user_id', 'category')
    )
    
    # ============================================
    # CITY-PROVINCE MAPPING
    # ============================================
    op.create_table(
        'city_province_mapping',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('city', sa.String(100), nullable=False),
        sa.Column('province', sa.String(100), nullable=False),
        sa.Column('country', sa.String(100), default='Pakistan'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('city', 'province', name='uq_city_province')
    )
    
    op.create_index('idx_city_province_city', 'city_province_mapping', ['city'])
    op.create_index('idx_city_province_province', 'city_province_mapping', ['province'])


def downgrade() -> None:
    op.drop_table('city_province_mapping')
    op.drop_table('offline_personalized_ranking')
    op.drop_table('offline_item_affinity')
    op.drop_table('offline_similar_items')
    op.drop_table('offline_user_recommendations')
