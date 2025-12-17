"""create offline recommendation tables

Revision ID: 8f2a1b9c3d4e
Revises: a1b2c3d4e5f6
Create Date: 2025-12-17 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '8f2a1b9c3d4e'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    # offline_user_recommendations
    op.create_table('offline_user_recommendations',
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('recommendations', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('recipe_name', sa.String(length=100), server_default='aws-user-personalization', nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('user_id')
    )
    op.create_index('idx_offline_user_recs_updated', 'offline_user_recommendations', ['updated_at'], unique=False)

    # offline_similar_items
    op.create_table('offline_similar_items',
        sa.Column('product_id', sa.String(length=255), nullable=False),
        sa.Column('similar_products', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('recipe_name', sa.String(length=100), server_default='aws-similar-items', nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('product_id')
    )
    op.create_index('idx_offline_similar_items_updated', 'offline_similar_items', ['updated_at'], unique=False)

    # offline_item_affinity
    op.create_table('offline_item_affinity',
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('item_affinities', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('recipe_name', sa.String(length=100), server_default='aws-item-affinity', nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('user_id')
    )
    op.create_index('idx_offline_item_affinity_updated', 'offline_item_affinity', ['updated_at'], unique=False)

    # offline_personalized_ranking
    op.create_table('offline_personalized_ranking',
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('input_items', sa.Text(), nullable=False),
        sa.Column('ranked_products', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('recipe_name', sa.String(length=100), server_default='aws-personalized-ranking', nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('user_id', 'input_items')
    )

    # batch_job_metadata
    op.create_table('batch_job_metadata',
        sa.Column('job_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('job_name', sa.String(length=255), nullable=False),
        sa.Column('job_arn', sa.String(length=512), nullable=True),
        sa.Column('recipe_name', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=50), server_default='PENDING', nullable=True),
        sa.Column('s3_input_path', sa.Text(), nullable=True),
        sa.Column('s3_output_path', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('job_id'),
        sa.UniqueConstraint('job_name')
    )
    op.create_index('idx_batch_jobs_status', 'batch_job_metadata', ['status', 'started_at'], unique=False)


def downgrade():
    op.drop_index('idx_batch_jobs_status', table_name='batch_job_metadata')
    op.drop_table('batch_job_metadata')
    op.drop_table('offline_personalized_ranking')
    op.drop_index('idx_offline_item_affinity_updated', table_name='offline_item_affinity')
    op.drop_table('offline_item_affinity')
    op.drop_index('idx_offline_similar_items_updated', table_name='offline_similar_items')
    op.drop_table('offline_similar_items')
    op.drop_index('idx_offline_user_recs_updated', table_name='offline_user_recommendations')
    op.drop_table('offline_user_recommendations')
