"""Seed admin user for dashboard access

Revision ID: 004_seed_admin_user
Revises: 003_stored_procedures
Create Date: 2025-11-29

This migration seeds the default admin user for dashboard access.
IMPORTANT: Change the password after first login in production!

Default Credentials:
- Email: admin@mastergroup.com
- Password: MG@2024#Secure!Pass
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '004_seed_admin_user'
down_revision: Union[str, None] = '003_stored_procedures'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Password: MG@2024#Secure!Pass (bcrypt hashed)
    op.execute("""
        INSERT INTO users (email, password_hash, full_name, is_active) 
        VALUES (
            'admin@mastergroup.com',
            '$2b$12$8ALlBQw1UrHePD2QyYRy0uGz/mMEOsay4HzCwvPjMt8nOmGlQ/8MO',
            'Admin User',
            true
        )
        ON CONFLICT (email) DO UPDATE SET
            password_hash = EXCLUDED.password_hash,
            full_name = EXCLUDED.full_name,
            is_active = EXCLUDED.is_active;
    """)


def downgrade() -> None:
    op.execute("DELETE FROM users WHERE email = 'admin@mastergroup.com';")
