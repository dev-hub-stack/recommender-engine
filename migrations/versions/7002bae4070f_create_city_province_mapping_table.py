"""create_city_province_mapping_table

Revision ID: 7002bae4070f
Revises: 35576334970d
Create Date: 2025-11-25 18:46:03.629142

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7002bae4070f'
down_revision: Union[str, Sequence[str], None] = '35576334970d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create city_province_mapping lookup table
    op.create_table(
        'city_province_mapping',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('city', sa.String(100), nullable=False, unique=True),
        sa.Column('province', sa.String(50), nullable=False),
        sa.Column('region', sa.String(50), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP'))
    )
    
    # Create index for fast city lookups
    op.create_index('idx_city_mapping_city', 'city_province_mapping', ['city'])
    
    # Insert Pakistani city mappings
    # Punjab (Central Pakistan)
    cities_punjab = [
        ('Lahore', 'Punjab', 'Central'),
        ('Faisalabad', 'Punjab', 'Central'),
        ('Rawalpindi', 'Punjab', 'North'),
        ('Multan', 'Punjab', 'South'),
        ('Gujranwala', 'Punjab', 'Central'),
        ('Sialkot', 'Punjab', 'North'),
        ('Bahawalpur', 'Punjab', 'South'),
        ('Sargodha', 'Punjab', 'Central'),
        ('Sahiwal', 'Punjab', 'Central'),
        ('Jhang', 'Punjab', 'Central'),
        ('Sheikhupura', 'Punjab', 'Central'),
        ('Gujrat', 'Punjab', 'North'),
        ('Kasur', 'Punjab', 'Central'),
        ('Rahim Yar Khan', 'Punjab', 'South'),
        ('Okara', 'Punjab', 'Central'),
        ('Mianwali', 'Punjab', 'North'),
        ('Chiniot', 'Punjab', 'Central'),
        ('Jhelum', 'Punjab', 'North'),
        ('Attock', 'Punjab', 'North'),
        ('Chakwal', 'Punjab', 'North'),
    ]
    
    # Sindh (South Pakistan)
    cities_sindh = [
        ('Karachi', 'Sindh', 'South'),
        ('Hyderabad', 'Sindh', 'South'),
        ('Sukkur', 'Sindh', 'North'),
        ('Larkana', 'Sindh', 'North'),
        ('Nawabshah', 'Sindh', 'Central'),
        ('Mirpur Khas', 'Sindh', 'South'),
        ('Jacobabad', 'Sindh', 'North'),
        ('Shikarpur', 'Sindh', 'North'),
        ('Khairpur', 'Sindh', 'North'),
        ('Dadu', 'Sindh', 'Central'),
        ('Badin', 'Sindh', 'South'),
        ('Thatta', 'Sindh', 'South'),
        ('Tando Allahyar', 'Sindh', 'South'),
        ('Sanghar', 'Sindh', 'Central'),
    ]
    
    # Khyber Pakhtunkhwa (North Pakistan)
    cities_kpk = [
        ('Peshawar', 'Khyber Pakhtunkhwa', 'North'),
        ('Mardan', 'Khyber Pakhtunkhwa', 'North'),
        ('Abbottabad', 'Khyber Pakhtunkhwa', 'North'),
        ('Mingora', 'Khyber Pakhtunkhwa', 'North'),
        ('Kohat', 'Khyber Pakhtunkhwa', 'Central'),
        ('Dera Ismail Khan', 'Khyber Pakhtunkhwa', 'South'),
        ('Mansehra', 'Khyber Pakhtunkhwa', 'North'),
        ('Swabi', 'Khyber Pakhtunkhwa', 'North'),
        ('Charsadda', 'Khyber Pakhtunkhwa', 'North'),
        ('Nowshera', 'Khyber Pakhtunkhwa', 'North'),
        ('Haripur', 'Khyber Pakhtunkhwa', 'North'),
    ]
    
    # Balochistan (West Pakistan)
    cities_balochistan = [
        ('Quetta', 'Balochistan', 'West'),
        ('Turbat', 'Balochistan', 'South'),
        ('Gwadar', 'Balochistan', 'South'),
        ('Khuzdar', 'Balochistan', 'Central'),
        ('Sibi', 'Balochistan', 'East'),
        ('Zhob', 'Balochistan', 'North'),
        ('Hub', 'Balochistan', 'East'),
        ('Loralai', 'Balochistan', 'North'),
    ]
    
    # Islamabad Capital Territory
    cities_ict = [
        ('Islamabad', 'Islamabad Capital Territory', 'North'),
    ]
    
    # Azad Kashmir
    cities_azad_kashmir = [
        ('Muzaffarabad', 'Azad Kashmir', 'North'),
        ('Mirpur', 'Azad Kashmir', 'North'),
        ('Kotli', 'Azad Kashmir', 'North'),
        ('Bhimber', 'Azad Kashmir', 'North'),
    ]
    
    # Gilgit-Baltistan
    cities_gilgit = [
        ('Gilgit', 'Gilgit-Baltistan', 'North'),
        ('Skardu', 'Gilgit-Baltistan', 'North'),
        ('Hunza', 'Gilgit-Baltistan', 'North'),
    ]
    
    # Combine all cities
    all_cities = (cities_punjab + cities_sindh + cities_kpk + 
                  cities_balochistan + cities_ict + cities_azad_kashmir + cities_gilgit)
    
    # Insert data using individual statements
    for city, province, region in all_cities:
        op.execute(
            f"INSERT INTO city_province_mapping (city, province, region) "
            f"VALUES ('{city}', '{province}', '{region}')"
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_city_mapping_city', 'city_province_mapping')
    op.drop_table('city_province_mapping')
