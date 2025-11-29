# Database Setup Guide

This guide explains how to set up the database for the MasterGroup Recommendation System.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/dev-hub-stack/recommender-engine.git
cd recommender-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your database credentials

# 5. Run database setup
python scripts/setup_database.py
```

## Environment Variables

Create a `.env` file with the following variables:

```env
# Database Configuration (Option 1: Individual variables)
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=mastergroup_recommendations
PG_USER=postgres
PG_PASSWORD=your_password
PG_SSLMODE=prefer

# OR Database URL (Option 2)
DATABASE_URL=postgresql://user:password@host:5432/database

# API Configuration
MASTERGROUP_API_KEY=your_api_key
MASTERGROUP_API_URL=https://api.mastergroup.com

# AWS Personalize (Optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
```

## Migration System

The project uses [Alembic](https://alembic.sqlalchemy.org/) for database migrations.

### Migration Files

```
migrations/versions/
├── 001_initial_schema.py       # Core tables (orders, users, etc.)
├── 002_aws_personalize_tables.py  # AWS Personalize cache tables
├── 003_stored_procedures.py    # Stored procedures for data processing
├── 004_seed_admin_user.py      # Default admin user
└── ... (existing migrations)
```

### Running Migrations Manually

```bash
# Apply all migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Check current version
alembic current

# Show migration history
alembic history
```

## Database Schema

### Core Tables

| Table | Description |
|-------|-------------|
| `orders` | Source order data from POS/OE systems |
| `order_items` | Individual items from each order |
| `users` | Dashboard authentication |
| `sync_metadata` | Data sync tracking |

### Recommendation Tables

| Table | Description |
|-------|-------------|
| `customer_purchases` | Aggregated purchase history |
| `product_pairs` | Co-purchase relationships |
| `product_statistics` | Product popularity metrics |
| `customer_statistics` | Customer analytics |
| `recommendation_cache` | Performance caching |

### AWS Personalize Tables

| Table | Description |
|-------|-------------|
| `offline_user_recommendations` | Cached user recommendations |
| `offline_similar_items` | Cached similar items |
| `offline_item_affinity` | Cached item affinity |
| `city_province_mapping` | Geographic data mapping |

## Stored Procedures

After migrations run, these functions are available:

```sql
-- Populate order_items from orders.items_json
SELECT populate_order_items_from_orders();

-- Rebuild recommendation tables
SELECT rebuild_customer_purchases();
SELECT rebuild_product_pairs();
SELECT rebuild_product_statistics();
SELECT rebuild_customer_statistics();
```

## Default Admin Credentials

After setup, login with:

- **Email:** admin@mastergroup.com
- **Password:** MG@2024#Secure!Pass

⚠️ **IMPORTANT:** Change the password after first login!

## Troubleshooting

### Migration Errors

```bash
# Reset to fresh state (CAUTION: Drops all data!)
alembic downgrade base
alembic upgrade head
```

### Connection Issues

```bash
# Test database connection
python -c "from scripts.setup_database import connect_db; connect_db(); print('OK')"
```

### Rebuild Tables

```bash
# Connect to database and run:
psql -U postgres -d mastergroup_recommendations

# Rebuild all tables
SELECT populate_order_items_from_orders();
SELECT rebuild_customer_purchases();
SELECT rebuild_product_pairs();
SELECT rebuild_product_statistics();
SELECT rebuild_customer_statistics();
```

## Legacy SQL Files

The following SQL files are kept for reference but are now superseded by migrations:

```
sql/legacy/
├── deploy_database.sql
├── create_correct_orders_table.sql
├── setup_recommendation_tables.sql
├── optimize_product_pairs.sql
├── seed_users.sql
└── aws_personalize/offline_recommendations.sql
```

These files are no longer needed as all schema changes are managed through Alembic migrations.
