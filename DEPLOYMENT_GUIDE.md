# Recommendation Engine Service - Deployment Guide

## Overview
This guide ensures complete deployment of the Recommendation Engine Service with all required dependencies and database setup.

## Prerequisites

### 1. System Requirements
- Python 3.8+
- PostgreSQL 12+
- Redis (optional, for caching)
- Git

### 2. Database Requirements
- PostgreSQL database with connection permissions
- Access to the main `orders` table (should contain order data with `items_json` field)

## Deployment Steps

### Step 1: Clone and Setup Repository
```bash
git clone <repository-url>
cd recommendation-engine-service
```

### Step 2: Environment Setup
1. Create Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Database Configuration
1. Copy environment configuration:
```bash
cp ../.env.example .env
```

2. Edit `.env` file with your database settings:
```env
# Database Configuration
DATABASE_URL=postgresql://username:password@host:port/database_name

# Or use individual components:
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mastergroup_recommendations
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
```

### Step 4: Database Setup and Verification
1. Run the database verification script:
```bash
python verify_database_setup.py
```

2. If tables are missing, the script will offer to run the setup automatically, or you can run manually:
```bash
psql -d your_database -f setup_recommendation_tables.sql
```

3. Verify the setup completed successfully:
```bash
python verify_database_setup.py
```

### Step 5: Data Population
After database setup, populate the recommendation tables:

```sql
-- Connect to your PostgreSQL database and run:
SELECT populate_order_items_from_orders();
SELECT rebuild_customer_purchases();
SELECT rebuild_product_pairs();
SELECT rebuild_product_statistics();
SELECT rebuild_customer_statistics();
```

### Step 6: Service Testing
1. Test the service configuration:
```bash
python -c "from src.main import app; print('Service imports successfully')"
```

2. Run performance tests (optional):
```bash
python test_complete_system.py
```

## Required Database Tables

The following tables will be created by `setup_recommendation_tables.sql`:

### Core Tables
1. **order_items** - Extracted product items from orders
2. **customer_purchases** - Aggregated customer purchase history
3. **product_pairs** - Co-purchase relationships for cross-selling
4. **product_statistics** - Product popularity and performance metrics
5. **customer_statistics** - Customer behavior and segmentation data
6. **recommendation_cache** - Caching layer for recommendations

### Required Functions
1. `populate_order_items_from_orders()` - Extract items from orders.items_json
2. `rebuild_customer_purchases()` - Build customer purchase history
3. `rebuild_product_pairs()` - Calculate product co-purchase relationships  
4. `rebuild_product_statistics()` - Calculate product performance metrics
5. `rebuild_customer_statistics()` - Build customer profiles and segments

## Environment Variables Reference

### Required Variables
```env
DATABASE_URL=postgresql://user:password@host:port/dbname
```

### Optional Variables
```env
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
DEBUG=false
ENVIRONMENT=production
```

## Verification Checklist

### ✅ Pre-deployment Verification
- [ ] Python 3.8+ installed
- [ ] PostgreSQL accessible
- [ ] Environment variables configured
- [ ] Dependencies installed (`pip install -r requirements.txt`)

### ✅ Database Verification
- [ ] Database connection successful
- [ ] All 6 tables created
- [ ] All 5 functions created
- [ ] Sample data populated
- [ ] Indexes created for performance

### ✅ Service Verification
- [ ] Service imports without errors
- [ ] API endpoints responding (if applicable)
- [ ] Logs generating correctly
- [ ] Cache working (if Redis configured)

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check database is running
sudo systemctl status postgresql

# Test connection manually
psql -h hostname -U username -d database_name
```

#### 2. Missing Tables
```bash
# Run setup script manually
python verify_database_setup.py
# Or directly with psql
psql -d your_database -f setup_recommendation_tables.sql
```

#### 3. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 4. No Data in Recommendation Tables
```sql
-- Check if orders table has data
SELECT COUNT(*) FROM orders WHERE items_json IS NOT NULL;

-- Populate recommendation tables
SELECT populate_order_items_from_orders();
SELECT rebuild_customer_purchases();
```

## Performance Optimization

### Database Indexes
All required indexes are created by the setup script:
- Order items: order_id, product_id
- Customer purchases: customer_id, product_id, purchase_count
- Product pairs: product_1, product_2, co_purchase_count
- Statistics tables: various performance indexes

### Recommended Settings
- Connection pooling: 10-20 connections
- Redis for caching (optional but recommended)
- Regular data refresh (daily/weekly depending on volume)

## Maintenance

### Regular Tasks
1. **Data Refresh**: Run rebuild functions daily/weekly
```bash
# Create a cron job for regular updates
0 2 * * * /path/to/update_recommendations.sh
```

2. **Performance Monitoring**: Monitor query performance and optimize as needed

3. **Backup**: Regular backup of recommendation tables

## Support

For issues during deployment:
1. Check logs in `logs/` directory
2. Run `python verify_database_setup.py` for diagnostics
3. Verify environment variables are correctly set
4. Check database permissions and connectivity

## Security Notes
- Never commit `.env` files to version control
- Use strong database passwords
- Restrict database access to necessary IPs only
- Consider SSL connections for production databases
