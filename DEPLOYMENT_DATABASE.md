# Database Deployment Guide

## Overview

This guide explains how to set up the complete database schema for the Master Group Recommendation System in production.

## Files

1. **deploy_database.sql** - Complete database schema (all 9 tables)
2. **seed_users.sql** - Seeds the default admin user
3. **setup_recommendation_tables.sql** - Original setup file (updated with users table)

## Quick Deployment

### Option 1: Complete Setup (Recommended)

```bash
# 1. Create database
createdb -U postgres mastergroup_recommendations

# 2. Deploy all tables
psql -U postgres -d mastergroup_recommendations -f deploy_database.sql

# 3. Seed admin user
psql -U postgres -d mastergroup_recommendations -f seed_users.sql
```

### Option 2: Step by Step

```bash
# 1. Create database
createdb -U postgres mastergroup_recommendations

# 2. Create tables
psql -U postgres -d mastergroup_recommendations -f setup_recommendation_tables.sql

# 3. Create orders table (if not exists)
psql -U postgres -d mastergroup_recommendations << 'SQL'
CREATE TABLE IF NOT EXISTS orders (
    id VARCHAR(100) PRIMARY KEY,
    customer_id VARCHAR(100),
    unified_customer_id VARCHAR(100),
    customer_name VARCHAR(255),
    customer_phone VARCHAR(50),
    customer_city VARCHAR(100),
    customer_address TEXT,
    customer_email VARCHAR(255),
    order_date TIMESTAMP NOT NULL,
    total_amount DECIMAL(12, 2),
    total_price DECIMAL(12, 2),
    order_type VARCHAR(20) DEFAULT 'POS',
    payment_mode VARCHAR(50),
    brand_name VARCHAR(100),
    order_status VARCHAR(50),
    order_name VARCHAR(255),
    items_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    synced_at TIMESTAMP
);
SQL

# 4. Seed users
psql -U postgres -d mastergroup_recommendations -f seed_users.sql
```

## Tables Created

### Core Tables
1. **orders** - Main orders table (source data from POS/OE)
2. **order_items** - Individual items from each order
3. **users** - Authentication for dashboard access

### Recommendation Tables
4. **customer_purchases** - Aggregated purchase history
5. **product_pairs** - Frequently bought together
6. **product_statistics** - Product popularity metrics
7. **customer_statistics** - Customer analytics
8. **recommendation_cache** - Performance cache

### System Tables
9. **sync_metadata** - Data synchronization tracking

## Default Admin User

After running `seed_users.sql`:

```
Email: admin@mastergroup.com
Password: MG@2024#Secure!Pass
```

**⚠️ IMPORTANT:** Change this password after first login in production!

## Verify Installation

```sql
-- Check all tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check admin user
SELECT id, email, full_name, is_active, created_at 
FROM users;

-- Check table counts
SELECT 
    (SELECT COUNT(*) FROM orders) as orders,
    (SELECT COUNT(*) FROM order_items) as order_items,
    (SELECT COUNT(*) FROM users) as users,
    (SELECT COUNT(*) FROM customer_purchases) as customer_purchases,
    (SELECT COUNT(*) FROM product_statistics) as product_statistics;
```

## Production Checklist

- [ ] Database created
- [ ] All 9 tables created
- [ ] Admin user seeded
- [ ] Admin password changed
- [ ] Database backups configured
- [ ] Connection pooling configured
- [ ] SSL/TLS enabled for database connections
- [ ] Environment variables set in .env
- [ ] API sync configured
- [ ] Test login working

## Changing Admin Password

### Method 1: Using Python

```python
import bcrypt

# Your new password
new_password = "YourNewSecurePassword123!"

# Generate hash
hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
print(f"Hashed: {hashed.decode()}")
```

Then update database:

```sql
UPDATE users 
SET password_hash = 'YOUR_HASHED_PASSWORD_HERE'
WHERE email = 'admin@mastergroup.com';
```

### Method 2: Using psql and Python inline

```bash
NEW_HASH=$(python3 -c "import bcrypt; print(bcrypt.hashpw(b'NewPassword123!', bcrypt.gensalt()).decode())")
psql -U postgres -d mastergroup_recommendations -c "UPDATE users SET password_hash = '$NEW_HASH' WHERE email = 'admin@mastergroup.com';"
```

## Adding More Users

```sql
-- Generate password hash first using Python
-- Then insert:

INSERT INTO users (email, password_hash, full_name, is_active)
VALUES (
    'user@mastergroup.com',
    '$2b$12$...your_hash_here...',
    'User Full Name',
    true
);
```

## Database Maintenance

### Rebuild Recommendation Tables

```sql
-- After new orders are synced, rebuild analytics:
SELECT rebuild_customer_purchases();
SELECT rebuild_product_pairs();
SELECT rebuild_product_statistics();
SELECT rebuild_customer_statistics();
```

### Clear Cache

```sql
-- Clear expired cache entries
DELETE FROM recommendation_cache 
WHERE expires_at < NOW();

-- Clear all cache
TRUNCATE recommendation_cache;
```

### View Sync History

```sql
SELECT 
    sync_type,
    last_sync_timestamp,
    orders_synced,
    sync_status,
    error_message
FROM sync_metadata
ORDER BY last_sync_timestamp DESC
LIMIT 10;
```

## Backup and Restore

### Backup

```bash
# Full database backup
pg_dump -U postgres mastergroup_recommendations > backup_$(date +%Y%m%d).sql

# Schema only
pg_dump -U postgres --schema-only mastergroup_recommendations > schema_backup.sql

# Data only
pg_dump -U postgres --data-only mastergroup_recommendations > data_backup.sql
```

### Restore

```bash
# Restore full backup
psql -U postgres -d mastergroup_recommendations < backup_20241119.sql

# Restore schema
psql -U postgres -d mastergroup_recommendations < schema_backup.sql

# Restore data
psql -U postgres -d mastergroup_recommendations < data_backup.sql
```

## Troubleshooting

### Tables Not Created

```sql
-- Check for errors
\dt

-- Recreate specific table
DROP TABLE IF EXISTS users CASCADE;
-- Then run deploy_database.sql again
```

### User Cannot Login

```sql
-- Check user exists and is active
SELECT * FROM users WHERE email = 'admin@mastergroup.com';

-- Reset password
UPDATE users 
SET password_hash = '$2b$12$8ALlBQw1UrHePD2QyYRy0uGz/mMEOsay4HzCwvPjMt8nOmGlQ/8MO'
WHERE email = 'admin@mastergroup.com';
```

### Performance Issues

```sql
-- Check missing indexes
SELECT schemaname, tablename, indexname
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename;

-- Analyze tables
ANALYZE orders;
ANALYZE order_items;
ANALYZE customer_purchases;
```

## Security Best Practices

1. ✅ Use strong passwords (16+ characters)
2. ✅ Change default password immediately
3. ✅ Enable SSL for database connections
4. ✅ Use connection pooling
5. ✅ Regular backups
6. ✅ Monitor failed login attempts
7. ✅ Keep database software updated
8. ✅ Restrict database access by IP
9. ✅ Use environment variables for credentials
10. ✅ Enable audit logging

## Support

For issues or questions:
1. Check logs: `tail -f /var/log/postgresql/postgresql.log`
2. Verify environment variables in `.env`
3. Test database connection: `psql -U postgres -d mastergroup_recommendations`
4. Check application logs for errors

---

**Last Updated**: November 19, 2025
**Version**: 1.0.0
