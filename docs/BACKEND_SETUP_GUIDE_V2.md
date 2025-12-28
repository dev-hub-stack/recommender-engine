# Backend Setup Guide (EC2 & Local)
*Updated: December 28, 2025*

This guide covers the complete step-by-step setup of the MasterGroup Recommendation Engine backend, including database initialization, ML pipeline training, API deployment, and performance optimizations.

## 1. Prerequisites

- **Python 3.10+**
- **PostgreSQL 14+**
- **Redis 6+**
- **Git**

## 2. Initial Setup

### Clone Repository
```bash
git clone https://github.com/dev-hub-stack/recommender-engine.git
cd recommender-engine
```

### Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure Environment Variables
Copy the example environment file and edit it with your credentials:
```bash
cp .env.example .env
nano .env
```

**Critical Variables:**
```ini
# Database
PG_HOST=localhost
PG_PORT=5432
PG_DB=mastergroup_recommendations
PG_USER=postgres
PG_PASSWORD=your_password

# Master Group API (For Data Sync)
MASTER_GROUP_API_BASE=https://mes.master.com.pk
MASTER_GROUP_AUTH_TOKEN=your_token

# ML Settings
USE_LOCAL_ML=true
```

## 3. Database Setup

We have a unified script that handles schema creation, migrations, and seeding.

**Run the setup script:**
```bash
python3 scripts/setup_database.py
```

This will:
1. Check database connection.
2. Run **Alembic migrations** (including offline recommendation tables).
3. Populate auxiliary tables.
4. Seed the admin user (`admin@mastergroup.com` / `MG@2024#Secure!Pass`).

## 4. Training Pipeline Setup

The ML pipeline fetches data, trains models (SVD, Similarity, Popularity), and caches recommendations.

### Run Full Historical Sync (First Time)
To fetch 4 years of data (approx 1500 days) and train models:
```bash
python3 scripts/local_ml_pipeline.py --sync-days 1500
```
*Note: This runs in 90-day batches to prevent API timeouts.*

### Run Standard Sync (Daily Update)
For daily updates, sync only the last 7 days:
```bash
python3 scripts/local_ml_pipeline.py --sync-days 7
```

### Training Only (No Sync)
If you already have data and just want to retrain models:
```bash
python3 scripts/local_ml_pipeline.py --train-only
```

### Verify Pipeline Success
Check the logs or database:
```bash
tail -f logs/ml_pipeline.log
```

## 5. Running the API

### Development
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### Production (EC2)
Use `systemd` to keep the service running.

1. **Create Service File:** `/etc/systemd/system/mastergroup-api.service`
```ini
[Unit]
Description=MasterGroup API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/mastergroup-ml
Environment="PATH=/opt/mastergroup-ml/venv/bin"
ExecStart=/opt/mastergroup-ml/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

2. **Start Service:**
```bash
sudo systemctl enable mastergroup-api
sudo systemctl start mastergroup-api
```

## 6. Automation (Cron Jobs)

Set up automatic data syncing and retraining.

Edit crontab:
```bash
crontab -e
```

Add these lines:
```bash
# 1. Daily ML Pipeline (Sync & Train) at 2:00 AM
0 2 * * * cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python3 scripts/local_ml_pipeline.py --sync-days 2 >> /opt/mastergroup-ml/logs/pipeline.log 2>&1

# 2. Frequent Data Sync (Orders only) every 4 hours
0 */4 * * * cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python3 services/sync_service.py >> /opt/mastergroup-ml/logs/sync.log 2>&1
```

## 6. Performance Optimizations

### 6.1 Database Optimizations

#### Fix Duplicate Order Items (Critical!)
The `order_items` table requires a unique constraint to prevent duplicates:

```bash
ssh ubuntu@your-ec2-ip "cd /opt/mastergroup-ml && source venv/bin/activate && python3 << 'EOF'
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv('.env')

conn = psycopg2.connect(
    host=os.getenv('PG_HOST'),
    port=int(os.getenv('PG_PORT', 5432)),
    database=os.getenv('PG_DB'),
    user=os.getenv('PG_USER'),
    password=os.getenv('PG_PASSWORD'),
    sslmode='require'
)
cur = conn.cursor()

print('Removing duplicate order_items...')
# Keep only the row with highest quantity
cur.execute('''
    DELETE FROM order_items a
    USING order_items b
    WHERE a.order_id = b.order_id
    AND a.product_id = b.product_id
    AND (a.quantity < b.quantity OR (a.quantity = b.quantity AND a.id < b.id));
''')
print(f'Deleted {cur.rowcount} duplicates')

print('Creating unique index...')
cur.execute('DROP INDEX IF EXISTS idx_order_items_order_product;')
cur.execute('CREATE UNIQUE INDEX idx_order_items_order_product ON order_items(order_id, product_id);')
conn.commit()
print('✅ Done!')
cur.close()
conn.close()
EOF
"
```

**Why This Matters:**
- Prevents `ON CONFLICT` errors during data sync
- Reduces database size (can save 50%+ storage)
- Improves query performance

### 6.2 Redis Cache Configuration

#### Increase Cache TTL for Heavy Queries
The "all" time filter queries scan the entire database. Configure longer cache TTL:

**In `src/main.py`:**
```python
# For "all" time filter: 2 hours cache (7200 seconds)
# For other filters: 5 minutes cache (300 seconds)
ttl = 7200 if time_filter == "all" else 300
```

#### Pre-warm Cache for Common Queries
Run this script after each data sync or training to pre-populate the cache:

```python
#!/usr/bin/env python3
"""Pre-warm Redis cache for heavy queries"""
import os
import sys
sys.path.insert(0, '/opt/mastergroup-ml')
from dotenv import load_dotenv
load_dotenv('/opt/mastergroup-ml/.env')

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime, timedelta

# Connect
r = redis.Redis(host='localhost', port=6379, db=0)
conn = psycopg2.connect(
    host=os.getenv('PG_HOST'),
    port=int(os.getenv('PG_PORT')),
    database=os.getenv('PG_DB'),
    user=os.getenv('PG_USER'),
    password=os.getenv('PG_PASSWORD'),
    sslmode='require'
)
cursor = conn.cursor(cursor_factory=RealDictCursor)

print('1. Caching dashboard metrics for ALL TIME...')
cursor.execute("""
    SELECT 
        COUNT(DISTINCT o.id) as total_orders,
        COUNT(DISTINCT o.unified_customer_id) as total_customers,
        SUM(o.total_price) as total_revenue,
        AVG(o.total_price) as avg_order_value
    FROM orders o
""")
result = cursor.fetchone()

dashboard_data = {
    "success": True,
    "total_orders": result["total_orders"] or 0,
    "total_customers": result["total_customers"] or 0,
    "total_revenue": float(result["total_revenue"] or 0),
    "avg_order_value": float(result["avg_order_value"] or 0),
    "cached": True,
    "timestamp": datetime.now().isoformat()
}
r.setex("analytics:dashboard:all:all", 7200, json.dumps(dashboard_data))
print(f'   Orders: {dashboard_data["total_orders"]:,}, Customers: {dashboard_data["total_customers"]:,}')

print('2. Caching popular products for ALL TIME...')
cursor.execute("""
    SELECT 
        oi.product_id,
        oi.product_name,
        COUNT(DISTINCT oi.order_id) as order_count,
        SUM(oi.quantity) as total_quantity,
        AVG(oi.unit_price) as avg_price
    FROM order_items oi
    GROUP BY oi.product_id, oi.product_name
    ORDER BY order_count DESC
    LIMIT 30
""")
products = cursor.fetchall()
products_data = {
    "success": True,
    "products": [dict(p) for p in products],
    "cached": True
}
for p in products_data["products"]:
    p["avg_price"] = float(p["avg_price"] or 0)
r.setex("popular_products:30:all:all", 7200, json.dumps(products_data))
print(f'   Cached {len(products)} products')

cursor.close()
conn.close()
print('✅ Cache pre-warming complete!')
```

**Save as `scripts/prewarm_cache.py` and run after training:**
```bash
python3 scripts/prewarm_cache.py
```

### 6.3 Database Connection Settings

#### Set Statement Timeout (Already Configured)
In `src/main.py`, all DB connections have a 30-second timeout:

```python
params = {
    'host': PG_HOST,
    'port': PG_PORT,
    'database': PG_DB,
    'user': PG_USER,
    'password': PG_PASSWORD,
    'options': '-c statement_timeout=30000'  # 30 seconds
}
```

This prevents hung queries from blocking the server.

### 6.4 Time Filter Options

#### Support Multiple Date Ranges
The backend now supports these filters:
- `today` - Current day only
- `7days` - Last 7 days  
- `30days` - Last 30 days
- `mtd` - Month to date
- `90days` - Last 90 days
- `6months` - Last 6 months
- `1year` - Last 1 year
- `2years` - Last 2 years (fast, recommended)
- `3years` - Last 3 years
- `all` - **ALL data** (slower, 2-hour cache)

**Frontend Implementation:**
```tsx
<select value={timeFilter} onChange={(e) => setTimeFilter(e.target.value)}>
  <option value="today">Today</option>
  <option value="7days">Last 7 Days</option>
  <option value="30days">Last 30 Days</option>
  <option value="mtd">Month to Date</option>
  <option value="90days">Last 3 Months</option>
  <option value="6months">Last 6 Months</option>
  <option value="1year">Last 1 Year</option>
  <option value="2years">Last 2 Years</option>
  <option value="3years">Last 3 Years</option>
  <option value="all">All Time (Slower)</option>
</select>
```

### 6.5 Monitoring & Debugging

#### Check Cache Hit Rate
```bash
redis-cli INFO stats | grep -E 'keyspace_hits|keyspace_misses'
```

**Ideal ratio:** 80%+ cache hits

#### View Cached Keys
```bash
redis-cli KEYS '*' | grep -i all
```

#### Clear Cache (If Needed)
```bash
redis-cli FLUSHDB
```

Then re-run `scripts/prewarm_cache.py`

### 6.6 Cron Job for Cache Pre-warming

Add to crontab to pre-warm cache after daily sync:

```bash
crontab -e
```

Add:
```bash
# Pre-warm cache after daily ML sync (2:30 AM UTC)
30 2 * * * cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python scripts/prewarm_cache.py >> /opt/mastergroup-ml/logs/cache_prewarm.log 2>&1
```

## 7. Troubleshooting

### Issue: Server Unresponsive with "all" Filter
**Cause:** Heavy queries with no cache

**Solution:**
1. Ensure Redis is running: `redis-cli ping`
2. Pre-warm cache: `python3 scripts/prewarm_cache.py`
3. Check cache TTL is 7200 seconds (2 hours)
4. Use `2years` filter instead of `all` for better performance

### Issue: Login Returns 401
**Cause:** Wrong database connection or missing DATABASE_URL

**Solution:**
```bash
# Ensure .env has correct PG_* variables
PG_HOST=your-rds-endpoint.rds.amazonaws.com
PG_PASSWORD=your_actual_password

# Restart API
sudo systemctl restart mastergroup-api
```

### Issue: Data Sync Fails with ON CONFLICT Error
**Cause:** Missing unique constraint on order_items

**Solution:** Run Section 6.1 (Database Optimizations)

### Issue: High Memory Usage
**Cause:** Too many concurrent requests with heavy queries

**Solution:**
1. Increase cache TTL for "all" queries
2. Add request rate limiting in nginx/edge functions
3. Scale to larger EC2 instance (t3.medium → t3.large)

## 8. Performance Benchmarks

After optimizations:

| Metric | Before | After |
|--------|--------|-------|
| Dashboard load (all filter) | 25-30s | 0.5-2s (cached) |
| Cache hit rate | 10% | 85%+ |
| Database queries per request | 5-10 | 0-1 |
| API response time (cached) | 5-10s | <200ms |
| Server stability | Crashes on load | Stable |
