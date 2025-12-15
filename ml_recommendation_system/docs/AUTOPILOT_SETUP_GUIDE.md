# Auto-Pilot Setup Guide

**ML Recommendation System - Automated Training Pipeline**

---

## Overview

The Auto-Pilot system automatically retrains the ML model monthly with fresh data, validates performance, and deploys improved models without manual intervention.

---

## System Architecture

```
┌──────────────┐
│  Scheduler   │ Monthly trigger (1st at 2 AM)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  AUTO-PILOT ORCHESTRATOR                                  │
├──────────────────────────────────────────────────────────┤
│  Step 1: Data Fetcher                                     │
│    - Fetch from API (last_sync to yesterday)             │
│    - INSERT into PostgreSQL                               │
│    - Update sync_logs table                               │
│                                                            │
│  Step 2: Pipeline Runner                                  │
│    - Query all data from PostgreSQL                       │
│    - Clean customer IDs                                   │
│    - Process & save to CSV                                │
│                                                            │
│  Step 3: Model Trainer                                    │
│    - Train new model                                      │
│    - Save to data/models/candidate/                       │
│                                                            │
│  Step 4: Model Validator                                  │
│    - Compare with production model                        │
│    - Decision: Deploy or Reject                           │
│                                                            │
│  Step 5: Model Deployer                                   │
│    - Archive old model                                    │
│    - Deploy new model                                     │
│    - Restart API (optional)                               │
│                                                            │
│  Step 6: Notifier                                         │
│    - Send email with results                              │
│    - Include metrics & logs                               │
└──────────────────────────────────────────────────────────┘
```

---

## Database Setup

### 1. Create PostgreSQL Database

```bash
# Create database
createdb ml_recommendations

# Or using psql
psql -U postgres
CREATE DATABASE ml_recommendations;
\q
```

### 2. Configure Environment Variables

Create `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit with your details
DATABASE_URL=postgresql://username:password@localhost:5432/ml_recommendations
POS_API_URL=https://api.masterverse.com/v1/pos-orders
OE_API_URL=https://api.masterverse.com/v1/oe-orders
API_AUTH_TOKEN=Bearer your_token_here
```

### 3. Run Migrations

```bash
# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

This creates tables:
- `pos_orders` - POS order data
- `oe_orders` - OE order data  
- `sync_logs` - Sync history tracking

---

## Auto-Pilot Components

### Component Files

```
src/autopilot/
├── __init__.py
├── data_fetcher.py       # Fetch from API (incremental)
├── pipeline_runner.py    # Run data pipeline
├── model_comparator.py   # Compare models
├── model_deployer.py     # Deploy new model
├── notifier.py           # Email notifications
├── scheduler.py          # Monthly scheduling
└── orchestrator.py       # Main auto-pilot logic
```

---

## Data Fetching Strategy (Incremental Sync)

### First Run:
```sql
-- No data in database
-- Fetch: 2000-01-01 to yesterday (2024-12-06)
-- INSERT 223,806 orders
-- UPDATE sync_logs: last_sync_date = 2024-12-06
```

### Subsequent Runs (Monthly):
```sql
-- Query sync_logs for last_sync_date
SELECT MAX(sync_end_date) FROM sync_logs WHERE status = 'success';
-- Result: 2024-12-06

-- Fetch: 2024-12-07 to yesterday (2025-01-06)
-- INSERT ~10,000 new orders
-- UPDATE sync_logs: last_sync_date = 2025-01-06
```

### Query for Training:
```sql
-- Get all orders for training
SELECT * FROM pos_orders WHERE order_date >= '2021-01-01'
UNION ALL
SELECT * FROM oe_orders WHERE order_date >= '2021-01-01'
ORDER BY order_date;
```

---

## Scheduler Configuration

### Monthly Schedule (Recommended):
```python
import schedule

# Run on 1st of every month at 2 AM
schedule.every().month.at("02:00").do(run_autopilot)

# Keep running
while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

### Alternative: Cron Job
```bash
# Edit crontab
crontab -e

# Add: Run 1st of month at 2 AM
0 2 1 * * /path/to/venv/bin/python /path/to/run_autopilot.py >> /var/log/ml_autopilot.log 2>&1
```

---

## Safety Features

### 1. Model Validation Before Deployment

```python
def should_deploy_new_model(new_metrics, old_metrics):
    """
    Deploy only if new model is better
    
    Criteria:
    - Precision improved by 5%+ OR
    - Hit rate improved by 10%+ OR
    - Both metrics didn't decrease
    """
    new_precision = new_metrics['precision@10']
    old_precision = old_metrics['precision@10']
    
    new_hit_rate = new_metrics['hit_rate']
    old_hit_rate = old_metrics['hit_rate']
    
    # Check improvements
    precision_improved = new_precision >= old_precision * 1.05
    hit_rate_improved = new_hit_rate >= old_hit_rate * 1.10
    
    # Check no degradation
    precision_ok = new_precision >= old_precision * 0.95
    hit_rate_ok = new_hit_rate >= old_hit_rate * 0.90
    
    if (precision_improved or hit_rate_improved) and precision_ok and hit_rate_ok:
        return True, "Model improved"
    else:
        return False, f"Model not better (P: {new_precision:.4f} vs {old_precision:.4f}, HR: {new_hit_rate:.4f} vs {old_hit_rate:.4f})"
```

### 2. Model Archiving

```python
# Before deploying new model, archive current
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
shutil.move(
    'data/models/production',
    f'data/models/archive/model_{timestamp}'
)
```

### 3. Rollback Capability

```bash
# If new model has issues, rollback
python scripts/rollback_model.py --to=20241207_020000
```

---

## Email Notifications

### Success Email:
```
Subject: ✅ ML Training SUCCESS - Model Deployed

Auto-Pilot Training Report
Date: 2025-01-07 02:35:42

Status: ✅ SUCCESS - New model deployed

Data Statistics:
- Orders fetched: 12,450 new orders
- Total orders in DB: 236,256
- Training data: 185,000 interactions

Model Performance:
- Precision@10: 0.0125 (1.25%) [+0.29% improvement]
- Hit Rate: 0.1150 (11.50%) [+1.85% improvement]

Action Taken: Deployed new model to production
Old model archived to: data/models/archive/model_20241207

Training Duration: 32 minutes
```

### Failure Email:
```
Subject: ⚠️ ML Training FAILED

Auto-Pilot Training Report
Date: 2025-01-07 02:15:23

Status: ❌ FAILED

Error: API connection timeout after 300 seconds

Action Taken: Kept current production model

Please investigate and retry manually.

Logs attached.
```

---

## Running Auto-Pilot

### Manual Run (Testing):
```bash
python run_autopilot.py
```

### Scheduled Run (Production):
```bash
# Start scheduler (runs in background)
nohup python run_autopilot_scheduler.py &

# Or use systemd service
sudo systemctl start ml-autopilot
sudo systemctl enable ml-autopilot
```

---

## Monitoring

### Check Sync History:
```sql
SELECT * FROM sync_logs ORDER BY started_at DESC LIMIT 10;
```

### Check Database Stats:
```python
from src.database import OrderRepository, get_db

with get_db() as db:
    repo = OrderRepository(db)
    stats = repo.get_database_stats()
    print(stats)
```

---

## Next Steps to Complete

1. ✅ PostgreSQL models created
2. ✅ Repository layer implemented
3. ✅ Alembic initialized
4. 🔄 Create initial migration
5. 🔄 Implement auto-pilot components
6. 🔄 Create setup scripts
7. 🔄 Test end-to-end

**Ready to continue implementation!**
