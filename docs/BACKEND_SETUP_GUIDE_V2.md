# Backend Setup Guide (EC2 & Local)
*Updated: December 17, 2025*

This guide covers the complete step-by-step setup of the MasterGroup Recommendation Engine backend, including database initialization, ML pipeline training, and API deployment.

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

## 7. Troubleshooting

**Database Migrations Failed?**
Run Alembic manually to see errors:
```bash
alembic upgrade head
```

**Pipeline API Timeouts?**
The pipeline uses batching, but if it fails, try reducing the batch size in `scripts/local_ml_pipeline.py`.

**SVD Model Missing?**
Ensure `scikit-surprise` is installed:
```bash
pip install scikit-surprise
```
