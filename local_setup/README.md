# 🏠 Local Development Setup

This guide helps you set up a local environment that mirrors the AWS production pipeline.

## 📦 S3 Backup Location

**Bucket:** `s3://mastergroup-db-backups-303498144074/database-exports/`

Latest backup: `20251209` (December 9, 2025)
- 2,391,151 total rows
- 49 MB compressed

## ✅ Local Database Status

Your local PostgreSQL already has the data:

| Table | Rows |
|-------|------|
| `offline_user_recommendations` | 180,483 |
| `offline_similar_items` | 4,182 |
| `orders` | 234,959 |
| `order_items` | 1,971,527 |

**Connection:** `localhost:5432` / `mastergroup_recommendations`

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL (native, port 5432)
- AWS CLI configured (`aws configure`)

## 🚀 Quick Start

### 1. Start Local Services

```bash
cd local_setup
docker-compose up -d
```

This starts:
- **PostgreSQL 16** on port `5433`
- **Redis** on port `6379`
- **API Server** on port `8001`

### 2. Restore Data from S3 Backup (Recommended)

```bash
# Install dependencies
pip install psycopg2-binary boto3

# Restore from latest S3 backup
python restore_from_s3.py

# Or restore specific date
python restore_from_s3.py --date 20251209

# Or restore specific tables only
python restore_from_s3.py --tables offline_user_recommendations,orders
```

### Alternative: Sync Directly from Production

```bash
# Sync all tables directly from production (slower)
python sync_from_production.py

# Or sync specific tables with row limit
python sync_from_production.py --tables orders,order_items,offline_user_recommendations --limit 50000
```

### 3. Verify Setup

```bash
# Check database
psql -h localhost -p 5433 -U postgres -d mastergroup_recommendations -c "SELECT COUNT(*) FROM offline_user_recommendations;"

# Test API
curl http://localhost:8001/health
curl http://localhost:8001/api/v1/personalize/recommendations/03068667596_mr%20latif
```

## 📁 File Structure

```
local_setup/
├── docker-compose.yml      # Docker services configuration
├── Dockerfile              # API container definition
├── .env.local              # Local environment variables
├── sync_from_production.py # Data sync script
├── init-scripts/           # Database initialization scripts
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables

Copy `.env.local` to your project root as `.env`:

```bash
cp local_setup/.env.local .env
```

Key variables:
| Variable | Description | Default |
|----------|-------------|---------|
| `PG_HOST` | Local PostgreSQL host | `localhost` |
| `PG_PORT` | Local PostgreSQL port | `5433` |
| `USE_LOCAL_ML` | Use local ML models | `true` |

## 🔄 Data Sync Options

### Full Sync (All Tables)
```bash
python sync_from_production.py
```

### Partial Sync (Specific Tables)
```bash
python sync_from_production.py --tables orders,order_items,products
```

### Limited Sync (For Testing)
```bash
python sync_from_production.py --limit 10000
```

### Essential Tables Only
```bash
python sync_from_production.py --tables offline_user_recommendations,offline_similar_items,orders,order_items,products
```

## 🗄️ Database Access

### Local Database
```bash
# Connect via psql
psql -h localhost -p 5433 -U postgres -d mastergroup_recommendations

# Password: MasterGroup2024Local!
```

### Production Database (Read-Only for Sync)
```bash
psql -h ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com \
     -p 5432 -U postgres -d mastergroup_recommendations

# Password: MasterGroup2024Secure!
```

## 🔌 API Endpoints

Same endpoints as production:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /api/v1/personalize/recommendations/{user_id}` | User recommendations |
| `GET /api/v1/personalize/recommendations/similar/{product_id}` | Similar products |
| `GET /api/v1/ml/product-pairs` | Product pairs for cross-selling |
| `GET /api/v1/analytics/dashboard` | Dashboard metrics |

## 🐳 Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Reset database (delete all data)
docker-compose down -v
docker-compose up -d
```

## 🔍 Troubleshooting

### Database Connection Failed
```bash
# Check if PostgreSQL is running
docker-compose ps

# Check logs
docker-compose logs postgres
```

### Sync Script Fails
```bash
# Ensure you can connect to production
psql -h ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com \
     -p 5432 -U postgres -d mastergroup_recommendations -c "SELECT 1"
```

### API Not Starting
```bash
# Check API logs
docker-compose logs api

# Restart API
docker-compose restart api
```

## 📊 Production vs Local Comparison

| Feature | Production (AWS) | Local (Docker) |
|---------|------------------|----------------|
| Database | Lightsail PostgreSQL | Docker PostgreSQL |
| Port | 5432 | 5433 |
| ML Models | Cached from AWS Personalize | Same cached data |
| API | Lightsail Instance | Docker Container |
| Cost | ~$50/month | Free |

## 🔐 Security Notes

- Local passwords are different from production
- Never commit `.env` files with real credentials
- Production credentials are in `.env.local` for sync only

## 📝 Maintenance

### Update Local Data
```bash
# Re-sync from production
python sync_from_production.py
```

### Backup Local Database
```bash
docker exec mastergroup-postgres-local pg_dump -U postgres mastergroup_recommendations > backup.sql
```

### Restore Local Database
```bash
docker exec -i mastergroup-postgres-local psql -U postgres mastergroup_recommendations < backup.sql
```
