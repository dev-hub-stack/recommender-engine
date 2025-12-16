# MasterGroup Recommendation Engine - Complete Backend Setup Guide

## Overview

This guide covers the complete setup of the MasterGroup Recommendation Engine backend service, including database setup, ML pipeline configuration, API deployment, and Shopify integration.

**System Status (as of Dec 16, 2025):**

| Component | Status | Details |
|-----------|--------|---------|
| PostgreSQL | ✅ Active | 234,959 orders, 1,971,527 order items |
| Redis Cache | ✅ Active | Running on localhost:6379 |
| ML Models | ✅ Trained | 74,827 users, 4,182 products |
| API Server | ✅ Running | Port 8001 |
| AWS Personalize | ❌ Disabled | Replaced by local ML (saves $170/month) |

---

## Prerequisites

### System Requirements
- **Python:** 3.10+
- **PostgreSQL:** 14+
- **Redis:** 6+
- **Disk Space:** 5GB+ for models and data

### Required Python Packages
```bash
pip install -r requirements.txt
```

Key packages:
- `fastapi`, `uvicorn` - Web framework
- `psycopg2-binary` - PostgreSQL driver
- `redis` - Cache layer
- `scikit-learn`, `scikit-surprise` - ML algorithms
- `pandas`, `numpy` - Data processing

---

## Step 1: Database Setup

### Local PostgreSQL Setup

```bash
# Create database
createdb mastergroup_recommendations

# Set password (if not localhost)
export PGPASSWORD='your_password'

# Run migrations
cd recommendation-engine-service
psql -h localhost -U postgres -d mastergroup_recommendations -f migrations/schema.sql
```

### Database Schema

| Table | Records | Purpose |
|-------|---------|---------|
| `orders` | 234,959 | Customer orders (POS + OE) |
| `order_items` | 1,971,527 | Order line items |
| `offline_user_recommendations` | 74,827 | Cached user recommendations |
| `offline_similar_items` | 4,182 | Similar products cache |
| `product_statistics` | 3,793 | Product analytics |
| `product_pairs` | varies | Co-purchase patterns |
| `sync_metadata` | varies | API sync tracking |

### Verify Database

```bash
psql -h localhost -U postgres -d mastergroup_recommendations -c "
SELECT 'orders' as table_name, COUNT(*) FROM orders
UNION ALL SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL SELECT 'offline_user_recommendations', COUNT(*) FROM offline_user_recommendations
UNION ALL SELECT 'offline_similar_items', COUNT(*) FROM offline_similar_items;
"
```

---

## Step 2: Environment Configuration

### Create `.env` File

```bash
# Copy template
cp .env.example .env
```

### Required Environment Variables

```bash
# ===========================================
# PostgreSQL Configuration
# ===========================================
PG_HOST=localhost
PG_PORT=5432
PG_DB=mastergroup_recommendations
PG_USER=postgres
PG_PASSWORD=

# ===========================================
# Redis Configuration
# ===========================================
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ===========================================
# Master Group API Configuration
# ===========================================
MASTER_GROUP_API_BASE=https://mes.master.com.pk
MASTER_GROUP_POS_ENDPOINT=/get_pos_orders
MASTER_GROUP_OE_ENDPOINT=/get_oe_orders
MASTER_GROUP_AUTH_TOKEN=

# ===========================================
# Sync Configuration
# ===========================================
SYNC_POS_ORDERS=true
SYNC_OE_ORDERS=true
SYNC_INTERVAL_MINUTES=360
ENABLE_AUTO_SYNC=true

# ===========================================
# ML Configuration
# ===========================================
USE_LOCAL_ML=true
ML_MODEL_PATH=./models

# ===========================================
# Shopify Configuration (Optional)
# ===========================================
SHOPIFY_STORE=masterverse-project.myshopify.com
SHOPIFY_API_KEY=
SHOPIFY_API_SECRET=
SHOPIFY_ACCESS_TOKEN=
SHOPIFY_API_VERSION=2024-01
```

---

## Step 3: Data Sync from Master APIs

### Data Flow

```
Master OE API ─┬─► sync_service.py ─► PostgreSQL
               │
Master POS API ─┘    (every 6 hours)
```

### Manual Sync

```bash
# Full sync from Master APIs
python3 services/sync_service.py

# Check sync status
psql -c "SELECT sync_type, last_sync_timestamp, orders_synced FROM sync_metadata ORDER BY last_sync_timestamp DESC LIMIT 5;"
```

### Automated Sync (Cron)

```bash
# Current schedule: Every 6 hours
0 */6 * * * cd /path/to/recommendation-engine-service && python3 services/sync_service.py >> logs/sync.log 2>&1
```

---

## Step 4: ML Pipeline Setup

### Pipeline Overview

```
STEP 1: Sync Data ──► STEP 2: Export CSV ──► STEP 3: Train Models ──► STEP 4: Cache Results
  (Optional)          interactions.csv         SVD, Similarity          PostgreSQL tables
                      items.csv, users.csv     Popularity model         74,827 user recs
                                                                        4,182 similar items
```

### Run ML Pipeline

```bash
# Full pipeline (sync + export + train + cache)
python3 scripts/local_ml_pipeline.py

# Skip sync (use existing data)
python3 scripts/local_ml_pipeline.py --skip-sync

# Only train
python3 scripts/local_ml_pipeline.py --train-only
```

### Expected Output

```
STEP 1: SYNC DATA (Optional - use existing database)
STEP 2: EXPORT DATA TO CSV
   ✅ Exported 1,971,527 interactions
   ✅ Exported 4,182 items
   ✅ Exported 180,483 users
STEP 3: TRAIN LOCAL ML MODELS
   ✅ SVD trained (74,827 users, 4,182 items)
STEP 4: GENERATE BATCH RECOMMENDATIONS & CACHE
   ✅ Generated recommendations for 74,827 users
   ✅ Generated similar items for 4,182 products
PIPELINE COMPLETE ✅
   Duration: ~77 minutes
```

### Model Files Location

```
models/
├── svd_recommender_latest.joblib      # SVD model (~1.3MB)
├── item_similarity_latest.joblib      # Similarity matrix (~116KB)
├── popularity_scores_latest.joblib    # Popularity scores (~26KB)
└── metadata.json                       # Model metadata
```

### Scheduled Training (Cron)

```bash
# Daily training at 2:00 AM
0 2 * * * cd /path/to/recommendation-engine-service && python3 scripts/local_ml_pipeline.py --skip-sync >> logs/ml_pipeline.log 2>&1
```

---

## Step 5: Start API Server

### Development Mode

```bash
cd recommendation-engine-service

# Start server
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### Production Mode

```bash
# Using gunicorn
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001

# Or systemd service
sudo systemctl start mastergroup-api
sudo systemctl enable mastergroup-api
```

### Verify API

```bash
# Health check
curl http://localhost:8001/health

# Test recommendations
curl "http://localhost:8001/api/v1/recommendations/popular?limit=5"

# Test ML recommendations
curl "http://localhost:8001/api/v1/personalize/recommendations/03001234567_john"
```

---

## Step 6: Shopify Integration (Optional)

### Prerequisites
- Shopify store access
- Admin API access token

### Configure Shopify

```bash
# Add to .env
SHOPIFY_STORE=masterverse-project.myshopify.com
SHOPIFY_ACCESS_TOKEN=shpat_xxx
```

### Available Shopify Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/shopify/recommendations` | Unified recommendations |
| `GET /api/v1/shopify/similar/{product_id}` | Similar products |
| `GET /api/v1/shopify/popular` | Popular by location |
| `GET /api/v1/shopify/products` | List Shopify products |

### Test Shopify Integration

```bash
# Get recommendations for customer
curl -X POST "http://localhost:8001/api/v1/shopify/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"customer_phone": "03001234567", "city": "Lahore", "limit": 5}'

# Get similar products
curl "http://localhost:8001/api/v1/shopify/similar/1328?limit=5"
```

See [SHOPIFY_INTEGRATION.md](./SHOPIFY_INTEGRATION.md) for full integration guide.

---

## API Endpoints Reference

### Core Recommendation Endpoints

```
GET  /api/v1/recommendations/popular
GET  /api/v1/recommendations/collaborative/{customer_id}
GET  /api/v1/recommendations/similar/{product_id}
POST /api/v1/recommendations/for-user
```

### Personalize-Compatible Endpoints (from ML cache)

```
GET  /api/v1/personalize/recommendations/{user_id}
GET  /api/v1/personalize/recommendations/similar/{product_id}
GET  /api/v1/personalize/recommendations/by-location
GET  /api/v1/personalize/recommendations/by-segment
```

### Analytics Endpoints

```
GET  /api/v1/analytics/dashboard
GET  /api/v1/analytics/product-categories
GET  /api/v1/analytics/brands/performance
GET  /api/v1/analytics/customer/distribution
```

### Health & Status

```
GET  /health
GET  /api/v1/stats
GET  /api/v1/cache/stats
```

---

## Monitoring & Maintenance

### Check System Status

```bash
# API health
curl http://localhost:8001/health

# Database status
psql -c "SELECT COUNT(*) FROM orders WHERE order_date > NOW() - INTERVAL '24 hours';"

# Cache stats
curl http://localhost:8001/api/v1/cache/stats
```

### Logs Location

```
logs/
├── api.log          # API server logs
├── sync.log         # Data sync logs
└── ml_pipeline.log  # ML training logs
```

### Clear Cache (if needed)

```bash
# Clear Redis cache
redis-cli FLUSHDB

# Clear specific cache key
redis-cli DEL "popular_products:*"
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Database connection failed | Check PG_HOST, PG_PASSWORD in .env |
| Redis connection failed | Verify Redis is running: `redis-cli ping` |
| No recommendations | Run ML pipeline: `python3 scripts/local_ml_pipeline.py` |
| API 500 errors | Check logs: `tail -f logs/api.log` |
| Categories not filtering | Restart server after code changes |

### Reset and Rebuild

```bash
# Clear all caches
redis-cli FLUSHALL

# Re-run ML pipeline
python3 scripts/local_ml_pipeline.py --skip-sync

# Restart API
sudo systemctl restart mastergroup-api
```

---

## Cost Summary

| Item | AWS Personalize | Local ML | Savings |
|------|-----------------|----------|---------|
| ML Inference | $144/month | $0 | $144/month |
| Batch Jobs | $25/month | $0 | $25/month |
| Storage | $2/month | $0 | $2/month |
| **Total** | **$171/month** | **$0** | **$171/month** |

**Annual Savings: ~$2,052**

---

## Next Steps for Production

1. [ ] Deploy to production server (Lightsail/EC2)
2. [ ] Configure SSL/HTTPS
3. [ ] Set up monitoring (CloudWatch/Prometheus)
4. [ ] Configure automated backups
5. [ ] Set up CI/CD pipeline
6. [ ] Configure Shopify webhooks for real-time sync

---

## Support Files

- [LOCAL_ML_SETUP.md](./LOCAL_ML_SETUP.md) - ML pipeline details
- [SHOPIFY_INTEGRATION.md](./SHOPIFY_INTEGRATION.md) - Shopify setup
- [AWS_PERSONALIZE_SHUTDOWN.md](./AWS_PERSONALIZE_SHUTDOWN.md) - AWS migration notes
- [../custom_ml/CAPABILITY_IMPLEMENTATION.md](../custom_ml/CAPABILITY_IMPLEMENTATION.md) - Algorithm details
