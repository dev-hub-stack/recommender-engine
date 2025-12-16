# Local ML Recommendation System

## Overview

This system **replaces AWS Personalize** with locally trained ML models. It uses the same data flow and stores recommendations in the same cache tables, so existing API endpoints work without modification.

## Architecture Comparison

| Component | AWS Personalize | Local ML |
|-----------|-----------------|----------|
| **Data Source** | Master APIs → PostgreSQL | Master APIs → PostgreSQL |
| **Storage** | S3 + AWS Personalize | PostgreSQL + Local Files |
| **Models** | AWS-proprietary | SVD, Item Similarity, Popularity |
| **Training Time** | 4-6 hours | 30-40 minutes |
| **Cost** | ~$7.50/month | $0/month |
| **Cache Tables** | `offline_user_recommendations`, `offline_similar_items` | Same tables! |
| **API Endpoints** | `/api/v1/personalize/*` | Same endpoints (serve from cache) |

## Pipeline Steps

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL ML PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: SYNC DATA                                              │
│  ├── Fetch from Master OE API                                  │
│  ├── Fetch from Master POS API                                 │
│  └── Insert/Update PostgreSQL                                  │
│                                                                 │
│  STEP 2: EXPORT TO CSV                                          │
│  ├── interactions.csv (USER_ID, ITEM_ID, TIMESTAMP)            │
│  ├── items.csv (ITEM_ID, ITEM_NAME, PRICE)                     │
│  └── users.csv (USER_ID, CITY, ORDER_COUNT)                    │
│                                                                 │
│  STEP 3: TRAIN MODELS                                           │
│  ├── SVD Collaborative Filtering                               │
│  ├── Item-Item Cosine Similarity                               │
│  └── Popularity (recency-weighted)                             │
│                                                                 │
│  STEP 4: GENERATE & CACHE                                       │
│  ├── Generate recommendations for ALL users                    │
│  ├── Generate similar items for ALL products                   │
│  ├── Save to offline_user_recommendations                      │
│  └── Save to offline_similar_items                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Run Full Pipeline

```bash
cd recommendation-engine-service

# Full pipeline: Sync → Export → Train → Cache
python3 scripts/local_ml_pipeline.py

# Skip API sync (use existing DB data)
python3 scripts/local_ml_pipeline.py --skip-sync

# Only export to CSV
python3 scripts/local_ml_pipeline.py --export-only

# Only train (skip sync & export)
python3 scripts/local_ml_pipeline.py --train-only
```

### 2. Set Up Automated Schedule

```bash
# Install cron jobs (daily at 2 AM, weekly full on Sunday at 3 AM)
chmod +x scripts/cron_setup.sh
./scripts/cron_setup.sh install

# Check status
./scripts/cron_setup.sh status

# Remove cron jobs
./scripts/cron_setup.sh remove
```

### 3. Verify Cache

```bash
# Check user recommendations
psql -c "SELECT COUNT(*) FROM offline_user_recommendations"

# Check similar items
psql -c "SELECT COUNT(*) FROM offline_similar_items"

# Sample recommendation
psql -c "SELECT user_id, recommendations->>0 as top_rec FROM offline_user_recommendations LIMIT 1"
```

## File Structure

```
recommendation-engine-service/
├── scripts/
│   ├── local_ml_pipeline.py    # Main pipeline script
│   ├── train_and_cache.py      # Standalone training script
│   └── cron_setup.sh           # Cron job installer
├── data/
│   └── local_ml/
│       ├── interactions_latest.csv
│       ├── items_latest.csv
│       ├── users_latest.csv
│       └── pipeline_log_*.json
├── models/                      # Trained model files
└── docs/
    └── LOCAL_ML_SETUP.md        # This file
```

## API Endpoints (Unchanged)

The existing endpoints automatically serve from the local ML cache:

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/personalize/recommendations/{user_id}` | User recommendations |
| `GET /api/v1/personalize/recommendations/similar/{product_id}` | Similar items |
| `GET /api/v1/personalize/recommendations/by-location` | Location-based |
| `GET /api/v1/personalize/recommendations/by-segment` | Segment-based |

## Model Details

### 1. SVD Collaborative Filtering
- Uses Surprise library
- 50 latent factors
- Predicts user preferences for unseen items

### 2. Item-Item Similarity
- Cosine similarity on user-item matrix
- Returns most similar products based on purchase patterns

### 3. Popularity (Recency-Weighted)
- Exponential decay: `exp(-days_ago / 30)`
- Fallback for cold-start users

## Monitoring

### Pipeline Logs
```bash
# Check latest log
cat data/local_ml/pipeline_log_*.json | jq .

# Check cron logs
tail -f logs/pipeline.log
```

### Cache Stats
```sql
-- User recommendations
SELECT 
    recipe_name,
    COUNT(*) as users,
    MAX(updated_at) as last_updated
FROM offline_user_recommendations
GROUP BY recipe_name;

-- Similar items
SELECT 
    recipe_name,
    COUNT(*) as products,
    MAX(updated_at) as last_updated
FROM offline_similar_items
GROUP BY recipe_name;
```

## Migration from AWS Personalize

1. **No code changes needed** - Same cache tables, same API endpoints
2. **Run pipeline once** - Populates cache with local ML results
3. **Set up cron** - Automated daily/weekly retraining
4. **Disable AWS Personalize** - Stop the AWS batch jobs
5. **Cost savings** - $7.50/month → $0/month

## Troubleshooting

### Pipeline fails at "Sync from APIs"
```bash
# Skip sync and use existing DB
python3 scripts/local_ml_pipeline.py --skip-sync
```

### Out of memory during training
```bash
# Reduce training data window
python3 scripts/local_ml_pipeline.py --sync-days 90
```

### Cache tables empty
```sql
-- Check if tables exist
\dt offline_*

-- Create if missing
CREATE TABLE IF NOT EXISTS offline_user_recommendations (
    user_id TEXT,
    recommendations JSONB,
    recipe_name TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS offline_similar_items (
    product_id TEXT,
    similar_products JSONB,
    recipe_name TEXT,
    updated_at TEXT
);
```
