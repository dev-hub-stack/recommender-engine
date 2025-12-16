# 🔄 Local ML Recommendation Pipeline

> **Last Updated:** December 16, 2025  
> **Status:** ✅ ACTIVE - Replaced AWS Personalize  
> **Pipeline Last Run:** Dec 16, 2025 (77 minutes, 74,827 users)

## Overview

This document describes the recommendation pipeline that replaces AWS Personalize with a local ML solution. The pipeline was successfully run on Dec 16, 2025, generating recommendations for 74,827 users and 4,182 products.

## Current Statistics

| Metric | Value |
|--------|-------|
| **Total Users with Recommendations** | 74,827 |
| **Total Products with Similar Items** | 4,182 |
| **Total Interactions Processed** | 1,971,527 |
| **Pipeline Duration** | ~77 minutes |
| **Cost** | $0/month (vs $170/month AWS Personalize) |

## 📊 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐         ┌──────────────┐                                 │
│   │   OE API     │         │   POS API    │                                 │
│   │ (E-commerce) │         │(Point of Sale)│                                │
│   └──────┬───────┘         └──────┬───────┘                                 │
│          │                        │                                          │
│          └────────────┬───────────┘                                          │
│                       ▼                                                      │
│              ┌────────────────┐                                              │
│              │  Data Sync Job │  ← Runs every 6 hours                       │
│              │  (sync_data.py)│                                              │
│              └────────┬───────┘                                              │
│                       ▼                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                         STORAGE LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     PostgreSQL Database                              │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────────┐ │   │
│   │  │   orders    │  │ order_items │  │ offline_user_recommendations │ │   │
│   │  │  (234,959)  │  │ (1,971,527) │  │        (180,483)             │ │   │
│   │  └─────────────┘  └─────────────┘  └──────────────────────────────┘ │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────────┐ │   │
│   │  │  products   │  │   users     │  │    offline_similar_items     │ │   │
│   │  └─────────────┘  └─────────────┘  │          (4,182)             │ │   │
│   │                                     └──────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ML TRAINING LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                    ML Training Pipeline                             │    │
│   │                    (Runs daily at 2:00 AM)                          │    │
│   │                                                                     │    │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │    │
│   │  │  Collaborative   │  │  Content-Based   │  │    Matrix       │   │    │
│   │  │   Filtering      │  │   Filtering      │  │ Factorization   │   │    │
│   │  └────────┬─────────┘  └────────┬─────────┘  └───────┬─────────┘   │    │
│   │           │                     │                     │             │    │
│   │           └─────────────────────┼─────────────────────┘             │    │
│   │                                 ▼                                   │    │
│   │                    ┌────────────────────┐                           │    │
│   │                    │   Hybrid Ensemble  │                           │    │
│   │                    │      Model         │                           │    │
│   │                    └─────────┬──────────┘                           │    │
│   │                              ▼                                      │    │
│   │                    ┌────────────────────┐                           │    │
│   │                    │  Save to S3/Local  │                           │    │
│   │                    │   (model.pkl)      │                           │    │
│   │                    └────────────────────┘                           │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         BATCH INFERENCE LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                  Batch Inference Pipeline                           │    │
│   │                  (Runs daily at 3:00 AM)                            │    │
│   │                                                                     │    │
│   │  1. Load trained model                                              │    │
│   │  2. Get all unique users from orders                                │    │
│   │  3. Generate top-10 recommendations per user                        │    │
│   │  4. Store in offline_user_recommendations table                     │    │
│   │  5. Generate similar items for all products                         │    │
│   │  6. Store in offline_similar_items table                            │    │
│   │                                                                     │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         API SERVING LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      FastAPI Server                                 │    │
│   │                      (Port 8001)                                    │    │
│   │                                                                     │    │
│   │  GET /api/v1/personalize/recommendations/{user_id}                  │    │
│   │      → Reads from offline_user_recommendations                      │    │
│   │                                                                     │    │
│   │  GET /api/v1/personalize/recommendations/similar/{product_id}       │    │
│   │      → Reads from offline_similar_items                             │    │
│   │                                                                     │    │
│   │  GET /api/v1/ml/product-pairs                                       │    │
│   │      → Reads from product_pairs table                               │    │
│   │                                                                     │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Pipeline Steps (Same as AWS Personalize)

### Step 1: Data Ingestion (Every 6 hours)

```python
# Fetch orders from OE API
oe_orders = fetch_oe_orders(since=last_sync_time)

# Fetch orders from POS API  
pos_orders = fetch_pos_orders(since=last_sync_time)

# Merge and deduplicate
all_orders = merge_orders(oe_orders, pos_orders)

# Insert into PostgreSQL
insert_orders(all_orders)
insert_order_items(all_orders)
```

**Tables Updated:**
- `orders` - Order header data
- `order_items` - Order line items
- `sync_metadata` - Last sync timestamp

### Step 2: ML Model Training (Daily at 2:00 AM)

```python
# Load interaction data
interactions = load_interactions()  # user_id, item_id, timestamp, event_value

# Train multiple algorithms
cf_model = CollaborativeFiltering().fit(interactions)
cb_model = ContentBasedFiltering().fit(interactions, product_features)
mf_model = MatrixFactorization().fit(interactions)

# Create hybrid ensemble
hybrid_model = HybridEnsemble([cf_model, cb_model, mf_model])

# Save model
save_model(hybrid_model, 'custom_ml/models/latest_model.pkl')
```

**Output:**
- `custom_ml/models/latest_model.pkl` - Trained model file

### Step 3: Batch Inference (Daily at 3:00 AM)

```python
# Load trained model
model = load_model('custom_ml/models/latest_model.pkl')

# Get all unique users
users = get_all_users()

# Generate recommendations for each user
for user_id in users:
    recommendations = model.recommend(user_id, n=10)
    save_to_offline_cache(user_id, recommendations)

# Generate similar items for each product
products = get_all_products()
for product_id in products:
    similar = model.get_similar_items(product_id, n=10)
    save_to_similar_items_cache(product_id, similar)
```

**Tables Updated:**
- `offline_user_recommendations` - User → Product recommendations
- `offline_similar_items` - Product → Similar products

### Step 4: API Serving (Real-time)

```python
@app.get("/api/v1/personalize/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    # Read from pre-computed cache
    recommendations = db.query(
        "SELECT recommendations FROM offline_user_recommendations WHERE user_id = %s",
        user_id
    )
    return {"recommendations": recommendations}
```

## 📅 Schedule Summary

| Job | Schedule | Duration | Description |
|-----|----------|----------|-------------|
| **Data Sync** | Every 6 hours | ~5 min | Fetch OE/POS orders |
| **ML Training** | Daily 2:00 AM | ~30 min | Train recommendation models |
| **Batch Inference** | Daily 3:00 AM | ~1 hour | Generate all recommendations |

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Data Sources
OE_API_BASE_URL=https://your-oe-api.com/api
POS_API_BASE_URL=https://your-pos-api.com/api

# Database
PG_HOST=localhost
PG_PORT=5432
PG_DB=mastergroup_recommendations

# ML Pipeline
USE_LOCAL_ML=true
ML_MODEL_PATH=./custom_ml/models
ML_TRAINING_SCHEDULE=0 2 * * *
ML_BATCH_INFERENCE_SCHEDULE=0 3 * * *
```

## 📊 Comparison: AWS Personalize vs Local ML

| Feature | AWS Personalize | Local ML |
|---------|-----------------|----------|
| **Cost** | ~$170/month | $0 |
| **Training** | AWS managed | Local Python |
| **Inference** | Batch + Real-time | Batch (cached) |
| **Latency** | ~100ms | ~10ms (from cache) |
| **Customization** | Limited | Full control |
| **Data Privacy** | AWS servers | Your servers |
| **Algorithms** | Black box | Transparent |

## 🚀 Running the Pipeline Locally

### Manual Execution

```bash
# 1. Sync data from APIs
python scripts/sync_data.py

# 2. Train ML model
python scripts/train_model.py

# 3. Generate batch recommendations
python scripts/batch_inference.py

# 4. Start API server
uvicorn src.main:app --host 0.0.0.0 --port 8001
```

### Automated (Cron)

```bash
# Add to crontab
0 */6 * * * /path/to/venv/bin/python /path/to/scripts/sync_data.py
0 2 * * * /path/to/venv/bin/python /path/to/scripts/train_model.py
0 3 * * * /path/to/venv/bin/python /path/to/scripts/batch_inference.py
```

## 📁 File Structure

```
recommendation-engine-service/
├── .env                          # Configuration
├── src/
│   ├── main.py                   # FastAPI server
│   └── algorithms/
│       ├── collaborative_filtering.py
│       ├── content_based_filtering.py
│       ├── matrix_factorization.py
│       └── ml_recommendation_service.py
├── custom_ml/
│   ├── models/                   # Trained model files
│   └── CUSTOM_ML_PLAYBOOK.md
├── scripts/
│   ├── sync_data.py              # Data ingestion
│   ├── train_model.py            # ML training
│   └── batch_inference.py        # Generate recommendations
└── docs/
    └── LOCAL_ML_PIPELINE.md      # This document
```

## ✅ Current Status

- ✅ **Data Synced**: 234,959 orders, 1.97M order items
- ✅ **Recommendations Cached**: 180,483 users
- ✅ **Similar Items Cached**: 4,182 products
- ✅ **API Serving**: From PostgreSQL cache
- ✅ **ML Training**: Implemented (see below)
- ✅ **Batch Inference**: Implemented (see below)

---

## 🆕 New File-Based Local ML Implementation

### Overview

The new implementation uses **joblib files** instead of database storage for faster model loading and simpler management.

### Components Created

| File | Purpose |
|------|---------|
| `src/services/local_model_storage.py` | File-based model storage using joblib |
| `src/services/local_recommender.py` | Training & inference for SVD, similarity, popularity |
| `scripts/train_local_models.py` | CLI script to train models |
| `models/` | Directory where trained models are saved |

### Models Trained

1. **SVD Recommender** (`svd_recommender_latest.joblib`)
   - Algorithm: Singular Value Decomposition (scikit-surprise)
   - Purpose: User personalization (like AWS User-Personalization)
   - Metrics: RMSE, MAE from cross-validation

2. **Item Similarity** (`item_similarity_latest.joblib`)
   - Algorithm: Cosine similarity on user-item matrix
   - Purpose: Similar items (like AWS Similar-Items)
   - Output: N×N similarity matrix

3. **Popularity Scores** (`popularity_scores_latest.joblib`)
   - Algorithm: Weighted by recency (exponential decay)
   - Purpose: Cold-start fallback
   - Output: Item → score dictionary

### API Endpoints

```
POST /api/v1/local-ml/train
     Train all models (SVD, similarity, popularity)
     Query params: days (default: 90)

GET  /api/v1/local-ml/status
     Check model status and versions

GET  /api/v1/local-ml/recommendations/{user_id}
     Get personalized recommendations
     Query params: limit, exclude_purchased

GET  /api/v1/local-ml/similar-items/{item_id}
     Get similar items

GET  /api/v1/local-ml/frequently-bought-together/{item_id}
     Get co-purchase items

GET  /api/v1/local-ml/popular
     Get most popular items
```

### Usage

#### CLI Training
```bash
# Train with default 90 days of data
python scripts/train_local_models.py

# Train with 30 days, cleanup old versions
python scripts/train_local_models.py --days 30 --cleanup

# Test only (no training)
python scripts/train_local_models.py --test-only
```

#### API Training
```bash
# Trigger training via API
curl -X POST "http://localhost:8001/api/v1/local-ml/train?days=90"

# Check status
curl "http://localhost:8001/api/v1/local-ml/status"

# Get recommendations
curl "http://localhost:8001/api/v1/local-ml/recommendations/customer123"
```

### Model Storage

Models are saved in the `models/` directory:

```
models/
├── svd_recommender_latest.joblib      # Always points to newest
├── svd_recommender_20231216_020000.joblib  # Versioned backup
├── item_similarity_latest.joblib
├── item_similarity_20231216_020000.joblib
├── popularity_scores_latest.joblib
├── popularity_scores_20231216_020000.joblib
└── metadata.json                       # Index of all models
```

### Dependencies

```
# requirements.txt additions
scikit-surprise==1.1.3   # Collaborative filtering (SVD, KNN)
scipy==1.11.4            # Sparse matrices, similarity calculations
joblib==1.3.2            # Model serialization (already included)
```

### Comparison: Old vs New Implementation

| Feature | Old (DB Storage) | New (File Storage) |
|---------|-----------------|-------------------|
| **Storage** | PostgreSQL BYTEA | Joblib files |
| **Load Time** | ~500ms (network) | ~50ms (local disk) |
| **Debugging** | Hard (binary in DB) | Easy (inspect files) |
| **Versioning** | Single version | Multiple versions |
| **Portability** | Tied to DB | Copy files anywhere |
| **Size Limit** | DB row limits | Filesystem only |
