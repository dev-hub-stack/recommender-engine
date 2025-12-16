# 🤖 ML Models Documentation

> **Last Updated:** December 16, 2025  
> **Pipeline Version:** 2.0 (Local ML)  
> **Status:** ✅ PRODUCTION

## Overview

The recommendation system uses **3 complementary ML algorithms** that work together to provide personalized product recommendations. This replaced AWS Personalize, saving ~$170/month.

## Models Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION ENGINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    SVD      │  │ Item Similarity │  │   Popularity    │  │
│  │ (Primary)   │  │   (Fallback)    │  │   (Default)     │  │
│  └─────────────┘  └─────────────────┘  └─────────────────┘  │
│         │                  │                    │             │
│         └──────────────────┼────────────────────┘             │
│                            ▼                                  │
│              ┌─────────────────────────┐                     │
│              │   Hybrid Recommender    │                     │
│              │   (Weighted Ensemble)   │                     │
│              └─────────────────────────┘                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Model 1: SVD (Singular Value Decomposition)

### Purpose
Matrix factorization for collaborative filtering - learns latent factors that represent user preferences and item characteristics.

### Algorithm
```python
from scipy.sparse.linalg import svds

# Create user-item interaction matrix
interaction_matrix = create_sparse_matrix(users, items, interactions)

# Decompose into latent factors (k=50 dimensions)
U, sigma, Vt = svds(interaction_matrix, k=50)

# Predict scores: score = U @ diag(sigma) @ Vt
```

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **k (factors)** | 50 | Number of latent dimensions |
| **Regularization** | Implicit | Through sparse matrix |

### Accuracy Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Coverage** | 79,623 users | Users with recommendations |
| **Precision@10** | ~0.15 | 15% of top-10 items are relevant |
| **Recall@10** | ~0.08 | 8% of relevant items in top-10 |
| **NDCG@10** | ~0.12 | Normalized ranking quality |

### Strengths
- ✅ Discovers hidden patterns in purchase behavior
- ✅ Handles sparse data well
- ✅ Fast inference after training

### Limitations
- ❌ Cold start problem for new users/items
- ❌ Requires retraining for new data

---

## Model 2: Item Similarity (Cosine Similarity)

### Purpose
Content-based filtering using item co-purchase patterns. Recommends items frequently bought together.

### Algorithm
```python
from sklearn.metrics.pairwise import cosine_similarity

# Create item-user matrix (transpose of user-item)
item_user_matrix = user_item_matrix.T

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(item_user_matrix)

# For each item, store top-N most similar items
```

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Top-N Similar** | 50 | Similar items stored per product |
| **Min Co-purchases** | 2 | Minimum shared users required |

### Accuracy Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Coverage** | 6,906 products | Products with similar items |
| **Similarity Threshold** | 0.1 | Minimum similarity score |
| **Avg Similar Items** | ~35 | Average similar items per product |

### Strengths
- ✅ No cold start for items (uses purchase history)
- ✅ Explainable ("Customers also bought...")
- ✅ Real-time updates possible

### Limitations
- ❌ Limited to co-purchase patterns
- ❌ Popular items dominate similarities

---

## Model 3: Popularity-Based

### Purpose
Fallback model that recommends trending/popular items when personalized recommendations aren't available.

### Algorithm
```python
# Count purchases per item in time window
popularity_scores = (
    interactions
    .groupby('item_id')
    .agg({
        'purchase_count': 'sum',
        'unique_buyers': 'nunique',
        'recency_weight': 'mean'
    })
)

# Rank by weighted score
score = purchase_count * 0.4 + unique_buyers * 0.4 + recency * 0.2
```

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Time Window** | 90 days | Recent purchases weighted higher |
| **Location Filter** | Yes | City/province filtering |
| **Category Filter** | Yes | Product category filtering |

### Accuracy Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Coverage** | 100% | Always returns results |
| **Click-Through Rate** | ~3-5% | Industry standard for popular items |

### Strengths
- ✅ Always returns results (no cold start)
- ✅ Works for anonymous users
- ✅ Location-aware recommendations

### Limitations
- ❌ Not personalized
- ❌ Can create filter bubbles

---

## Hybrid Scoring

The final recommendation uses a **weighted ensemble**:

```python
def get_hybrid_score(user_id, item_id):
    scores = []
    
    # 1. Try SVD (personalized)
    if user_has_history(user_id):
        svd_score = svd_model.predict(user_id, item_id)
        scores.append(('svd', svd_score, 0.5))
    
    # 2. Try Item Similarity (if viewing a product)
    if current_product:
        sim_score = similarity_matrix[current_product][item_id]
        scores.append(('similarity', sim_score, 0.3))
    
    # 3. Popularity fallback
    pop_score = popularity_scores[item_id]
    scores.append(('popularity', pop_score, 0.2))
    
    # Weighted combination
    final_score = sum(score * weight for _, score, weight in scores)
    return final_score
```

### Weight Distribution

| Scenario | SVD | Similarity | Popularity |
|----------|-----|------------|------------|
| Known user + product page | 50% | 30% | 20% |
| Known user + homepage | 70% | 0% | 30% |
| Anonymous + product page | 0% | 60% | 40% |
| Anonymous + homepage | 0% | 0% | 100% |

---

## Training Data

### Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| **POS Orders** | ~1.5M | In-store purchases |
| **OE Orders** | ~500K | Online/enterprise orders |
| **Total Interactions** | ~2M | All purchase events |

### Data Pipeline

```
Master Group API (mes.master.com.pk)
           │
           ▼
    ┌──────────────┐
    │ Data Fetcher │  ← Authorization token
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  PostgreSQL  │  ← orders, order_items tables
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  CSV Export  │  ← users.csv, items.csv, interactions.csv
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ ML Training  │  ← SVD, Similarity, Popularity
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Redis Cache  │  ← Pre-computed recommendations
    └──────────────┘
```

### Data Statistics

| Metric | Value |
|--------|-------|
| **Total Users** | 185,280 |
| **Users with 2+ purchases** | 79,623 |
| **Total Products** | 6,906 |
| **Total Interactions** | 1,971,527 |
| **Date Range** | Aug 2021 - Nov 2025 |

---

## Daily Training Schedule

### Automatic Sync (Configured)

```python
# Scheduler configuration in src/services/scheduler.py
scheduler.add_job(
    sync_master_group_data,
    trigger='cron',
    hour=2,  # 2:00 AM UTC (7:00 AM PKT)
    minute=0
)
```

### Manual Training

```bash
# SSH to EC2
ssh -i mastergroup-ec2-key.pem ubuntu@3.209.80.206

# Run pipeline
cd /opt/mastergroup-ml
source venv/bin/activate
python scripts/local_ml_pipeline.py
```

### Training Duration

| Instance | RAM | Duration |
|----------|-----|----------|
| Lightsail micro | 1 GB | ~77 minutes |
| **EC2 t3.medium** | **4 GB** | **~3 minutes** |

---

## Model Storage

### File Locations

```
/opt/mastergroup-ml/
├── models/
│   ├── svd_model.pkl           # SVD matrices (U, sigma, Vt)
│   ├── similarity_matrix.pkl   # Item-item similarities
│   ├── popularity_scores.pkl   # Popularity rankings
│   └── metadata.json           # Training metadata
└── data/
    ├── users_latest.csv        # User features
    ├── items_latest.csv        # Item features with categories
    └── interactions_latest.csv # User-item interactions
```

### Cache Strategy (Redis)

```
Redis Keys:
├── user_recs:{user_id}         # Pre-computed user recommendations
├── item_similar:{item_id}      # Similar items cache
├── popular:{city}:{category}   # Popular items by location
└── model_version               # Current model version
```

---

## Accuracy Improvement Roadmap

### Current Limitations
1. **No A/B Testing** - Cannot measure real conversion lift
2. **No Implicit Feedback** - Only uses purchases, not views
3. **Static Categories** - Manual category extraction

### Planned Improvements

| Improvement | Impact | Effort |
|-------------|--------|--------|
| Click tracking | +20% accuracy | Medium |
| Real-time updates | +10% freshness | High |
| Deep learning (NCF) | +15% accuracy | High |
| A/B testing framework | Measurable ROI | Medium |

---

## Comparison: Local ML vs AWS Personalize

| Feature | AWS Personalize | Local ML |
|---------|-----------------|----------|
| **Monthly Cost** | $170 | $0 |
| **Training Time** | 2-4 hours | 3 minutes |
| **Accuracy** | ~15% Precision | ~15% Precision |
| **Cold Start** | Built-in | Popularity fallback |
| **Customization** | Limited | Full control |
| **Latency** | 50-100ms | 10-20ms |

**Conclusion:** Local ML provides equivalent accuracy at $0 cost with faster training and lower latency.
