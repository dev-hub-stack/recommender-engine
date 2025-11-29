# MasterGroup Recommendation System - Complete Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MASTERGROUP RECOMMENDATION SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Shopify    │    │  Dashboard   │    │   Mobile     │    │   Email      │  │
│  │    Store     │    │   (React)    │    │    App       │    │  Campaigns   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │                   │           │
│         └───────────────────┼───────────────────┼───────────────────┘           │
│                             │                   │                               │
│                             ▼                   ▼                               │
│                    ┌─────────────────────────────────┐                          │
│                    │     RECOMMENDATION API          │                          │
│                    │     (FastAPI - Port 8001)       │                          │
│                    │     44.201.11.243:8001          │                          │
│                    └──────────────┬──────────────────┘                          │
│                                   │                                             │
│         ┌─────────────────────────┼─────────────────────────┐                  │
│         │                         │                         │                  │
│         ▼                         ▼                         ▼                  │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐           │
│  │    Redis     │         │  PostgreSQL  │         │     AWS      │           │
│  │   (Cache)    │         │    (RDS)     │         │  Personalize │           │
│  │  localhost   │         │   us-east-1  │         │   us-east-1  │           │
│  └──────────────┘         └──────────────┘         └──────────────┘           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow - Complete Cycle

### 1️⃣ Data Ingestion (MasterGroup API → PostgreSQL)

```
┌────────────────────┐
│  MasterGroup API   │
│ (ERP/POS System)   │
└─────────┬──────────┘
          │
          │ Sync Service (Scheduled)
          │ Every 6 hours
          ▼
┌────────────────────┐
│    sync_service.py │
│  - Fetch orders    │
│  - Fetch customers │
│  - Fetch products  │
└─────────┬──────────┘
          │
          │ Transform & Load
          ▼
┌────────────────────────────────────────┐
│           PostgreSQL (RDS)              │
├─────────────────┬──────────────────────┤
│     orders      │    order_items       │
│  (234K rows)    │   (1.97M rows)       │
├─────────────────┴──────────────────────┤
│  unified_customer_id, customer_name,   │
│  province, city, product_id, price...  │
└────────────────────────────────────────┘
```

### 2️⃣ Data Export to AWS (PostgreSQL → S3 → Personalize)

```
┌────────────────────────────────────────┐
│           PostgreSQL (RDS)              │
└─────────────────┬──────────────────────┘
                  │
                  │ export_training_data.py
                  │ (Creates CSV files)
                  ▼
┌────────────────────────────────────────┐
│              S3 Bucket                  │
│    s3://mastergroup-personalize/       │
├────────────────────────────────────────┤
│  /training-data/                       │
│    ├── users.csv (180K users)          │
│    ├── items.csv (4K products)         │
│    └── interactions.csv (2M+ events)   │
└─────────────────┬──────────────────────┘
                  │
                  │ AWS Personalize Import
                  ▼
┌────────────────────────────────────────┐
│         AWS Personalize Dataset        │
│  - Users Dataset                       │
│  - Items Dataset                       │
│  - Interactions Dataset                │
└────────────────────────────────────────┘
```

### 3️⃣ Model Training (AWS Personalize)

```
┌────────────────────────────────────────┐
│       AWS Personalize Solutions        │
├────────────────────────────────────────┤
│                                        │
│  ┌──────────────────────────────────┐  │
│  │  User-Personalization Solution   │  │
│  │  Recipe: aws-user-personalization│  │
│  │  → "What will this user buy?"    │  │
│  └──────────────────────────────────┘  │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │    Similar-Items Solution        │  │
│  │  Recipe: aws-similar-items       │  │
│  │  → "Products like this one"      │  │
│  └──────────────────────────────────┘  │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │    Item-Affinity Solution        │  │
│  │  Recipe: aws-item-affinity       │  │
│  │  → "Products that drive sales"   │  │
│  └──────────────────────────────────┘  │
│                                        │
└────────────────────────────────────────┘
```

### 4️⃣ Batch Inference (Generate Recommendations)

```
┌────────────────────────────────────────┐
│     run_batch_inference.py             │
│  (Runs daily via cron/scheduler)       │
└─────────────────┬──────────────────────┘
                  │
                  │ Creates batch jobs
                  ▼
┌────────────────────────────────────────┐
│    AWS Personalize Batch Jobs          │
├────────────────────────────────────────┤
│  Input: s3://bucket/batch-input/       │
│    └── users.json (all user IDs)       │
│                                        │
│  Output: s3://bucket/batch-output/     │
│    ├── user-personalization/           │
│    ├── similar-items/                  │
│    └── item-affinity/                  │
└─────────────────┬──────────────────────┘
                  │
                  │ load_batch_results.py
                  ▼
┌────────────────────────────────────────┐
│           PostgreSQL Cache             │
├────────────────────────────────────────┤
│  offline_user_recommendations          │
│    └── 180,483 rows                    │
│  offline_similar_items                 │
│    └── 4,182 rows                      │
│  offline_item_affinity                 │
│    └── (pending batch job)             │
└────────────────────────────────────────┘
```

### 5️⃣ API Serving (Cache → API → Clients)

```
┌────────────────────────────────────────┐
│         Client Request                  │
│  GET /api/v1/personalize/              │
│      recommendations/similar/1328      │
└─────────────────┬──────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────┐
│          FastAPI Server                 │
│         (main.py:8001)                  │
└─────────────────┬──────────────────────┘
                  │
          ┌───────┴───────┐
          ▼               ▼
┌──────────────┐  ┌──────────────┐
│    Redis     │  │  PostgreSQL  │
│   (1hr TTL)  │  │   (Cache)    │
│   Hit? ──────┼──│   Tables     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
                ▼
┌────────────────────────────────────────┐
│            JSON Response                │
│  {                                      │
│    "product_id": "1328",               │
│    "recommendations": [                 │
│      {"product_id": "1715",            │
│       "product_name": "GOLD PILLOW"}   │
│    ]                                    │
│  }                                      │
└────────────────────────────────────────┘
```

---

## API Endpoints Summary

### AWS Personalize Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/personalize/recommendations/{user_id}` | GET | User personalized recommendations |
| `/api/v1/personalize/recommendations/similar/{product_id}` | GET | Similar products (cross-sell) |
| `/api/v1/personalize/recommendations/item-affinity/{user_id}` | GET | Products that drive conversions |
| `/api/v1/personalize/recommendations/by-location` | GET | Trending products by region |
| `/api/v1/personalize/status` | GET | System status & data counts |

### Analytics Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/analytics/collaborative-products` | GET | Top collaborative products |
| `/api/v1/analytics/collaborative-pairs` | GET | Frequently bought together |
| `/api/v1/analytics/customer-similarity` | GET | Customer similarity insights |
| `/api/v1/analytics/collaborative-metrics` | GET | Overall metrics |

---

## AWS Personalize Recipes Explained

### 1. User-Personalization (ACTIVE ✅)
```
Purpose: "What products should we show THIS user?"
Input: User ID
Output: Ranked list of products personalized for that user

Use Cases:
- Shopify checkout page recommendations
- "Recommended for you" section
- Personalized email campaigns

Status: ✅ Active - 180,483 users cached
```

### 2. Similar-Items (ACTIVE ✅)
```
Purpose: "What products are similar to THIS product?"
Input: Product ID
Output: Ranked list of similar products

Use Cases:
- Product page "You may also like"
- Cart page cross-selling
- "Customers also viewed"

Status: ✅ Active - 4,182 products cached
```

### 3. Item-Affinity (REAL-TIME ONLY ⚠️)
```
Purpose: "Which products INFLUENCE this user to buy?"
Input: User ID
Output: Products that drive conversion for this user

Use Cases:
- Homepage hero banners
- Retargeting ads
- Email win-back campaigns
- High-value product promotion

Status: ⚠️ Real-time campaigns only (no batch support)
        Requires ~$400/month for real-time inference
        
Alternative: Use similar-items + user-personalization for same effect
```

---

## Item Affinity - Activation Options

### Option 1: Real-time Campaign (~$400/month)
```bash
# Create a campaign (AWS Console or CLI)
aws personalize create-campaign \
  --name mastergroup-item-affinity-campaign \
  --solution-version-arn arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-item-affinity/0abd6fe3 \
  --min-provisioned-tps 1

# This enables real-time recommendations but costs ~$400/month
```

### Option 2: Use Existing Recipes (Recommended - FREE)
Instead of item-affinity, combine existing batch recipes:
```
User-Personalization → "What should we recommend to this user?"
Similar-Items → "What products are similar to what they're viewing?"

Combined effect = Item Affinity behavior without extra cost
```

---

## Scheduled Jobs (Automated)

### Daily Batch Refresh (2 AM UTC)
```bash
# Cron job runs daily at 2 AM
0 2 * * * /opt/mastergroup-api/run_daily_batch.sh

# What it does:
# 1. Creates batch inference jobs for user-personalization and similar-items
# 2. Waits 1 hour for completion
# 3. Loads results into PostgreSQL cache
```

### Manual Run
```bash
ssh ubuntu@44.201.11.243
cd /opt/mastergroup-api
./run_daily_batch.sh
```

---

## Cost Breakdown

| Component | Cost/Month | Notes |
|-----------|------------|-------|
| AWS Personalize (Batch) | ~$7.50 | No real-time campaigns |
| PostgreSQL RDS | ~$15 | db.t3.micro |
| Lightsail Server | ~$10 | 1GB RAM + swap |
| S3 Storage | ~$1 | Training data |
| **Total** | **~$33.50** | 98% savings vs real-time |

---

## File Structure

```
recommendation-engine-service/
├── src/
│   └── main.py                    # FastAPI application (3500+ lines)
├── aws_personalize/
│   ├── personalize_service.py     # Service class for batch cache
│   ├── export_training_data.py    # Export to S3
│   ├── run_batch_inference.py     # Create batch jobs
│   ├── load_batch_results.py      # Load results to PostgreSQL
│   ├── README.md                  # AWS Personalize docs
│   ├── PLAYBOOK.md                # Operations guide
│   └── QUICK_START.md             # Getting started
├── config/
│   └── master_group_api.py        # Database & API config
├── services/
│   ├── sync_service.py            # MasterGroup data sync
│   └── scheduler.py               # Cron jobs
├── API_COLLECTION.md              # API documentation
└── ARCHITECTURE.md                # This file
```

---

## Deployment Checklist

- [x] PostgreSQL RDS configured
- [x] Redis cache running
- [x] FastAPI server on port 8001
- [x] User-Personalization batch loaded
- [x] Similar-Items batch loaded
- [ ] Item-Affinity batch (pending)
- [x] Dashboard connected
- [x] Shopify integration ready

---

## Future Enhancements

1. **Real-time Event Tracking** - Send user clicks/views to Personalize
2. **Filters** - Exclude out-of-stock, already purchased
3. **Promotions** - Boost specific products
4. **A/B Testing** - Compare recommendation strategies
5. **Real-time Campaigns** - If budget allows (~$400/mo)
