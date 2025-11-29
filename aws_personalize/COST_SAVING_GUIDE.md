# AWS Personalize Cost-Saving Implementation Guide

## 💰 Cost Savings Summary

### Before (Real-Time Campaigns)
- **4 Campaigns running 24/7**: ~$432/month
  - User Personalization: $0.20/hour × 730 hours = $146/month
  - Similar Items: $0.20/hour × 730 hours = $146/month  
  - Item Affinity: $0.20/hour × 730 hours = $146/month
  - (Personalized Ranking used on-demand only)

### After (Batch Inference)
- **Training**: $0.24/hour × ~2 hours/month = $0.48/month
- **Batch Inference**: $0.0883/TPS-hour × 4 jobs/month = ~$5/month
- **Storage (S3 + PostgreSQL)**: ~$2/month

**Total Monthly Cost: ~$7.50 vs $432 = 98% Cost Reduction! 💸**

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     COST-SAVING WORKFLOW                     │
└─────────────────────────────────────────────────────────────┘

1. DATA PREPARATION (Once/Monthly)
   PostgreSQL → generate_batch_inputs.py → S3 JSON files

2. MODEL TRAINING (Once/Monthly or on-demand)
   AWS Personalize Solutions (4 recipes) → Train → Active Models

3. BATCH INFERENCE (Monthly/Weekly)
   Models + S3 Inputs → Batch Jobs → S3 Outputs

4. LOAD TO DATABASE (After batch completion)
   S3 Results → load_batch_results.py → PostgreSQL Tables

5. SERVE RECOMMENDATIONS (Real-time, from cache)
   API Request → PostgreSQL Query → Fast Response
```

---

## 🔧 Setup Instructions

### Step 1: Create Database Tables

```bash
cd /Users/clustox_1/Documents/MasterGroup-RecommendationSystem/recommendation-engine-service

# Install dependencies if needed
pip install psycopg2-binary boto3 python-dotenv

# Create offline tables
python aws_personalize/setup_offline_tables.py
```

**Expected Output:**
```
✅ Tables created successfully!

Created tables:
  - batch_job_metadata
  - offline_item_affinity
  - offline_personalized_ranking
  - offline_similar_items
  - offline_user_recommendations
```

### Step 2: Generate Batch Input Files

```bash
# This queries your PostgreSQL database and uploads input files to S3
python aws_personalize/generate_batch_inputs.py
```

**Expected Output:**
```
✅ BATCH INPUT FILES READY!
Users:     s3://mastergroup-personalize-data/batch/input/users.json
Items:     s3://mastergroup-personalize-data/batch/input/items.json
Affinity:  s3://mastergroup-personalize-data/batch/input/affinity.json
```

### Step 3: Train Models & Run Batch Inference

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1

# Run hybrid training (creates solutions + batch jobs)
python aws_personalize/train_hybrid_model.py
```

**Expected Timeline:**
- Solutions created: ~5 minutes
- Training: ~30-60 minutes per recipe
- Batch jobs started: ~5 minutes
- **Total: ~2-4 hours** (runs asynchronously on AWS)

### Step 4: Load Results into PostgreSQL

**Wait for batch jobs to complete** (check AWS Console or use AWS CLI):

```bash
# Check batch job status
aws personalize list-batch-inference-jobs \
  --solution-version-arn <your-solution-version-arn>
```

Once status is `ACTIVE`:

```bash
# Load results from S3 into PostgreSQL
python aws_personalize/load_batch_results.py
```

**Expected Output:**
```
📥 Loading User Recommendations...
  ✅ Loaded 1,234 user recommendations
📥 Loading Similar Items...
  ✅ Loaded 567 product similarities
📥 Loading Item Affinity...
  ✅ Loaded 1,234 user affinities
```

---

## 📋 Recipe Details

### 1. User Personalization (`aws-user-personalization`)
**Purpose:** Personalized product recommendations for each user  
**Input:** `{"userId": "123"}`  
**Output:** List of recommended products with scores  
**Use Case:** Homepage "Recommended for You", Email campaigns  
**Storage:** `offline_user_recommendations` table

### 2. Similar Items (`aws-similar-items`)
**Purpose:** Find products similar to a given product  
**Input:** `{"itemId": "ABC"}`  
**Output:** List of similar products  
**Use Case:** Product detail page "You may also like", Cross-selling  
**Storage:** `offline_similar_items` table

### 3. Item Affinity (`aws-item-affinity`)
**Purpose:** User's affinity/interest scores for products/categories  
**Input:** `{"userId": "123"}`  
**Output:** Products ranked by user's predicted interest  
**Use Case:** Category pages, Browse personalization  
**Storage:** `offline_item_affinity` table

### 4. Personalized Ranking (`aws-personalized-ranking`)
**Purpose:** Rank a specific list of items for a user  
**Input:** `{"userId": "123", "itemList": ["A", "B", "C"]}`  
**Output:** Same items, reordered by relevance  
**Use Case:** Search results, Category page sorting  
**Note:** Used on-demand via API (not batch)

---

## 🔄 Maintenance Schedule

### Monthly Tasks (Automated via Cron)
1. **Day 1:** Run `generate_batch_inputs.py` (updates with new users/products)
2. **Day 1:** Run `train_hybrid_model.py` (retrains with fresh data)
3. **Day 2:** Wait for batch jobs to complete (~2-4 hours)
4. **Day 2:** Run `load_batch_results.py` (refresh recommendations)

### Weekly Tasks (Optional, for fast-changing catalogs)
- Run batch inference only (skip training if models are recent)

---

## 🎯 API Integration

The backend will automatically serve from offline tables instead of real-time campaigns:

```python
# Example: Get recommendations for user
GET /api/v1/personalize/recommendations/{user_id}

# Backend queries:
SELECT recommendations FROM offline_user_recommendations 
WHERE user_id = '123'

# Returns cached recommendations instantly (no AWS API call)
```

---

## 📊 Monitoring & Alerts

### Key Metrics to Track
- **Recommendation Freshness:** Check `updated_at` timestamps
- **Coverage:** % of users/products with recommendations
- **Batch Job Success Rate:** Monitor failed jobs
- **Query Performance:** PostgreSQL query times

### Set Up Alerts
```sql
-- Find stale recommendations (>30 days old)
SELECT COUNT(*) FROM offline_user_recommendations 
WHERE updated_at < NOW() - INTERVAL '30 days';
```

---

## 🐛 Troubleshooting

### Batch Job Failed
1. Check CloudWatch logs for the batch job
2. Verify S3 permissions (PersonalizeRole must have S3 access)
3. Check input file format (must be JSONL: one JSON per line)

### No Recommendations Loaded
1. Verify S3 output path contains `.json.out` files
2. Check batch job status: `aws personalize describe-batch-inference-job --batch-inference-job-arn <arn>`
3. Ensure database credentials are correct

### Slow Query Performance
1. Check indexes: `\d offline_user_recommendations`
2. Add product name enrichment index if needed
3. Consider partitioning for very large datasets (>10M records)

---

## 🚀 Next Steps

1. ✅ Set up database tables
2. ✅ Generate batch inputs
3. ⏳ Run first training cycle (wait ~4 hours)
4. ✅ Load results
5. 🔄 Update backend API endpoints
6. 📈 Monitor & optimize

**Estimated Time to Complete:** 4-6 hours (mostly waiting for AWS)  
**Monthly Savings:** ~$424 💰
