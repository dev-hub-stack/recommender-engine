# AWS Personalize - Cost-Optimized Batch Inference

**Status:** ✅ **PRODUCTION - ACTIVE**  
**Date Deployed:** November 29, 2025  
**Architecture:** Batch Inference + PostgreSQL Caching  
**Cost:** **$7.50/month** (98% savings vs real-time campaigns)

---

## 🎯 Current Production Setup

**MasterGroup is using AWS Personalize with BATCH INFERENCE** (not real-time campaigns)

### **Why Batch Inference?**

| Real-Time Campaigns | Batch Inference (Current) |
|---------------------|---------------------------|
| ❌ $432/month | ✅ **$7.50/month** |
| 24/7 campaign running | Monthly batch jobs |
| ~100ms latency | **<10ms (from cache)** |
| AWS API calls | PostgreSQL queries |
| High cost | 98% cost reduction |

---

## 💰 Actual Production Costs

**MasterGroup's Current Scale:**
- **Interactions:** 1,971,527 orders
- **Users:** 180,484 customers (Recommendations Loaded ✅)
- **Products:** 4,182 items (Similarity Loaded ✅)

**Monthly Cost Breakdown:**
- **Training:** $0.50/month (monthly retraining)
- **Batch Inference:** $5.00/month (3 recipes)
- **S3 Storage:** $2.00/month
- **Total:** **$7.50/month**

**vs Real-Time Alternative:** $432/month (4 campaigns × $0.20/hour × 730 hours)

**Annual Savings:** **$5,094/year** 💰

---

## 🏗️ Production Architecture

### **Current System (Deployed & Active)**

```
┌──────────────────────────────────────────────────────────┐
│          COST-OPTIMIZED PRODUCTION ARCHITECTURE          │
└──────────────────────────────────────────────────────────┘

[Daily at 2 AM] Data Sync ✅
    Shopify → Master Group API → PostgreSQL RDS
    
[Monthly on 1st] AWS Personalize Batch Training ✅
    PostgreSQL → S3 Input Files → AWS Personalize
    ↓
    Training: 30-60 mins per recipe
    ↓
    Batch Inference: 2-4 hours for 180K users
    ↓
    S3 Output Files → load_batch_results.py (Optimized)
    ↓
    PostgreSQL Cache Tables:
      • offline_user_recommendations (180,483 records)
      • offline_similar_items (4,182 records)
      • offline_item_affinity (0 records - Optional)

[Real-time] API Serving ⚡
    Backend API → PostgreSQL Cache → <10ms response
    ↓
    Frontend Dashboard (Netlify)
    ↓
    Shopify Store (Checkout/Cart)
```

### **Key Components**

| Component | Status | Schedule | Purpose |
|-----------|--------|----------|---------|
| **Data Sync** | ✅ Active | Daily 2 AM | Keep database fresh |
| **AWS Personalize Training** | ✅ Active | Monthly (1st) | Retrain ML models |
| **Batch Inference** | ✅ Active | Monthly (1st) | Generate all recommendations |
| **PostgreSQL Cache** | ✅ Active | Always | Fast recommendation serving |
| **Backend API** | ✅ Running | 24/7 | Serve recommendations |
| **Custom ML (Auto-Pilot)** | 🚫 Disabled | None | Replaced by AWS Personalize |
| **Lightsail Server** | ✅ Active | 24/7 | 2GB Swap Added for Stability |

### **3 Active Recipes**

1. **User Personalization** - Personalized product recommendations per user
2. **Similar Items** - Product-to-product similarity for cross-selling
3. **Item Affinity** - User interest scores for categories/products

---

## 🛒 Shopify Integration (Real-Time)

The system is ready for Shopify integration. Since recommendations are cached in PostgreSQL, API response times are extremely fast (<10ms), making it perfect for checkout pages.

### **Available Endpoints**

| Page | Use Case | Endpoint |
|------|----------|----------|
| **Checkout** | "Recommended for You" | `GET /api/v1/personalize/recommendations/user/{user_id}` |
| **Product Page** | "Similar Products" | `GET /api/v1/personalize/recommendations/similar/{product_id}` |
| **Cart** | "Frequently Bought Together" | `GET /api/v1/analytics/collaborative-pairs` |

---

## Prerequisites

1. **AWS Account** with billing enabled
2. **AWS CLI** installed and configured
3. **IAM Permissions** for Personalize, S3

## 🚀 Quick Start (Batch Inference)

**⚠️ Note:** This is for NEW setups. MasterGroup's production system is ALREADY deployed and running.

### **For Production (Already Done)** ✅

The system is deployed on Lightsail at: `44.201.11.243:8001`

**Current Status:**
- ✅ Database tables created
- ✅ Auto-sync enabled (daily 2 AM)
- ✅ AWS Personalize solutions created
- ✅ Batch training completed
- ✅ Data loaded into PostgreSQL (180K+ records)
- ✅ Backend API serving recommendations

**Check Current Status:**
```bash
# SSH to server
ssh -i your-key.pem ubuntu@44.201.11.243

# Check batch training log
tail -f /opt/mastergroup-api/aws_personalize/training.log

# Check API status
curl http://44.201.11.243:8001/health
```

---

### **For New Deployment (Reference)**

If you need to set this up from scratch:

#### **Step 1: Setup Database Tables**

```bash
cd aws_personalize
python3 setup_offline_tables.py
```

Creates tables:
- `offline_user_recommendations`
- `offline_similar_items`
- `offline_item_affinity`
- `batch_job_metadata`

#### **Step 2: Generate Batch Inputs**

```bash
python3 generate_batch_inputs.py
```

Uploads to S3:
- `s3://mastergroup-personalize-data/batch/input/users.json`
- `s3://mastergroup-personalize-data/batch/input/items.json`
- `s3://mastergroup-personalize-data/batch/input/affinity.json`

#### **Step 3: Train & Run Batch Inference**

```bash
python3 train_hybrid_model.py
```

This will:
1. Create AWS Personalize solutions (if needed)
2. Train models (~30-60 mins per recipe)
3. Run batch inference jobs (~2-4 hours)
4. Output to S3

**⏳ Go grab coffee!** This takes 4-6 hours total.

#### **Step 4: Load Results to Cache**

After batch jobs complete:

```bash
python3 load_batch_results.py
```

Loads recommendations into PostgreSQL cache tables.

#### **Step 5: Verify**

```sql
-- Check loaded recommendations
SELECT COUNT(*) FROM offline_user_recommendations;
SELECT COUNT(*) FROM offline_similar_items;
SELECT COUNT(*) FROM offline_item_affinity;
```

---

### **One-Click Setup (Automated)**

```bash
cd aws_personalize
./run_cost_saving_setup.sh
```

Runs all steps automatically with prompts.

## 🔄 Monthly Maintenance

The system runs automatically, but you can trigger updates manually:

### **Check Batch Training Status**

```bash
# SSH to server
ssh -i your-key.pem ubuntu@44.201.11.243

# View training log
tail -f /opt/mastergroup-api/aws_personalize/training.log

# Or check AWS Console
# https://console.aws.amazon.com/personalize
```

### **Manually Trigger Batch Job**

```bash
cd /opt/mastergroup-api/aws_personalize

# Generate inputs
python3 generate_batch_inputs.py

# Train & run batch inference
python3 train_hybrid_model.py

# Load results (after jobs complete ~4 hours)
python3 load_batch_results.py
```

### **Verify Cache Freshness**

```sql
-- Check last update time
SELECT MAX(updated_at) FROM offline_user_recommendations;
SELECT MAX(updated_at) FROM offline_similar_items;

-- Check coverage
SELECT COUNT(*) FROM offline_user_recommendations;  -- Should be ~180K
SELECT COUNT(*) FROM offline_similar_items;         -- Should be ~4K
```

---

## 📊 Monitoring

### **Key Metrics to Track**

| Metric | Command/Location | Target |
|--------|-----------------|--------|
| **API Health** | `curl http://44.201.11.243:8001/health` | `healthy` |
| **Cache Freshness** | SQL query above | <35 days |
| **Coverage** | Count query | >90% users |
| **Monthly Cost** | AWS Cost Explorer | ~$7.50 |

### **AWS Console Links**

- **Personalize**: https://console.aws.amazon.com/personalize
- **S3 Bucket**: https://s3.console.aws.amazon.com/s3/buckets/mastergroup-personalize-data
- **Cost Explorer**: https://console.aws.amazon.com/cost-management

---

## 🐛 Troubleshooting

### **"No recommendations found for user"**
**Cause:** User not in batch (new customer after last batch run)  
**Solution:** Fallback to popular products or wait for next monthly batch

### **"Batch training failed"**
**Cause:** AWS credentials, data issues, or insufficient permissions  
**Solution:**
```bash
# Check AWS credentials
aws sts get-caller-identity

# Check S3 access
aws s3 ls s3://mastergroup-personalize-data/

# Review training logs
tail -100 /opt/mastergroup-api/aws_personalize/training.log
```

### **"Database connection error"**
**Cause:** RDS credentials or network issue  
**Solution:**
```bash
# Test PostgreSQL connection
psql -h ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com \
     -U postgres -d mastergroup_recommendations

# Check .env file
cat /opt/mastergroup-api/.env | grep PG_
```

### **"Stale recommendations (>35 days)"**
**Cause:** Batch job didn't run or failed  
**Solution:** Manually trigger batch job (see Monthly Maintenance above)

---

## 📚 Documentation

| Document | Description | When to Read |
|----------|-------------|--------------|
| **README.md** (this file) | Overview & quick start | First time |
| **QUICK_START.md** | Fast-track setup guide | New deployment |
| **PLAYBOOK.md** | Complete technical guide | Deep dive |
| **COST_SAVING_GUIDE.md** | Detailed cost optimization | Implementation |
| **ARCHITECTURE_CLEANUP.md** | Auto-Pilot disable explanation | Architecture changes |

---

## 📁 Key Files

| File | Purpose | Status |
|------|---------|--------|
| `setup_offline_tables.py` | Create PostgreSQL cache tables | ✅ Run once |
| `generate_batch_inputs.py` | Export users/products to S3 | ✅ Monthly |
| `train_hybrid_model.py` | Train models & batch inference | ✅ Monthly |
| `load_batch_results.py` | Load S3 results to PostgreSQL | ✅ Monthly |
| `run_cost_saving_setup.sh` | Automated setup script | ✅ One-click |
| `offline_recommendations.sql` | Database schema | ✅ Reference |

---

## 🎯 Success Criteria

✅ **System is working if:**
- API health check returns `healthy`
- `offline_user_recommendations` has >180K records
- Last update timestamp is <35 days old
- API response time is <10ms
- Monthly AWS cost is ~$7.50

---

**Status:** ✅ **Production Ready**  
**Last Updated:** November 29, 2025  
**Deployed:** Lightsail (44.201.11.243:8001)  
**Cost:** $7.50/month (98% savings)
