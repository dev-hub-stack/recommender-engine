# 🚀 Quick Start: Cost-Saving AWS Personalize

## 💰 The Big Picture
- **Current Cost**: ~$432/month (4 real-time campaigns)
- **New Cost**: ~$7.50/month (batch inference)
- **Savings**: 98% reduction! 💸

## 📁 What We Created

| File | Purpose |
|------|---------|
| `setup_offline_tables.py` | Creates PostgreSQL tables for cached recommendations |
| `generate_batch_inputs.py` | Exports user/product IDs to S3 for batch processing |
| `train_hybrid_model.py` | Trains 4 AWS Personalize recipes & runs batch inference |
| `load_batch_results.py` | Imports batch results from S3 to PostgreSQL |
| `run_cost_saving_setup.sh` | One-click setup script (runs all steps) |
| `COST_SAVING_GUIDE.md` | Detailed documentation |

## ⚡ Option 1: One-Click Setup

```bash
cd recommendation-engine-service/aws_personalize

# Make sure environment variables are set
export PG_HOST=your_db_host
export PG_DATABASE=mastergroup
export PG_USER=postgres
export PG_PASSWORD=your_password
export AWS_ACCESS_KEY_ID=your_aws_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret
export AWS_REGION=us-east-1

# Run the setup
./run_cost_saving_setup.sh
```

## 🔧 Option 2: Manual Step-by-Step

### 1️⃣ Create Database Tables (2 mins)
```bash
python setup_offline_tables.py
```

### 2️⃣ Generate Batch Inputs (5 mins)
```bash
python generate_batch_inputs.py
```
This uploads files to S3:
- `s3://mastergroup-personalize-data/batch/input/users.json`
- `s3://mastergroup-personalize-data/batch/input/items.json`
- `s3://mastergroup-personalize-data/batch/input/affinity.json`

### 3️⃣ Train & Run Batch Inference (2-4 hours ⏳)
```bash
python train_hybrid_model.py
```
This will:
- ✅ Create 4 AWS Personalize solutions (if not exists)
- ⏳ Train models (~30-60 mins each)
- 🚀 Start batch inference jobs (~2-4 hours)

**Go grab lunch/coffee** ☕ - AWS runs this asynchronously!

### 4️⃣ Check Batch Job Status

**Option A: AWS Console**
1. Go to [AWS Personalize Console](https://console.aws.amazon.com/personalize)
2. Click "Batch inference jobs"
3. Wait for status = `ACTIVE`

**Option B: AWS CLI**
```bash
aws personalize list-batch-inference-jobs \
  --max-results 10 \
  --region us-east-1
```

### 5️⃣ Load Results (5 mins)
Once batch jobs complete:
```bash
python load_batch_results.py
```

### 6️⃣ Verify Data
```bash
# Check if recommendations are loaded
psql -h $PG_HOST -U $PG_USER -d $PG_DATABASE -c "
SELECT 
  (SELECT COUNT(*) FROM offline_user_recommendations) as users,
  (SELECT COUNT(*) FROM offline_similar_items) as products,
  (SELECT COUNT(*) FROM offline_item_affinity) as affinity;
"
```

## 📊 The 4 Recipes Explained

| Recipe | What It Does | Example Use |
|--------|-------------|-------------|
| **User Personalization** | Personalized recommendations per user | "Recommended for You" section |
| **Similar Items** | Find products similar to item X | "You may also like" on product pages |
| **Item Affinity** | User's interest scores for products | Personalized category browsing |
| **Personalized Ranking** | Rank a list for specific user | Personalized search results |

## 🔄 Monthly Maintenance

**Set a monthly cron job:**
```bash
# Run on 1st of each month at 2 AM
0 2 1 * * cd /path/to/recommendation-engine-service/aws_personalize && \
  ./run_cost_saving_setup.sh
```

Or use AWS EventBridge to trigger automatically!

## 🐛 Common Issues

### "No batch result files found"
- Wait longer - batch jobs take 2-4 hours
- Check AWS Console for job status
- Verify S3 bucket permissions

### "Database connection failed"
- Check environment variables are set
- Test connection: `psql -h $PG_HOST -U $PG_USER -d $PG_DATABASE`

### "AWS credentials not configured"
```bash
aws configure
# Enter your credentials when prompted
```

## 📈 Next: Update Backend API

After loading recommendations, update your backend to serve from these tables instead of AWS campaigns. This gives you:
- ⚡ Faster responses (PostgreSQL is faster than AWS API)
- 💰 Zero AWS inference costs
- 🔄 Monthly refresh keeps recommendations fresh

## 🎯 Success Criteria

✅ Database tables created  
✅ Batch input files uploaded to S3  
✅ 3 solutions trained (user_personalization, similar_items, item_affinity)  
✅ Batch jobs completed  
✅ Recommendations loaded to PostgreSQL  
✅ Backend API updated to use offline tables  

**Result:** ~$424/month saved! 🎉
