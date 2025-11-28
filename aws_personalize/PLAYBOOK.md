# AWS Personalize Migration Playbook
## MasterGroup Recommendation System

**Date**: November 28, 2025  
**Author**: AI Assistant  
**Purpose**: Migrate from Heroku-hosted ML models to AWS Personalize

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: AWS CLI Configuration](#step-1-aws-cli-configuration)
4. [Step 2: Create S3 Bucket](#step-2-create-s3-bucket)
5. [Step 3: Create IAM Role](#step-3-create-iam-role)
6. [Step 4: Export Data from PostgreSQL](#step-4-export-data-from-postgresql)
7. [Step 5: Upload Data to S3](#step-5-upload-data-to-s3)
8. [Step 6: Create Personalize Resources](#step-6-create-personalize-resources)
9. [Step 7: Import Data](#step-7-import-data)
10. [Step 8: Create Solution (Train Model)](#step-8-create-solution-train-model)
11. [Step 9: Create Campaign (Deploy)](#step-9-create-campaign-deploy)
12. [Step 10: Update Backend](#step-10-update-backend)
13. [Monitoring & Maintenance](#monitoring--maintenance)
14. [Troubleshooting](#troubleshooting)
15. [Cost Optimization](#cost-optimization)

---

## Overview

### Why AWS Personalize?
| Heroku Problem | AWS Personalize Solution |
|----------------|-------------------------|
| Memory crashes with ML models | Fully managed, auto-scales |
| Dyno restarts lose trained models | Persistent model storage |
| Limited compute for training | Enterprise ML infrastructure |
| High latency during training | Async training, real-time inference |

### Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   PostgreSQL    │────▶│   S3 Bucket      │────▶│ AWS Personalize │
│   (Source DB)   │     │ (Data Storage)   │     │  (ML Engine)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│   Frontend      │◀────│   Backend API    │◀─────────────┘
│   Dashboard     │     │   (FastAPI)      │
└─────────────────┘     └──────────────────┘
```

### Data Stats (MasterGroup)
- **Interactions**: 1,971,527 records
- **Items (Products)**: 4,182 products
- **Users (Customers)**: 180,484 customers

---

## Prerequisites

### Required
- [ ] AWS Account with billing enabled
- [ ] AWS CLI installed (`brew install awscli` on Mac)
- [ ] Python 3.8+ with pip
- [ ] PostgreSQL database access
- [ ] boto3 library (`pip install boto3`)

### AWS Account Details
- **Account ID**: `657020414783`
- **Region**: `us-east-1`
- **Profile Name**: `mastergroup`

---

## Step 1: AWS CLI Configuration

### 1.1 Create Access Keys in AWS Console
1. Go to AWS Console → IAM → Users → Your User
2. Click "Security credentials" tab
3. Click "Create access key"
4. Select "Command Line Interface (CLI)"
5. Download or copy the keys

### 1.2 Configure AWS CLI
```bash
aws configure --profile mastergroup
```

Enter when prompted:
```
AWS Access Key ID: AKIAZR6LX5M7XXXXXXXX
AWS Secret Access Key: uuMn5DiuEHcnPB9ejlxvXXXXXXXXXXXXXXXX
Default region name: us-east-1
Default output format: json
```

### 1.3 Verify Configuration
```bash
aws sts get-caller-identity --profile mastergroup
```

Expected output:
```json
{
    "UserId": "657020414783",
    "Account": "657020414783",
    "Arn": "arn:aws:iam::657020414783:root"
}
```

---

## Step 2: Create S3 Bucket

### 2.1 Create Bucket
```bash
aws s3 mb s3://mastergroup-personalize-data --region us-east-1 --profile mastergroup
```

### 2.2 Add Bucket Policy for Personalize Access
```bash
cat > /tmp/bucket-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PersonalizeAccess",
            "Effect": "Allow",
            "Principal": {
                "Service": "personalize.amazonaws.com"
            },
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::mastergroup-personalize-data",
                "arn:aws:s3:::mastergroup-personalize-data/*"
            ]
        }
    ]
}
EOF

aws s3api put-bucket-policy \
    --bucket mastergroup-personalize-data \
    --policy file:///tmp/bucket-policy.json \
    --profile mastergroup
```

---

## Step 3: Create IAM Role

### 3.1 Create Role with Trust Policy
```bash
aws iam create-role \
    --role-name PersonalizeRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "personalize.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' \
    --profile mastergroup
```

### 3.2 Attach Personalize Policy
```bash
aws iam attach-role-policy \
    --role-name PersonalizeRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonPersonalizeFullAccess \
    --profile mastergroup
```

### 3.3 Add S3 Access Policy
```bash
aws iam put-role-policy \
    --role-name PersonalizeRole \
    --policy-name S3Access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket", "s3:PutObject"],
            "Resource": [
                "arn:aws:s3:::mastergroup-personalize-data",
                "arn:aws:s3:::mastergroup-personalize-data/*"
            ]
        }]
    }' \
    --profile mastergroup
```

---

## Step 4: Export Data from PostgreSQL

### 4.1 Set Database Environment Variables
```bash
export PG_HOST=localhost
export PG_PORT=5432
export PG_DB=mastergroup_recommendations
export PG_USER=postgres
export PG_PASSWORD=postgres
```

### 4.2 Run Export Script
```bash
cd /path/to/recommendation-engine-service
python3 aws_personalize/export_data_for_personalize.py
```

### 4.3 Verify Exported Files
```bash
ls -la aws_personalize/data/
# Should show:
# - interactions.csv (~91 MB)
# - items.csv (~200 KB)
# - users.csv (~7.6 MB)

# Check file headers
head -3 aws_personalize/data/interactions.csv
head -3 aws_personalize/data/items.csv
head -3 aws_personalize/data/users.csv
```

### Data Format Requirements

**interactions.csv** (Required):
```csv
USER_ID,ITEM_ID,TIMESTAMP,EVENT_TYPE,EVENT_VALUE
customer_123,product_456,1701187200,purchase,1
```

**items.csv** (Optional but recommended):
```csv
ITEM_ID,ITEM_NAME,CATEGORY,PRICE,PURCHASE_COUNT
product_456,Widget Pro,Electronics,299.99,150
```

**users.csv** (Optional but recommended):
```csv
USER_ID,CITY,PROVINCE,ORDER_COUNT,TOTAL_SPEND
customer_123,Lahore,Punjab,15,45000.00
```

---

## Step 5: Upload Data to S3

### 5.1 Upload All Files
```bash
aws s3 cp aws_personalize/data/interactions.csv \
    s3://mastergroup-personalize-data/data/interactions.csv \
    --profile mastergroup

aws s3 cp aws_personalize/data/items.csv \
    s3://mastergroup-personalize-data/data/items.csv \
    --profile mastergroup

aws s3 cp aws_personalize/data/users.csv \
    s3://mastergroup-personalize-data/data/users.csv \
    --profile mastergroup
```

### 5.2 Verify Upload
```bash
aws s3 ls s3://mastergroup-personalize-data/data/ --profile mastergroup
```

---

## Step 6: Create Personalize Resources

### 6.1 Create Dataset Group
```bash
aws personalize create-dataset-group \
    --name mastergroup-recommendations \
    --profile mastergroup \
    --region us-east-1
```

Wait for ACTIVE status:
```bash
aws personalize describe-dataset-group \
    --dataset-group-arn arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations \
    --profile mastergroup \
    --region us-east-1
```

### 6.2 Create Schemas

**Interactions Schema:**
```bash
aws personalize create-schema \
    --name mastergroup-interactions-schema \
    --schema '{
        "type": "record",
        "name": "Interactions",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {"name": "USER_ID", "type": "string"},
            {"name": "ITEM_ID", "type": "string"},
            {"name": "TIMESTAMP", "type": "long"},
            {"name": "EVENT_TYPE", "type": "string"},
            {"name": "EVENT_VALUE", "type": "float"}
        ],
        "version": "1.0"
    }' \
    --profile mastergroup \
    --region us-east-1
```

**Items Schema:**
```bash
aws personalize create-schema \
    --name mastergroup-items-schema \
    --schema '{
        "type": "record",
        "name": "Items",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {"name": "ITEM_ID", "type": "string"},
            {"name": "ITEM_NAME", "type": "string"},
            {"name": "CATEGORY", "type": "string", "categorical": true},
            {"name": "PRICE", "type": "float"},
            {"name": "PURCHASE_COUNT", "type": "int"}
        ],
        "version": "1.0"
    }' \
    --profile mastergroup \
    --region us-east-1
```

**Users Schema:**
```bash
aws personalize create-schema \
    --name mastergroup-users-schema \
    --schema '{
        "type": "record",
        "name": "Users",
        "namespace": "com.amazonaws.personalize.schema",
        "fields": [
            {"name": "USER_ID", "type": "string"},
            {"name": "CITY", "type": "string", "categorical": true},
            {"name": "PROVINCE", "type": "string", "categorical": true},
            {"name": "ORDER_COUNT", "type": "int"},
            {"name": "TOTAL_SPEND", "type": "float"}
        ],
        "version": "1.0"
    }' \
    --profile mastergroup \
    --region us-east-1
```

### 6.3 Create Datasets

```bash
# Interactions Dataset
aws personalize create-dataset \
    --name mastergroup-recommendations-interactions \
    --dataset-group-arn arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations \
    --dataset-type Interactions \
    --schema-arn arn:aws:personalize:us-east-1:657020414783:schema/mastergroup-interactions-schema \
    --profile mastergroup \
    --region us-east-1

# Items Dataset
aws personalize create-dataset \
    --name mastergroup-recommendations-items \
    --dataset-group-arn arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations \
    --dataset-type Items \
    --schema-arn arn:aws:personalize:us-east-1:657020414783:schema/mastergroup-items-schema \
    --profile mastergroup \
    --region us-east-1

# Users Dataset
aws personalize create-dataset \
    --name mastergroup-recommendations-users \
    --dataset-group-arn arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations \
    --dataset-type Users \
    --schema-arn arn:aws:personalize:us-east-1:657020414783:schema/mastergroup-users-schema \
    --profile mastergroup \
    --region us-east-1
```

---

## Step 7: Import Data

### 7.1 Import Interactions (Required)
```bash
TIMESTAMP=$(date +%Y%m%d%H%M%S)

aws personalize create-dataset-import-job \
    --job-name "interactions-import-$TIMESTAMP" \
    --dataset-arn "arn:aws:personalize:us-east-1:657020414783:dataset/mastergroup-recommendations/INTERACTIONS" \
    --data-source dataLocation="s3://mastergroup-personalize-data/data/interactions.csv" \
    --role-arn "arn:aws:iam::657020414783:role/PersonalizeRole" \
    --profile mastergroup \
    --region us-east-1
```

### 7.2 Import Items
```bash
aws personalize create-dataset-import-job \
    --job-name "items-import-$TIMESTAMP" \
    --dataset-arn "arn:aws:personalize:us-east-1:657020414783:dataset/mastergroup-recommendations/ITEMS" \
    --data-source dataLocation="s3://mastergroup-personalize-data/data/items.csv" \
    --role-arn "arn:aws:iam::657020414783:role/PersonalizeRole" \
    --profile mastergroup \
    --region us-east-1
```

### 7.3 Import Users
```bash
aws personalize create-dataset-import-job \
    --job-name "users-import-$TIMESTAMP" \
    --dataset-arn "arn:aws:personalize:us-east-1:657020414783:dataset/mastergroup-recommendations/USERS" \
    --data-source dataLocation="s3://mastergroup-personalize-data/data/users.csv" \
    --role-arn "arn:aws:iam::657020414783:role/PersonalizeRole" \
    --profile mastergroup \
    --region us-east-1
```

### 7.4 Check Import Status
```bash
aws personalize list-dataset-import-jobs \
    --profile mastergroup \
    --region us-east-1
```

**Expected Duration:**
- Interactions (2M rows): ~15-20 minutes
- Items (4K rows): ~5 minutes
- Users (180K rows): ~5-10 minutes

---

## Step 8: Create Solution (Train Model)

### 8.1 Available Recipes

| Recipe | Use Case | Best For |
|--------|----------|----------|
| `aws-user-personalization` | Personalized recommendations | Main recommendation engine |
| `aws-similar-items` | Related products | Cross-selling |
| `aws-personalized-ranking` | Re-rank lists | Search results |
| `aws-popularity-count` | Trending items | New users |

### 8.2 Create User Personalization Solution
```bash
aws personalize create-solution \
    --name mastergroup-user-personalization \
    --dataset-group-arn arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations \
    --recipe-arn arn:aws:personalize:::recipe/aws-user-personalization \
    --profile mastergroup \
    --region us-east-1
```

### 8.3 Create Solution Version (Train)
```bash
aws personalize create-solution-version \
    --solution-arn arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-user-personalization \
    --profile mastergroup \
    --region us-east-1
```

### 8.4 Check Training Status
```bash
aws personalize describe-solution-version \
    --solution-version-arn arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-user-personalization/SOLUTION_VERSION_ID \
    --profile mastergroup \
    --region us-east-1
```

**Expected Duration:** 45-60 minutes

---

## Step 9: Create Campaign (Deploy)

### 9.1 Create Campaign
```bash
aws personalize create-campaign \
    --name mastergroup-campaign \
    --solution-version-arn arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-user-personalization/SOLUTION_VERSION_ID \
    --min-provisioned-tps 1 \
    --profile mastergroup \
    --region us-east-1
```

### 9.2 Check Campaign Status
```bash
aws personalize describe-campaign \
    --campaign-arn arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign \
    --profile mastergroup \
    --region us-east-1
```

**Expected Duration:** ~15 minutes

### 9.3 Test Recommendations
```bash
aws personalize-runtime get-recommendations \
    --campaign-arn arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign \
    --user-id "test_customer_123" \
    --num-results 10 \
    --profile mastergroup \
    --region us-east-1
```

---

## Step 10: Update Backend

### 10.1 Set Environment Variables
```bash
export PERSONALIZE_CAMPAIGN_ARN=arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign
export AWS_REGION=us-east-1
export AWS_PROFILE=mastergroup
```

### 10.2 Use Personalize Service in Code
```python
from aws_personalize.personalize_service import get_personalize_service

# Get recommendations for a user
service = get_personalize_service()
recommendations = service.get_recommendations_for_user(
    user_id="customer_123",
    num_results=10
)

# Returns:
# [
#     {"product_id": "1328", "score": 0.95, "algorithm": "aws_personalize"},
#     {"product_id": "1331", "score": 0.87, "algorithm": "aws_personalize"},
#     ...
# ]
```

### 10.3 Update API Endpoints
Modify `src/main.py` to use Personalize:

```python
from aws_personalize import get_personalize_service

@app.get("/api/v1/recommendations/{customer_id}")
async def get_recommendations(customer_id: str, limit: int = 10):
    service = get_personalize_service()
    
    if service.is_configured:
        return service.get_recommendations_for_user(customer_id, limit)
    else:
        # Fallback to existing ML service
        return existing_ml_recommendations(customer_id, limit)
```

---

## Monitoring & Maintenance

### Check Campaign Metrics
```bash
aws personalize describe-campaign \
    --campaign-arn arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign \
    --profile mastergroup \
    --region us-east-1
```

### Retrain Model (Weekly)
```bash
# Create new solution version
aws personalize create-solution-version \
    --solution-arn arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-user-personalization \
    --profile mastergroup \
    --region us-east-1

# After training, update campaign
aws personalize update-campaign \
    --campaign-arn arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign \
    --solution-version-arn NEW_SOLUTION_VERSION_ARN \
    --profile mastergroup \
    --region us-east-1
```

### Real-Time Event Tracking
```python
service.record_event(
    user_id="customer_123",
    item_id="product_456",
    event_type="purchase"
)
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Insufficient privileges for S3` | Missing bucket policy | Add bucket policy (Step 2.2) |
| `Schema validation failed` | CSV format mismatch | Check column names match schema |
| `User not found` | New user no history | Use popularity fallback |
| `Campaign not active` | Still deploying | Wait 15 minutes |

### Debug Commands
```bash
# Check all resources
aws personalize list-dataset-groups --profile mastergroup --region us-east-1
aws personalize list-solutions --profile mastergroup --region us-east-1
aws personalize list-campaigns --profile mastergroup --region us-east-1

# Check import job details
aws personalize describe-dataset-import-job \
    --dataset-import-job-arn JOB_ARN \
    --profile mastergroup \
    --region us-east-1
```

---

## Cost Optimization

### Estimated Monthly Cost
| Resource | Usage | Cost |
|----------|-------|------|
| Training | 1 hour/week | ~$1/week |
| Campaign (1 TPS) | Always on | ~$0.20/hour = $144/month |
| Data storage (S3) | 100 MB | <$1/month |
| **Total** | | **~$150-200/month** |

### Cost Saving Tips
1. Use `minProvisionedTPS: 1` for low traffic
2. Enable auto-scaling for campaigns
3. Retrain weekly instead of daily
4. Delete old solution versions

---

## File Reference

| File | Purpose |
|------|---------|
| `aws_personalize/export_data_for_personalize.py` | Export PostgreSQL → CSV |
| `aws_personalize/setup_personalize.py` | Create AWS resources |
| `aws_personalize/personalize_service.py` | Python API client |
| `aws_personalize/data/interactions.csv` | User-item interactions |
| `aws_personalize/data/items.csv` | Product metadata |
| `aws_personalize/data/users.csv` | Customer metadata |

---

## Quick Reference ARNs

```
Dataset Group:  arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations
IAM Role:       arn:aws:iam::657020414783:role/PersonalizeRole
S3 Bucket:      s3://mastergroup-personalize-data
Solution:       arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-user-personalization
Campaign:       arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign
```

---

---

## Appendix: Data Pipeline Integration

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE FLOW                          │
└────────────────────────────────────────────────────────────────┘

   Master Group APIs              PostgreSQL              AWS Personalize
   ┌─────────────┐               ┌──────────┐            ┌─────────────┐
   │  POS API    │──┐            │  orders  │            │ Real-time   │
   │  OE API     │──┼──► sync ──►│  order_  │──► sync ──►│ Events API  │
   └─────────────┘  │  service   │  items   │   service  │             │
                    │            └──────────┘            └──────┬──────┘
   ┌─────────────┐  │                 │                         │
   │ Auto-Sync   │──┘                 │                         ▼
   │ (Scheduler) │                    │                  ┌─────────────┐
   └─────────────┘                    │   batch_sync     │ ML Model    │
                                      └────────────────► │ (Retrain)   │
                                           (Daily)       └─────────────┘
```

### Two Sync Modes

| Mode | Frequency | Purpose | Script |
|------|-----------|---------|--------|
| **Real-time** | Every sync | Send purchase events immediately | `personalize_sync.py` |
| **Batch** | Daily/Weekly | Export data, retrain model | `batch_sync.py` |

### Real-Time Sync (Automatic)

When `sync_service.py` runs (auto-sync from Master Group APIs):

1. Orders fetched from POS/OE APIs
2. Orders inserted into PostgreSQL
3. **NEW**: Events sent to AWS Personalize via `personalize_sync.py`

**Code Integration** (already added to `sync_service.py`):
```python
# After inserting orders
from aws_personalize.personalize_sync import sync_orders_to_personalize
personalize_result = sync_orders_to_personalize(transformed_orders)
```

### Setting Up Real-Time Events

#### 1. Create Event Tracker (One-time)
```bash
aws personalize create-event-tracker \
    --name mastergroup-event-tracker \
    --dataset-group-arn arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations \
    --profile mastergroup \
    --region us-east-1
```

#### 2. Get Tracking ID
```bash
aws personalize list-event-trackers \
    --dataset-group-arn arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations \
    --profile mastergroup \
    --region us-east-1
```

#### 3. Set Environment Variable
```bash
export PERSONALIZE_TRACKING_ID=<tracking-id-from-above>
```

### Batch Sync (Scheduled)

Run daily to refresh data and retrain:

```bash
# Daily incremental sync (recommended)
python aws_personalize/batch_sync.py

# Weekly full sync
python aws_personalize/batch_sync.py --full
```

### Setting Up Scheduled Jobs

#### Option 1: Cron Job
```bash
# Edit crontab
crontab -e

# Add daily sync at 2 AM
0 2 * * * cd /path/to/recommendation-engine-service && python aws_personalize/batch_sync.py >> /var/log/personalize_sync.log 2>&1

# Add weekly full sync on Sundays at 3 AM
0 3 * * 0 cd /path/to/recommendation-engine-service && python aws_personalize/batch_sync.py --full >> /var/log/personalize_full_sync.log 2>&1
```

#### Option 2: AWS CloudWatch Events (Better for AWS deployment)
```bash
# Create EventBridge rule for daily sync
aws events put-rule \
    --name "personalize-daily-sync" \
    --schedule-expression "cron(0 2 * * ? *)" \
    --profile mastergroup \
    --region us-east-1
```

### Environment Variables Summary

```bash
# AWS Configuration
export AWS_REGION=us-east-1
export AWS_PROFILE=mastergroup

# PostgreSQL (for export)
export PG_HOST=localhost
export PG_DB=mastergroup_recommendations
export PG_USER=postgres
export PG_PASSWORD=postgres

# AWS Personalize
export PERSONALIZE_TRACKING_ID=<your-tracking-id>
export PERSONALIZE_CAMPAIGN_ARN=arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign
export PERSONALIZE_DATASET_GROUP_ARN=arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations
```

### File Reference (Pipeline)

| File | Purpose |
|------|---------|
| `services/sync_service.py` | Master Group API → PostgreSQL sync (existing) |
| `aws_personalize/personalize_sync.py` | Real-time event sync to Personalize |
| `aws_personalize/batch_sync.py` | Daily/weekly batch export & retrain |
| `aws_personalize/personalize_service.py` | Get recommendations from Personalize |

### Monitoring the Pipeline

#### Check Sync Status
```bash
# Check event tracker status
aws personalize describe-event-tracker \
    --event-tracker-arn <tracker-arn> \
    --profile mastergroup \
    --region us-east-1

# Check recent imports
aws personalize list-dataset-import-jobs \
    --profile mastergroup \
    --region us-east-1 \
    --max-results 5
```

#### CloudWatch Metrics
- `personalize:PutEvents` - Real-time events sent
- `personalize:GetRecommendations` - API calls made
- Monitor for errors and latency

---

**Last Updated**: November 28, 2025
