# AWS Personalize Migration Guide

## Why AWS Personalize?

| Heroku Issues | AWS Personalize Solution |
|---------------|-------------------------|
| Memory crashes with large ML models | Fully managed, auto-scales |
| Slow training times | Optimized ML infrastructure |
| Limited compute resources | Enterprise-grade capacity |
| Dyno restart issues | 99.9% SLA availability |

## Cost Estimate

For MasterGroup's scale (~50K interactions, ~4K users, ~1.5K items):
- **Training**: ~$0.24/hour (runs once per day or week)
- **Inference**: ~$0.20/1000 recommendations
- **Data storage**: Minimal (S3 charges)
- **Estimated monthly**: $50-100 (vs Heroku $50+ with crashes)

## Prerequisites

1. **AWS Account** with billing enabled
2. **AWS CLI** installed and configured
3. **IAM Permissions** for Personalize, S3

## Quick Start

### Step 1: Configure AWS CLI

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key  
# Region: us-east-1 (recommended)
```

### Step 2: Create S3 Bucket

```bash
aws s3 mb s3://mastergroup-personalize-data --region us-east-1
```

### Step 3: Create IAM Role

Create a role with these policies:
- `AmazonPersonalizeFullAccess`
- Custom S3 policy for your bucket

```bash
# Create role (use AWS Console or this command)
aws iam create-role \
  --role-name PersonalizeRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "personalize.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach policies
aws iam attach-role-policy \
  --role-name PersonalizeRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonPersonalizeFullAccess
```

### Step 4: Export Data

```bash
cd recommendation-engine-service

# Set database connection
export PG_HOST=localhost
export PG_DB=mastergroup_recommendations
export PG_USER=postgres
export PG_PASSWORD=postgres

# Export data
python aws_personalize/export_data_for_personalize.py
```

This creates:
- `aws_personalize/data/interactions.csv`
- `aws_personalize/data/items.csv`
- `aws_personalize/data/users.csv`

### Step 5: Setup AWS Personalize

```bash
# Set environment variables
export AWS_REGION=us-east-1
export PERSONALIZE_S3_BUCKET=mastergroup-personalize-data
export PERSONALIZE_ROLE_ARN=arn:aws:iam::657020414783:role/PersonalizeRole

# Run setup
python aws_personalize/setup_personalize.py
```

### Step 6: Wait for Training

Training takes 30-60 minutes. Monitor in AWS Console:
1. Go to Amazon Personalize
2. Select "mastergroup-recommendations" dataset group
3. Check solution version status

### Step 7: Create Campaign

After training completes:
1. Go to Campaigns in AWS Console
2. Click "Create Campaign"
3. Select your solution version
4. Set minimum TPS (start with 1)
5. Note the Campaign ARN

### Step 8: Update Backend

Set these environment variables:

```bash
export PERSONALIZE_CAMPAIGN_ARN=arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign
export AWS_REGION=us-east-1
```

### Step 9: Deploy to AWS

Options:
1. **AWS ECS** (recommended for containers)
2. **AWS EC2** (simple VM)
3. **AWS Lambda** (serverless, for low traffic)

## Using the Personalize Service

```python
from aws_personalize.personalize_service import get_personalize_service

# Get recommendations
service = get_personalize_service()
recommendations = service.get_recommendations_for_user(
    user_id="customer_123",
    num_results=10
)

# Get similar items (cross-selling)
similar = service.get_similar_items(
    item_id="product_456",
    num_results=5
)

# Record real-time event
service.record_event(
    user_id="customer_123",
    item_id="product_789",
    event_type="purchase"
)
```

## Recipes Available

| Recipe | Use Case |
|--------|----------|
| `aws-user-personalization` | Personalized recommendations per user |
| `aws-similar-items` | Cross-selling, related products |
| `aws-personalized-ranking` | Re-rank search results |
| `aws-popularity-count` | Trending/popular items |

## Monitoring

1. **CloudWatch Metrics**: Latency, throughput, errors
2. **AWS Console**: Campaign status, solution metrics
3. **Cost Explorer**: Track spending

## Troubleshooting

### "Campaign not found"
- Ensure Campaign ARN is correct
- Check campaign is ACTIVE status

### "User not found"
- New users need interaction history
- Use popularity-based fallback

### "Slow responses"
- Increase campaign TPS
- Use caching (Redis)

## Files

| File | Purpose |
|------|---------|
| `export_data_for_personalize.py` | Export DB to CSV |
| `setup_personalize.py` | Create AWS resources |
| `personalize_service.py` | Python client for recommendations |
