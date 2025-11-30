# AWS Personalize Model and Data Pipeline Documentation

## Overview
Master Group uses AWS Personalize for ML-powered product recommendations. This document explains the complete data pipeline from API ingestion to recommendation delivery.

## AWS Personalize Components Used

### 1. Dataset Group
- **ARN**: `arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations`
- **Purpose**: Container for all datasets and models
- **Region**: us-east-1

### 2. Datasets
#### Interactions Dataset
- **Source**: Orders table from PostgreSQL
- **Schema**: 
  ```
  USER_ID (string) - Customer ID
  ITEM_ID (string) - Product ID  
  EVENT_TYPE (string) - "purchase"
  EVENT_VALUE (float) - Quantity purchased
  TIMESTAMP (int) - Order timestamp
  ```
- **Update Frequency**: Real-time via Event Tracker

#### Items Dataset
- **Source**: Products table
- **Schema**:
  ```
  ITEM_ID (string) - Product ID
  PRODUCT_NAME (string) - Product name
  CATEGORY (string) - Product category
  PRICE (float) - Product price
  ```

#### Users Dataset (Optional)
- **Source**: Customers table
- **Schema**:
  ```
  USER_ID (string) - Customer ID
  AGE (int) - Customer age (if available)
  GENDER (string) - Customer gender (if available)
  LOCATION (string) - Customer city/province
  ```

### 3. Models (Recipes)
#### User Personalization (aws-user-personalization)
- **Purpose**: Recommend products to specific users
- **Algorithm**: Collaborative filtering with implicit feedback
- **Output**: Personalized product recommendations per user

#### Similar Items (aws-sims)
- **Purpose**: Find similar products for cross-selling
- **Algorithm**: Item-to-item collaborative filtering
- **Output**: Similar products for each product

#### Personalized Ranking (aws-personalized-ranking)
- **Purpose**: Re-rank a list of products for a specific user
- **Algorithm**: Learning-to-rank
- **Output**: Ranked product list per user

### 4. Campaigns
- **User Personalization Campaign**: Generates recommendations
- **Similar Items Campaign**: Generates similar products
- **Ranking Campaign**: Re-ranks product lists

### 5. Event Tracker
- **Tracking ID**: Configured via environment variable
- **Purpose**: Real-time event ingestion
- **Events**: Purchase events sent immediately

## Data Pipeline Steps

### Step 1: Data Ingestion from Company API
```
Company API → Sync Service → PostgreSQL (orders table)
```

**Process:**
1. **API Pull**: Sync service pulls orders from company's API
2. **Data Transformation**: 
   - Map company fields to internal schema
   - Standardize customer IDs (unified_customer_id)
   - Extract product details
3. **Database Storage**: Store in PostgreSQL `orders` table

**Key Tables:**
- `orders`: Order headers with customer info
- `order_items`: Product details per order
- `products`: Product catalog
- `customers`: Customer information

### Step 2: AWS Dataset Synchronization
```
PostgreSQL → S3 → AWS Personalize Datasets
```

**Batch Process (Daily/Weekly):**
1. **Export Data**: Extract from PostgreSQL to CSV files
2. **Upload to S3**: Place in `s3://mastergroup-personalize-data/`
3. **Import to Personalize**: Create dataset import jobs
4. **Validation**: AWS validates schema and data quality

**Real-time Process:**
1. **Event Capture**: New orders trigger events
2. **Send to Event Tracker**: Immediate sync to AWS
3. **Model Updates**: Models learn from new patterns

### Step 3: Model Training
```
AWS Personalize → Model Training → Campaign Deployment
```

**Training Process:**
1. **Data Preparation**: AWS processes imported data
2. **Feature Engineering**: Automatic feature extraction
3. **Model Training**: Collaborative filtering algorithms
4. **Evaluation**: Accuracy metrics calculation
5. **Campaign Creation**: Deploy trained models

**Training Frequency:**
- **Full Retraining**: Weekly with complete dataset
- **Incremental Updates**: Daily via real-time events

### Step 4: Batch Inference (Cost-Optimized)
```
AWS Personalize → Batch Jobs → PostgreSQL Cache Tables
```

**Offline Cache Generation:**
1. **User Recommendations**: Generate for all active users
2. **Similar Items**: Generate for all products
3. **Store Results**: Save in PostgreSQL cache tables
4. **Update Frequency**: Every 6 hours

**Cache Tables:**
```sql
offline_user_recommendations (
    user_id VARCHAR(255) PRIMARY KEY,
    recommendations JSONB, -- [{product_id, score}]
    updated_at TIMESTAMP
)

offline_similar_items (
    product_id VARCHAR(255) PRIMARY KEY,
    similar_products JSONB, -- [{product_id, score}]
    updated_at TIMESTAMP
)
```

### Step 5: API Response Generation
```
Frontend Request → Backend API → PostgreSQL Cache → Response
```

**Request Flow:**
1. **Frontend Request**: User visits recommendations page
2. **Location Sampling**: Sample 50 users from selected region
3. **Cache Lookup**: Read pre-computed recommendations
4. **Aggregation**: Calculate match rates and affinity scores
5. **Response**: Return formatted recommendations

## Metrics Calculation

### Match Rate
```
Match Rate = (Number of users who received this recommendation / Total sampled users) × 100
```

**Example:**
- Product recommended to 45 out of 50 sampled users
- Match Rate = (45/50) × 100 = 90%

### Regional Affinity
```
High Affinity: Match Rate ≥ 80%
Medium Affinity: Match Rate 40-79%
Low Affinity: Match Rate < 40%
```

### Recommendation Score
```
AWS Personalize Score = Confidence × Relevance
Range: 0.0 to 1.0 (higher = better)
```

## Cost Optimization Strategy

### 1. Batch Inference vs Real-time API
- **Batch**: Pre-compute recommendations, store in cache
- **Real-time**: Only for new users or special cases
- **Savings**: 90% cost reduction vs real-time API calls

### 2. Sampling Strategy
- **Sample Size**: 50 users per region (statistically significant)
- **Sampling Method**: Random selection from active customers
- **Update Frequency**: Every 6 hours

### 3. Cache Management
- **TTL**: 6 hours for recommendations
- **Invalidation**: Manual trigger for major updates
- **Storage**: PostgreSQL JSONB for fast retrieval

## Performance Metrics

### Model Accuracy
- **Precision@10**: % of relevant items in top 10
- **Recall@10**: Coverage of relevant items
- **NDCG@10**: Ranking quality metric

### Business Metrics
- **Click-through Rate**: Recommendation engagement
- **Conversion Rate**: Purchase after recommendation
- **Revenue Impact**: Additional revenue from recommendations

### Technical Metrics
- **API Response Time**: <200ms for cached results
- **Cache Hit Rate**: >95%
- **Model Update Latency**: <24 hours

## Monitoring and Maintenance

### Data Quality Checks
1. **Missing Values**: Check for null user_id or product_id
2. **Duplicate Events**: Remove duplicate purchases
3. **Outlier Detection**: Flag unusual purchase patterns
4. **Coverage Analysis**: Ensure all products have interactions

### Model Performance
1. **Accuracy Tracking**: Monitor precision/recall trends
2. **Cold Start**: Handle new users/products
3. **Concept Drift**: Detect model degradation
4. **A/B Testing**: Compare model versions

### Infrastructure Health
1. **API Availability**: Monitor endpoint health
2. **Cache Performance**: Track hit rates and latency
3. **Cost Tracking**: Monitor AWS Personalize costs
4. **Error Rates**: Track failed recommendations

## Troubleshooting Guide

### Common Issues
1. **Empty Recommendations**: Check cache freshness
2. **Low Match Rates**: Verify sampling parameters
3. **Missing Products**: Check dataset synchronization
4. **High Latency**: Verify cache performance

### Debug Steps
1. **Check Logs**: Review AWS Personalize logs
2. **Validate Data**: Ensure datasets are up-to-date
3. **Test API**: Verify backend endpoints
4. **Monitor Cache**: Check PostgreSQL cache tables

## Future Enhancements

### Planned Improvements
1. **Real-time Ranking**: On-demand product re-ranking
2. **Contextual Recommendations**: Time/location aware
3. **Multi-Objective Optimization**: Balance revenue and satisfaction
4. **Explainable AI**: Recommendation reasoning

### Scalability Considerations
1. **Regional Models**: Separate models per geographic region
2. **Category Specialization**: Models for product categories
3. **User Segmentation**: Different models for customer segments
4. **Hybrid Approach**: Combine collaborative and content-based filtering
