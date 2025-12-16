# 🔄 Data Sync & Daily Training

> **Last Updated:** December 16, 2025  
> **Sync Source:** Master Group APIs (mes.master.com.pk)  
> **Training Server:** EC2 (3.209.80.206)

## Overview

The recommendation system fetches data from Master Group APIs and retrains models daily to ensure recommendations stay fresh and accurate.

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MASTER GROUP SYSTEMS                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐          ┌─────────────┐                   │
│  │   POS API   │          │   OE API    │                   │
│  │ (Retail)    │          │ (Enterprise)│                   │
│  └─────────────┘          └─────────────┘                   │
│         │                        │                           │
│         └────────────┬───────────┘                           │
│                      ▼                                        │
│         ┌─────────────────────────┐                          │
│         │  https://mes.master.com.pk                        │
│         │  /pos-order-detail-items                          │
│         │  /oe-order-detail-items                           │
│         └─────────────────────────┘                          │
│                                                               │
└──────────────────────────┼────────────────────────────────────┘
                           │
                           │ Authorization: H2rcLQPfzYoV55k9...
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA SYNC SERVICE                           │
│                (local_ml_pipeline.py)                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Fetch POS orders (with date range)                       │
│  2. Fetch OE orders (with date range)                        │
│  3. Transform & deduplicate                                   │
│  4. Insert into PostgreSQL                                    │
│  5. Export to CSV for training                                │
│  6. Train ML models                                           │
│  7. Cache recommendations in Redis                            │
│                                                               │
└──────────────────────────┼────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ PostgreSQL  │  │   Redis     │  │   Files     │          │
│  │ (Raw Data)  │  │  (Cache)    │  │  (Models)   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Master Group API Integration

### API Endpoints

| Endpoint | Purpose | Data |
|----------|---------|------|
| `/pos-order-detail-items` | Retail POS orders | Customer, products, quantities |
| `/oe-order-detail-items` | Enterprise orders | B2B orders |

### Authentication

```python
# Headers required for API requests
headers = {
    'Authorization': 'H2rcLQPfzYoV55k9ZyT5aWkyyMKEyxHhX1r3ntrkrvrGeVL4dOsGv3EcQMY2',
    'Content-Type': 'application/json'
}
```

### Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date` | YYYY-MM-DD | Start of date range |
| `end_date` | YYYY-MM-DD | End of date range |

### Example Request

```python
import requests

response = requests.get(
    'https://mes.master.com.pk/pos-order-detail-items',
    params={
        'start_date': '2021-08-01',
        'end_date': '2025-12-16'
    },
    headers={
        'Authorization': 'H2rcLQPfzYoV55k9ZyT5aWkyyMKEyxHhX1r3ntrkrvrGeVL4dOsGv3EcQMY2'
    },
    timeout=120
)

orders = response.json()
```

### Response Structure

```json
{
  "data": [
    {
      "order_id": "POS-2024-12345",
      "customer_name": "John Doe",
      "customer_phone": "03001234567",
      "customer_city": "Lahore",
      "product_id": "1553",
      "product_name": "CELESTE CLASSIQUE 72-48",
      "quantity": 1,
      "unit_price": 45000,
      "order_date": "2024-12-15"
    }
  ]
}
```

---

## Data Storage

### PostgreSQL Tables

```sql
-- Orders table
CREATE TABLE orders (
    id VARCHAR(100) PRIMARY KEY,
    customer_name VARCHAR(255),
    customer_phone VARCHAR(50),
    customer_city VARCHAR(100),
    customer_province VARCHAR(100),
    order_type VARCHAR(20),  -- 'pos' or 'oe'
    order_date TIMESTAMP,
    total_amount DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Order items table
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) REFERENCES orders(id),
    product_id VARCHAR(100),
    product_name VARCHAR(255),
    quantity INTEGER,
    unit_price DECIMAL(12,2),
    sku VARCHAR(100)
);

-- Indexes for performance
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_customer ON orders(customer_phone);
CREATE INDEX idx_items_product ON order_items(product_id);
```

### Current Data Statistics

| Metric | Value |
|--------|-------|
| **Total Orders** | ~450,000 |
| **Total Order Items** | ~1,970,000 |
| **Unique Customers** | 185,280 |
| **Unique Products** | 6,906 |
| **Date Range** | Aug 2021 - Dec 2025 |
| **Database Size** | ~2 GB |

---

## Daily Training Schedule

### Automatic Scheduler

The API includes a built-in scheduler that runs daily:

```python
# In src/services/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

# Daily sync at 2:00 AM UTC (7:00 AM PKT)
scheduler.add_job(
    sync_and_train,
    trigger='cron',
    hour=2,
    minute=0,
    id='daily_sync'
)
```

### Manual Training

```bash
# SSH to EC2
ssh -i mastergroup-ec2-key.pem ubuntu@3.209.80.206

# Run full pipeline
cd /opt/mastergroup-ml
source venv/bin/activate
python scripts/local_ml_pipeline.py
```

### Cron Job Alternative

For more reliable scheduling, set up a cron job:

```bash
# Edit crontab
crontab -e

# Add daily training at 2:00 AM
0 2 * * * cd /opt/mastergroup-ml && source venv/bin/activate && python scripts/local_ml_pipeline.py >> /tmp/ml_cron.log 2>&1
```

---

## Pipeline Steps

### Step 1: Fetch Data from APIs

```python
def fetch_from_master_apis():
    """Fetch new orders from Master Group APIs."""
    
    # Get last sync date from database
    last_sync = get_last_sync_date()  # e.g., "2025-12-15"
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch POS orders
    pos_orders = requests.get(
        f'{API_BASE}/pos-order-detail-items',
        params={'start_date': last_sync, 'end_date': today},
        headers=headers
    ).json()
    
    # Fetch OE orders
    oe_orders = requests.get(
        f'{API_BASE}/oe-order-detail-items',
        params={'start_date': last_sync, 'end_date': today},
        headers=headers
    ).json()
    
    return pos_orders, oe_orders
```

### Step 2: Insert into Database

```python
def insert_orders(orders, order_type):
    """Insert orders into PostgreSQL."""
    
    for order in orders:
        # Insert order
        cursor.execute("""
            INSERT INTO orders (id, customer_name, customer_phone, ...)
            VALUES (%s, %s, %s, ...)
            ON CONFLICT (id) DO NOTHING
        """, (order['order_id'], order['customer_name'], ...))
        
        # Insert order items
        cursor.execute("""
            INSERT INTO order_items (order_id, product_id, ...)
            VALUES (%s, %s, ...)
        """, (order['order_id'], order['product_id'], ...))
```

### Step 3: Export Training Data

```python
def export_to_csv():
    """Export data for ML training."""
    
    # Users (customers with 2+ purchases)
    cursor.execute("""
        SELECT customer_phone as user_id, 
               customer_name,
               customer_city
        FROM orders 
        GROUP BY customer_phone 
        HAVING COUNT(*) >= 2
    """)
    
    # Items (products with purchase history)
    cursor.execute("""
        SELECT DISTINCT product_id as item_id,
               product_name,
               category
        FROM order_items
    """)
    
    # Interactions (user-item purchases)
    cursor.execute("""
        SELECT customer_phone as user_id,
               product_id as item_id,
               SUM(quantity) as event_value,
               MAX(order_date) as timestamp
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        GROUP BY customer_phone, product_id
    """)
```

### Step 4: Train Models

```python
def train_models():
    """Train all ML models."""
    
    # Load training data
    interactions = pd.read_csv('interactions_latest.csv')
    
    # Create user-item matrix
    matrix = create_interaction_matrix(interactions)
    
    # Train SVD
    U, sigma, Vt = svds(matrix, k=50)
    
    # Compute item similarities
    similarity = cosine_similarity(matrix.T)
    
    # Calculate popularity scores
    popularity = interactions.groupby('item_id').size()
    
    # Save models
    save_models(U, sigma, Vt, similarity, popularity)
```

### Step 5: Generate & Cache Recommendations

```python
def cache_recommendations():
    """Pre-compute and cache recommendations."""
    
    redis_client = redis.Redis()
    
    # For each user
    for user_id in users:
        # Generate top-50 recommendations
        recs = generate_recommendations(user_id, limit=50)
        
        # Cache in Redis (24-hour TTL)
        redis_client.setex(
            f'user_recs:{user_id}',
            86400,  # 24 hours
            json.dumps(recs)
        )
    
    # Cache item similarities
    for item_id in items:
        similar = get_similar_items(item_id, limit=50)
        redis_client.setex(
            f'item_similar:{item_id}',
            86400,
            json.dumps(similar)
        )
```

---

## Monitoring & Logging

### Pipeline Logs

```bash
# View pipeline output
tail -f /tmp/ml_pipeline.log

# Example output:
# =====================================
# STEP 1: FETCH DATA FROM MASTER APIs
# =====================================
#   📊 POS Orders: 1,500
#   📊 OE Orders: 300
#   📊 Total: 1,800
#
# STEP 2: INSERT INTO DATABASE
#   ✅ Orders inserted: 1,800
#   ✅ Items inserted: 4,200
#
# STEP 3: EXPORT TRAINING DATA
#   📄 Users: 185,280
#   📄 Items: 6,906
#   📄 Interactions: 1,971,527
#
# STEP 4: TRAIN MODELS
#   🤖 SVD: ✅
#   🤖 Similarity: ✅
#   🤖 Popularity: ✅
#
# STEP 5: CACHE RECOMMENDATIONS
#   💾 Users cached: 79,623
#   💾 Items cached: 6,906
#
# PIPELINE COMPLETE ✅
# Duration: 175 seconds
```

### Sync Metadata Table

```sql
CREATE TABLE sync_metadata (
    id SERIAL PRIMARY KEY,
    sync_type VARCHAR(50),
    start_date DATE,
    end_date DATE,
    records_synced INTEGER,
    duration_seconds FLOAT,
    status VARCHAR(20),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Error Handling

### API Timeout

```python
try:
    response = requests.get(url, timeout=120)
except requests.Timeout:
    logger.error("API timeout - will retry in next sync")
    # Continue with existing data
```

### Database Connection

```python
def get_db_connection():
    """Get database connection with retry."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return psycopg2.connect(DATABASE_URL)
        except psycopg2.OperationalError:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
```

### Redis Fallback

```python
def get_recommendations(user_id):
    """Get recommendations with fallback."""
    
    # Try Redis cache
    cached = redis_client.get(f'user_recs:{user_id}')
    if cached:
        return json.loads(cached)
    
    # Fallback: Generate on-the-fly
    return generate_recommendations(user_id)
```

---

## Performance Metrics

### Training Performance

| Instance | RAM | Users | Duration |
|----------|-----|-------|----------|
| Lightsail micro | 1 GB | 79,623 | 77 min |
| **EC2 t3.medium** | **4 GB** | **79,623** | **3 min** |

### API Response Times

| Endpoint | Cached | Uncached |
|----------|--------|----------|
| `/recommendations` | 15ms | 150ms |
| `/similar/{id}` | 10ms | 100ms |
| `/popular` | 20ms | 200ms |

---

## Maintenance Tasks

### Weekly
- Check sync logs for errors
- Verify Redis memory usage
- Review API response times

### Monthly
- Full database vacuum
- Archive old sync metadata
- Review model accuracy metrics

### Quarterly
- Evaluate model improvements
- Update category mappings
- Review data quality
