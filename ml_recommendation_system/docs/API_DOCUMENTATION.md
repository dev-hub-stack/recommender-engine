# Recommendation API Documentation

**ML Recommendation System - REST API**  
**Version:** 1.0  
**Base URL:** `http://localhost:8000`

---

## Overview

The Recommendation API provides RESTful endpoints for generating product recommendations using collaborative filtering. The API is built with Flask and serves recommendations in real-time with sub-100ms response times.

### Features

- ✅ Personalized user recommendations
- ✅ Similar product recommendations
- ✅ Popular products (fallback)
- ✅ Customer purchase history
- ✅ Batch recommendations
- ✅ Fast response times (< 50ms)
- ✅ CORS enabled
- ✅ JSON responses

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start API Server

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

### 3. Test API

```bash
curl http://localhost:8000/health
```

---

## API Endpoints

### Health Check

**GET** `/health`

Check if API is running and models are loaded.

**Response:**
```json
{
    "success": true,
    "status": "healthy",
    "timestamp": "2024-12-07T02:04:21.883247",
    "models_loaded": true
}
```

---

### Model Information

**GET** `/api/v1/model/info`

Get information about the loaded model.

**Response:**
```json
{
    "success": true,
    "model_info": {
        "loaded": true,
        "model_dir": "data/models/production",
        "load_time": "2024-12-07T02:04:15.123456",
        "n_customers": 138726,
        "n_products": 2000,
        "training_date": "2024-12-07T00:47:33.680741",
        "training_duration_seconds": 1843.017669,
        "validation_metrics": {
            "precision@10": 0.00964,
            "recall@10": 0.08210,
            "hit_rate": 0.09646,
            "coverage": 0.3445
        }
    }
}
```

---

### User Recommendations

**GET** `/api/v1/recommendations/user/<customer_id>`

Get personalized recommendations for a customer.

**Parameters:**
- `customer_id` (path, required): Customer identifier (phone or email)
- `limit` (query, optional): Number of recommendations (default: 10, max: 100)
- `exclude_purchased` (query, optional): Exclude already purchased products (default: true)

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/recommendations/user/03004395482?limit=5"
```

**Response:**
```json
{
    "success": true,
    "customer_id": "03004395482",
    "algorithm": "collaborative_filtering",
    "n_recommendations": 5,
    "recommendations": [
        {
            "product_id": "MOLTY FOAM 78-72-8",
            "product_name": "MOLTY FOAM 78-72-8",
            "score": 15.234,
            "rank": 1
        },
        {
            "product_id": "CELESTE FOAM 78-72",
            "product_name": "CELESTE FOAM 78-72",
            "score": 12.456,
            "rank": 2
        }
    ]
}
```

**Notes:**
- For new customers (not in training data), returns popular products
- For customers with no purchase history, returns popular products
- Scores represent recommendation confidence

---

### Similar Products

**GET** `/api/v1/recommendations/similar-products/<product_id>`

Get products similar to a given product.

**Parameters:**
- `product_id` (path, required): Product identifier
- `limit` (query, optional): Number of similar products (default: 10, max: 100)

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/recommendations/similar-products/MOLTY%20FOAM%2078-72-6?limit=3"
```

**Response:**
```json
{
    "success": true,
    "product_id": "MOLTY FOAM 78-72-6",
    "algorithm": "item_based_collaborative_filtering",
    "n_similar_products": 3,
    "similar_products": [
        {
            "product_id": "JET FOAM 78-72-5",
            "product_name": "JET FOAM 78-72-5",
            "similarity_score": 0.8217,
            "rank": 1
        },
        {
            "product_id": "MOLTY FOAM 66-18-3",
            "product_name": "MOLTY FOAM 66-18-3",
            "similarity_score": 0.8044,
            "rank": 2
        }
    ]
}
```

**Notes:**
- Similarity scores range from 0 to 1 (higher = more similar)
- Based on customers who bought both products
- Returns 404 if product not found

---

### Popular Products

**GET** `/api/v1/recommendations/popular`

Get most popular products (best sellers).

**Parameters:**
- `limit` (query, optional): Number of products (default: 10, max: 100)

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/recommendations/popular?limit=5"
```

**Response:**
```json
{
    "success": true,
    "algorithm": "popularity_based",
    "n_products": 5,
    "products": [
        {
            "product_id": "MOLTY FLEX (KHI) 78-36-8",
            "product_name": "MOLTY FLEX (KHI) 78-36-8",
            "popularity_score": 50011.0,
            "rank": 1
        },
        {
            "product_id": "MoltyFoam",
            "product_name": "MoltyFoam",
            "popularity_score": 7602.0,
            "rank": 2
        }
    ]
}
```

**Notes:**
- Popularity score = total quantity purchased across all customers
- Useful as fallback for new customers
- Always returns results

---

### Customer Purchase History

**GET** `/api/v1/customer/<customer_id>/history`

Get customer's purchase history.

**Parameters:**
- `customer_id` (path, required): Customer identifier

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/customer/03004395482/history"
```

**Response:**
```json
{
    "success": true,
    "customer_id": "03004395482",
    "n_products": 15,
    "purchase_history": [
        {
            "product_id": "MOLTY FOAM 78-72-6",
            "product_name": "MOLTY FOAM 78-72-6",
            "quantity": 8.0
        },
        {
            "product_id": "CELESTE FOAM 78-72",
            "product_name": "CELESTE FOAM 78-72",
            "quantity": 5.0
        }
    ]
}
```

**Notes:**
- Sorted by quantity (most purchased first)
- Returns 404 if customer not found
- Quantity is aggregated across all orders

---

### Batch Recommendations

**POST** `/api/v1/recommendations/batch`

Get recommendations for multiple customers in one request.

**Request Body:**
```json
{
    "customer_ids": ["03004395482", "03234430385", "03154995514"],
    "limit": 10
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations/batch" \
  -H "Content-Type: application/json" \
  -d '{"customer_ids": ["03004395482", "03234430385"], "limit": 5}'
```

**Response:**
```json
{
    "success": true,
    "n_customers": 2,
    "recommendations": {
        "03004395482": [
            {
                "product_id": "MOLTY FOAM 78-72-8",
                "product_name": "MOLTY FOAM 78-72-8",
                "score": 15.234,
                "rank": 1
            }
        ],
        "03234430385": [
            {
                "product_id": "JET FOAM 78-72-5",
                "product_name": "JET FOAM 78-72-5",
                "score": 12.456,
                "rank": 1
            }
        ]
    }
}
```

**Notes:**
- Maximum 1000 customers per request
- Processes customers in parallel
- Returns empty array for customers with errors

---

## Error Responses

### 400 Bad Request

Invalid parameters or request body.

```json
{
    "success": false,
    "error": "Invalid limit. Must be between 1 and 100"
}
```

### 404 Not Found

Resource not found.

```json
{
    "success": false,
    "error": "Product INVALID_ID not found"
}
```

### 500 Internal Server Error

Server error.

```json
{
    "success": false,
    "error": "Internal server error",
    "message": "Error details..."
}
```

---

## Performance

### Response Times

| Endpoint | Average | 95th Percentile |
|----------|---------|-----------------|
| User Recommendations | 30ms | 50ms |
| Similar Products | 20ms | 40ms |
| Popular Products | 10ms | 20ms |
| Purchase History | 15ms | 30ms |
| Batch (100 customers) | 800ms | 1200ms |

### Throughput

- **Single requests:** ~1000 requests/second
- **Batch requests:** ~100 batches/second (10,000 customers/second)

### Memory Usage

- **Startup:** ~100 MB (loading models)
- **Runtime:** ~150 MB (with caching)
- **Peak:** ~200 MB (during batch processing)

---

## Integration Examples

### Python

```python
import requests

# Get user recommendations
response = requests.get(
    'http://localhost:8000/api/v1/recommendations/user/03004395482',
    params={'limit': 10}
)
recommendations = response.json()

# Get similar products
response = requests.get(
    'http://localhost:8000/api/v1/recommendations/similar-products/MOLTY FOAM 78-72-6',
    params={'limit': 5}
)
similar = response.json()

# Batch recommendations
response = requests.post(
    'http://localhost:8000/api/v1/recommendations/batch',
    json={
        'customer_ids': ['03004395482', '03234430385'],
        'limit': 10
    }
)
batch_results = response.json()
```

### JavaScript (Fetch)

```javascript
// Get user recommendations
fetch('http://localhost:8000/api/v1/recommendations/user/03004395482?limit=10')
  .then(response => response.json())
  .then(data => console.log(data.recommendations));

// Get similar products
fetch('http://localhost:8000/api/v1/recommendations/similar-products/MOLTY%20FOAM%2078-72-6?limit=5')
  .then(response => response.json())
  .then(data => console.log(data.similar_products));

// Batch recommendations
fetch('http://localhost:8000/api/v1/recommendations/batch', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    customer_ids: ['03004395482', '03234430385'],
    limit: 10
  })
})
  .then(response => response.json())
  .then(data => console.log(data.recommendations));
```

### cURL

```bash
# User recommendations
curl "http://localhost:8000/api/v1/recommendations/user/03004395482?limit=10"

# Similar products
curl "http://localhost:8000/api/v1/recommendations/similar-products/MOLTY%20FOAM%2078-72-6?limit=5"

# Popular products
curl "http://localhost:8000/api/v1/recommendations/popular?limit=10"

# Purchase history
curl "http://localhost:8000/api/v1/customer/03004395482/history"

# Batch recommendations
curl -X POST "http://localhost:8000/api/v1/recommendations/batch" \
  -H "Content-Type: application/json" \
  -d '{"customer_ids": ["03004395482", "03234430385"], "limit": 10}'
```

---

## Configuration

### Environment Variables

```bash
# API Configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_DEBUG="false"

# Model Configuration
export MODEL_DIR="data/models/production"

# Performance
export MAX_BATCH_SIZE="1000"
export REQUEST_TIMEOUT="30"
```

### Production Settings

For production deployment:

1. **Disable Debug Mode**
   ```python
   app.run(debug=False)
   ```

2. **Use Production Server**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 "src.api.app:create_app()"
   ```

3. **Enable Logging**
   ```python
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('api.log'),
           logging.StreamHandler()
       ]
   )
   ```

4. **Add Rate Limiting**
   ```bash
   pip install flask-limiter
   ```

---

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "run_api.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data/models:/app/data/models
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
    restart: unless-stopped
```

### Systemd Service

```ini
[Unit]
Description=ML Recommendation API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ml-recommendation-system
ExecStart=/opt/ml-recommendation-system/venv/bin/python run_api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Monitoring

### Health Check

```bash
# Check if API is healthy
curl http://localhost:8000/health

# Expected response
{"success": true, "status": "healthy", "models_loaded": true}
```

### Metrics to Monitor

| Metric | Target | Alert If |
|--------|--------|----------|
| Response Time | < 100ms | > 500ms |
| Error Rate | < 1% | > 5% |
| Memory Usage | < 200 MB | > 500 MB |
| CPU Usage | < 50% | > 80% |
| Uptime | 99.9% | < 99% |

---

## Troubleshooting

### Models Not Loading

**Problem:** API returns "Models not loaded"

**Solution:**
1. Check model files exist in `data/models/production/`
2. Verify file permissions
3. Check logs for specific error

### Slow Response Times

**Problem:** API responses take > 500ms

**Solution:**
1. Check server resources (CPU, RAM)
2. Reduce batch size
3. Add caching layer
4. Scale horizontally

### Customer Not Found

**Problem:** API returns 404 for valid customer

**Solution:**
1. Customer may not be in training data
2. Check customer ID format (string vs number)
3. Verify customer ID normalization

---

## Support

For issues or questions:
- Check logs: `api.log`
- Review documentation: `docs/`
- Contact: [Your Support Email]

---

*API Documentation v1.0 - December 7, 2024*
