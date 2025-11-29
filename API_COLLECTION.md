# AWS Personalize API Collection

**Base URL:** `http://44.201.11.243:8001`

---

## 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "recommendation-engine",
  "version": "1.0.0",
  "redis_connected": true,
  "postgres_connected": true
}
```

---

## 2. Similar Products (Shopify Product Page)

Get products similar to a given product - use for "You May Also Like" section.

```http
GET /api/v1/personalize/recommendations/similar/{product_id}?num_results=5
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| product_id | path | Product ID to find similar items for |
| num_results | query | Number of similar products (default: 10) |

**Sample Request:**
```
GET /api/v1/personalize/recommendations/similar/1328?num_results=5
```

**Sample Response:**
```json
{
  "product_id": "1328",
  "recommendations": [
    {
      "product_id": "1715",
      "score": 0.0,
      "algorithm": "aws_similar_items_batch",
      "product_name": "GOLD PILLOW"
    },
    {
      "product_id": "1329",
      "score": 0.0,
      "algorithm": "aws_similar_items_batch",
      "product_name": "MOLTY FOAM 78-66-6"
    },
    {
      "product_id": "1331",
      "score": 0.0,
      "algorithm": "aws_similar_items_batch",
      "product_name": "MOLTY FOAM 78-72-8"
    }
  ],
  "count": 5,
  "source": "aws_personalize_batch"
}
```

**Shopify Integration (Liquid):**
```liquid
{% comment %} Add to product page {% endcomment %}
<div id="similar-products"></div>
<script>
  fetch('http://44.201.11.243:8001/api/v1/personalize/recommendations/similar/{{ product.id }}?num_results=4')
    .then(r => r.json())
    .then(data => {
      const html = data.recommendations.map(p => 
        `<div class="product-card">${p.product_name}</div>`
      ).join('');
      document.getElementById('similar-products').innerHTML = html;
    });
</script>
```

---

## 3. User Recommendations (Shopify Checkout)

Get personalized recommendations for a specific user/customer.

```http
GET /api/v1/personalize/recommendations/{user_id}?num_results=5
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| user_id | path | Customer ID (format: phone_name) |
| num_results | query | Number of recommendations (default: 10) |

**Sample Request:**
```
GET /api/v1/personalize/recommendations/03001234567_John%20Doe?num_results=5
```

**Sample Response:**
```json
{
  "user_id": "03001234567_John Doe",
  "recommendations": [
    {
      "product_id": "1328",
      "score": 0.95,
      "algorithm": "aws_personalize_batch",
      "product_name": "MOLTY FOAM 78-72-6"
    },
    {
      "product_id": "1715",
      "score": 0.87,
      "algorithm": "aws_personalize_batch",
      "product_name": "GOLD PILLOW"
    }
  ],
  "count": 5,
  "source": "aws_personalize"
}
```

---

## 4. Item Affinity (Products That Drive Conversions)

Get products that influence a user's buying decision. Unlike user-personalization (what they'll buy), item-affinity shows what products DRIVE their purchase behavior.

```http
GET /api/v1/personalize/recommendations/item-affinity/{user_id}?num_results=5
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| user_id | path | Customer ID (format: phone_name) |
| num_results | query | Number of products (default: 10) |

**Use Cases:**
- Homepage hero banners
- Email marketing campaigns
- Retargeting ads
- Win-back campaigns

**Sample Request:**
```
GET /api/v1/personalize/recommendations/item-affinity/03001234567_John%20Doe?num_results=5
```

**Sample Response (when data available):**
```json
{
  "user_id": "03001234567_John Doe",
  "recommendations": [
    {
      "product_id": "1328",
      "affinity_score": 0.95,
      "algorithm": "aws_item_affinity",
      "product_name": "MOLTY FOAM 78-72-6"
    }
  ],
  "count": 5,
  "source": "aws_item_affinity",
  "status": "success"
}
```

**Sample Response (when batch not run):**
```json
{
  "user_id": "03001234567_John Doe",
  "recommendations": [],
  "count": 0,
  "source": "item_affinity",
  "status": "no_data",
  "message": "Item affinity batch job not yet run. Run aws_personalize/run_batch_inference.py with --recipe item-affinity"
}
```

**How to Activate:**
```bash
# 1. Run batch inference
python3 aws_personalize/run_batch_inference.py --recipe item-affinity

# 2. Wait ~30 min for completion

# 3. Load results
python3 aws_personalize/load_batch_results.py --recipe item-affinity
```

---

## 5. Trending Products by Location

Get trending products for a specific province/city.

```http
GET /api/v1/personalize/recommendations/by-location?province={province}&city={city}&limit=10
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| province | query | Province name (e.g., Punjab, Sindh) |
| city | query | Optional city name |
| limit | query | Number of products (default: 10) |

**Sample Request:**
```
GET /api/v1/personalize/recommendations/by-location?province=Punjab&limit=5
```

**Sample Response:**
```json
{
  "aggregated_recommendations": [
    {
      "product_id": "1328",
      "avg_score": 0.0,
      "recommended_to_users": 2,
      "product_name": "MOLTY FOAM 78-72-6"
    },
    {
      "product_id": "1331",
      "avg_score": 0.0,
      "recommended_to_users": 3,
      "product_name": "MOLTY FOAM 78-72-8"
    }
  ],
  "location_filter": {"province": "Punjab"},
  "total_users_in_location": 150
}
```

---

## 5. Collaborative Products (Dashboard)

Get products with highest collaborative filtering signals.

```http
GET /api/v1/analytics/collaborative-products?time_filter=30days&limit=10
```

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| time_filter | query | today, 7days, 30days, all |
| limit | query | Number of products (default: 10) |

**Sample Response:**
```json
{
  "products": [
    {
      "product_id": "1328",
      "product_name": "MOLTY FOAM 78-72-6",
      "category": "Mattresses & Foam",
      "price": 0,
      "recommendation_count": 265,
      "avg_similarity_score": 0.2,
      "total_revenue": 82506896.28
    },
    {
      "product_id": "1331",
      "product_name": "MOLTY FOAM 78-72-8",
      "category": "Mattresses & Foam",
      "price": 0,
      "recommendation_count": 280,
      "avg_similarity_score": 0.22,
      "total_revenue": 99271650.45
    }
  ]
}
```

---

## 6. Frequently Bought Together (Collaborative Pairs)

Get product pairs that are frequently purchased together.

```http
GET /api/v1/analytics/collaborative-pairs?time_filter=30days&limit=10
```

**Sample Response:**
```json
{
  "pairs": [
    {
      "product_a_id": "1328",
      "product_a_name": "MOLTY FOAM 78-72-6",
      "product_b_id": "1331",
      "product_b_name": "MOLTY FOAM 78-72-8",
      "co_recommendation_count": 19,
      "combined_revenue": 283234671.42
    }
  ]
}
```

**Shopify Integration (Cart Page):**
```liquid
{% comment %} Frequently Bought Together {% endcomment %}
<script>
  const productId = '{{ cart.items.first.product_id }}';
  fetch(`http://44.201.11.243:8001/api/v1/analytics/collaborative-pairs?limit=3`)
    .then(r => r.json())
    .then(data => {
      // Find pairs containing current product
      const pairs = data.pairs.filter(p => 
        p.product_a_id === productId || p.product_b_id === productId
      );
      // Render "Frequently Bought Together" section
    });
</script>
```

---

## 7. Customer Similarity (Dashboard)

Get customers with similar purchase patterns.

```http
GET /api/v1/analytics/customer-similarity?time_filter=30days&limit=10
```

**Sample Response:**
```json
{
  "customers": [
    {
      "customer_id": "03001234567_John Doe",
      "customer_name": "John Doe",
      "similar_customers_count": 45,
      "actual_recommendations": 12,
      "recommendations_generated": 12,
      "top_shared_products": [
        {"product_name": "MOLTY FOAM 78-72-6", "shared_count": 8},
        {"product_name": "GOLD PILLOW", "shared_count": 5}
      ]
    }
  ]
}
```

---

## 8. Collaborative Metrics (Dashboard)

Get overall collaborative filtering metrics.

```http
GET /api/v1/analytics/collaborative-metrics?time_filter=30days
```

**Sample Response:**
```json
{
  "total_recommendations": 5864,
  "avg_similarity_score": 0.222,
  "active_customer_pairs": 1630,
  "algorithm_accuracy": 0.0,
  "total_users": 4116,
  "total_products": 1441,
  "coverage": 0.0,
  "time_filter": "30days"
}
```

---

## 9. AWS Personalize Status

Check AWS Personalize configuration status.

```http
GET /api/v1/personalize/status
```

**Sample Response:**
```json
{
  "is_configured": true,
  "region": "us-east-1",
  "campaign_arn": null,
  "similar_items_configured": false
}
```

---

## Data Sources

| Endpoint | Data Source |
|----------|-------------|
| `/similar/{product_id}` | `offline_similar_items` table (AWS Batch) |
| `/recommendations/{user_id}` | `offline_user_recommendations` table (AWS Batch) |
| `/by-location` | `offline_user_recommendations` + orders |
| `/collaborative-*` | Real-time SQL analytics on orders |

---

## Batch Data Stats

- **Users with recommendations:** 180,483
- **Products with similar items:** 4,182
- **Last batch run:** November 29, 2025
- **Refresh frequency:** Daily (configurable)

---

## Error Handling

All endpoints return standard HTTP status codes:

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 404 | Resource not found |
| 500 | Internal server error |

Error response format:
```json
{
  "detail": "Error message here"
}
```
