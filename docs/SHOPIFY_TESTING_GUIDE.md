# 🛒 Shopify Integration Testing Guide

> **Last Updated:** December 16, 2025  
> **API Endpoint:** http://3.209.80.206:8001  
> **Shopify Store:** masterverse-project.myshopify.com

## Overview

This guide explains how to integrate and test the recommendation system with the Shopify store.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SHOPIFY STORE                            │
│                masterverse-project.myshopify.com             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Product Page │  │  Cart Page   │  │   Checkout       │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│         │                  │                   │              │
│         └──────────────────┼───────────────────┘              │
│                            ▼                                  │
│              ┌─────────────────────────┐                     │
│              │  Liquid Theme Template  │                     │
│              │  + JavaScript Widget    │                     │
│              └─────────────────────────┘                     │
│                            │                                  │
└────────────────────────────┼──────────────────────────────────┘
                             │ HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              RECOMMENDATION API (EC2)                        │
│                   3.209.80.206:8001                          │
├─────────────────────────────────────────────────────────────┤
│  /api/v1/shopify/recommendations   → Personalized recs      │
│  /api/v1/shopify/similar/{id}      → Similar products       │
│  /api/v1/shopify/popular           → Popular by location    │
│  /api/v1/shopify/webhook/order     → Order tracking         │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Test API Endpoints

Before integrating with Shopify, verify the API is working:

### Health Check
```bash
curl http://3.209.80.206:8001/health
```
Expected: `{"status":"healthy","redis_connected":true,"postgres_connected":true}`

### Test Personalized Recommendations
```bash
curl -X POST "http://3.209.80.206:8001/api/v1/shopify/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_phone": "03001234567",
    "city": "Lahore",
    "limit": 5
  }'
```

### Test Similar Products
```bash
curl "http://3.209.80.206:8001/api/v1/shopify/similar/1553?limit=5"
```

### Test Popular Products
```bash
curl "http://3.209.80.206:8001/api/v1/shopify/popular?city=Karachi&limit=5"
```

---

## Step 2: Add JavaScript Widget to Shopify Theme

### Option A: Add to Theme.liquid (Global)

1. Go to **Shopify Admin** → **Online Store** → **Themes**
2. Click **Actions** → **Edit Code**
3. Open `layout/theme.liquid`
4. Add before `</head>`:

```html
<!-- MasterGroup Recommendation Widget -->
<script>
  window.MG_RECS_CONFIG = {
    apiBase: 'http://3.209.80.206:8001/api/v1/shopify',
    storeId: 'masterverse-project'
  };
</script>
<script src="https://cdn.example.com/mg-recommendations.js" defer></script>
```

### Option B: Product Page Widget (Recommended)

1. Open `sections/main-product.liquid` or `templates/product.liquid`
2. Add after the product description:

```liquid
<!-- Similar Products Recommendations -->
<div id="mg-similar-products" data-product-id="{{ product.id }}">
  <h3>You May Also Like</h3>
  <div class="mg-recs-container"></div>
</div>

<script>
(function() {
  const productId = '{{ product.id }}';
  const container = document.querySelector('#mg-similar-products .mg-recs-container');
  
  fetch(`http://3.209.80.206:8001/api/v1/shopify/similar/${productId}?limit=4`)
    .then(res => res.json())
    .then(data => {
      if (data.success && data.similar_products) {
        container.innerHTML = data.similar_products.map(item => `
          <div class="mg-rec-item">
            <a href="/products/${item.item_id}">
              <div class="mg-rec-name">${item.item_name}</div>
              <div class="mg-rec-score">Match: ${Math.round(item.score * 100)}%</div>
            </a>
          </div>
        `).join('');
      }
    })
    .catch(err => console.error('Recommendations error:', err));
})();
</script>

<style>
  #mg-similar-products { margin: 2rem 0; }
  .mg-recs-container { display: flex; gap: 1rem; flex-wrap: wrap; }
  .mg-rec-item { 
    flex: 1 1 200px; 
    padding: 1rem; 
    border: 1px solid #eee; 
    border-radius: 8px;
    text-align: center;
  }
  .mg-rec-name { font-weight: bold; margin-bottom: 0.5rem; }
  .mg-rec-score { color: #666; font-size: 0.9rem; }
</style>
```

---

## Step 3: Cart Page Recommendations

Add to `templates/cart.liquid` or `sections/main-cart.liquid`:

```liquid
<!-- Personalized Recommendations -->
<div id="mg-cart-recommendations">
  <h3>Recommended For You</h3>
  <div class="mg-recs-container"></div>
</div>

<script>
(function() {
  const container = document.querySelector('#mg-cart-recommendations .mg-recs-container');
  
  // Get customer info from Shopify
  const customerPhone = '{{ customer.phone | default: "" }}';
  const customerEmail = '{{ customer.email | default: "" }}';
  const city = '{{ customer.default_address.city | default: "" }}';
  
  // Get cart items
  const cartItems = [
    {% for item in cart.items %}
      '{{ item.product_id }}'{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ];
  
  fetch('http://3.209.80.206:8001/api/v1/shopify/recommendations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      customer_phone: customerPhone,
      customer_email: customerEmail,
      cart_items: cartItems,
      city: city,
      limit: 4
    })
  })
    .then(res => res.json())
    .then(data => {
      if (data.success && data.recommendations) {
        container.innerHTML = data.recommendations.map(item => `
          <div class="mg-rec-item">
            <a href="/products/${item.item_id}">
              <div class="mg-rec-name">${item.item_name}</div>
              <div class="mg-rec-algo">${item.algorithm}</div>
            </a>
          </div>
        `).join('');
      }
    })
    .catch(err => console.error('Recommendations error:', err));
})();
</script>
```

---

## Step 4: Homepage Popular Products

Add to `templates/index.liquid` or homepage section:

```liquid
<!-- Popular Products -->
<div id="mg-popular-products">
  <h2>Trending in Your Area</h2>
  <div class="mg-recs-container"></div>
</div>

<script>
(function() {
  const container = document.querySelector('#mg-popular-products .mg-recs-container');
  const city = '{{ customer.default_address.city | default: "Lahore" }}';
  
  fetch(`http://3.209.80.206:8001/api/v1/shopify/popular?city=${city}&limit=8`)
    .then(res => res.json())
    .then(data => {
      if (data.success && data.popular_products) {
        container.innerHTML = data.popular_products.map(item => `
          <div class="mg-rec-item">
            <a href="/products/${item.item_id}">
              <div class="mg-rec-name">${item.item_name}</div>
              <div class="mg-rec-score">${item.score} orders</div>
            </a>
          </div>
        `).join('');
      }
    })
    .catch(err => console.error('Popular products error:', err));
})();
</script>
```

---

## Step 5: Order Webhook (For Training Data)

### Register Webhook in Shopify Admin

1. Go to **Settings** → **Notifications** → **Webhooks**
2. Click **Create webhook**
3. Configure:
   - **Event:** Order creation
   - **URL:** `http://3.209.80.206:8001/api/v1/shopify/webhook/order-created`
   - **Format:** JSON

### Alternative: Use Shopify API

```bash
curl -X POST "https://masterverse-project.myshopify.com/admin/api/2024-01/webhooks.json" \
  -H "X-Shopify-Access-Token: YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "webhook": {
      "topic": "orders/create",
      "address": "http://3.209.80.206:8001/api/v1/shopify/webhook/order-created",
      "format": "json"
    }
  }'
```

---

## Step 6: Testing Checklist

### API Tests
- [ ] Health endpoint returns healthy
- [ ] `/shopify/recommendations` returns personalized results
- [ ] `/shopify/similar/{id}` returns similar products
- [ ] `/shopify/popular` returns popular products
- [ ] Webhook endpoint accepts POST requests

### Shopify Theme Tests
- [ ] Product page shows similar products
- [ ] Cart page shows personalized recommendations
- [ ] Homepage shows popular products
- [ ] No JavaScript console errors
- [ ] Recommendations load within 2 seconds

### User Scenarios
- [ ] **Anonymous user** sees popular products
- [ ] **Logged-in user** sees personalized recommendations
- [ ] **Product page** shows "You May Also Like"
- [ ] **Cart page** shows "Complete Your Purchase"
- [ ] **Empty cart** shows "Popular Items"

---

## CORS Configuration

If you encounter CORS errors, the API already has CORS enabled for all origins:

```python
# In src/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Troubleshooting

### API Not Responding
```bash
# Check if API is running
ssh -i mastergroup-ec2-key.pem ubuntu@3.209.80.206 "ps aux | grep uvicorn"

# Restart API
ssh -i mastergroup-ec2-key.pem ubuntu@3.209.80.206 "cd /opt/mastergroup-ml && source venv/bin/activate && pkill -f uvicorn; nohup uvicorn src.main:app --host 0.0.0.0 --port 8001 --workers 2 > /tmp/api.log 2>&1 &"
```

### No Recommendations Returned
- Check if user exists in database
- Verify Redis is running: `redis-cli ping`
- Check API logs: `tail -f /tmp/api.log`

### Slow Response Times
- Normal: 10-50ms
- If slow, check Redis connection
- Run ML pipeline to refresh cache

---

## Production Considerations

### 1. Use HTTPS
Currently using HTTP. For production:
- Add SSL certificate (Let's Encrypt)
- Or use AWS ALB with ACM certificate

### 2. Rate Limiting
Implement rate limiting to prevent abuse:
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
```

### 3. Authentication
Add API key authentication for Shopify requests:
```python
@app.middleware("http")
async def verify_shopify_request(request, call_next):
    api_key = request.headers.get("X-MG-API-Key")
    if not api_key or api_key != settings.SHOPIFY_API_KEY:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)
```

### 4. Monitoring
- Set up CloudWatch alarms for API health
- Track recommendation click-through rates
- Monitor Redis memory usage
