# Shopify Store Integration Guide

## Overview

This guide explains how to integrate the MasterGroup recommendation system with your Shopify store (masterverse-project.myshopify.com) to show personalized product recommendations during checkout and on product pages.

**Key Features:**
- Personalized recommendations based on customer purchase history
- Similar product suggestions ("You might also like")
- Location-based popular products
- RFM segment-based recommendations
- Real-time order sync via webhooks

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SHOPIFY INTEGRATION FLOW                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐         ┌─────────────────┐                       │
│  │  SHOPIFY STORE  │◄───────►│  RECOMMENDATION │                       │
│  │  (Frontend)     │   API   │     ENGINE      │                       │
│  └────────┬────────┘         └────────┬────────┘                       │
│           │                           │                                 │
│           │                           ▼                                 │
│           │                  ┌─────────────────┐                       │
│           │                  │   PostgreSQL    │                       │
│           │                  │  Cache Tables   │                       │
│           │                  │ ─────────────── │                       │
│           │                  │ • 74,827 users  │                       │
│           │                  │ • 4,182 items   │                       │
│           │                  └─────────────────┘                       │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │   WEBHOOKS      │───────► Order Created ───► Training Data          │
│  │   (Optional)    │───────► Customer Updated                          │
│  └─────────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Credentials & Configuration

### Shopify App Details

| Setting | Value |
|---------|-------|
| **Store** | masterverse-project.myshopify.com |
| **App Name** | pods-recommendation |
| **API Key** | (see .env file) |
| **API Version** | 2024-01 |
| **Scopes** | read_products, read_customers, write_inventory |

### Environment Variables

```bash
# .env file
SHOPIFY_STORE=masterverse-project.myshopify.com
SHOPIFY_API_KEY=<your-api-key>
SHOPIFY_API_SECRET=<your-api-secret>
SHOPIFY_ACCESS_TOKEN=<your-access-token>
SHOPIFY_API_VERSION=2024-01
```

> **Note:** Get your credentials from Shopify Admin → Settings → Apps and sales channels → Develop apps

## Available Endpoints

### 1. Unified Recommendations (POST)

**Best for:** Cart page, checkout page, or any page where you have customer context.

```
POST /api/v1/shopify/recommendations
```

**Request Body:**
```json
{
    "customer_email": "user@example.com",
    "customer_phone": "03001234567",
    "cart_items": ["product_id_1", "product_id_2"],
    "current_product": "product_id",
    "city": "Lahore",
    "province": "Punjab",
    "rfm_segment": "champions",
    "limit": 10
}
```

**All fields are optional.** The system uses a fallback strategy:
1. **Personalized** → If customer phone/email matches a known user
2. **Similar Items** → If cart items or current product provided
3. **Location-based** → If city/province provided
4. **Segment-based** → If RFM segment provided
5. **Popular** → Fallback for anonymous users

**Response:**
```json
{
    "success": true,
    "recommendation_type": "personalized",
    "user_identified": true,
    "user_id": "03001234567_asim",
    "recommendations": [
        {
            "item_id": "1331",
            "item_name": "MOLTY FOAM 78-72-8",
            "score": 0.95,
            "algorithm": "SVD"
        }
    ],
    "count": 10
}
```

### 2. Similar Products (GET)

**Best for:** Product pages - "You might also like" section.

```
GET /api/v1/shopify/similar/{product_id}?limit=10
```

**Response:**
```json
{
    "success": true,
    "product_id": "1331",
    "similar_products": [
        {
            "item_id": "1328",
            "item_name": "MOLTY FOAM 78-72-6",
            "score": 0.32
        }
    ],
    "count": 10
}
```

### 3. Popular Products (GET)

**Best for:** Homepage, anonymous users, or location-based recommendations.

```
GET /api/v1/shopify/popular?city=Lahore&province=Punjab&days=30&limit=10
```

**Response:**
```json
{
    "success": true,
    "popular_products": [
        {
            "item_id": "1331",
            "item_name": "MOLTY FOAM 78-72-8",
            "score": 1523
        }
    ],
    "count": 10,
    "filters": {
        "city": "Lahore",
        "province": "Punjab",
        "days": 30
    }
}
```

## Shopify Integration Examples

### 1. JavaScript (Fetch API)

```javascript
// On checkout page - get personalized recommendations
async function getRecommendations() {
    // Get customer info from Shopify checkout
    const customerEmail = Shopify.checkout?.email || null;
    const customerPhone = Shopify.checkout?.phone || null;
    
    // Get cart items
    const cartItems = Shopify.checkout?.line_items?.map(item => item.product_id.toString()) || [];
    
    // Get location from cookies or IP geolocation
    const city = getCookie('user_city') || null;
    const province = getCookie('user_province') || null;
    
    // Get RFM segment (if you calculate it on frontend)
    const rfmSegment = localStorage.getItem('rfm_segment') || null;
    
    const response = await fetch('https://your-api-domain.com/api/v1/shopify/recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            customer_email: customerEmail,
            customer_phone: customerPhone,
            cart_items: cartItems,
            city: city,
            province: province,
            rfm_segment: rfmSegment,
            limit: 4
        })
    });
    
    const data = await response.json();
    
    if (data.success) {
        displayRecommendations(data.recommendations, data.recommendation_type);
    }
}

function displayRecommendations(recommendations, type) {
    const container = document.getElementById('recommendations-container');
    
    let html = `<h3>Recommended for You</h3>`;
    html += `<div class="recommendation-grid">`;
    
    recommendations.forEach(rec => {
        html += `
            <div class="recommendation-item">
                <a href="/products/${rec.item_id}">
                    <h4>${rec.item_name}</h4>
                </a>
            </div>
        `;
    });
    
    html += `</div>`;
    container.innerHTML = html;
}
```

### 2. Product Page - Similar Items

```javascript
// On product page
async function getSimilarProducts(productId) {
    const response = await fetch(
        `https://your-api-domain.com/api/v1/shopify/similar/${productId}?limit=4`
    );
    
    const data = await response.json();
    
    if (data.success) {
        displaySimilarProducts(data.similar_products);
    }
}

// Get product ID from Shopify
const productId = {{ product.id }};
getSimilarProducts(productId);
```

### 3. Location Detection from Cookies

```javascript
// Set location cookie (call once when user visits)
async function detectAndSaveLocation() {
    // Option 1: Use IP geolocation service
    try {
        const response = await fetch('https://ipapi.co/json/');
        const data = await response.json();
        
        setCookie('user_city', data.city, 30);
        setCookie('user_province', data.region, 30);
    } catch (e) {
        // Fallback: ask user or use default
    }
}

function setCookie(name, value, days) {
    const expires = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/`;
}

function getCookie(name) {
    return document.cookie.split('; ').reduce((r, v) => {
        const parts = v.split('=');
        return parts[0] === name ? decodeURIComponent(parts[1]) : r;
    }, null);
}
```

### 4. RFM Segmentation on Frontend

```javascript
// Simple RFM segment estimation based on cookies
function estimateRFMSegment() {
    const lastVisit = getCookie('last_visit');
    const visitCount = parseInt(getCookie('visit_count') || '0');
    const hasPurchased = getCookie('has_purchased') === 'true';
    
    const daysSinceLastVisit = lastVisit 
        ? Math.floor((Date.now() - new Date(lastVisit)) / 86400000)
        : 999;
    
    // Simple segment logic
    if (hasPurchased && daysSinceLastVisit < 30 && visitCount > 5) {
        return 'champions';
    } else if (hasPurchased && daysSinceLastVisit < 60) {
        return 'loyal';
    } else if (hasPurchased && daysSinceLastVisit > 90) {
        return 'at_risk';
    } else if (!hasPurchased && visitCount > 3) {
        return 'potential';
    } else {
        return 'new';
    }
}

// Update visit tracking
function updateVisitTracking() {
    const visitCount = parseInt(getCookie('visit_count') || '0') + 1;
    setCookie('visit_count', visitCount, 365);
    setCookie('last_visit', new Date().toISOString(), 365);
}
```

## Shopify Liquid Template Example

```liquid
<!-- In product.liquid or cart.liquid -->
<div id="recommendations-container" class="recommendations-section">
    <h3>You Might Also Like</h3>
    <div class="recommendation-grid" id="rec-grid">
        <!-- Recommendations loaded via JavaScript -->
        <p>Loading recommendations...</p>
    </div>
</div>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        {% if customer %}
            // Known customer
            fetchRecommendations({
                customer_email: "{{ customer.email }}",
                customer_phone: "{{ customer.phone }}",
                limit: 4
            });
        {% elsif cart.items.size > 0 %}
            // Has cart items
            fetchRecommendations({
                cart_items: [{% for item in cart.items %}"{{ item.product_id }}"{% unless forloop.last %},{% endunless %}{% endfor %}],
                limit: 4
            });
        {% else %}
            // Anonymous - use popular
            fetchPopularProducts();
        {% endif %}
    });
    
    async function fetchRecommendations(params) {
        try {
            const response = await fetch('https://your-api.com/api/v1/shopify/recommendations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            const data = await response.json();
            renderRecommendations(data.recommendations);
        } catch (e) {
            console.error('Recommendation error:', e);
        }
    }
    
    async function fetchPopularProducts() {
        const city = getCookie('user_city');
        let url = 'https://your-api.com/api/v1/shopify/popular?limit=4';
        if (city) url += `&city=${encodeURIComponent(city)}`;
        
        try {
            const response = await fetch(url);
            const data = await response.json();
            renderRecommendations(data.popular_products);
        } catch (e) {
            console.error('Popular products error:', e);
        }
    }
    
    function renderRecommendations(products) {
        const grid = document.getElementById('rec-grid');
        if (!products || products.length === 0) {
            grid.innerHTML = '<p>No recommendations available</p>';
            return;
        }
        
        grid.innerHTML = products.map(p => `
            <div class="recommendation-item">
                <a href="/products/${p.item_id}">
                    <span class="product-name">${p.item_name}</span>
                </a>
            </div>
        `).join('');
    }
</script>

<style>
    .recommendations-section {
        margin: 20px 0;
        padding: 20px;
        background: #f9f9f9;
    }
    .recommendation-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
    }
    .recommendation-item {
        padding: 10px;
        background: white;
        border-radius: 8px;
        text-align: center;
    }
    .recommendation-item a {
        text-decoration: none;
        color: #333;
    }
</style>
```

## Testing

### Test with cURL

```bash
# Test unified recommendations
curl -X POST "http://localhost:8001/api/v1/shopify/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"customer_phone": "03001234567", "city": "Lahore", "limit": 5}'

# Test similar products
curl "http://localhost:8001/api/v1/shopify/similar/1331?limit=5"

# Test popular products
curl "http://localhost:8001/api/v1/shopify/popular?city=Lahore&limit=5"
```

## CORS Configuration

If calling from Shopify frontend, ensure CORS is enabled on your API server:

```python
# In main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-shopify-store.myshopify.com", "https://your-custom-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Webhook Setup (Real-Time Order Sync)

To automatically sync new Shopify orders for ML training:

### Step 1: Configure Webhook in Shopify Admin

1. Go to **Settings** → **Notifications** → **Webhooks**
2. Click **Create webhook**
3. Configure:
   - **Event:** Order creation
   - **Format:** JSON
   - **URL:** `https://your-api-domain.com/api/v1/shopify/webhook/order-created`
   - **Webhook API version:** 2024-01

### Step 2: Verify Webhook (Optional but Recommended)

```python
# Add to main.py for webhook verification
import hmac
import hashlib

def verify_shopify_webhook(data: bytes, hmac_header: str) -> bool:
    secret = os.getenv('SHOPIFY_API_SECRET')
    computed_hmac = base64.b64encode(
        hmac.new(secret.encode(), data, hashlib.sha256).digest()
    ).decode()
    return hmac.compare_digest(computed_hmac, hmac_header)
```

### Webhook Endpoint Response

When an order is created, the webhook saves it to the database:

```json
{
    "success": true,
    "order_id": "shopify_12345678",
    "customer_id": "03001234567_john",
    "items_count": 3
}
```

## Product ID Mapping

### Shopify vs MasterGroup Product IDs

| System | Product ID Format | Example |
|--------|------------------|---------|
| Shopify | 16-digit numeric | 10045012017458 |
| MasterGroup | 4-digit numeric | 1328 |

### Mapping Strategy

The recommendation system uses MasterGroup product IDs internally. When integrating with Shopify:

1. **Option A: Direct Mapping Table** - Create a mapping table linking Shopify IDs to MasterGroup IDs
2. **Option B: SKU Matching** - Use SKU field to match products
3. **Option C: Title Matching** - Match by product title (less reliable)

### API Endpoint for Product List

```bash
# Get all Shopify products for mapping
curl "http://localhost:8001/api/v1/shopify/products?limit=100"
```

Response includes product IDs, titles, and handles for mapping.

## Deployment Guide

### Option 1: Heroku (Recommended for Quick Setup)

```bash
# Deploy to Heroku
git push heroku main

# Set environment variables
heroku config:set SHOPIFY_STORE=masterverse-project.myshopify.com
heroku config:set SHOPIFY_ACCESS_TOKEN=shpat_xxx
```

### Option 2: AWS Lightsail

```bash
# SSH to Lightsail instance
ssh -i LightsailKey.pem ubuntu@your-ip

# Pull latest code
cd /opt/mastergroup-api
git pull origin main

# Restart service
sudo systemctl restart mastergroup-api
```

### Option 3: Docker

```dockerfile
# Dockerfile included in repo
docker build -t mastergroup-api .
docker run -p 8001:8001 --env-file .env mastergroup-api
```

## Shopify Theme Files to Modify

### 1. Product Page (`sections/product-template.liquid`)

Add similar products section:

```liquid
{% comment %} Similar Products Section {% endcomment %}
<div class="similar-products-section" data-product-id="{{ product.id }}">
  <h3>You Might Also Like</h3>
  <div class="similar-products-grid" id="similar-products"></div>
</div>

<script>
  document.addEventListener('DOMContentLoaded', function() {
    const productId = '{{ product.id }}';
    loadSimilarProducts(productId);
  });
  
  async function loadSimilarProducts(productId) {
    const API_URL = 'https://your-api.com/api/v1/shopify';
    try {
      const res = await fetch(`${API_URL}/similar/${productId}?limit=4`);
      const data = await res.json();
      renderProducts(data.similar_products, 'similar-products');
    } catch (e) {
      console.error('Failed to load similar products', e);
    }
  }
</script>
```

### 2. Cart Page (`sections/cart-template.liquid`)

Add cross-sell recommendations:

```liquid
{% comment %} Cart Recommendations {% endcomment %}
<div class="cart-recommendations" id="cart-recs">
  <h3>Complete Your Order</h3>
  <div class="recommendation-grid" id="cart-rec-grid"></div>
</div>

<script>
  document.addEventListener('DOMContentLoaded', function() {
    const cartItems = [
      {% for item in cart.items %}
        "{{ item.product_id }}"{% unless forloop.last %},{% endunless %}
      {% endfor %}
    ];
    loadCartRecommendations(cartItems);
  });
  
  async function loadCartRecommendations(cartItems) {
    const API_URL = 'https://your-api.com/api/v1/shopify';
    try {
      const res = await fetch(`${API_URL}/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cart_items: cartItems, limit: 4 })
      });
      const data = await res.json();
      renderProducts(data.recommendations, 'cart-rec-grid');
    } catch (e) {
      console.error('Failed to load cart recommendations', e);
    }
  }
</script>
```

### 3. Checkout (Thank You Page)

For post-purchase recommendations (requires Shopify Plus or checkout.liquid access):

```liquid
{% comment %} Post-Purchase Recommendations {% endcomment %}
<script>
  Shopify.Checkout.OrderStatus.addContentBox(
    '<h3>Recommended For You</h3><div id="post-purchase-recs"></div>'
  );
  
  // Load personalized recommendations
  fetch('https://your-api.com/api/v1/shopify/recommendations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      customer_email: '{{ order.email }}',
      limit: 4
    })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('post-purchase-recs').innerHTML = 
      data.recommendations.map(p => `<a href="/products/${p.item_id}">${p.item_name}</a>`).join('');
  });
</script>
```

## API Rate Limiting

For production, add rate limiting:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/shopify/recommendations")
@limiter.limit("100/minute")
async def get_recommendations(...):
    ...
```

## Monitoring & Analytics

### Track Recommendation Performance

```javascript
// Track click-through on recommendations
function trackRecommendationClick(productId, recommendationType) {
  fetch('https://your-api.com/api/v1/analytics/click', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      product_id: productId,
      recommendation_type: recommendationType,
      timestamp: new Date().toISOString()
    })
  });
}
```

### Key Metrics to Monitor

| Metric | Description | Target |
|--------|-------------|--------|
| CTR | Click-through rate on recommendations | >5% |
| Conversion | Orders from recommended products | >2% |
| Response Time | API latency | <200ms |
| Coverage | % of users with recommendations | >90% |

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CORS Error | API not allowing Shopify domain | Add domain to CORS whitelist |
| 403 Forbidden | Invalid access token | Regenerate token in Shopify Admin |
| Empty recommendations | User not in cache | Fallback to popular products |
| Slow response | Large dataset | Add Redis caching layer |

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --log-level debug
```

## Production Checklist

- [ ] Deploy API to production server (Heroku/Lightsail/Docker)
- [ ] Configure CORS for Shopify domain (`masterverse-project.myshopify.com`)
- [ ] Set up SSL/HTTPS certificate
- [ ] Add API rate limiting (100 req/min recommended)
- [ ] Run local ML pipeline to populate cache (74,827 users, 4,182 items)
- [ ] Test all endpoints with real Shopify product IDs
- [ ] Add error handling and fallbacks in frontend code
- [ ] Set up webhook for real-time order sync
- [ ] Configure monitoring and alerting
- [ ] Document product ID mapping between systems

## Support

For issues or questions:
- Check API logs: `tail -f /var/log/mastergroup-api.log`
- Test endpoints: Use the cURL examples above
- Shopify Admin: Settings → Apps → pods-recommendation
