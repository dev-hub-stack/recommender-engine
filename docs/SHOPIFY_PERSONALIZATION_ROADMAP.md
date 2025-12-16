# 🎯 Shopify Personalization Roadmap

> **Last Updated:** December 17, 2025  
> **Current Status:** Phase 1 Complete (Trending Products)  
> **Next Phase:** Product Mapping for True Personalization

---

## Executive Summary

We have successfully integrated the recommendation system with Shopify, but **true personalization is not yet active** due to a fundamental data mismatch between Shopify product IDs and Master Group internal product IDs.

### Current State
| Feature | Status | Notes |
|---------|--------|-------|
| API Integration | ✅ Working | Via Netlify proxy (HTTPS) |
| Trending Products | ✅ Working | Shows popular items from POS/OE data |
| Similar Products | ⚠️ Partial | Works with internal IDs only |
| Personalized Recs | ❌ Not Active | Requires customer mapping |
| Order Sync | ❌ Not Active | Webhook not configured |

---

## The Core Problem

### ID Mismatch

```
┌─────────────────────────────────────────────────────────────┐
│                     THE PROBLEM                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  SHOPIFY STORE                    MASTER GROUP (POS/OE)      │
│  ─────────────                    ─────────────────────      │
│  Product ID: 10045012017458       Product ID: 1553           │
│  Title: "Beauty Rest"             Name: "CELESTE 72-48"      │
│  SKU: "beauty-rest-78*42-5"       SKU: null                  │
│                                                               │
│           ┌───────────────┐                                  │
│           │  NO MAPPING   │ ◄── This is the problem          │
│           └───────────────┘                                  │
│                                                               │
│  Customer ID: 9281038123          Customer Phone: 03001234   │
│  Email: user@email.com            Name: "Mr Sateeq"          │
│                                                               │
│           ┌───────────────┐                                  │
│           │  NO MAPPING   │ ◄── Also this                    │
│           └───────────────┘                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Impact

1. **Similar Products:** Can't find similar items because Shopify product ID `10045012017458` doesn't exist in our database.
2. **Personalized Recs:** Can't identify the customer because Shopify customer ID doesn't match our phone-based user IDs.
3. **Order Sync:** Can't update our models because Shopify orders reference unknown products/customers.

---

## Current Workaround

We're using **Trending/Popular Products** as a fallback:

```javascript
// Instead of:
fetch('/api/v1/shopify/similar/' + shopifyProductId)  // ❌ 404 Error

// We use:
fetch('/api/v1/shopify/popular?limit=4')  // ✅ Always works
```

**Result:** Shows best-selling products from POS/OE data with "🔥 Trending" label.

---

## Roadmap to True Personalization

### Phase 1: Trending Products ✅ COMPLETE
**Timeline:** Done  
**Effort:** 1 day

- [x] API deployed on EC2
- [x] Netlify proxy for HTTPS
- [x] Shopify theme integration
- [x] Popular products endpoint
- [x] Daily training scheduled

### Phase 2: Product Mapping
**Timeline:** 1-2 weeks  
**Effort:** Medium

**Goal:** Link Shopify products to Master Group products.

#### Option A: Manual SKU Mapping (Recommended)
1. Export all Master Group products with SKUs
2. Import products to Shopify with matching SKUs
3. Use SKU as the common identifier

```python
# In Shopify theme:
var sku = '{{ product.variants.first.sku }}';
fetch('/api/v1/shopify/similar-by-sku/' + sku);

# API implementation:
@router.get("/shopify/similar-by-sku/{sku}")
async def similar_by_sku(sku: str):
    # Look up internal product ID by SKU
    product_id = await db.fetch_one(
        "SELECT product_id FROM products WHERE sku = $1", sku
    )
    if product_id:
        return await get_similar_items(product_id)
    return await get_popular_products()
```

#### Option B: Name Similarity Matching
Use fuzzy matching to link products by name.

```python
from fuzzywuzzy import fuzz

def find_matching_product(shopify_title):
    """Find internal product that best matches Shopify title."""
    products = await db.fetch_all("SELECT id, name FROM products")
    
    best_match = None
    best_score = 0
    
    for product in products:
        score = fuzz.token_sort_ratio(shopify_title, product['name'])
        if score > best_score and score > 70:  # 70% threshold
            best_score = score
            best_match = product
    
    return best_match
```

#### Option C: Product Sync via Shopify API
Automatically sync Shopify products to our database.

```python
# Nightly job to sync Shopify products
async def sync_shopify_products():
    shopify_products = await shopify_api.get_all_products()
    
    for product in shopify_products:
        await db.execute("""
            INSERT INTO shopify_products (
                shopify_id, title, sku, handle, internal_product_id
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (shopify_id) DO UPDATE SET ...
        """, product['id'], product['title'], ...)
```

### Phase 3: Customer Mapping
**Timeline:** 1 week  
**Effort:** Medium

**Goal:** Link Shopify customers to Master Group customers.

#### Implementation
1. Match by phone number (primary)
2. Match by email (secondary)
3. Create new record if no match

```python
async def identify_customer(shopify_customer):
    """Find or create internal customer from Shopify data."""
    
    phone = normalize_phone(shopify_customer.get('phone'))
    email = shopify_customer.get('email')
    
    # Try phone first
    if phone:
        customer = await db.fetch_one(
            "SELECT user_id FROM customers WHERE phone = $1", phone
        )
        if customer:
            return customer['user_id']
    
    # Try email
    if email:
        customer = await db.fetch_one(
            "SELECT user_id FROM customers WHERE email = $1", email
        )
        if customer:
            return customer['user_id']
    
    # No match - new Shopify customer
    return None  # Use popularity-based recs
```

### Phase 4: Order Webhook
**Timeline:** 2 days  
**Effort:** Low

**Goal:** Capture Shopify orders for model retraining.

```python
@router.post("/shopify/webhook/order-created")
async def order_webhook(request: Request):
    """Handle Shopify order webhook."""
    
    order = await request.json()
    
    # Map customer
    customer_id = await identify_customer(order['customer'])
    
    # Map products
    for item in order['line_items']:
        product_id = await find_product_by_sku(item['sku'])
        
        if customer_id and product_id:
            await db.execute("""
                INSERT INTO interactions (user_id, item_id, event_type)
                VALUES ($1, $2, 'purchase')
            """, customer_id, product_id)
    
    return {"status": "processed"}
```

### Phase 5: Full Personalization
**Timeline:** Ongoing  
**Effort:** Low (maintenance)

Once mapping is complete:

1. **Product Page:** Show truly similar products based on purchase patterns
2. **Cart Page:** Show personalized upsells based on customer history
3. **Homepage:** Show personalized recommendations for logged-in users
4. **Email:** Power abandoned cart emails with personalized products

---

## Technical Requirements

### Database Changes

```sql
-- New table: Shopify product mapping
CREATE TABLE shopify_product_mapping (
    id SERIAL PRIMARY KEY,
    shopify_product_id BIGINT UNIQUE NOT NULL,
    shopify_title VARCHAR(255),
    shopify_sku VARCHAR(100),
    shopify_handle VARCHAR(255),
    internal_product_id VARCHAR(100),  -- Our product ID
    match_confidence FLOAT,  -- 0-1 score
    match_method VARCHAR(50),  -- 'sku', 'name_fuzzy', 'manual'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- New table: Shopify customer mapping
CREATE TABLE shopify_customer_mapping (
    id SERIAL PRIMARY KEY,
    shopify_customer_id BIGINT UNIQUE NOT NULL,
    shopify_email VARCHAR(255),
    shopify_phone VARCHAR(50),
    internal_user_id VARCHAR(100),  -- Our user ID
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_shopify_sku ON shopify_product_mapping(shopify_sku);
CREATE INDEX idx_shopify_phone ON shopify_customer_mapping(shopify_phone);
```

### API Changes

```python
# New endpoints needed:
GET  /api/v1/shopify/similar-by-sku/{sku}
GET  /api/v1/shopify/recommendations-by-email/{email}
POST /api/v1/shopify/sync-products
POST /api/v1/shopify/sync-customers
```

### Shopify Theme Changes

```javascript
// Update to use SKU instead of product ID
var sku = '{{ product.variants.first.sku }}';
var customerEmail = '{{ customer.email }}';
var customerPhone = '{{ customer.phone }}';

fetch('/api/v1/shopify/recommendations', {
  method: 'POST',
  body: JSON.stringify({
    sku: sku,
    customer_email: customerEmail,
    customer_phone: customerPhone,
    city: '{{ customer.default_address.city }}'
  })
});
```

---

## Effort Estimates

| Phase | Effort | Business Value | Priority |
|-------|--------|----------------|----------|
| Phase 1: Trending | ✅ Done | Medium | - |
| Phase 2: Product Mapping | 1-2 weeks | High | 🔴 High |
| Phase 3: Customer Mapping | 1 week | High | 🔴 High |
| Phase 4: Order Webhook | 2 days | Medium | 🟡 Medium |
| Phase 5: Full Personalization | Ongoing | Very High | 🟢 After 2-4 |

---

## Quick Wins

### 1. Add SKUs to Shopify Products
If your Shopify products don't have SKUs that match Master Group:
- Export Master Group products
- Add matching SKUs to Shopify products
- Immediately enables similar product recommendations

### 2. Collect Customer Phone at Checkout
- Add phone as required field
- Format: Pakistani format (03XX-XXXXXXX)
- Enables personalized recommendations on next visit

### 3. Enable Customer Accounts
- Required for cross-session personalization
- Captures email and phone for matching

---

## Metrics to Track

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Recommendation CTR | Unknown | 3-5% | Add tracking clicks |
| Cart Additions | Unknown | 2-3% | Track "Add to Cart" from recs |
| Revenue from Recs | $0 | 5-10% of total | Tag orders with rec source |
| Coverage | 0% products | 80%+ products | Product mapping completion |

---

## Conclusion

**Current State:** Working integration with trending products.  
**Blocker:** Product and customer ID mapping.  
**Next Step:** Implement SKU-based product mapping (Phase 2).

The foundation is solid. Once product mapping is complete, true personalization is just a configuration change.
