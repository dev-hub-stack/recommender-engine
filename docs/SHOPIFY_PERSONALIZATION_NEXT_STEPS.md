# 🚀 Shopify Personalization - Next Steps Action Plan

> **Created:** January 18, 2026  
> **Priority:** High  
> **Estimated Effort:** 4-5 hours total  
> **Status:** Ready to implement

---

## Background

After analyzing the product catalogs, we discovered that **95% of Shopify products (164/171)** can be matched to MasterGroup products. This opens the door to enabling true personalization.

---

## Phase 1: Create Product Mapping Table ⏱️ 1 hour

### Task 1.1: Create Database Table

```sql
-- Create the mapping table
CREATE TABLE IF NOT EXISTS shopify_product_mapping (
    id SERIAL PRIMARY KEY,
    shopify_product_id BIGINT UNIQUE NOT NULL,
    shopify_title VARCHAR(255),
    shopify_sku VARCHAR(100),
    shopify_handle VARCHAR(255),
    mastergroup_product_id VARCHAR(100),
    mastergroup_product_name VARCHAR(255),
    match_confidence FLOAT DEFAULT 1.0,
    match_method VARCHAR(50),  -- 'title', 'keywords', 'sku', 'manual'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_shopify_mapping_shopify_id ON shopify_product_mapping(shopify_product_id);
CREATE INDEX idx_shopify_mapping_mg_id ON shopify_product_mapping(mastergroup_product_id);
CREATE INDEX idx_shopify_mapping_sku ON shopify_product_mapping(shopify_sku);
```

### Task 1.2: Populate with Matched Products

Run a script to insert the 164 matched products:

```python
# scripts/populate_shopify_mapping.py
# This will be created and run to populate the mapping table
```

### Task 1.3: Handle Unmatched Products

For the 7 unmatched products, either:
- **Option A:** Manually map them to closest products
- **Option B:** Mark as "fallback_to_popular" 
- **Option C:** Add them to MasterGroup database

---

## Phase 2: Update Shopify API Endpoints ⏱️ 2-3 hours

### Task 2.1: Update Similar Products Endpoint

**Current:**
```python
GET /api/v1/shopify/similar/{product_id}
# Expects MasterGroup internal product ID
```

**New:**
```python
GET /api/v1/shopify/similar/{shopify_product_id}
# Accepts Shopify product ID, translates internally
```

**Implementation:**
```python
@router.get("/shopify/similar/{product_id}")
async def get_similar_products(product_id: str, limit: int = 10):
    # Check if it's a Shopify ID (long numeric) or MasterGroup ID
    if len(product_id) > 10 and product_id.isdigit():
        # It's a Shopify ID - look up mapping
        mapping = await db.fetch_one(
            "SELECT mastergroup_product_id FROM shopify_product_mapping WHERE shopify_product_id = $1",
            int(product_id)
        )
        if mapping:
            internal_id = mapping['mastergroup_product_id']
        else:
            # No mapping - return popular products
            return await get_popular_products(limit=limit)
    else:
        internal_id = product_id
    
    # Get similar products using internal ID
    similar = await get_cached_similar_items(internal_id, limit)
    
    # Optionally: Map back to Shopify IDs for display
    return {"success": True, "similar_products": similar}
```

### Task 2.2: Update Recommendations Endpoint

**Add Shopify product ID support:**
```python
@router.post("/shopify/recommendations")
async def get_recommendations(
    customer_email: str = None,
    customer_phone: str = None,
    cart_items: List[str] = None,  # Can be Shopify or MG IDs
    current_product: str = None,   # Can be Shopify or MG ID
    ...
):
    # Translate any Shopify IDs in cart_items to MasterGroup IDs
    translated_cart = []
    for item_id in cart_items or []:
        translated_cart.append(await translate_product_id(item_id))
    
    # Continue with existing logic using translated IDs
    ...
```

### Task 2.3: Add Product Mapping API

**New endpoint to refresh mappings:**
```python
POST /api/v1/shopify/sync-products
# Fetches products from Shopify, updates mapping table
```

---

## Phase 3: Customer Mapping (Optional) ⏱️ 2-3 hours

### Task 3.1: Create Customer Mapping Table

```sql
CREATE TABLE IF NOT EXISTS shopify_customer_mapping (
    id SERIAL PRIMARY KEY,
    shopify_customer_id BIGINT UNIQUE,
    shopify_email VARCHAR(255),
    shopify_phone VARCHAR(50),
    mastergroup_user_id VARCHAR(100),
    match_method VARCHAR(50),  -- 'phone', 'email', 'name'
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Task 3.2: Match During Recommendation Request

```python
async def identify_customer(email: str = None, phone: str = None):
    """Find MasterGroup customer from Shopify data."""
    
    # Normalize phone (remove country code, spaces)
    if phone:
        phone = normalize_pakistan_phone(phone)
        
        # Check existing mapping
        mapping = await db.fetch_one(
            "SELECT mastergroup_user_id FROM shopify_customer_mapping WHERE shopify_phone = $1",
            phone
        )
        if mapping:
            return mapping['mastergroup_user_id']
        
        # Check MasterGroup orders by phone
        customer = await db.fetch_one(
            "SELECT DISTINCT unified_customer_id FROM orders WHERE customer_phone LIKE $1 LIMIT 1",
            f"%{phone[-10:]}"
        )
        if customer:
            return customer['unified_customer_id']
    
    # Try email
    if email:
        customer = await db.fetch_one(
            "SELECT DISTINCT unified_customer_id FROM orders WHERE customer_email = $1 LIMIT 1",
            email.lower()
        )
        if customer:
            return customer['unified_customer_id']
    
    return None  # Unknown customer - use popularity-based recs
```

---

## Phase 4: Update Shopify Theme ⏱️ 30 minutes

### Task 4.1: Update Product Page JavaScript

Change from using internal IDs to Shopify IDs:

```javascript
// BEFORE:
const productId = '1328';  // Hardcoded or internal ID

// AFTER:
const productId = '{{ product.id }}';  // Shopify product ID

fetch(`${API_URL}/shopify/similar/${productId}?limit=4`)
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      renderProducts(data.similar_products);
    }
  });
```

### Task 4.2: No Changes Needed for Cart/Checkout

The cart recommendations already use customer phone/email which works with the current system.

---

## Phase 5: Testing & Validation ⏱️ 1 hour

### Test Cases

| Test | Expected Result |
|------|-----------------|
| Similar products for matched Shopify ID | Returns similar items |
| Similar products for unmatched Shopify ID | Returns popular items |
| Recommendations with phone match | Returns personalized recs |
| Recommendations without phone | Returns popular recs |
| Cart with Shopify product IDs | Translates and returns cross-sells |

### Test Commands

```bash
# Test similar products with Shopify ID
curl "http://3.209.80.206:8001/api/v1/shopify/similar/10045012017458?limit=5"

# Test recommendations with customer phone
curl -X POST "http://3.209.80.206:8001/api/v1/shopify/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"customer_phone": "03001234567", "limit": 5}'

# Test with cart items (Shopify IDs)
curl -X POST "http://3.209.80.206:8001/api/v1/shopify/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"cart_items": ["10045012017458", "10078985060658"], "limit": 5}'
```

---

## Summary Timeline

| Phase | Task | Time | Priority |
|-------|------|------|----------|
| 1 | Create mapping table | 30 min | 🔴 Critical |
| 1 | Populate mappings | 30 min | 🔴 Critical |
| 2 | Update similar products API | 1 hour | 🔴 Critical |
| 2 | Update recommendations API | 1 hour | 🟡 High |
| 2 | Add product sync API | 30 min | 🟢 Medium |
| 3 | Customer mapping table | 30 min | 🟢 Medium |
| 3 | Customer matching logic | 1 hour | 🟢 Medium |
| 4 | Update Shopify theme | 30 min | 🟡 High |
| 5 | Testing | 1 hour | 🔴 Critical |

**Total: 6-7 hours for full implementation**

---

## Quick Win (30 minutes)

If you want immediate results with minimal changes:

1. **Create the mapping table** (10 min)
2. **Populate with matched products** (10 min)
3. **Modify `/shopify/similar/` to auto-translate IDs** (10 min)

This enables "You Might Also Like" for 95% of products immediately!

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `scripts/create_shopify_mapping.py` | Create - one-time table setup |
| `scripts/populate_shopify_mapping.py` | Create - populate 164 matches |
| `src/main.py` | Modify - update Shopify endpoints |
| `src/services/shopify_mapper.py` | Create - product ID translation service |
| Shopify theme files | Modify - use Shopify product IDs |

---

## Ready to Proceed?

Let me know if you'd like me to:

1. ✅ **Create the mapping table and populate it** - Phase 1
2. ✅ **Update the API endpoints** - Phase 2  
3. ✅ **Create the customer mapping** - Phase 3
4. ✅ **All of the above** - Full implementation

---

*Action Plan created by MasterGroup Recommendation Engine Team*
