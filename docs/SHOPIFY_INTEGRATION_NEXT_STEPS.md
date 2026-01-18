# Shopify Integration - Next Steps Guide

## ✅ Already Completed

1. **Product Mapping Tables** - Created via Alembic migration
2. **164/171 Shopify Products Matched** - 95.9% match rate
3. **API Endpoints Updated** - `/shopify/similar`, `/shopify/recommendations`
4. **Shopify Theme Templates Updated** - `product-template.liquid` and `cart-template.liquid`

---

## 📋 Step 1: Setup Order Webhook in Shopify Admin

The order webhook captures new Shopify orders to improve ML recommendations over time.

### Steps:

1. **Login to Shopify Admin**
   - Go to: https://masterverse-project.myshopify.com/admin

2. **Navigate to Webhooks**
   - Go to **Settings** (gear icon at bottom left)
   - Click **Notifications** (in the sidebar)
   - Scroll down to **Webhooks** section
   - Click **Create webhook**

3. **Configure the Webhook**
   
   | Field | Value |
   |-------|-------|
   | **Event** | `Order creation` |
   | **Format** | `JSON` |
   | **URL** | `http://3.209.80.206:8001/api/v1/shopify/webhook/order-created` |
   | **API version** | `2024-01` (or latest) |

4. **Click "Save"**

5. **Test the Webhook**
   - Click "Send test notification" button
   - Check EC2 logs to verify it was received:
   ```bash
   ssh ubuntu@3.209.80.206 "sudo journalctl -u mastergroup-api -n 20"
   ```

### What the Webhook Does:
- Captures order details (products, quantities, customer info)
- Stores in database for ML retraining
- Auto-populates customer mapping for personalization
- Updates product pair statistics

---

## 📋 Step 2: Add Customer Phone/Email for Personalized Recommendations

The Shopify templates already include customer data capture. Here's how it works:

### How It's Already Implemented:

In both `product-template.liquid` and `cart-template.liquid`, we capture:

```javascript
// Customer data is automatically captured from Shopify's customer object
var customerData = {
  phone: '{{ customer.phone | default: "" }}',
  email: '{{ customer.email | default: "" }}',
  city: '{{ customer.default_address.city | default: "" }}',
  province: '{{ customer.default_address.province | default: "" }}'
};
```

When the customer is logged in, this data is sent to the recommendations API:

```javascript
fetch(API_BASE + '/recommendations', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    customer_phone: customerData.phone,  // ← Used for matching
    customer_email: customerData.email,   // ← Used for matching
    current_product: productId,
    limit: 4
  })
})
```

### How Customer Matching Works:

1. **Phone Number Normalization**
   - Pakistani numbers like `03001234567` are normalized to `+923001234567`
   - Shopify phone like `+923001234567` matches POS phone `03001234567`

2. **Matching Priority**
   - First tries to match by phone (most reliable in Pakistan)
   - Then tries email match
   - Falls back to popular products if no match

3. **Personalized Recommendations**
   - If customer is matched, shows products based on their purchase history
   - Uses collaborative filtering: "Customers who bought X also bought Y"

### Testing Personalization:

1. **Customer must be logged in** to see personalized recommendations
2. Open browser console and check for:
   ```
   customerData = {phone: "+923001234567", email: "customer@example.com", ...}
   ```
3. The API will then return personalized results

---

## 📋 Step 3: Upload Templates to Shopify Theme

### Option A: Through Shopify Admin (Recommended)

1. **Login to Shopify Admin**
   - https://masterverse-project.myshopify.com/admin

2. **Go to Online Store → Themes**
   - Click on your live theme
   - Click **"..."** → **"Edit code"**

3. **Update Product Template**
   - Navigate to **Templates** or **Sections**
   - Find `product-template.liquid` or `main-product.liquid`
   - Add the MasterGroup AI Recommendations code block **BEFORE** the closing `</div>` or at the end of the product section

4. **Update Cart Template**
   - Find `cart-template.liquid` or `main-cart.liquid`  
   - Add the code block similarly

5. **Save and Preview**

### Option B: Copy-Paste the Code Block

Add this to your product page template right before the closing script tags:

```liquid
<!-- MasterGroup AI Recommendations - START -->
<div class="mg-recommendations-wrapper" style="margin: 40px auto; padding: 30px; background: linear-gradient(135deg, #fff8f0 0%, #ffe9c0 100%); border-radius: 12px; max-width: 1200px;">
  <h3 style="margin-bottom: 20px; font-size: 1.4rem; font-weight: bold; color: #ed1c24; text-align: center;">You Might Also Like</h3>
  <div id="mg-similar-products" style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;"></div>
</div>

<script>
(function() {
  var container = document.getElementById('mg-similar-products');
  if (!container) return;
  container.innerHTML = '<p style="color: #666; text-align: center;">Finding similar products...</p>';
  
  var API_BASE = 'http://3.209.80.206:8001/api/v1/shopify';
  var productId = '{{ product.id }}';
  var customerData = {
    phone: '{{ customer.phone | default: "" }}',
    email: '{{ customer.email | default: "" }}'
  };
  
  function renderProducts(products) {
    if (!products || products.length === 0) {
      container.parentElement.style.display = 'none';
      return;
    }
    container.innerHTML = products.slice(0, 4).map(function(item) {
      var name = item.item_name || item.product_name || 'Product';
      var score = item.score || 0;
      return '<div style="flex: 1 1 220px; max-width: 260px; padding: 20px; background: white; border: 1px solid #e0e0e0; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">' +
        '<div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 10px; color: #333;">' + name.substring(0, 40) + '</div>' +
        '<div style="color: #ed1c24; font-size: 0.85rem;">' + (score > 0 && score <= 1 ? 'AI Match: ' + Math.round(score * 100) + '%' : '🔥 Popular') + '</div>' +
      '</div>';
    }).join('');
  }
  
  // First try similar products, then personalized, then popular
  fetch(API_BASE + '/similar/' + productId + '?limit=4')
    .then(function(res) { return res.json(); })
    .then(function(data) {
      if (data.success && data.similar_products && data.similar_products.length > 0) {
        renderProducts(data.similar_products);
      } else if (customerData.phone || customerData.email) {
        return fetch(API_BASE + '/recommendations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ customer_phone: customerData.phone, customer_email: customerData.email, current_product: productId, limit: 4 })
        }).then(function(res) { return res.json(); });
      } else {
        return fetch(API_BASE + '/popular?limit=4').then(function(res) { return res.json(); });
      }
    })
    .then(function(data) {
      if (data) renderProducts(data.recommendations || data.popular_products || data.similar_products);
    })
    .catch(function() { container.parentElement.style.display = 'none'; });
})();
</script>
<!-- MasterGroup AI Recommendations - END -->
```

---

## 📋 Step 4: Verify Everything Works

### Test Similar Products API:
```bash
# Test with a Shopify product ID
curl "http://3.209.80.206:8001/api/v1/shopify/similar/10045012017458?limit=4"
```

### Test Recommendations API:
```bash
# Test with customer phone
curl -X POST "http://3.209.80.206:8001/api/v1/shopify/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"customer_phone": "03001234567", "limit": 4}'
```

### Test Product Mapping:
```bash
# Check mappings
curl "http://3.209.80.206:8001/api/v1/shopify/product-mappings"
```

---

## 🔐 Security Note

Currently using HTTP (port 8001). For production:
1. Add SSL certificate (Let's Encrypt via Certbot)
2. Configure Nginx as HTTPS proxy
3. Update API_BASE in templates to use `https://`

---

## 📊 Summary of API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/shopify/similar/{product_id}` | GET | Similar products (uses Shopify ID) |
| `/api/v1/shopify/recommendations` | POST | Personalized recommendations |
| `/api/v1/shopify/popular` | GET | Popular products fallback |
| `/api/v1/shopify/product-mappings` | GET | View all product mappings |
| `/api/v1/shopify/translate-product/{id}` | GET | Debug: translate Shopify ID |
| `/api/v1/shopify/webhook/order-created` | POST | Order webhook endpoint |

---

## ✅ Checklist

- [ ] Upload updated templates to Shopify theme
- [ ] Create order webhook in Shopify Admin
- [ ] Test recommendations on product page
- [ ] Test recommendations on cart page  
- [ ] Login as customer and verify personalized recommendations
- [ ] (Optional) Add SSL/HTTPS for production
