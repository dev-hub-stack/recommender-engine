# 🔒 Secure API Access via Netlify Proxy

> **Last Updated:** December 17, 2025
> **Status:** ✅ Active

## Problem
Shopify stores run on HTTPS. Our EC2 API runs on HTTP.
Browsers block "mixed content" (HTTP requests from HTTPS pages), causing `net::ERR_SSL_PROTOCOL_ERROR`.

## Solution
Use the Analytics Dashboard (hosted on Netlify) as a secure reverse proxy.

```
┌─────────────┐       HTTPS        ┌─────────────┐       HTTP        ┌─────────────┐
│   Shopify   │  ────────────────► │   Netlify   │  ───────────────► │   EC2 API   │
│   Store     │                    │  (Dashboard)│                   │ 3.209.80.206│
└─────────────┘                    └─────────────┘                   └─────────────┘
```

## Configuration

### 1. Netlify Proxy (`netlify.toml`)
Located in `mastergroup-analytics-dashboard/netlify.toml`:

```toml
[[redirects]]
  from = "/api/*"
  to = "http://3.209.80.206:8001/api/:splat"
  status = 200
  force = true
```

### 2. Shopify Integration
Update your theme code to use the Netlify URL instead of the EC2 IP.

**Before:**
```javascript
fetch('http://3.209.80.206:8001/api/v1/shopify/similar/...')
```

**After:**
```javascript
// Replace with your actual Netlify site URL
const API_BASE = 'https://your-dashboard-url.netlify.app/api/v1';
fetch(API_BASE + '/shopify/similar/...')
```

## Setup Steps

1. **Deploy Dashboard:**
   - Push changes to `mastergroup-analytics-dashboard` repo
   - Netlify automatically builds and deploys
   - Verify: Visit `https://your-dashboard.netlify.app/health` -> should return JSON

2. **Update Shopify Theme:**
   - Edit `sections/product-template.liquid`
   - Edit `sections/cart-template.liquid`
   - Replace `http://3.209.80.206:8001/api/v1` with `https://your-dashboard.netlify.app/api/v1`

## Troubleshooting

- **502 Bad Gateway:** EC2 API is down. Check with `ssh ubuntu@3.209.80.206`
- **404 Not Found:** Proxy path is wrong. Check `netlify.toml`
- **CORS Error:** Should not happen as Netlify handles headers, but check EC2 `main.py` CORS settings if issues persist.
