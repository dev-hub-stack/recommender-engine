# API Migration Guide

## Overview

This guide explains how to switch the data pipeline from CSV files to API endpoints for production deployment.

The pipeline is **already built** to support both CSV and API modes. Switching requires only configuration changes - no code modifications needed!

---

## Quick Start (TL;DR)

1. Open `ml-recommendation-system/config/config.py`
2. Change `SOURCE_TYPE = 'csv'` to `SOURCE_TYPE = 'api'`
3. Update `API_CONFIG` with your API details
4. Run `python run_pipeline.py`

---

## Step-by-Step Migration

### Step 1: Gather API Information

Before switching, collect these details:

#### Required Information:
- [ ] **POS Orders API Endpoint URL**
- [ ] **OE Orders API Endpoint URL**
- [ ] **Authentication Token/Key**
- [ ] **API Response Format** (sample JSON)

#### Optional Information:
- [ ] Pagination details (if API paginates results)
- [ ] Rate limits (requests per minute/hour)
- [ ] Custom headers required
- [ ] Query parameters needed

---

### Step 2: Update Configuration File

Edit `ml-recommendation-system/config/config.py`:

```python
# Change source type from 'csv' to 'api'
SOURCE_TYPE = 'api'  # Changed from 'csv'

# Update API configuration
API_CONFIG = {
    'pos_endpoint': 'https://api.masterverse.com/v1/pos-orders',  # Your POS API URL
    'oe_endpoint': 'https://api.masterverse.com/v1/oe-orders',    # Your OE API URL
    'auth_token': 'Bearer YOUR_ACTUAL_TOKEN_HERE',                # Your authentication token
    'headers': {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
        # Add any other required headers
    }
}
```

---

### Step 3: Test API Connection

Before running the full pipeline, test your API connection:

```python
# test_api.py
import requests
from config.config import API_CONFIG

# Test POS endpoint
print("Testing POS API...")
response = requests.get(
    API_CONFIG['pos_endpoint'],
    headers={**API_CONFIG['headers'], 'Authorization': API_CONFIG['auth_token']}
)
print(f"Status: {response.status_code}")
print(f"Sample data: {response.json()[:2]}")  # First 2 records

# Test OE endpoint
print("\nTesting OE API...")
response = requests.get(
    API_CONFIG['oe_endpoint'],
    headers={**API_CONFIG['headers'], 'Authorization': API_CONFIG['auth_token']}
)
print(f"Status: {response.status_code}")
print(f"Sample data: {response.json()[:2]}")  # First 2 records
```

---

### Step 4: Run Pipeline

```bash
python ml-recommendation-system/run_pipeline.py
```

The pipeline will automatically fetch data from API endpoints instead of CSV files.

---

## API Response Format Requirements

### Expected JSON Structure

The API should return a JSON array with these fields:

```json
[
  {
    "customer_phone": "03004547681",
    "customer_email": "customer@example.com",
    "has_items": "{'id': 1297, 'title': 'MOLTY FOAM 2IN1 78-72-6', 'quantity': 1, 'price': 28700}",
    "order_date": "2022-04-19",
    "id": "710"
  },
  {
    "customer_phone": "03214674910",
    "customer_email": "another@example.com",
    "has_items": "{'id': 1300, 'title': 'MOLTY FOAM 2IN1 78-72-8', 'quantity': 1, 'price': 39800}",
    "order_date": "2022-04-20",
    "id": "711"
  }
]
```

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `customer_phone` | string | Customer phone number | "03004547681" |
| `customer_email` | string | Customer email | "customer@example.com" |
| `has_items` | string/object | Product details (JSON) | See below |
| `order_date` | string | Order date (YYYY-MM-DD) | "2022-04-19" |
| `id` | string/number | Order ID | "710" |

### `has_items` Format

Can be either:

**Option 1: JSON String**
```json
"has_items": "{'id': 1297, 'title': 'MOLTY FOAM', 'quantity': 1, 'price': 28700}"
```

**Option 2: JSON Object**
```json
"has_items": {
  "id": 1297,
  "title": "MOLTY FOAM",
  "quantity": 1,
  "price": 28700
}
```

**Option 3: Array of Products**
```json
"has_items": [
  {"id": 1297, "title": "MOLTY FOAM", "quantity": 1, "price": 28700},
  {"id": 1298, "title": "MOLTY PLUS", "quantity": 2, "price": 34600}
]
```

All formats are supported by the pipeline!

---

## Authentication Methods

### Bearer Token (Recommended)

```python
API_CONFIG = {
    'pos_endpoint': 'https://api.example.com/pos-orders',
    'oe_endpoint': 'https://api.example.com/oe-orders',
    'auth_token': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
    'headers': {
        'Content-Type': 'application/json'
    }
}
```

### API Key in Header

```python
API_CONFIG = {
    'pos_endpoint': 'https://api.example.com/pos-orders',
    'oe_endpoint': 'https://api.example.com/oe-orders',
    'auth_token': '',  # Leave empty
    'headers': {
        'Content-Type': 'application/json',
        'X-API-Key': 'your-api-key-here'
    }
}
```

### Basic Authentication

```python
API_CONFIG = {
    'pos_endpoint': 'https://api.example.com/pos-orders',
    'oe_endpoint': 'https://api.example.com/oe-orders',
    'auth_token': '',  # Leave empty
    'headers': {
        'Content-Type': 'application/json',
        'Authorization': 'Basic dXNlcm5hbWU6cGFzc3dvcmQ='  # base64(username:password)
    }
}
```

---

## Handling Different API Structures

### Nested Response

If your API returns nested data:

```json
{
  "success": true,
  "data": [
    { "customer_phone": "...", "has_items": "...", ... }
  ],
  "total": 223806
}
```

**Update `data_loader.py`:**

```python
# In load_from_api method, after getting response:
pos_response = requests.get(api_config['pos_endpoint'], headers=headers)
pos_response.raise_for_status()
pos_data = pos_response.json()

# Add this line to extract nested data:
if 'data' in pos_data:
    pos_data = pos_data['data']

pos_orders = pd.DataFrame(pos_data)
```

### Pagination

If your API paginates results (e.g., 1000 records per page):

**Update `data_loader.py`:**

```python
def load_from_api_paginated(self, api_config: Dict[str, str]) -> pd.DataFrame:
    """Load data from paginated API"""
    headers = api_config.get('headers', {})
    if 'auth_token' in api_config:
        headers['Authorization'] = api_config['auth_token']
    
    all_pos_orders = []
    page = 1
    
    # Fetch POS orders with pagination
    while True:
        logger.info(f"Fetching POS orders page {page}...")
        response = requests.get(
            f"{api_config['pos_endpoint']}?page={page}&limit=1000",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        if not data or len(data) == 0:
            break
        
        all_pos_orders.extend(data)
        page += 1
    
    pos_orders = pd.DataFrame(all_pos_orders)
    logger.info(f"Fetched {len(pos_orders)} POS orders")
    
    # Repeat for OE orders...
    # (similar logic)
    
    return combined_orders
```

### Different Field Names

If your API uses different field names:

**Update `data_loader.py`:**

```python
# After loading data, rename columns:
pos_orders = pos_orders.rename(columns={
    'phone': 'customer_phone',
    'email': 'customer_email',
    'items': 'has_items',
    'date': 'order_date',
    'order_id': 'id'
})
```

---

## Common Issues and Solutions

### Issue 1: SSL Certificate Errors

**Error:**
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution:**
```python
# In data_loader.py, add verify=False (NOT recommended for production)
response = requests.get(api_config['pos_endpoint'], headers=headers, verify=False)

# Better solution: Provide certificate path
response = requests.get(api_config['pos_endpoint'], headers=headers, verify='/path/to/cert.pem')
```

### Issue 2: Timeout Errors

**Error:**
```
requests.exceptions.Timeout
```

**Solution:**
```python
# Add timeout parameter
response = requests.get(
    api_config['pos_endpoint'], 
    headers=headers, 
    timeout=300  # 5 minutes
)
```

### Issue 3: Rate Limiting

**Error:**
```
429 Too Many Requests
```

**Solution:**
```python
import time

# Add delay between requests
time.sleep(1)  # Wait 1 second between requests

# Or use exponential backoff
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

response = session.get(api_config['pos_endpoint'], headers=headers)
```

---

## Testing Checklist

Before deploying to production:

- [ ] Test API connection with sample request
- [ ] Verify authentication works
- [ ] Check API response format matches expected structure
- [ ] Test with small dataset first (limit to 100 records)
- [ ] Verify all required fields are present
- [ ] Test pagination (if applicable)
- [ ] Check rate limits don't cause failures
- [ ] Verify processed data looks correct
- [ ] Compare results with CSV mode (should be similar)
- [ ] Test error handling (what happens if API is down?)

---

## Rollback to CSV Mode

If you need to switch back to CSV mode:

```python
# In config/config.py
SOURCE_TYPE = 'csv'  # Change back from 'api'
```

That's it! The pipeline will use CSV files again.

---

## Production Deployment Checklist

- [ ] Store API credentials securely (environment variables, not in code)
- [ ] Set up monitoring for API failures
- [ ] Configure retry logic for transient failures
- [ ] Set up alerts for pipeline failures
- [ ] Schedule regular pipeline runs (cron job, Airflow, etc.)
- [ ] Monitor API rate limits and costs
- [ ] Set up logging for debugging
- [ ] Test failover to CSV backup if API is down

---

## Environment Variables (Recommended for Production)

Instead of hardcoding credentials in `config.py`, use environment variables:

**Update `config/config.py`:**

```python
import os

# API configuration (for production)
API_CONFIG = {
    'pos_endpoint': os.getenv('POS_API_ENDPOINT', 'https://api.masterverse.com/pos-orders'),
    'oe_endpoint': os.getenv('OE_API_ENDPOINT', 'https://api.masterverse.com/oe-orders'),
    'auth_token': os.getenv('API_AUTH_TOKEN', 'Bearer YOUR_TOKEN'),
    'headers': {
        'Content-Type': 'application/json'
    }
}
```

**Set environment variables:**

```bash
# Linux/Mac
export POS_API_ENDPOINT="https://api.masterverse.com/v1/pos-orders"
export OE_API_ENDPOINT="https://api.masterverse.com/v1/oe-orders"
export API_AUTH_TOKEN="Bearer your-actual-token-here"

# Windows
set POS_API_ENDPOINT=https://api.masterverse.com/v1/pos-orders
set OE_API_ENDPOINT=https://api.masterverse.com/v1/oe-orders
set API_AUTH_TOKEN=Bearer your-actual-token-here
```

**Or use `.env` file:**

```bash
# .env file
POS_API_ENDPOINT=https://api.masterverse.com/v1/pos-orders
OE_API_ENDPOINT=https://api.masterverse.com/v1/oe-orders
API_AUTH_TOKEN=Bearer your-actual-token-here
```

```python
# Install python-dotenv
# pip install python-dotenv

# In config/config.py
from dotenv import load_dotenv
load_dotenv()

API_CONFIG = {
    'pos_endpoint': os.getenv('POS_API_ENDPOINT'),
    'oe_endpoint': os.getenv('OE_API_ENDPOINT'),
    'auth_token': os.getenv('API_AUTH_TOKEN'),
    'headers': {'Content-Type': 'application/json'}
}
```

---

## Example: Complete API Migration

Here's a complete example of switching to API mode:

### Before (CSV Mode):

```python
# config/config.py
SOURCE_TYPE = 'csv'
POS_ORDERS_CSV = 'all_pos_orders.csv'
OE_ORDERS_CSV = 'all_oe_orders.csv'
```

### After (API Mode):

```python
# config/config.py
SOURCE_TYPE = 'api'

API_CONFIG = {
    'pos_endpoint': 'https://api.masterverse.com/v1/pos-orders',
    'oe_endpoint': 'https://api.masterverse.com/v1/oe-orders',
    'auth_token': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U',
    'headers': {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
}
```

### Run Pipeline:

```bash
python ml-recommendation-system/run_pipeline.py
```

### Expected Output:

```
======================================================================
ML RECOMMENDATION SYSTEM - DATA PIPELINE
======================================================================

Loading from API endpoints
INFO:pipeline.data_loader:Loading data from API endpoints
INFO:pipeline.data_loader:Fetching POS orders from https://api.masterverse.com/v1/pos-orders
INFO:pipeline.data_loader:Fetched 97617 POS orders
INFO:pipeline.data_loader:Fetching OE orders from https://api.masterverse.com/v1/oe-orders
INFO:pipeline.data_loader:Fetched 126189 OE orders
INFO:pipeline.data_loader:Combined total: 223806 orders
...
```

---

## Need Help?

If you encounter issues during migration:

1. **Check API documentation** - Verify endpoint URLs and authentication
2. **Test API manually** - Use Postman or curl to test API
3. **Check logs** - Look for error messages in pipeline output
4. **Compare with CSV** - Ensure API returns same data structure
5. **Contact support** - Reach out to API provider if issues persist

---

## Summary

**To switch to API mode:**

1. Update `SOURCE_TYPE = 'api'` in `config/config.py`
2. Fill in `API_CONFIG` with your API details
3. Run `python run_pipeline.py`

**That's it!** The pipeline handles everything else automatically.

The code is already built to support both CSV and API modes - you just need to configure it!
