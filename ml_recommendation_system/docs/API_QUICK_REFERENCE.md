# API Mode - Quick Reference

## Switch to API Mode in 3 Steps

### 1. Edit `config/config.py`

```python
SOURCE_TYPE = 'api'  # Change from 'csv'

API_CONFIG = {
    'pos_endpoint': 'YOUR_POS_API_URL',
    'oe_endpoint': 'YOUR_OE_API_URL',
    'auth_token': 'Bearer YOUR_TOKEN',
    'headers': {
        'Content-Type': 'application/json'
    }
}
```

### 2. Run Pipeline

```bash
python ml-recommendation-system/run_pipeline.py
```

### 3. Done!

The pipeline will fetch data from API instead of CSV files.

---

## What You Need

- [ ] POS API endpoint URL
- [ ] OE API endpoint URL  
- [ ] Authentication token
- [ ] API should return JSON with these fields:
  - `customer_phone`
  - `customer_email`
  - `has_items`
  - `order_date`
  - `id`

---

## Rollback to CSV

```python
SOURCE_TYPE = 'csv'  # Change back from 'api'
```

---

## Full Documentation

See `API_MIGRATION_GUIDE.md` for complete details, troubleshooting, and examples.
