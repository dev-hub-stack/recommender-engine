# Reminder: How to Switch Pipeline to API Mode

## Quick Prompt for Kiro

If you need to switch the pipeline to API mode later, just say:

> "Please switch the data pipeline to API mode. Here are my API details:
> - POS Endpoint: [YOUR_URL]
> - OE Endpoint: [YOUR_URL]  
> - Auth Token: [YOUR_TOKEN]"

Or simply:

> "Switch to API mode using the details in API_MIGRATION_GUIDE.md"

---

## What Kiro Will Do

Kiro will:
1. Update `config/config.py` with your API details
2. Change `SOURCE_TYPE` from `'csv'` to `'api'`
3. Test the configuration
4. Run the pipeline to verify it works

---

## What You Need to Provide

Just provide these 3 things:
1. **POS API Endpoint URL**
2. **OE API Endpoint URL**
3. **Authentication Token**

Example:
```
POS Endpoint: https://api.masterverse.com/v1/pos-orders
OE Endpoint: https://api.masterverse.com/v1/oe-orders
Auth Token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Manual Switch (If You Want to Do It Yourself)

1. Open `ml-recommendation-system/config/config.py`
2. Change line 18: `SOURCE_TYPE = 'api'`
3. Update lines 21-28 with your API details
4. Run: `python ml-recommendation-system/run_pipeline.py`

Done!

---

## Documentation

- Full guide: `docs/API_MIGRATION_GUIDE.md`
- Quick reference: `docs/API_QUICK_REFERENCE.md`
