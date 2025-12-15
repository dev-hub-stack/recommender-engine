# Quick Start Guide: Data Ingestion Pipeline

**Get started in 5 minutes!**

---

## Step 1: Setup Environment

```bash
# Navigate to project
cd ml-recommendation-system

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

---

## Step 2: Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials
nano .env  # or use any text editor
```

**Required settings in .env:**
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/ml_recommendations
POS_API_URL=https://api.masterverse.com/v1/pos-orders
OE_API_URL=https://api.masterverse.com/v1/oe-orders
API_AUTH_TOKEN=Bearer your_actual_token_here
```

---

## Step 3: Initialize Database

```bash
# Create database tables
python run_ingestion.py --init-db
```

---

## Step 4: Test Connections

```bash
# Test API connections
python run_ingestion.py --test-connection
```

Expected output:
```
Testing API connections...
✅ POS API: Success
✅ OE API: Success
```

---

## Step 5: Run Ingestion

```bash
# Fetch last 30 days of data
python run_ingestion.py --days 30

# Or fetch all data
python run_ingestion.py
```

---

## Common Commands

```bash
# Fetch last 7 days
python run_ingestion.py --days 7

# Fetch only POS orders
python run_ingestion.py --source pos --days 30

# Fetch only OE orders
python run_ingestion.py --source oe --days 30

# Fetch specific date range
python run_ingestion.py --start-date 2024-01-01 --end-date 2024-12-31

# Test with limited pages (for testing)
python run_ingestion.py --max-pages 2
```

---

## Verify Data

```bash
# Connect to database
psql -h localhost -U username -d ml_recommendations

# Check record counts
SELECT 'POS' as source, COUNT(*) FROM pos_orders
UNION ALL
SELECT 'OE' as source, COUNT(*) FROM oe_orders;

# Check recent syncs
SELECT * FROM sync_logs ORDER BY started_at DESC LIMIT 5;
```

---

## Troubleshooting

**Issue: Database connection failed**
```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Verify DATABASE_URL in .env
```

**Issue: API authentication failed**
```bash
# Verify API_AUTH_TOKEN in .env
# Make sure it starts with "Bearer "
```

**Issue: No data fetched**
```bash
# Check date filters
# Verify API has data for the date range
# Check logs for errors
```

---

## Next Steps

1. ✅ Data ingestion working
2. ⏭️  Data processing (location ID generation)
3. ⏭️  ML model training
4. ⏭️  API deployment

---

## Documentation

- **Full Documentation:** `docs/DATA_INGESTION_PIPELINE.md`
- **API Documentation:** `docs/API_DOCUMENTATION.md`
- **Database Models:** `src/database/models.py`

---

## Support

**Issues?** Check the full documentation or contact the team.

**Success?** Move on to the next phase: Data Processing!
