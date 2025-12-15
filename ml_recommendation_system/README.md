# ML Recommendation System

## Overview

This system implements a collaborative filtering recommendation engine for product recommendations based on customer purchase history.

## Directory Structure

```
ml-recommendation-system/
├── data/
│   ├── raw/              # Raw data files (not tracked)
│   ├── processed/        # Cleaned, processed data ready for training
│   └── models/           # Trained model artifacts
├── src/
│   ├── pipeline/         # Data pipeline modules
│   │   ├── data_loader.py      # Load data from CSV or API
│   │   ├── data_processor.py   # Clean and transform data
│   │   └── pipeline.py         # Main pipeline orchestrator
│   ├── training/         # Model training (coming next)
│   └── api/              # API for serving recommendations (coming next)
├── config/
│   └── config.py         # Configuration settings
├── run_pipeline.py       # Script to run the data pipeline
└── requirements.txt      # Python dependencies
```

## Data Pipeline

### What It Does

The pipeline processes raw order data and prepares it for ML training:

1. **Load Data**: Reads from CSV files (development) or API endpoints (production)
2. **Extract Columns**: Keeps only necessary columns:
   - `customer_phone` / `customer_email` → Customer ID
   - `has_items` → Product information (JSON)
   - `order_date` → Purchase date
   - `id` → Order ID
3. **Parse Products**: Extracts product details from JSON
4. **Time Filter**: Uses only recent data (default: last 2 years)
5. **Clean Data**: Removes duplicates, invalid records
6. **Aggregate**: Combines multiple purchases of same product
7. **Save**: Stores processed data in `data/processed/`

### Output Files

The pipeline creates:
- `processed_orders_YYYYMMDD_HHMMSS.csv` - Timestamped processed data
- `processed_orders_YYYYMMDD_HHMMSS.parquet` - Parquet format (efficient)
- `processed_orders_latest.csv` - Latest version (easy access)
- `processed_orders_latest.parquet` - Latest parquet version
- `metadata_YYYYMMDD_HHMMSS.json` - Processing metadata and statistics

### Processed Data Schema

| Column | Type | Description |
|--------|------|-------------|
| customer_id | string | Unique customer identifier (phone or email) |
| product_id | string | Product identifier (product title) |
| product_name | string | Product name |
| quantity | int | Total quantity purchased |
| price | float | Average price |
| order_date | datetime | Most recent purchase date |
| id | string | Order ID |
| source | string | Data source (POS or OE) |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Pipeline (CSV Mode)

```bash
# Make sure CSV files are in the root directory
python run_pipeline.py
```

### Configuration

Edit `config/config.py` to customize:

```python
# Data source
SOURCE_TYPE = 'csv'  # or 'api' for production

# Time window (days)
TIME_WINDOW_DAYS = 730  # 2 years

# CSV file paths
POS_ORDERS_CSV = 'all_pos_orders.csv'
OE_ORDERS_CSV = 'all_oe_orders.csv'

# API configuration (for production)
API_CONFIG = {
    'pos_endpoint': 'https://api.masterverse.com/pos-orders',
    'oe_endpoint': 'https://api.masterverse.com/oe-orders',
    'auth_token': 'Bearer YOUR_TOKEN_HERE'
}
```

### Switch to API Mode

**Quick Start:**
1. Edit `config/config.py` - change `SOURCE_TYPE = 'api'`
2. Update `API_CONFIG` with your API details
3. Run `python run_pipeline.py`

**📖 Full Documentation:**
- **[API Migration Guide](docs/API_MIGRATION_GUIDE.md)** - Complete guide with examples
- **[API Quick Reference](docs/API_QUICK_REFERENCE.md)** - Quick reference card

The pipeline is already built to support both CSV and API modes!

## Next Steps

1. ✅ **Data Pipeline** (COMPLETE)
2. ✅ **Model Training** (COMPLETE)
3. ✅ **API Service** (COMPLETE)
4. 🔄 **Dashboard/Deployment** (Next)
5. ⏳ **Auto-pilot Scheduling** (Coming)

## Key Features

- **Flexible Data Source**: Works with CSV files or API endpoints
- **Time-Based Filtering**: Uses only recent data to avoid outdated trends
- **Efficient Storage**: Saves in both CSV and Parquet formats
- **Metadata Tracking**: Records processing statistics
- **Production Ready**: Easy switch from CSV to API mode
