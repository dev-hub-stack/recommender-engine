# Data Ingestion Pipeline Documentation

**Version:** 1.0  
**Last Updated:** December 11, 2024  
**Status:** Active Development

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Database Schema](#database-schema)
5. [Setup & Configuration](#setup--configuration)
6. [Usage](#usage)
7. [API Integration](#api-integration)
8. [Error Handling](#error-handling)
9. [Monitoring](#monitoring)
10. [Future Enhancements](#future-enhancements)

---

## 1. Overview

### Purpose

The Data Ingestion Pipeline fetches order data from external POS (Point of Sale) and OE (Order Entry) APIs and stores it in a PostgreSQL database for ML model training.

### Key Features

✅ **Dual Source Ingestion** - Fetches from both POS and OE APIs  
✅ **Automatic Pagination** - Handles large datasets automatically  
✅ **Upsert Logic** - Inserts new records, updates existing ones  
✅ **Data Validation** - Validates data before storing  
✅ **Error Handling** - Graceful error handling with logging  
✅ **Sync Tracking** - Tracks all sync operations in database  
✅ **Date Filtering** - Supports date range filtering  
✅ **CLI Interface** - Easy command-line execution  

### Data Flow

```
External APIs (POS + OE)
    ↓
API Client (Fetch with pagination)
    ↓
Data Transformer (Transform to DB format)
    ↓
Data Validator (Validate required fields)
    ↓
Ingestion Service (Upsert to database)
    ↓
PostgreSQL Database
    ↓
Sync Log (Track operation)
```

---

## 2. Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCES                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   POS API        │         │    OE API        │         │
│  │  (Point of Sale) │         │  (Order Entry)   │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                            │                     │
└───────────┼────────────────────────────┼─────────────────────┘
            │                            │
            │                            │
┌───────────▼────────────────────────────▼─────────────────────┐
│                    INGESTION LAYER                            │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              API Client Layer                          │  │
│  │  • POSAPIClient - Handles POS API requests            │  │
│  │  • OEAPIClient - Handles OE API requests              │  │
│  │  • Automatic pagination                               │  │
│  │  • Rate limiting                                      │  │
│  │  • Error handling & retries                           │  │
│  └────────────────────┬───────────────────────────────────┘  │
│                       │                                       │
│  ┌────────────────────▼───────────────────────────────────┐  │
│  │           Data Transformation Layer                    │  │
│  │  • Transform API response to DB format                │  │
│  │  • Parse JSON fields (has_items, assigned_tags)       │  │
│  │  • Convert data types                                 │  │
│  │  • Handle missing fields                              │  │
│  └────────────────────┬───────────────────────────────────┘  │
│                       │                                       │
│  ┌────────────────────▼───────────────────────────────────┐  │
│  │            Data Validation Layer                       │  │
│  │  • Validate required fields                           │  │
│  │  • Check data integrity                               │  │
│  │  • Skip invalid records                               │  │
│  └────────────────────┬───────────────────────────────────┘  │
│                       │                                       │
│  ┌────────────────────▼───────────────────────────────────┐  │
│  │           Ingestion Service Layer                      │  │
│  │  • Orchestrates entire process                        │  │
│  │  • Upsert logic (insert or update)                    │  │
│  │  • Transaction management                             │  │
│  │  • Statistics tracking                                │  │
│  │  • Sync logging                                       │  │
│  └────────────────────┬───────────────────────────────────┘  │
│                       │                                       │
└───────────────────────┼───────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                    DATABASE LAYER                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ pos_orders   │  │  oe_orders   │  │  sync_logs   │        │
│  │   table      │  │    table     │  │    table     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│                PostgreSQL Database                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
ml-recommendation-system/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── api_client.py          # API communication
│   │   ├── data_transformer.py    # Data transformation
│   │   └── ingestion_service.py   # Main orchestration
│   │
│   └── database/
│       ├── models.py               # SQLAlchemy models
│       ├── connection.py           # Database connection
│       └── repository.py           # Data access layer
│
├── migrations/
│   └── versions/
│       └── 001_create_orders_tables.py  # Database migration
│
├── run_ingestion.py                # CLI entry point
├── .env.example                    # Configuration template
└── docs/
    └── DATA_INGESTION_PIPELINE.md  # This document
```

---

## 3. Components

### 3.1 API Client (`api_client.py`)

**Purpose:** Handles communication with external APIs

**Classes:**
- `APIClient` - Base client with common functionality
- `POSAPIClient` - Specialized for POS API
- `OEAPIClient` - Specialized for OE API

**Key Methods:**
```python
fetch_orders(start_date, end_date, page, per_page)
    # Fetch single page of orders

fetch_all_orders(start_date, end_date, per_page, max_pages)
    # Fetch all orders with automatic pagination

test_connection()
    # Test API connectivity
```

**Features:**
- Automatic pagination
- Rate limiting (0.5s delay between requests)
- Timeout handling (30s default)
- Error handling and retries
- Progress logging

### 3.2 Data Transformer (`data_transformer.py`)

**Purpose:** Transforms API response data to database model format

**Classes:**
- `DataTransformer` - Handles all transformations

**Key Methods:**
```python
transform_pos_order(api_data)
    # Transform POS API response to POSOrder model

transform_oe_order(api_data)
    # Transform OE API response to OEOrder model

validate_order_data(data, order_type)
    # Validate transformed data
```

**Transformations:**
- Convert JSON strings to proper format
- Parse dates (multiple formats supported)
- Handle missing/null values
- Type conversions (string to float, etc.)
- Extract nested fields

### 3.3 Ingestion Service (`ingestion_service.py`)

**Purpose:** Orchestrates the entire ingestion process

**Classes:**
- `IngestionService` - Main service class

**Key Methods:**
```python
ingest_pos_orders(start_date, end_date, max_pages)
    # Ingest POS orders

ingest_oe_orders(start_date, end_date, max_pages)
    # Ingest OE orders

ingest_all_orders(start_date, end_date, max_pages)
    # Ingest both POS and OE orders

test_connections()
    # Test all API connections
```

**Features:**
- Upsert logic (insert new, update existing)
- Transaction management
- Statistics tracking
- Sync logging
- Error handling

---

## 4. Database Schema

### 4.1 pos_orders Table

Stores Point of Sale orders

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(50) | Primary key, order ID |
| customer_phone | VARCHAR(50) | Customer phone (indexed) |
| customer_email | VARCHAR(255) | Customer email (indexed) |
| customer_name | VARCHAR(255) | Customer name |
| customer_address | TEXT | Customer address |
| customer_city | VARCHAR(100) | Customer city |
| customer_state | VARCHAR(100) | Customer state/province |
| customer_country | VARCHAR(100) | Customer country |
| order_date | DATETIME | Order date (indexed) |
| order_source | VARCHAR(50) | Order source |
| order_status | VARCHAR(50) | Order status |
| order_status_id | INTEGER | Order status ID |
| has_items | TEXT | JSON string of products |
| dealer_id | VARCHAR(50) | Dealer ID |
| dealer_name | VARCHAR(255) | Dealer name |
| dealership_id | VARCHAR(50) | Dealership ID |
| total_price | FLOAT | Total order price |
| discount | FLOAT | Discount amount |
| dealer_discount | FLOAT | Dealer discount |
| payment_mode | VARCHAR(50) | Payment method |
| brand_name | VARCHAR(100) | Brand name |
| courier_id | VARCHAR(50) | Courier ID |
| is_split | BOOLEAN | Split order flag |
| created_at | DATETIME | Record creation time |
| updated_at | DATETIME | Record update time |

**Indexes:**
- `idx_pos_customer_phone` on `customer_phone`
- `idx_pos_customer_email` on `customer_email`
- `idx_pos_order_date` on `order_date`
- `idx_pos_customer_date` on `(customer_phone, order_date)`

### 4.2 oe_orders Table

Stores Order Entry orders

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(50) | Primary key, order ID |
| customer_phone | VARCHAR(50) | Customer phone (indexed) |
| customer_email | VARCHAR(255) | Customer email (indexed) |
| customer_name | VARCHAR(255) | Customer name |
| customer_address | TEXT | Customer address |
| customer_city | VARCHAR(100) | Customer city |
| customer_state | VARCHAR(100) | Customer state/province |
| customer_country | VARCHAR(100) | Customer country |
| order_date | DATETIME | Order date (indexed) |
| order_name | VARCHAR(100) | Order name |
| order_status | VARCHAR(50) | Order status |
| order_status_id | INTEGER | Order status ID |
| order_comments | TEXT | Order comments |
| has_items | TEXT | JSON string of products |
| total_price | FLOAT | Total order price |
| discount | FLOAT | Discount amount |
| payment_mode | VARCHAR(50) | Payment method |
| brand_name | VARCHAR(100) | Brand name |
| courier_id | VARCHAR(50) | Courier ID |
| is_split | BOOLEAN | Split order flag |
| assigned_tags | TEXT | JSON string of tags |
| created_at | DATETIME | Record creation time |
| updated_at | DATETIME | Record update time |

**Indexes:**
- `idx_oe_customer_phone` on `customer_phone`
- `idx_oe_customer_email` on `customer_email`
- `idx_oe_order_date` on `order_date`
- `idx_oe_customer_date` on `(customer_phone, order_date)`

### 4.3 sync_logs Table

Tracks all sync operations

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key (auto-increment) |
| sync_type | VARCHAR(20) | 'pos' or 'oe' |
| sync_start_date | DATETIME | Sync start date filter |
| sync_end_date | DATETIME | Sync end date filter |
| records_fetched | INTEGER | Number of records fetched |
| records_inserted | INTEGER | Number of records inserted |
| records_updated | INTEGER | Number of records updated |
| records_failed | INTEGER | Number of failed records |
| status | VARCHAR(20) | 'success', 'failed', 'partial' |
| error_message | TEXT | Error message if failed |
| duration_seconds | FLOAT | Sync duration |
| started_at | DATETIME | Sync start time |
| completed_at | DATETIME | Sync completion time |
| created_at | DATETIME | Record creation time |

**Indexes:**
- `idx_sync_type` on `sync_type`
- `idx_sync_status` on `status`
- `idx_sync_started_at` on `started_at`

---

## 5. Setup & Configuration

### 5.1 Prerequisites

- Python 3.9+
- PostgreSQL 14+
- pip (Python package manager)
- Virtual environment (recommended)

### 5.2 Installation

```bash
# Navigate to project directory
cd ml-recommendation-system

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 5.3 Configuration

**1. Create .env file:**

```bash
cp .env.example .env
```

**2. Edit .env file:**

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/ml_recommendations

# API Configuration
POS_API_URL=https://api.masterverse.com/v1/pos-orders
OE_API_URL=https://api.masterverse.com/v1/oe-orders
API_AUTH_TOKEN=Bearer your_actual_token_here

# API Settings
API_PER_PAGE=100
API_TIMEOUT=30
```

**3. Initialize Database:**

```bash
# Run migrations
alembic upgrade head

# Or use CLI
python run_ingestion.py --init-db
```

**4. Test Connections:**

```bash
# Test database connection
python -c "from src.database.connection import test_connection; test_connection()"

# Test API connections
python run_ingestion.py --test-connection
```

---

## 6. Usage

### 6.1 Command Line Interface

**Basic Usage:**

```bash
# Ingest all orders (POS + OE)
python run_ingestion.py

# Ingest only POS orders
python run_ingestion.py --source pos

# Ingest only OE orders
python run_ingestion.py --source oe
```

**Date Filtering:**

```bash
# Fetch last 30 days
python run_ingestion.py --days 30

# Fetch specific date range
python run_ingestion.py --start-date 2024-01-01 --end-date 2024-12-31

# Fetch from specific start date to now
python run_ingestion.py --start-date 2024-06-01
```

**Testing & Debugging:**

```bash
# Test API connections only
python run_ingestion.py --test-connection

# Fetch only first 5 pages (for testing)
python run_ingestion.py --max-pages 5

# Initialize database tables
python run_ingestion.py --init-db
```

**Combined Examples:**

```bash
# Fetch last 7 days of POS orders only
python run_ingestion.py --source pos --days 7

# Fetch specific date range for OE orders
python run_ingestion.py --source oe --start-date 2024-11-01 --end-date 2024-11-30

# Test with limited pages
python run_ingestion.py --days 30 --max-pages 2
```

### 6.2 Programmatic Usage

```python
from src.ingestion.ingestion_service import IngestionService
from datetime import datetime, timedelta

# Initialize service
service = IngestionService(
    pos_api_url="https://api.masterverse.com/v1/pos-orders",
    oe_api_url="https://api.masterverse.com/v1/oe-orders",
    auth_token="Bearer your_token"
)

# Test connections
service.test_connections()

# Ingest last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

stats = service.ingest_all_orders(
    start_date=start_date,
    end_date=end_date
)

print(f"Fetched: {stats['total_fetched']}")
print(f"Inserted: {stats['total_inserted']}")
print(f"Updated: {stats['total_updated']}")

# Close connections
service.close()
```

---

## 7. API Integration

### 7.1 Expected API Response Format

**POS API Response:**

```json
{
  "data": [
    {
      "id": 69930,
      "order_name": "MF-40863",
      "order_date": "2024-01-31",
      "customer_name": "John Doe",
      "customer_city": "Lahore",
      "customer_state": "Punjab",
      "customer_country": "Pakistan",
      "customer_phone": "03001234567",
      "customer_email": "john@example.com",
      "customer_address": "123 Main St",
      "total_price": "5000.00",
      "payment_mode": "COD",
      "discount": 0,
      "order_status": "Delivered",
      "order_status_id": 9,
      "brand_name": "Master MoltyFoam",
      "is_split": 0,
      "has_items": [
        {
          "id": 70538,
          "order_id": 69930,
          "sku": "Coccyx-cushion",
          "title": "Coccyx Cushion",
          "quantity": 1,
          "base_price": 4700,
          "price": "4700.00",
          "product_type": "Accessories"
        }
      ]
    }
  ],
  "total": 1000,
  "page": 1,
  "per_page": 100
}
```

**OE API Response:**

Same format as POS API

### 7.2 API Requirements

**Authentication:**
- Bearer token in Authorization header
- Token should be valid and not expired

**Pagination:**
- `page` parameter (1-indexed)
- `per_page` parameter (default: 100, max: 500)

**Date Filtering:**
- `start_date` parameter (YYYY-MM-DD format)
- `end_date` parameter (YYYY-MM-DD format)

**Response:**
- Must include `data` array
- Should include `total`, `page`, `per_page` for pagination

---

## 8. Error Handling

### 8.1 Error Types

**API Errors:**
- Connection timeout (30s)
- HTTP errors (4xx, 5xx)
- Invalid response format
- Authentication failures

**Data Errors:**
- Missing required fields
- Invalid data types
- Malformed JSON

**Database Errors:**
- Connection failures
- Constraint violations
- Transaction failures

### 8.2 Error Handling Strategy

**Retry Logic:**
- API timeouts: No automatic retry (log and continue)
- HTTP 5xx errors: No automatic retry (log and continue)
- HTTP 4xx errors: No retry (likely auth or bad request)

**Graceful Degradation:**
- Skip invalid records (log warning)
- Continue processing remaining records
- Report statistics at end

**Logging:**
- All errors logged with context
- Sync operations logged to database
- Failed records tracked in statistics

### 8.3 Monitoring Sync Operations

**Query sync logs:**

```sql
-- Recent syncs
SELECT * FROM sync_logs 
ORDER BY started_at DESC 
LIMIT 10;

-- Failed syncs
SELECT * FROM sync_logs 
WHERE status = 'failed' 
ORDER BY started_at DESC;

-- Sync statistics
SELECT 
    sync_type,
    COUNT(*) as total_syncs,
    SUM(records_fetched) as total_fetched,
    SUM(records_inserted) as total_inserted,
    SUM(records_updated) as total_updated,
    SUM(records_failed) as total_failed,
    AVG(duration_seconds) as avg_duration
FROM sync_logs
WHERE started_at >= NOW() - INTERVAL '30 days'
GROUP BY sync_type;
```

---

## 9. Monitoring

### 9.1 Key Metrics

**Ingestion Metrics:**
- Records fetched per sync
- Records inserted vs updated
- Failed records count
- Sync duration
- Success rate

**Database Metrics:**
- Total orders in database
- Orders per day/week/month
- Data growth rate
- Table sizes

**API Metrics:**
- API response times
- API error rates
- Rate limit hits

### 9.2 Monitoring Queries

**Total orders:**

```sql
SELECT 
    'POS' as source, COUNT(*) as total 
FROM pos_orders
UNION ALL
SELECT 
    'OE' as source, COUNT(*) as total 
FROM oe_orders;
```

**Orders by date:**

```sql
SELECT 
    DATE(order_date) as date,
    COUNT(*) as orders
FROM pos_orders
WHERE order_date >= NOW() - INTERVAL '30 days'
GROUP BY DATE(order_date)
ORDER BY date DESC;
```

**Recent sync performance:**

```sql
SELECT 
    sync_type,
    started_at,
    records_fetched,
    records_inserted,
    records_updated,
    records_failed,
    duration_seconds,
    status
FROM sync_logs
ORDER BY started_at DESC
LIMIT 20;
```

---

## 10. Future Enhancements

### Phase 1 (Next Sprint)
- [ ] Add incremental sync (only fetch new orders)
- [ ] Implement retry logic for failed records
- [ ] Add email notifications for sync failures
- [ ] Create dashboard for monitoring

### Phase 2 (Future)
- [ ] Real-time streaming ingestion
- [ ] Webhook support for instant updates
- [ ] Data quality checks and alerts
- [ ] Automated data cleanup/archival

### Phase 3 (Long-term)
- [ ] Multi-source support (additional APIs)
- [ ] Data versioning and history
- [ ] Advanced error recovery
- [ ] Performance optimization for large datasets

---

## Appendix A: Troubleshooting

### Issue: Database connection failed

**Solution:**
```bash
# Check DATABASE_URL in .env
# Verify PostgreSQL is running
pg_isready -h localhost -p 5432

# Test connection
psql -h localhost -U username -d ml_recommendations
```

### Issue: API authentication failed

**Solution:**
```bash
# Verify API_AUTH_TOKEN in .env
# Check token format: "Bearer YOUR_TOKEN"
# Test API manually:
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.masterverse.com/v1/pos-orders?page=1&per_page=1
```

### Issue: No orders fetched

**Solution:**
- Check date filters (start_date, end_date)
- Verify API has data for the date range
- Check API response format
- Review logs for errors

---

## Appendix B: Performance Tips

**1. Batch Size:**
- Default: 100 records per page
- Increase for faster ingestion: `--per-page 500`
- Decrease if API timeouts occur

**2. Date Filtering:**
- Always use date filters for incremental syncs
- Fetch only recent data (last 30 days)
- Full sync only when necessary

**3. Database Optimization:**
- Indexes already created for common queries
- Regular VACUUM ANALYZE recommended
- Monitor table sizes

**4. Parallel Processing:**
- Run POS and OE ingestion in parallel (future enhancement)
- Use separate processes for each source

---

**Document Version:** 1.0  
**Last Updated:** December 11, 2024  
**Maintained By:** ML Team  
**Next Review:** January 2025
