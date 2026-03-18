# MasterGroup Analytics — Complete Integration Guide

> **Project:** MasterGroup Recommendation System  
> **Date:** March 2026  
> **Repositories:** `dev-hub-stack/recommender-engine` (backend) · `dev-hub-stack/recommender-dashboard` (frontend)  
> **Environments:** EC2 (`http://3.209.80.206:8001`) · Netlify (dashboard)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Step 1 — Historical Data Cleaning](#2-step-1--historical-data-cleaning)
3. [Step 2 — Live Database Cleaning](#3-step-2--live-database-cleaning)
4. [Step 3 — Historical Data Ingestion](#4-step-3--historical-data-ingestion)
5. [Step 4 — Data & Coverage Report](#5-step-4--data--coverage-report)
6. [Step 5 — Backend API Additions](#6-step-5--backend-api-additions)
7. [Step 6 — Dashboard: Historical Store Channels Section](#7-step-6--dashboard-historical-store-channels-section)
8. [Step 7 — Dashboard: Custom RFM Campaign Builder](#8-step-7--dashboard-custom-rfm-campaign-builder)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Environment Variables](#10-environment-variables)
11. [Database Reference](#11-database-reference)
12. [API Reference](#12-api-reference)

---

## 1. Overview

The goal of this integration was to bring **237,597 legacy MasterVerse customers** (who previously existed only in an Excel file) into the live analytics system. The full pipeline covers:

| Phase | What was done |
|---|---|
| **Clean** | Normalise phone numbers, city names, emails in the Excel file |
| **Migrate** | Apply the same cleaning rules to the live PostgreSQL database |
| **Ingest** | Load the cleaned historical records into the `orders` table, tagged `source_type='HISTORICAL'` |
| **Analyse** | New API endpoints serving channel distribution, RFM, and campaign exports |
| **Visualise** | Two new dashboard sections: Historical Store Channels and Custom RFM Builder |

---

## 2. Step 1 — Historical Data Cleaning

**Script:** `scripts/clean_historical_data.py`

### What it does
Reads `CustomerDataMasterVerse.xlsx` and applies the following rules row by row:

| Field | Rule |
|---|---|
| **Phone** | Normalise to `+923XXXXXXXXX` format. Drop rows where number < 10 digits (e.g. Exhibition dummy data) |
| **City** | `.strip().title()` — fixes casing inconsistencies (e.g. `LAHORE` → `Lahore`) |
| **Email** | `.strip().lower()` — prevents duplicates caused by case differences |
| **Province** | Map city to province if province is missing |
| **Name** | Strip leading/trailing whitespace |

### Output
- `data/cleaned_historical_customers.csv` — ready for ingestion
- Console summary: total rows, dropped rows, final valid rows

### How to run
```bash
cd recommendation-engine-service
python scripts/clean_historical_data.py
```

---

## 3. Step 2 — Live Database Cleaning

**Script:** `scripts/clean_live_db.py`  
**Dry-run script:** `scripts/clean_live_db_dry_run.py`

### What it does
Applies the exact same cleaning rules to the existing rows in the live PostgreSQL database:

| Table | Columns updated |
|---|---|
| `orders` | `customer_phone`, `customer_city` |
| `customer_statistics` | `phone`, `city` |

The dry-run version prints what **would** change without committing anything. Always run the dry-run first.

### How to run
```bash
# Safe dry run first
python scripts/clean_live_db_dry_run.py

# Apply for real
python scripts/clean_live_db.py
```

---

## 4. Step 3 — Historical Data Ingestion

**Script:** `scripts/ingest_historical_data.py`

### What it does
Reads `data/cleaned_historical_customers.csv` and inserts each row into the `orders` table using the following strategy:

1. **Unified Customer ID** — derived from `{phone}_{first_name}` to enable customer deduplication
2. **Order ID** — generated as `HIST-{phone}-{idx}` for uniqueness
3. **`order_type`** — set to `'OE'` (required by the PostgreSQL CHECK constraint on the column)
4. **`source_type`** — set to `'HISTORICAL'` — this is the true distinguishing tag used by all queries
5. **`order_name`** — set to `'Historical import - {ChannelName}'` (e.g. `'Historical import - Exhibition'`)
6. **Batch inserts** — 5,000 rows per batch with `commit()` between batches. If one batch fails, only that batch is rolled back
7. **Skip duplicates** — uses `ON CONFLICT DO NOTHING` to handle re-runs safely

### Outcome
- **237,597** unique historical customer records successfully ingested
- **11 legacy channels** tagged: Exhibition, JobBox, Changan, Dealers, DuraFoam, MasterOffdays, MattHome, OE (historical), POS (historical), and others

### How to run
```bash
python scripts/ingest_historical_data.py
```

---

## 5. Step 4 — Data & Coverage Report

**File:** `docs/data_coverage_report.md`

### What it contains
- Source-wise record count breakdown (OE, POS, HISTORICAL)
- Gap analysis: how many customers exist in the historical data vs. what was accessible via live APIs before ingestion
- Channel breakdown table: each of the 11 historical channels with customer counts and top provinces

### Key finding
> Before ingestion, the live APIs served `~202,000` customers.  
> The MasterVerse Excel file contained `~310,000` records.  
> After cleaning and deduplication, **237,597 net-new customers** were added — a **117% increase** in addressable customers.

---

## 6. Step 5 — Backend API Additions

All changes in: `src/main.py` (branch: `dev`, repo: `recommender-engine`)

---

### 5.1 `get_order_source_filter()` — Modified

**Location:** `src/main.py` ~line 382

Added support for `order_source='historical'`. When this value is passed:
- Filters `WHERE source_type = 'HISTORICAL'`
- When `order_source='oe'` or `order_source='pos'`, historical records are **explicitly excluded** using `COALESCE(source_type, '') != 'HISTORICAL'`

This ensures historical data never pollutes live OE/POS metrics.

---

### 5.2 `GET /api/v1/analytics/historical/store-channels` — New

Returns per-channel customer distribution for all ingested historical data.

**Sample response:**
```json
{
  "channels": [
    {
      "channel": "Exhibition",
      "customers": 106134,
      "share_pct": 44.7,
      "provinces": [
        { "province": "Punjab", "count": 52000 },
        { "province": "Sindh",  "count": 29000 }
      ]
    }
  ],
  "total_customers": 237597
}
```

**How it works:**  
Extracts channel name from `order_name` by stripping the `'Historical import - '` prefix, groups by channel and province.

---

### 5.3 `GET /api/v1/analytics/customers/rfm-custom` — New

Live RFM segment preview using **user-supplied thresholds**.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `champion_r` | 30 | Max recency (days) for Champions |
| `champion_f` | 5 | Min frequency (orders) for Champions |
| `champion_m` | 50000 | Min monetary (PKR) for Champions |
| `loyal_r` | 60 | Max recency for Loyal |
| `loyal_f` | 3 | Min frequency for Loyal |
| `loyal_m` | 20000 | Min monetary for Loyal |
| `at_risk_r_min` | 90 | Start of At-Risk recency window |
| `at_risk_r_max` | 180 | End of At-Risk recency window |
| `at_risk_f` | 2 | Min frequency for At-Risk |
| `hibernating_r` | 180 | Recency > this = Hibernating |
| `lost_r` | 365 | Recency > this = Lost |
| `order_source` | all | `all` / `oe` / `pos` / `historical` |

**Sample response:**
```json
{
  "segments": [
    { "segment_name": "Champions",    "customer_count": 104,    "percentage": 0.0 },
    { "segment_name": "Lost",         "customer_count": 366981, "percentage": 83.5 }
  ],
  "total_customers": 439375,
  "order_source": "all"
}
```

---

### 5.4 `GET /api/v1/export/rfm-campaign-csv` — New

Exports a **campaign-ready CSV** for a single RFM segment.

**Key query parameters:**
- `segment` (required): `Champions` / `Loyal` / `New Customers` / `At Risk` / `Hibernating` / `Lost`
- Same threshold parameters as `rfm-custom` above
- `order_source`: same values

**CSV columns exported:**

| Column | Notes |
|---|---|
| Customer Name | |
| **Email** | For email campaigns |
| **Phone** | For SMS/WhatsApp campaigns |
| City | |
| Province | |
| Total Orders | |
| Total Spent (PKR) | |
| Last Purchase Date | `YYYY-MM-DD` |
| Days Since Purchase | |

**Filename format:** `rfm_campaign_{segment}_{source}_{YYYYMMDD}.csv`

---

### 5.5 `GET /api/v1/export/dashboard-csv` — Modified

Added two new parameters:

| Parameter | Description |
|---|---|
| `sections=historical_channels` | Include the Historical Channel Distribution section |
| `historical_channel=Exhibition` | Optional: filter the export to a single specific channel |

---

## 7. Step 6 — Dashboard: Historical Store Channels Section

**Component:** `src/screens/Wireframe/sections/HistoricalStoreChannelsSection/HistoricalStoreChannelsSection.tsx`  
**Location in dashboard:** Below the POS vs OE section on the main Dashboard view

### UI Features

1. **Section header** — Title, total customer count badge, global "Export Historical Channels" green button
2. **Channel legend bar** — Horizontal colour-coded legend showing all 11 channels and their approximate share
3. **Channel cards grid** — One card per channel, showing:
   - Channel icon and name
   - Share % badge
   - Customer count (large number)
   - Share progress bar
4. **Click to expand** — Clicking a card reveals:
   - Top 5 provinces for that channel with mini progress bars
   - **"Export [Channel Name]" button** — downloads a source-specific CSV for that single channel only

### How the source-specific export works
Each expand-view export button calls `DashboardExportButton` with:
```tsx
<DashboardExportButton
  sections={['historical_channels']}
  historicalChannel={ch.channel}   // e.g. "Exhibition"
/>
```
The backend then adds `AND REPLACE(order_name, 'Historical import - ', '') = 'Exhibition'` to the SQL.

---

## 8. Step 7 — Dashboard: Custom RFM Campaign Builder

**Component:** `src/screens/Wireframe/sections/CustomRFMSection/CustomRFMSection.tsx`  
**Location in dashboard:** RFM Segmentation page → scroll to bottom

### UI Features

#### Source Selector (top-right)
Dropdown to switch between `All Sources` / `OE` / `POS` / `Historical`

#### Threshold Sliders Panel
Interactive sliders for every RFM boundary, grouped by segment:

| Segment | Adjustable thresholds |
|---|---|
| **Champions** | Recency ≤ (days), Frequency ≥ (orders), Monetary ≥ (PKR) |
| **Loyal** | Same three thresholds (must not qualify as Champion) |
| **At Risk** | Recency window (min and max days), Frequency ≥ |
| **Hibernating** | Recency > (days) |
| **Lost** | Recency > (days) |

#### Live Segment Preview Cards
- Auto-refreshes 500ms after any slider moves (debounced)
- Each card shows: icon, segment name, customer count, share %, mini bar chart
- **Disabled Export** if segment count = 0

#### Export Campaign CSV Button (per segment card)
- Green button at the bottom of each segment card
- Calls `GET /api/v1/export/rfm-campaign-csv` with current threshold values and selected source
- Downloads a CSV with Name, Email, Phone, City, Province, Total Orders, Total Spent, Last Purchase, Days Since Purchase

---

## 9. Architecture Diagram

```
Excel File (CustomerDataMasterVerse.xlsx)
         │
         ▼
clean_historical_data.py
         │  (normalise phone/city/email, drop invalid rows)
         ▼
cleaned_historical_customers.csv
         │
         ▼
ingest_historical_data.py
         │  (batch insert, source_type='HISTORICAL', order_name='Historical import - {channel}')
         ▼
PostgreSQL orders table (EC2)
  ├── source_type = NULL  →  Live OE / POS orders (via Shopify / POS APIs)
  └── source_type = 'HISTORICAL'  →  237,597 legacy MasterVerse customers

         │
         ├──► GET /api/v1/analytics/historical/store-channels
         │         → Channel distribution + province breakdown
         │
         ├──► GET /api/v1/analytics/customers/rfm-custom
         │         → Live RFM segment counts (custom thresholds + source filter)
         │
         └──► GET /api/v1/export/rfm-campaign-csv
                   → Campaign CSV (Name / Email / Phone / Spend / Recency)

React Dashboard (Netlify)
  ├── Dashboard view
  │     └── HistoricalStoreChannelsSection
  │           (channel cards + province drill-down + per-channel export)
  │
  └── RFM Segmentation view
        ├── RFMSegmentationSection  (ML-powered, read-only)
        └── CustomRFMSection        (slider builder + per-segment campaign exports)
```

---

## 10. Environment Variables

### Backend (`recommendation-engine-service/.env`)
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mastergroup_db
DB_USER=mastergroup
DB_PASSWORD=mastergroup123
REDIS_URL=redis://localhost:6379
```

### Frontend (`mastergroup-analytics-dashboard/.env`)
```
VITE_API_BASE_URL=http://3.209.80.206:8001/api/v1
```

---

## 11. Database Reference

### `orders` table — key columns used by this integration

| Column | Type | Notes |
|---|---|---|
| `id` | `VARCHAR` | Primary key. Historical rows use `HIST-{phone}-{idx}` |
| `unified_customer_id` | `VARCHAR` | `{phone}_{first_name}` for historical rows |
| `customer_name` | `VARCHAR` | |
| `customer_email` | `VARCHAR` | Available for historical rows from the Excel file |
| `customer_phone` | `VARCHAR` | Normalised to `+923XXXXXXXXX` |
| `customer_city` | `VARCHAR` | Title-cased |
| `province` | `VARCHAR` | Derived from city where missing |
| `order_type` | `VARCHAR` | `'OE'` (required by CHECK constraint, even for historical) |
| **`source_type`** | `VARCHAR` | `'HISTORICAL'` for imported rows, `NULL` for live orders |
| **`order_name`** | `VARCHAR` | `'Historical import - {ChannelName}'` for historical rows |
| `order_date` | `TIMESTAMP` | Date from the Excel file |
| `total_price` | `NUMERIC` | Amount in PKR |

### How to identify historical rows
```sql
-- All historical customers
SELECT * FROM orders WHERE source_type = 'HISTORICAL';

-- By channel
SELECT * FROM orders
WHERE source_type = 'HISTORICAL'
  AND order_name = 'Historical import - Exhibition';

-- Count per channel
SELECT
  REPLACE(order_name, 'Historical import - ', '') AS channel,
  COUNT(*) AS customers
FROM orders
WHERE source_type = 'HISTORICAL'
GROUP BY order_name
ORDER BY customers DESC;
```

---

## 12. API Reference

### Quick Reference Table

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/analytics/historical/store-channels` | Channel distribution |
| `GET` | `/api/v1/analytics/customers/rfm-custom` | Custom threshold RFM preview |
| `GET` | `/api/v1/export/rfm-campaign-csv` | Campaign CSV by segment |
| `GET` | `/api/v1/export/dashboard-csv` | Full dashboard CSV export |
| `GET` | `/api/v1/analytics/customers/segment-details/{segment}` | Customer list per segment |
| `GET` | `/api/v1/ml/rfm-segments` | ML-powered RFM segments |

### Example: Export "At Risk" customers from Historical source only
```bash
curl "http://3.209.80.206:8001/api/v1/export/rfm-campaign-csv?\
segment=At Risk&\
order_source=historical&\
at_risk_r_min=90&\
at_risk_r_max=730&\
at_risk_f=1" \
-o campaign_at_risk_historical.csv
```

### Example: Preview segments with relaxed thresholds
```bash
curl "http://3.209.80.206:8001/api/v1/analytics/customers/rfm-custom?\
order_source=all&\
champion_r=90&champion_f=2&champion_m=10000&\
loyal_r=365&loyal_f=1&loyal_m=5000"
```

---

*Document generated: 2026-03-18 | Author: Antigravity AI Engineering Assistant*
