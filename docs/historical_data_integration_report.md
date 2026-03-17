# MasterGroup Recommendation System: Historical Data Integration Report

**Date:** March 17, 2026  
**Status:** Completed & Deployed to Production

---

## Executive Summary
This report details the end-to-end integration of historical MasterVerse customer data into the live Master Group Analytics system. The objective was to ingest legacy data, merge it with real-time API data, and surface actionable insights regarding offline store channels (Exhibition, JobBox, Changan, etc.) directly on the analytics dashboard.

---

## 1. Data Analysis and Cleaning
The project began by analyzing the raw `CustomerDataMasterVerse.xlsx` file, which contained over 300,000 legacy records.

- **Data Gap Analysis**: We identified that while the live APIs (`OE` and `POS`) provided deep transactional data for around ~1,500 recent customers, the legacy MasterVerse data contained over **310,000 unique customers**. This represented a massive data gap that the recommendation engine was missing out on. 
- **Scripted Cleaning Pipeline**: The raw Excel file was converted to CSV and passed through an automated Python cleaning script (`scripts/clean_historical_data.py`).
- **Standardization**:
  - `Phone Numbers` were normalized to E.164 (`+92...`) format.
  - `Names` were properly capitalized.
  - `Dates` were parsed and standardized (falling back to a default `1900-01-01` baseline for missing timestamps).
  - Dummy columns were added to satisfy database constraints (`order_type`, `status`, `total_price`).

---

## 2. Database Ingestion
The cleaned historical data was then formally ingested into the live `dev-hub-stack` PostgreSQL database hosted on EC2.

- **Ingestion Script**: A bespoke script (`scripts/ingest_historical_data.py`) was written to handle the massive upsert.
- **Data Tagging**: All imported records were explicitly tagged with `source_type='HISTORICAL'` and `order_name='Historical import - {Channel}'`. This ensured historical data could easily be segmented away from live transactional data.
- **Handling Constraints**: The script safely maneuvered around existing database checks, populating `orders` and recalculating metrics in the `customer_statistics` table.
- **Result**: **237,597** unique, high-quality legacy customer records were successfully inserted.

---

## 3. Backend API Development
With the data available, the ML/Analytics backend service (`recommendation-engine-service`) was expanded.

- **New Endpoint**: Created `GET /api/v1/analytics/historical/store-channels` which aggregates historical records by store channel using raw SQL.
- **Metrics Calculated**:
  - Total historical customer count.
  - Customer counts per store channel.
  - Percentage share for each channel.
  - Top 5 provinces breakdown per channel.
- **Global Data Filtering**: The existing `get_order_source_filter()` logic was updated so that live `OE` and `POS` metrics strictly exclude the new historical records, preventing legacy data from skewing live revenue metrics.
- **CSV Export Engine**: Integrated the `historical_channels` section into the master `/export/dashboard-csv` backend endpoint to allow downloading the legacy data in spreadsheet format.

---

## 4. Frontend Dashboard Integration
The user interface (`mastergroup-analytics-dashboard`) was upgraded to visualize the newly exposed data.

- **Filter Enhancement**: A new "📦 Historical (Imported)" option was added to the global `orderSource` drop-down, allowing users to run the entire dashboard against historical data only.
- **New Dashboard Section**: Created the rich, dynamic `HistoricalStoreChannelsSection` React component, positioned directly below the live POS vs OE revenue section.
- **Features of the New UI**:
  - **Bar Legend**: A horizontal share bar showing the top channels.
  - **Expandable Cards**: A grid of 11 store channels (Exhibition, JobBox, Changan, CFH, DuraFoam, etc.) with custom icons and color schemes.
  - **Province Drill-down**: Clicking a card expands to show mini-progress bars for the top 5 geographical provinces for that specific channel.
  - **Contextual Export**: Added a native green "Export Historical Channels" CSV download button directly to the section header.

---

## 5. Deployment
- **Backend**: Pushed to the `dev` branch on GitHub. Resolved a deployment port conflict (`uvicorn` port 8001), allowing the systemd service to successfully restart the API on the Heroku/EC2 infrastructure. 
- **Frontend**: Pushed to the `main` branch. The frontend React application was successfully built via Vite with zero TypeScript compilation errors.

---
## Conclusion
The recommendation engine and analytics dashboard now possess a complete 360-degree view of Master Group's customer base, successfully bridging the gap between legacy offline retail channels (Exhibition/JobBox/Dealers) and modern online/POS systems.
