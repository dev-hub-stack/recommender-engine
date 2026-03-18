# Historical Data Integration: OE & POS Customer Deduplication Report

## Executive Summary
This report addresses a key observation in the new Analytics Dashboard: **Why does the "Historical" dashboard view only show 94 "OE" and 6 "POS" customers, when the original historical Excel file contained hundreds of thousands of records?**

The short answer is: **We prevented duplicate data.** 

The vast majority of the OE and POS customers in the historical Excel file already exist as active shoppers in our live database. To avoid artificially inflating customer counts and double-counting revenue, the system intentionally merged them behind the scenes rather than creating duplicate "historical" profiles for them.

---

## The Data Merge Process Explained (In Simple Terms)

When we ingested the 310,000+ row historical Excel file (`CustomerDataMasterVerse_Cleaned.csv`), the system performed a smart match against our live database using the customer's phone number as the unique identifier.

Here is exactly what happened during the ingestion phase:

### 1. The Overlapping Customers (73,000+ Records)
Over **73,000 customers** in the historical Excel file were identified as people who **already actively shop** on our live OE (Online Ecommerce) or POS (Point of Sale) systems. 

- **What we did:** We **skipped** inserting a brand-new "Historical" row for them. Instead, we took their historical information (like a missing City name or a better spelling of their Name) and quietly **enriched** their live profile.
- **Why we did it:** If we inserted them again as "Historical" customers, they would be counted twice in the dashboard. Their total revenue and total order counts would be duplicated, heavily skewing our analytics and RFM segmentation.

### 2. The Exclusive Customers (237,597 Records)
The remaining **237,597 customers** were people who existed in the Excel file but had **never** made a purchase on our live OE or POS systems. 

- **What we did:** We inserted these as brand-new profiles into the database and tagged them strictly as `HISTORICAL`.
- **Why we did it:** Since these customers had zero digital/live presence, we needed to create their profiles from scratch so the Recommendation Engine could start analyzing them for future campaigns.

---

## Understanding the Dashboard View

When you look at the **"Historical Store Channels"** section on the dashboard, you are looking *exclusively* at those **237,597 new, offline-only customers**. 

Because we successfully stripped out the 73,000 active live shoppers, the remaining 237,597 customers are almost entirely from purely offline/legacy channels:
- **Exhibition:** 106,134 customers
- **JobBox:** 65,605 customers
- **Changan:** 31,853 customers
- **CFH:** 18,688 customers

### So why are there 94 OE and 6 POS customers there?
Out of those 237,597 exclusive offline customers, there were exactly **94** customers tagged as "OE" and **6** tagged as "POS" in the historical Excel file who *did not* match anyone in our live database. These are likely legacy records from an older system or anomalies where the customer provided a different phone number historically than they do today.

---

## The RFM Segmentation Anomaly: Why the "Lost" Segment holds 100% of Historical Customers

When analyzing the Custom RFM Campaign Builder with the source set to "Historical", executives will notice that **all 237,597 historical customers are categorized as "Lost."**

This is an expected outcome of the standard RFM (Recency, Frequency, Monetary) algorithm:

1. **Missing Purchase Dates:** The historical Excel file provided rich demographic data (Names, Phones, Cities, Source) but it **did not contain individual purchase dates**.
2. **The Recency Penalty:** RFM heavily weights "Recency" (days since the customer's last order). Because we did not know when the historical customers actually made their purchases, their profiles were safely seeded into the database with an extremely old default date.
3. **The Score Breakdown:** Since their simulated last purchase date guarantees a Recency of `> 365 Days Ago`, the RFM algorithm automatically assigns them a Recency score of `1` (the lowest possible score). A score this low instantly disqualifies a customer from segments like "Champions" or "Loyal Customers," pushing them directly into the "Lost" bucket.

**Next Steps / Solution:** 
To unlock RFM insights on purely historical data, we plan to implement a "Relative Recency" toggle. This will allow the dashboard to temporarily ignore Recency dates for historical customers and segment them purely based on Frequency (how many times they bought) and Monetary value (how much they spent).

---

## The Business Value of this Approach

By performing this deduplication merge, we achieved three critical goals for the Master Group executive team:

1. **Accurate Analytics:** We avoided "double-counting." The dashboard numbers represent unique human beings.
2. **Rich Single Customer View (SCV):** When an active POS customer logs into an OE system, or when we look at their RFM score, we are looking at one unified profile that now contains enriched historical location data.
3. **Massive Addressable Market:** We safely added 237,597 brand-new, previously unreachable customers to the Recommendation Engine, increasing our targeting pool by **118%** without polluting the live transactional data.
