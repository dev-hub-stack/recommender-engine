# Historical Data RFM Limitations & Future Expansion Guide

## 1. Current State: Why Historical Customers Show as "Lost"
As of the initial historical data integration (Deliverable 4), the Master Group Analytics Dashboard correctly displays **237,597 exclusive historical customers**, primarily from offline sources like JobBox and Exhibitions. 

When analyzing these customers using the **Historical RFM Campaign Builder**, you will notice that 100% of these customers are currently bucketed into the **"Lost"** segment.

### The Reason
This behavior is mathematically correct and working by design.
- The original provided dataset (`CustomerDataMasterVerse_Cleaned.csv`) was a **demographic dump**, not a transaction dump. It contained Names, Phone Numbers, and Cities, but lacked individual purchase records, timestamps, or spend amounts.
- To safely integrate these users into the live recommendation engine, the ingestion pipeline seeded an empty **"Profile Order"** for each customer. 
- As a result, every historical customer currently has a verifiable **Frequency of 1** and a verifiable **Monetary Value of PKR 0**.
- Because the system utilizes "Relative Recency" (ignoring Recency dates) for historical sources, the algorithm categorizes any customer with exactly 1 order as a one-time buyer, pushing them into the **Lost** bucket.

---

## 2. Future Expansion: Graduating Customers to "Champions"
The infrastructure is already built to organically graduate these customers from "Lost" into "Champions", "Loyalists", or "At Risk". The system is simply waiting for actual transaction data to analyze.

### The Required Action
When Master Group is ready to unlock full RFM on these legacy users, they must provide an **Actual Transaction Dump** from their legacy systems (e.g., JobBox POS, Exhibition receipts, or legacy CFH databases).

The minimum required fields for this transaction dump are:
1. `Customer Phone Number`
2. `Order/Purchase Date`
3. `Total Spent (PKR)`

### The Execution Process
Once that file is provided, an engineer simply needs to feed those records into the `orders` table with `source_type = 'HISTORICAL'`. 

1. **No deduplication logic is necessary** for the transaction rows. 
2. Because the original demographic dump seeded the `unified_customer_id` profiles based on phone numbers, the new orders will seamlessly attach to the existing 237,597 historical profiles.
3. Every night (or every 2 hours), the `prewarm_cache.py` script scans the database.
4. As soon as the script sees that a historical customer now has 5 combined orders totaling PKR 80,000, it will instantly move them into the **"Champions"** bucket.

### Summary
We have not lost or broken any data. We have simply built a 237,000-person auditorium, and the RFM engine is holding their seats as "Lost/Inactive" until we import their legacy receipts to prove they are high-value shoppers.
