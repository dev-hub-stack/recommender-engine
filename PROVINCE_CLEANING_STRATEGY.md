# Province Data Quality Strategy
## Achieving 100% Coverage & Pipeline Integration

**Date:** December 28, 2024  
**Status:** 🎯 **Ready for Implementation**

---

## 📊 Current Situation

### Data Analysis Results
- **Total NULL provinces:** 25,587 orders (from 21,058 customers)
- **Current coverage:** 87.7% via city mapping (22,436 orders)
- **Customer history inference:** +8.1% (2,079 orders)
- **Total achievable:** 95.8% (24,515 orders)
- **Remaining:** 4.2% (1,072 orders) - truly unknown/invalid

### What Cannot Be Fixed
- **Fake/test data**: "Fake City" (101 orders), "Test" (23 orders)
- **Invalid cities**: "Out Of Pakistan" (22 orders), "Master Head Office" (17 orders)
- **No city data**: 8 orders with NULL city
- **Typos/variants**: Require manual mapping or fuzzy matching

---

## 🎯 Recommended Strategy

### ✅ Phase 1: One-Time Cleanup (THIS WEEK)
**Goal:** Fix 95.8% of existing NULL provinces

**Actions:**
```bash
cd /Users/clustox_1/Documents/MasterGroup-RecommendationSystem/recommendation-engine-service

# 1. Run full analysis
python3 clean_provinces.py --analyze

# 2. Apply cleanup (fixes 24,515 orders)
python3 clean_provinces.py --apply
# Type 'yes' when prompted

# 3. Verify results
python3 clean_provinces.py --analyze
```

**Expected Results:**
- ✅ 22,436 orders fixed via city mapping
- ✅ 2,079 orders fixed via customer history
- ⚠️ 1,072 orders remain NULL (test data, invalid cities)

---

### ✅ Phase 2: Pipeline Integration (THIS WEEK)

**Goal:** Prevent future NULL provinces at data ingestion point

#### Option A: Real-Time (RECOMMENDED)
Integrate into `sync_orders_from_master_group()` function:

**File:** `recommendation-engine-service/src/main.py` (around line 1250)

**Add this code BEFORE inserting orders:**

```python
from province_utils import get_province_for_order

# In the order processing loop:
for order in fetched_orders:
    # ... existing code ...
    
    # Province inference (ADD THIS)
    if not order.get('province') or order['province'].strip() == '':
        inferred_province = get_province_for_order(
            city=order.get('customer_city'),
            existing_province=order.get('province')
        )
        order['province'] = inferred_province
        logger.info(f"Inferred province '{inferred_province}' for order {order.get('id')} from city '{order.get('customer_city')}'")
    
    # ... rest of insert logic ...
```

**Benefits:**
- ✅ Prevents NULL provinces from entering database
- ✅ Real-time, no batch jobs needed
- ✅ Leverages existing city data
- ✅ Minimal performance impact

---

#### Option B: Database Trigger (ALTERNATIVE)

Create PostgreSQL trigger to auto-fill province:

```sql
CREATE OR REPLACE FUNCTION infer_province_from_city()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.province IS NULL OR TRIM(NEW.province) = '' THEN
        -- Try to infer from city
        NEW.province := CASE 
            WHEN LOWER(TRIM(NEW.customer_city)) IN ('lahore', 'faisalabad', 'rawalpindi') THEN 'Punjab'
            WHEN LOWER(TRIM(NEW.customer_city)) IN ('karachi', 'hyderabad', 'sukkur') THEN 'Sindh'
            -- ... add all cities ...
            ELSE 'Unspecified'
        END;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER before_insert_order_province
BEFORE INSERT ON orders
FOR EACH ROW
EXECUTE FUNCTION infer_province_from_city();
```

**Benefits:**
- ✅ Database-level enforcement
- ✅ Works regardless of ingestion method
- ⚠️ Harder to maintain city mappings

---

### ✅ Phase 3: Scheduled Cleanup (ONGOING)

**Goal:** Safety net for any NULLs that slip through

**File:** `recommendation-engine-service/src/scheduled_tasks.py` (create new)

```python
"""
Scheduled maintenance tasks for data quality
"""
import psycopg2
import os
from dotenv import load_dotenv
from province_utils import CITY_TO_PROVINCE

load_dotenv()

def clean_null_provinces():
    """Weekly cleanup job for NULL provinces"""
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        database=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD")
    )
    
    cursor = conn.cursor()
    total_fixed = 0
    
    # Fix via city mapping
    for city, province in CITY_TO_PROVINCE.items():
        cursor.execute("""
            UPDATE orders
            SET province = %s
            WHERE province IS NULL 
                AND LOWER(TRIM(customer_city)) = %s
        """, (province, city))
        total_fixed += cursor.rowcount
    
    # Fix via customer history
    cursor.execute("""
        UPDATE orders o1
        SET province = (
            SELECT o2.province
            FROM orders o2
            WHERE o2.unified_customer_id = o1.unified_customer_id
                AND o2.province IS NOT NULL
            GROUP BY o2.province
            ORDER BY COUNT(*) DESC
            LIMIT 1
        )
        WHERE o1.province IS NULL
            AND o1.unified_customer_id IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM orders o2
                WHERE o2.unified_customer_id = o1.unified_customer_id
                    AND o2.province IS NOT NULL
            )
    """)
    total_fixed += cursor.rowcount
    
    conn.commit()
    conn.close()
    
    print(f"✅ Cleaned {total_fixed} NULL provinces")
    return total_fixed

if __name__ == "__main__":
    clean_null_provinces()
```

**Add to cron (on EC2):**
```bash
# Run every Sunday at 2 AM
0 2 * * 0 cd /opt/mastergroup-ml && /usr/bin/python3 src/scheduled_tasks.py >> /var/log/province_cleanup.log 2>&1
```

---

### ✅ Phase 4: Monitoring & Alerts (OPTIONAL)

**Goal:** Track data quality over time

**Add to dashboard:**
1. **Province Coverage Metric**: `(non-NULL provinces / total orders) * 100`
2. **Weekly NULL Province Alert**: Email if >100 new NULLs detected
3. **Data Quality Page**: Show unmapped cities for manual review

**API Endpoint** (add to `main.py`):
```python
@app.get("/api/v1/data-quality/provinces")
async def get_province_data_quality():
    """Get province data quality metrics"""
    cursor = pg_pool.getconn().cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE province IS NOT NULL) as with_province,
            COUNT(*) FILTER (WHERE province IS NULL) as without_province,
            COUNT(*) as total,
            ROUND(COUNT(*) FILTER (WHERE province IS NOT NULL)::numeric / COUNT(*) * 100, 2) as coverage_pct
        FROM orders
    """)
    
    return cursor.fetchone()
```

---

## 📋 Implementation Checklist

### Week 1: Foundation
- [x] Create `province_utils.py` module
- [ ] Run one-time cleanup (`python3 clean_provinces.py --apply`)
- [ ] Verify cleanup results
- [ ] Commit changes to `dev` branch
- [ ] Deploy to EC2

### Week 1-2: Pipeline Integration
- [ ] Integrate `province_utils` into `sync_orders_from_master_group()`
- [ ] Test data sync with province inference
- [ ] Monitor logs for inference success rate
- [ ] Add unit tests for province utils

### Week 2-3: Automation
- [ ] Create `scheduled_tasks.py` for cleanup
- [ ] Test manual cleanup script
- [ ] Add cron job to EC2
- [ ] Set up log rotation

### Week 3-4: Monitoring (Optional)
- [ ] Add data quality API endpoint
- [ ] Add dashboard widget
- [ ] Set up email alerts
- [ ] Document unmapped cities for manual review

---

## 🎯 Success Metrics

### Immediate (After Phase 1)
- ✅ **Coverage:** 95.8% (24,515 / 25,587)
- ✅ **NULL orders:** <1,100 (down from 25,587)

### 1 Month (After Phase 2)
- ✅ **New NULL orders:** <50 per week
- ✅ **Coverage maintained:** >95%

### 3 Months (After Phase 3)
- ✅ **New NULL orders:** <10 per week
- ✅ **Coverage:** >98%
- ✅ **Automated cleanup:** Running reliably

---

## 💡 Why Not 100%?

**Remaining 4.2% is acceptable because:**
1. **Invalid test data**: "Fake City", "Test", "Master Head Office"
2. **Out of scope**: "Out Of Pakistan"
3. **No data**: Orders with NULL city cannot be inferred
4. **Cost-benefit**: Diminishing returns for fuzzy matching

**Recommendation:** Mark remaining as `"Unspecified"` and filter from analytics:
```sql
WHERE province != 'Unspecified' 
  OR province IS NOT NULL
```

---

## 🚀 Deployment Steps

### Step 1: Deploy Cleanup Script
```bash
# Local
cd /Users/clustox_1/Documents/MasterGroup-RecommendationSystem/recommendation-engine-service
git add clean_provinces.py src/province_utils.py
git commit -m "Add province data cleaning pipeline"
git push origin dev

# EC2
ssh -i mastergroup-ec2-key.pem ubuntu@3.209.80.206
cd /opt/mastergroup-ml
git pull origin dev
python3 clean_provinces.py --apply  # Type 'yes'
```

### Step 2: Deploy Pipeline Integration
```bash
# After integrating into main.py
git add src/main.py src/province_utils.py
git commit -m "Integrate province inference into sync pipeline"
git push origin dev

# EC2
cd /opt/mastergroup-ml
git pull origin dev
sudo systemctl restart mastergroup-api
```

### Step 3: Deploy Scheduled Cleanup
```bash
# Create scheduled_tasks.py
git add src/scheduled_tasks.py
git commit -m "Add scheduled province cleanup task"
git push origin dev

# EC2
cd /opt/mastergroup-ml
git pull origin dev
crontab -e
# Add: 0 2 * * 0 cd /opt/mastergroup-ml && /usr/bin/python3 src/scheduled_tasks.py >> /var/log/province_cleanup.log 2>&1
```

---

## 📞 Support

**Questions?**
- Check logs: `/var/log/mastergroup-api.log`
- Run analysis: `python3 clean_provinces.py --analyze`
- Manual cleanup: `python3 clean_provinces.py --apply`

**Issues?**
- Review unmapped cities in analysis output
- Check Master Group API data quality
- Verify city name spellings in database

---

## 🎉 Summary

| Metric | Before | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|--------|---------------|---------------|---------------|
| NULL provinces | 25,587 | ~1,072 | ~50/week | ~10/week |
| Coverage | 0% | 95.8% | >95% | >98% |
| Manual work | High | One-time | None | Monitoring only |

**Time Investment:**
- Phase 1: 30 minutes
- Phase 2: 2-4 hours
- Phase 3: 1-2 hours
- Phase 4: 4-8 hours (optional)

**ROI:** ✅ Clean geographic data, better analytics, no duplicate provinces!
