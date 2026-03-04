# Customer Data Analysis & Integration Plan

> **Source:** CustomerDataMasterVerse.xlsx  
> **Records:** 312,658 customers  
> **Created:** January 27, 2026

---

## 📊 Data Summary

| Field | Coverage | Notes |
|-------|----------|-------|
| **Mobileno** | 100% | Primary identifier |
| **CityName** | 100% | Needs normalization |
| **DataSourceName** | 100% | 11 sources |
| **CompanyName** | 100% | 8 brands |
| **CustomerName** | 58% | 130K missing |
| **CustomerAddress** | 28% | 225K missing |
| **EmailAddress** | 4% | Only 12K have email |

---

## 🏢 Data Sources

| Source | Count | % |
|--------|-------|---|
| Exhibition | 109,180 | 35% |
| JobBox | 67,986 | 22% |
| OE | 47,825 | 15% |
| Changan | 33,065 | 11% |
| CFH | 21,828 | 7% |
| POS | 15,708 | 5% |
| Others | 17,066 | 5% |

---

## 🏙️ Top Cities

| City | Customers |
|------|-----------|
| Lahore | 83,405 |
| Karachi | 58,999 |
| Islamabad | 21,711 |
| Faisalabad | 20,911 |
| Gujranwala | 17,336 |
| Multan | 13,493 |

---

## 🔍 Deep Dive: Missing Data Analysis

### 1. Missing Customer Names (42% / 130k records)
- **Primary Sources:** Exhibition leads (75k) and POS (15k). 
- **Root Cause:** Quick data entry scenarios where staff only collected the Phone Number (primary ID).
- **Implication:** Personalization using names is not possible for nearly half the user base.

### 2. Missing Addresses (72% / 225k records)
- **Implication:** Hyper-local targeting (e.g., "Deliver to DHA") is limited.
- **Silver Lining:** **CityName is 100% complete**, so city-level targeting ("Trending in Lahore") works for everyone.

---

## 💡 Frontend Integration Recommendations

### 1. Handling Missing Names (Graceful Fallback)
Since we can't rely on `CustomerName` being present, the UI must adapt:

- **Scenario A (Name Exists):**
  > "Welcome back, **Ali**! Here are your recommended products."
- **Scenario B (Name Missing):**
  > "Welcome back! Here are **your** top picks." (Generic but personalized content)

### 2. Data Enrichment Strategy
- **"Complete Your Profile"**: When a user logs in with a phone number that has a missing name, prompt them: *"Tell us your name to get better service."*
- **Checkout Capture**: Shopify checkout naturally collects Name and Address. We can sync this back to MasterVerse to fill in the blanks.

### 3. Phone-Based Customer Recognition
- Match Shopify checkout phone with customer database
- Display: "Welcome back, [Name]!" (or fallback as above)
- **Feasibility:** ✅ High (100% phone coverage)

### 4. City-Based Recommendations
- Show "Bestsellers in [City]"
- Personalize by location (relies on CityName, not Address)
- **Feasibility:** ✅ High (100% city coverage)

### 5. Brand/Source Segmentation
- Changan customers → cross-sell MoltyFoam
- CFH customers → home fashion focus
- Exhibition leads → nurture campaigns
- **Feasibility:** ✅ High (100% source coverage)

### 4. Email Marketing
- Only 12,636 customers have email (4%)
- **Feasibility:** ⚠️ Low - need data collection

---

## ⚠️ Data Quality Issues

1. **City names not normalized** - "LAHORE" vs "Lahore" vs "lahore"
2. **Phone formats inconsistent** - "3001234567" vs "03001234567" vs "+923001234567"
3. **Duplicate phones across sources** - Same phone in OE + POS + Exhibition
4. **Missing names (42%)** - Cannot personalize for these

---

## 📋 Implementation Phases

### Phase 1: Data Cleaning
- [ ] Normalize phone numbers to standard format
- [ ] Normalize city names (title case)
- [ ] Deduplicate by phone number
- [ ] Create `customer_profiles` table

### Phase 2: API Integration
- [ ] Add `/customer/lookup` endpoint
- [ ] Add city filter to `/popular` endpoint
- [ ] Merge with existing order data

### Phase 3: Frontend Integration
- [ ] Update Shopify templates
- [ ] Add customer recognition
- [ ] Add city-based recommendations
