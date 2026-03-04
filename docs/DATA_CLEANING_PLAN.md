# Data Cleaning Plan

> **Scope:** PostgreSQL database + CustomerDataMasterVerse.xlsx  
> **Created:** January 27, 2026

---

## 🎯 Objective

Normalize and standardize critical data fields to enable accurate:
- Customer matching across Shopify ↔ MasterGroup
- Location-based recommendations
- Analytics and reporting

---

## 1. Phone Number Normalization

### Current Issues
| Format | Example | Count |
|--------|---------|-------|
| 10 digits | `3001234567` | 311,053 |
| 11 digits | `03001234567` | 1,351 |
| With country code | `+923001234567` | Unknown |
| Invalid (<10) | `123456` | 122 |
| Invalid (>15) | Multiple formats | 15 |

### Target Format
```
+923001234567 (E.164 International Format)
```

### Normalization Rules
```python
def normalize_phone(phone):
    """
    Normalize Pakistani phone to +92XXXXXXXXXX format.
    """
    if not phone:
        return None
    
    # Remove all non-digits
    digits = re.sub(r'[^0-9]', '', str(phone))
    
    # Handle different formats
    if len(digits) == 10 and digits.startswith('3'):
        # 3001234567 → +923001234567
        return f'+92{digits}'
    elif len(digits) == 11 and digits.startswith('03'):
        # 03001234567 → +923001234567
        return f'+92{digits[1:]}'
    elif len(digits) == 12 and digits.startswith('92'):
        # 923001234567 → +923001234567
        return f'+{digits}'
    elif len(digits) == 13 and digits.startswith('923'):
        # Already includes +92 prefix
        return f'+{digits}'
    else:
        # Invalid format, return original for manual review
        return None
```

### SQL Update (orders table)
```sql
UPDATE orders
SET customer_phone = 
    CASE
        WHEN LENGTH(REGEXP_REPLACE(customer_phone, '[^0-9]', '', 'g')) = 10 
             AND customer_phone ~ '^3' 
        THEN '+92' || REGEXP_REPLACE(customer_phone, '[^0-9]', '', 'g')
        
        WHEN LENGTH(REGEXP_REPLACE(customer_phone, '[^0-9]', '', 'g')) = 11 
             AND customer_phone ~ '^03' 
        THEN '+92' || SUBSTRING(REGEXP_REPLACE(customer_phone, '[^0-9]', '', 'g') FROM 2)
        
        ELSE customer_phone
    END
WHERE customer_phone IS NOT NULL;
```

---

## 2. City Name Normalization

### Current Issues
| Variation | Standard |
|-----------|----------|
| `LAHORE` | Lahore |
| `lahore` | Lahore |
| `Gujrawala` (typo) | Gujranwala |
| `Rahimyarkhan` | Rahim Yar Khan |
| `BehriaIslamabad` | Bahria Town, Islamabad |

### Target Format
```
Title Case: "Lahore", "Karachi", "Islamabad"
```

### City Mapping Dictionary
```python
CITY_CORRECTIONS = {
    'gujrawala': 'Gujranwala',
    'rahimyarkhan': 'Rahim Yar Khan',
    'rahimyar khan': 'Rahim Yar Khan',
    'behriaislambad': 'Bahria Town Islamabad',
    'd.g khan': 'Dera Ghazi Khan',
    'dgkhan': 'Dera Ghazi Khan',
    'd.i khan': 'Dera Ismail Khan',
    'dikhan': 'Dera Ismail Khan',
    'nankana sahib': 'Nankana Sahib',
    'r.y.khan': 'Rahim Yar Khan',
    # Add more as discovered
}

def normalize_city(city):
    if not city:
        return None
    
    city_lower = city.strip().lower()
    
    # Check for known corrections
    if city_lower in CITY_CORRECTIONS:
        return CITY_CORRECTIONS[city_lower]
    
    # Default: Title Case
    return city.strip().title()
```

### SQL Update
```sql
UPDATE orders
SET customer_city = INITCAP(TRIM(customer_city))
WHERE customer_city IS NOT NULL;
```

---

## 3. Province Name Normalization

### Current Issues
| Variation | Standard | Count |
|-----------|----------|-------|
| `Punjab` | Punjab | 197,067 |
| `Sindh` | Sindh | 53,640 |
| `Islamabad` | Islamabad | 18,534 |
| `Khyber Pakhtunkhwa` | KPK | 8,570 |
| `NULL` | *Missing* | 3,964 |
| `Islamabad Capital Territory` | Islamabad | 673 |
| `K.P.K` | KPK | 6 |

### Province Mapping
```python
PROVINCE_MAPPING = {
    'punjab': 'Punjab',
    'sindh': 'Sindh',
    'islamabad': 'Islamabad',
    'islamabad capital territory': 'Islamabad',
    'ict': 'Islamabad',
    'khyber pakhtunkhwa': 'KPK',
    'kpk': 'KPK',
    'k.p.k': 'KPK',
    'khyber-pakhtunkhwa': 'KPK',
    'balochistan': 'Balochistan',
    'azad kashmir': 'AJK',
    'ajk': 'AJK',
    'gilgit-baltistan': 'Gilgit-Baltistan',
    'gb': 'Gilgit-Baltistan',
}
```

### Infer Province from City
```python
CITY_TO_PROVINCE = {
    'lahore': 'Punjab',
    'karachi': 'Sindh',
    'islamabad': 'Islamabad',
    'rawalpindi': 'Punjab',
    'faisalabad': 'Punjab',
    'multan': 'Punjab',
    'peshawar': 'KPK',
    'quetta': 'Balochistan',
    'hyderabad': 'Sindh',
    'gujranwala': 'Punjab',
    'muzaffarabad': 'AJK',
    'gilgit': 'Gilgit-Baltistan',
    # Add more cities...
}

def infer_province(city, province):
    """Infer province from city if missing."""
    if province:
        return normalize_province(province)
    
    city_lower = (city or '').strip().lower()
    return CITY_TO_PROVINCE.get(city_lower, None)
```

### SQL Update
```sql
-- First normalize existing provinces
UPDATE orders
SET province = 
    CASE LOWER(TRIM(province))
        WHEN 'islamabad capital territory' THEN 'Islamabad'
        WHEN 'khyber pakhtunkhwa' THEN 'KPK'
        WHEN 'k.p.k' THEN 'KPK'
        WHEN 'azad kashmir' THEN 'AJK'
        ELSE INITCAP(TRIM(province))
    END
WHERE province IS NOT NULL;

-- Then infer missing provinces from city
UPDATE orders
SET province = 'Punjab'
WHERE province IS NULL 
AND LOWER(customer_city) IN ('lahore', 'faisalabad', 'multan', 'gujranwala', 'rawalpindi');

UPDATE orders
SET province = 'Sindh'
WHERE province IS NULL 
AND LOWER(customer_city) IN ('karachi', 'hyderabad', 'sukkur');

UPDATE orders
SET province = 'Islamabad'
WHERE province IS NULL 
AND LOWER(customer_city) = 'islamabad';
```

---

## 📋 Implementation Steps

### Step 1: Create Backup
```sql
CREATE TABLE orders_backup AS SELECT * FROM orders;
CREATE TABLE order_items_backup AS SELECT * FROM order_items;
```

### Step 2: Run Cleaning Script
```bash
cd /opt/mastergroup-ml
source venv/bin/activate
python scripts/clean_data.py
```

### Step 3: Verify Changes
```sql
-- Check phone normalization
SELECT customer_phone, COUNT(*) 
FROM orders 
WHERE customer_phone NOT LIKE '+92%'
GROUP BY customer_phone 
LIMIT 10;

-- Check city normalization
SELECT customer_city, COUNT(*) 
FROM orders 
GROUP BY customer_city 
ORDER BY COUNT(*) DESC 
LIMIT 20;

-- Check province distribution
SELECT province, COUNT(*) 
FROM orders 
GROUP BY province 
ORDER BY COUNT(*) DESC;
```

---

## 🔒 Safety Measures

1. **Always create backup before cleaning**
2. **Run on staging first** (if available)
3. **Log all changes** for audit trail
4. **Reversible operations** - keep original values in separate column if needed
