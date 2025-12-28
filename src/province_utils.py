"""
Province Utilities Module
=========================

Reusable module for province mapping and normalization.
Used by:
1. Data sync pipeline (main.py)
2. One-time cleanup script (clean_provinces.py)
3. API endpoints for validation
"""

# City to Province mapping for Pakistan
CITY_TO_PROVINCE = {
    # Punjab cities
    'lahore': 'Punjab', 'faisalabad': 'Punjab', 'rawalpindi': 'Punjab', 
    'multan': 'Punjab', 'gujranwala': 'Punjab', 'sialkot': 'Punjab',
    'bahawalpur': 'Punjab', 'sargodha': 'Punjab', 'shekhupura': 'Punjab',
    'jhang': 'Punjab', 'rahim yar khan': 'Punjab', 'gujrat': 'Punjab',
    'kasur': 'Punjab', 'sahiwal': 'Punjab', 'okara': 'Punjab',
    'wah': 'Punjab', 'wah cantt': 'Punjab', 'dera ghazi khan': 'Punjab',
    'chiniot': 'Punjab', 'kamoke': 'Punjab', 'mandi bahauddin': 'Punjab',
    'jhelum': 'Punjab', 'sadiqabad': 'Punjab', 'khanewal': 'Punjab',
    'hafizabad': 'Punjab', 'shorkot': 'Punjab', 'gujar khan': 'Punjab',
    'kharian': 'Punjab', 'dinga': 'Punjab', 'raiwind': 'Punjab',
    'chakwal': 'Punjab', 'gojra': 'Punjab', 'mailsi': 'Punjab',
    'chichawatni': 'Punjab', 'wazirabad': 'Punjab', 'bhakkar': 'Punjab',
    'khanpur': 'Punjab', 'pattoki': 'Punjab', 'theeng more': 'Punjab',
    'attock': 'Punjab', 'vehari': 'Punjab', 'kot addu': 'Punjab',
    'layyah': 'Punjab', 'muzaffargarh': 'Punjab', 'toba tek singh': 'Punjab',
    'jaranwala': 'Punjab', 'pakpattan': 'Punjab', 'lodhran': 'Punjab',
    'rajanpur': 'Punjab', 'khushab': 'Punjab', 'narowal': 'Punjab',
    'mianwali': 'Punjab', 'samundri': 'Punjab', 'talagang': 'Punjab',
    'nankana sahib': 'Punjab', 'sambrial': 'Punjab', 'hasilpur': 'Punjab',
    'chishtian': 'Punjab', 'jampur': 'Punjab', 'haroonabad': 'Punjab',
    'khuddian khas': 'Punjab', 'bhalwal': 'Punjab', 'dina': 'Punjab',
    'lalamusa': 'Punjab', 'lala musa': 'Punjab', 'arif wala': 'Punjab',
    'arifwala': 'Punjab', 'ahmed pur east': 'Punjab', 'ahmadpur east': 'Punjab',
    'phalia': 'Punjab', 'kamra': 'Punjab', 'minchinabad': 'Punjab',
    'jalalpur jattan': 'Punjab', 'mian channun': 'Punjab', 'mian channu': 'Punjab',
    'sheikhupura': 'Punjab', 'muridke': 'Punjab', 'rahimyar khan': 'Punjab',
    'rahimyarkhan': 'Punjab', 'fateh jang': 'Punjab', 'fatehjang': 'Punjab',
    'sangla hill': 'Punjab', 'jauharabad': 'Punjab', 'joharabad': 'Punjab',
    'alipur': 'Punjab', 'sargodah': 'Punjab', 'bhera': 'Punjab',
    'pind dadan khan': 'Punjab', 'kot radha kishan': 'Punjab', 'pir mahal': 'Punjab',
    'pasrur': 'Punjab', 'murree': 'Punjab', 'taxila': 'Punjab',
    'gujjar khan': 'Punjab', 'mandra': 'Punjab', 'sohawa': 'Punjab',
    'pindi gheb': 'Punjab', 'rawat': 'Punjab',
    
    # Sindh cities
    'karachi': 'Sindh', 'hyderabad': 'Sindh', 'sukkur': 'Sindh',
    'larkana': 'Sindh', 'nawabshah': 'Sindh', 'mirpur khas': 'Sindh',
    'jacobabad': 'Sindh', 'shikarpur': 'Sindh', 'khairpur': 'Sindh',
    'dadu': 'Sindh', 'thatta': 'Sindh', 'badin': 'Sindh',
    'tando allahyar': 'Sindh', 'matiari': 'Sindh', 'sanghar': 'Sindh',
    'umerkot': 'Sindh', 'tharparkar': 'Sindh', 'tando adam': 'Sindh',
    'ghotki': 'Sindh', 'kashmor': 'Sindh', 'kandhkot': 'Sindh',
    'shahdadpur': 'Sindh', 'ratodero': 'Sindh', 'daharki': 'Sindh',
    'moro': 'Sindh', 'mirpur mathelo': 'Sindh', 'sakrand': 'Sindh',
    'kunri': 'Sindh', 'ranipur': 'Sindh', 'matli': 'Sindh',
    'digri': 'Sindh', 'tando muhammad khan': 'Sindh', 'pano aqil': 'Sindh',
    'mehrabpur': 'Sindh', 'mithi': 'Sindh', 'rohri': 'Sindh',
    'kot diji': 'Sindh', 'gambat': 'Sindh',
    
    # Khyber Pakhtunkhwa cities
    'peshawar': 'Khyber Pakhtunkhwa', 'mardan': 'Khyber Pakhtunkhwa',
    'abbottabad': 'Khyber Pakhtunkhwa', 'mingora': 'Khyber Pakhtunkhwa',
    'kohat': 'Khyber Pakhtunkhwa', 'swabi': 'Khyber Pakhtunkhwa',
    'charsadda': 'Khyber Pakhtunkhwa', 'nowshera': 'Khyber Pakhtunkhwa',
    'mansehra': 'Khyber Pakhtunkhwa', 'haripur': 'Khyber Pakhtunkhwa',
    'bannu': 'Khyber Pakhtunkhwa', 'swat': 'Khyber Pakhtunkhwa',
    'batkhela': 'Khyber Pakhtunkhwa', 'timergara': 'Khyber Pakhtunkhwa',
    'karak': 'Khyber Pakhtunkhwa', 'hangu': 'Khyber Pakhtunkhwa',
    'lakki marwat': 'Khyber Pakhtunkhwa', 'abbotabad': 'Khyber Pakhtunkhwa',
    'abottabad': 'Khyber Pakhtunkhwa', 'dera ismail khan': 'Khyber Pakhtunkhwa',
    
    # Islamabad
    'islamabad': 'Islamabad',
    
    # Balochistan cities
    'quetta': 'Balochistan', 'turbat': 'Balochistan', 'gwadar': 'Balochistan',
    'khuzdar': 'Balochistan', 'chaman': 'Balochistan', 'hub': 'Balochistan',
    'sibi': 'Balochistan', 'zhob': 'Balochistan', 'loralai': 'Balochistan',
    'pishin': 'Balochistan',
    
    # Gilgit-Baltistan cities
    'gilgit': 'Gilgit-Baltistan', 'skardu': 'Gilgit-Baltistan',
    'chilas': 'Gilgit-Baltistan', 'hunza': 'Gilgit-Baltistan',
    
    # Azad Kashmir cities
    'muzaffarabad': 'Azad Kashmir', 'mirpur': 'Azad Kashmir',
    'rawalakot': 'Azad Kashmir', 'kotli': 'Azad Kashmir',
    'bhimber': 'Azad Kashmir', 'bagh': 'Azad Kashmir',
    'mangla': 'Azad Kashmir',
}

# Province normalization mapping
PROVINCE_VARIANTS = {
    'ISLAMABAD': 'Islamabad',
    'ISLAMABAD CAPITAL TERRITORY': 'Islamabad',
    'ISLAMABAD CAPITAL': 'Islamabad',
    'ICT': 'Islamabad',
    'KPK': 'Khyber Pakhtunkhwa',
    'K.P.K': 'Khyber Pakhtunkhwa',
    'NWFP': 'Khyber Pakhtunkhwa',
    'KHYBER PAKHTUNKHWA': 'Khyber Pakhtunkhwa',
    'PUNJAB': 'Punjab',
    'SINDH': 'Sindh',
    'BALOCHISTAN': 'Balochistan',
    'BALUCHISTAN': 'Balochistan',
    'GILGIT-BALTISTAN': 'Gilgit-Baltistan',
    'GB': 'Gilgit-Baltistan',
    'AZAD KASHMIR': 'Azad Kashmir',
    'AJK': 'Azad Kashmir',
    'AZAD JAMMU AND KASHMIR': 'Azad Kashmir',
}


def infer_province_from_city(city: str) -> str | None:
    """
    Infer province from city name.
    
    Args:
        city: City name (can be mixed case with whitespace)
        
    Returns:
        Province name (properly cased) or None if not found
    """
    if not city:
        return None
    
    city_normalized = city.lower().strip()
    return CITY_TO_PROVINCE.get(city_normalized)


def normalize_province(province: str) -> str | None:
    """
    Normalize province name (handle variants, case issues).
    
    Args:
        province: Raw province value from database
        
    Returns:
        Normalized province name or None if invalid
    """
    if not province or not province.strip():
        return None
    
    province_upper = province.upper().strip()
    
    # Remove dots (handles K.P.K → KPK)
    province_no_dots = province_upper.replace('.', '')
    
    # Check variants
    if province_no_dots in PROVINCE_VARIANTS:
        return PROVINCE_VARIANTS[province_no_dots]
    
    if province_upper in PROVINCE_VARIANTS:
        return PROVINCE_VARIANTS[province_upper]
    
    # Return title case if not in variants
    return province.strip().title()


def get_province_for_order(city: str, existing_province: str = None) -> str:
    """
    Get best province for an order, using city inference if province is missing.
    
    Args:
        city: City name from order
        existing_province: Existing province value (may be NULL/invalid)
        
    Returns:
        Province name (normalized) or 'Unspecified' if cannot determine
    """
    # First try to normalize existing province
    if existing_province:
        normalized = normalize_province(existing_province)
        if normalized and normalized.lower() not in ('unknown', 'n/a', 'na', 'null', 'none'):
            return normalized
    
    # Try to infer from city
    inferred = infer_province_from_city(city)
    if inferred:
        return inferred
    
    # Cannot determine
    return 'Unspecified'


# For use in SQL queries
def get_province_normalization_sql_case() -> str:
    """
    Generate SQL CASE statement for province normalization.
    
    Returns:
        SQL CASE statement string
    """
    return """
        CASE 
            WHEN UPPER(province) IN ('ISLAMABAD', 'ISLAMABAD CAPITAL TERRITORY', 'ISLAMABAD CAPITAL', 'ICT') THEN 'Islamabad'
            WHEN UPPER(REPLACE(province, '.', '')) IN ('KPK', 'NWFP', 'KHYBER PAKHTUNKHWA') THEN 'Khyber Pakhtunkhwa'
            WHEN UPPER(province) = 'PUNJAB' THEN 'Punjab'
            WHEN UPPER(province) = 'SINDH' THEN 'Sindh'
            WHEN UPPER(province) IN ('BALOCHISTAN', 'BALUCHISTAN') THEN 'Balochistan'
            WHEN UPPER(province) IN ('GILGIT-BALTISTAN', 'GB') THEN 'Gilgit-Baltistan'
            WHEN UPPER(province) IN ('AZAD KASHMIR', 'AJK', 'AZAD JAMMU AND KASHMIR') THEN 'Azad Kashmir'
            ELSE INITCAP(TRIM(province))
        END
    """
