"""
Province Data Cleaning Script
==============================

This script analyzes and optionally cleans the province data in the orders table:
1. Shows statistics on NULL/Unknown provinces
2. Attempts to infer provinces from city names
3. Provides options to update the database

IMPORTANT: Run with --dry-run first to see what would be changed!
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import argparse

load_dotenv()

# City to Province mapping for Pakistan (comprehensive)
CITY_TO_PROVINCE = {
    # Punjab cities (major + additional)
    'lahore': 'Punjab',
    'faisalabad': 'Punjab',
    'rawalpindi': 'Punjab',
    'multan': 'Punjab',
    'gujranwala': 'Punjab',
    'sialkot': 'Punjab',
    'bahawalpur': 'Punjab',
    'sargodha': 'Punjab',
    'shekhupura': 'Punjab',
    'jhang': 'Punjab',
    'rahim yar khan': 'Punjab',
    'gujrat': 'Punjab',
    'kasur': 'Punjab',
    'sahiwal': 'Punjab',
    'okara': 'Punjab',
    'wah': 'Punjab',
    'wah cantt': 'Punjab',
    'dera ghazi khan': 'Punjab',
    'chiniot': 'Punjab',
    'kamoke': 'Punjab',
    'mandi bahauddin': 'Punjab',
    'jhelum': 'Punjab',
    'sadiqabad': 'Punjab',
    'khanewal': 'Punjab',
    'hafizabad': 'Punjab',
    'shorkot': 'Punjab',
    'gujar khan': 'Punjab',
    'kharian': 'Punjab',
    'dinga': 'Punjab',
    'raiwind': 'Punjab',
    'chakwal': 'Punjab',
    'gojra': 'Punjab',
    'mailsi': 'Punjab',
    'chichawatni': 'Punjab',
    'wazirabad': 'Punjab',
    'bhakkar': 'Punjab',
    'khanpur': 'Punjab',
    'pattoki': 'Punjab',
    'theeng more': 'Punjab',
    'attock': 'Punjab',
    'vehari': 'Punjab',
    'kot addu': 'Punjab',
    'layyah': 'Punjab',
    'muzaffargarh': 'Punjab',
    'dera ismail khan': 'Punjab',
    'toba tek singh': 'Punjab',
    'jaranwala': 'Punjab',
    'pakpattan': 'Punjab',
    'lodhran': 'Punjab',
    'rajanpur': 'Punjab',
    'khushab': 'Punjab',
    'narowal': 'Punjab',
    'mianwali': 'Punjab',
    'samundri': 'Punjab',
    'talagang': 'Punjab',
    'nankana sahib': 'Punjab',
    'sambrial': 'Punjab',
    'hasilpur': 'Punjab',
    'chishtian': 'Punjab',
    'jampur': 'Punjab',
    'haroonabad': 'Punjab',
    'khuddian khas': 'Punjab',
    'bhalwal': 'Punjab',
    'dina': 'Punjab',
    'lalamusa': 'Punjab',
    'lala musa': 'Punjab',
    'arif wala': 'Punjab',
    'arifwala': 'Punjab',
    'ahmed pur east': 'Punjab',
    'ahmadpur east': 'Punjab',
    'phalia': 'Punjab',
    'kamra': 'Punjab',
    'minchinabad': 'Punjab',
    'jalalpur jattan': 'Punjab',
    'mian channun': 'Punjab',
    'mian channu': 'Punjab',
    'sheikhupura': 'Punjab',
    'muridke': 'Punjab',
    'rahimyar khan': 'Punjab',
    'rahimyarkhan': 'Punjab',
    'fateh jang': 'Punjab',
    'fatehjang': 'Punjab',
    'sangla hill': 'Punjab',
    'jauharabad': 'Punjab',
    'joharabad': 'Punjab',
    'alipur': 'Punjab',
    'sargodah': 'Punjab',
    'bhera': 'Punjab',
    'pind dadan khan': 'Punjab',
    'kot radha kishan': 'Punjab',
    'pir mahal': 'Punjab',
    'pasrur': 'Punjab',
    'murree': 'Punjab',
    'taxila': 'Punjab',
    'gujjar khan': 'Punjab',
    'mandra': 'Punjab',
    'sohawa': 'Punjab',
    'pindi gheb': 'Punjab',
    'rawat': 'Punjab',
    
    # Sindh cities
    'karachi': 'Sindh',
    'hyderabad': 'Sindh',
    'sukkur': 'Sindh',
    'larkana': 'Sindh',
    'nawabshah': 'Sindh',
    'mirpur khas': 'Sindh',
    'jacobabad': 'Sindh',
    'shikarpur': 'Sindh',
    'khairpur': 'Sindh',
    'dadu': 'Sindh',
    'thatta': 'Sindh',
    'badin': 'Sindh',
    'tando allahyar': 'Sindh',
    'matiari': 'Sindh',
    'sanghar': 'Sindh',
    'umerkot': 'Sindh',
    'tharparkar': 'Sindh',
    'tando adam': 'Sindh',
    'ghotki': 'Sindh',
    'kashmor': 'Sindh',
    'kandhkot': 'Sindh',
    'shahdadpur': 'Sindh',
    'ratodero': 'Sindh',
    'daharki': 'Sindh',
    'moro': 'Sindh',
    'mirpur mathelo': 'Sindh',
    'sakrand': 'Sindh',
    'kunri': 'Sindh',
    'ranipur': 'Sindh',
    'matli': 'Sindh',
    'digri': 'Sindh',
    'tando muhammad khan': 'Sindh',
    'pano aqil': 'Sindh',
    'mehrabpur': 'Sindh',
    'mithi': 'Sindh',
    'rohri': 'Sindh',
    'kot diji': 'Sindh',
    'gambat': 'Sindh',
    'umerkot': 'Sindh',
    
    # Khyber Pakhtunkhwa cities
    'peshawar': 'Khyber Pakhtunkhwa',
    'mardan': 'Khyber Pakhtunkhwa',
    'abbottabad': 'Khyber Pakhtunkhwa',
    'mingora': 'Khyber Pakhtunkhwa',
    'kohat': 'Khyber Pakhtunkhwa',
    'swabi': 'Khyber Pakhtunkhwa',
    'charsadda': 'Khyber Pakhtunkhwa',
    'nowshera': 'Khyber Pakhtunkhwa',
    'mansehra': 'Khyber Pakhtunkhwa',
    'haripur': 'Khyber Pakhtunkhwa',
    'bannu': 'Khyber Pakhtunkhwa',
    'swat': 'Khyber Pakhtunkhwa',
    'batkhela': 'Khyber Pakhtunkhwa',
    'timergara': 'Khyber Pakhtunkhwa',
    'karak': 'Khyber Pakhtunkhwa',
    'hangu': 'Khyber Pakhtunkhwa',
    'lakki marwat': 'Khyber Pakhtunkhwa',
    'abbotabad': 'Khyber Pakhtunkhwa',  # Spelling variant
    'abottabad': 'Khyber Pakhtunkhwa',  # Another spelling variant
    'dera ismail khan': 'Khyber Pakhtunkhwa',  # Move from Punjab to KPK (correct)
    
    # Islamabad
    'islamabad': 'Islamabad',
    
    # Balochistan cities
    'quetta': 'Balochistan',
    'turbat': 'Balochistan',
    'gwadar': 'Balochistan',
    'khuzdar': 'Balochistan',
    'chaman': 'Balochistan',
    'hub': 'Balochistan',
    'sibi': 'Balochistan',
    'zhob': 'Balochistan',
    'loralai': 'Balochistan',
    'pishin': 'Balochistan',
    
    # Gilgit-Baltistan cities
    'gilgit': 'Gilgit-Baltistan',
    'skardu': 'Gilgit-Baltistan',
    'chilas': 'Gilgit-Baltistan',
    'hunza': 'Gilgit-Baltistan',
    
    # Azad Kashmir cities
    'muzaffarabad': 'Azad Kashmir',
    'mirpur': 'Azad Kashmir',
    'rawalakot': 'Azad Kashmir',
    'kotli': 'Azad Kashmir',
    'bhimber': 'Azad Kashmir',
    'bagh': 'Azad Kashmir',
    'mangla': 'Azad Kashmir',
}


def analyze_null_provinces(conn):
    """Analyze NULL province records"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS: NULL Province Records")
    print("="*80)
    
    # Total NULL provinces
    cursor.execute("""
        SELECT 
            COUNT(*) as order_count,
            COUNT(DISTINCT unified_customer_id) as customer_count
        FROM orders
        WHERE province IS NULL
    """)
    stats = cursor.fetchone()
    total_null_orders = stats['order_count']
    total_null_customers = stats['customer_count']
    print(f"\n📊 Total NULL provinces: {total_null_orders:,} orders from {total_null_customers:,} customers")
    
    # Get ALL cities with NULL provinces (not just top 30)
    cursor.execute("""
        SELECT 
            LOWER(TRIM(customer_city)) as city,
            COUNT(*) as count,
            COUNT(DISTINCT unified_customer_id) as customers
        FROM orders
        WHERE province IS NULL AND customer_city IS NOT NULL AND TRIM(customer_city) != ''
        GROUP BY LOWER(TRIM(customer_city))
        ORDER BY count DESC
    """)
    
    all_rows = cursor.fetchall()
    
    # Statistics
    inferable_orders = 0
    not_inferable_orders = 0
    inferable_cities = []
    not_inferable_cities = []
    
    for row in all_rows:
        city = row['city']
        if city in CITY_TO_PROVINCE:
            inferable_orders += row['count']
            inferable_cities.append(row)
        else:
            not_inferable_orders += row['count']
            not_inferable_cities.append(row)
    
    # Summary statistics
    print("\n" + "-"*80)
    print("📈 SUMMARY STATISTICS")
    print("-"*80)
    print(f"Total cities with NULL province: {len(all_rows)}")
    print(f"Cities we can map: {len(inferable_cities)} ({len(inferable_cities)/len(all_rows)*100:.1f}%)")
    print(f"Cities we CANNOT map: {len(not_inferable_cities)} ({len(not_inferable_cities)/len(all_rows)*100:.1f}%)")
    print(f"\nOrders we can fix: {inferable_orders:,} ({inferable_orders/total_null_orders*100:.1f}%)")
    print(f"Orders we CANNOT fix: {not_inferable_orders:,} ({not_inferable_orders/total_null_orders*100:.1f}%)")
    print(f"Orders with no city data: {total_null_orders - inferable_orders - not_inferable_orders:,}")
    
    # Show inferable cities (TOP 50)
    print("\n" + "-"*80)
    print("✅ TOP 50 CITIES WE CAN FIX (Highest Order Count)")
    print("-"*80)
    print(f"{'City':<30} {'Province':<25} {'Orders':>8} {'Customers':>10}")
    print("-"*80)
    
    for row in inferable_cities[:50]:
        city = row['city']
        province = CITY_TO_PROVINCE.get(city)
        print(f"{city.title():<30} {province:<25} {row['count']:>8,} {row['customers']:>10,}")
    
    if len(inferable_cities) > 50:
        print(f"\n... and {len(inferable_cities) - 50} more cities (showing top 50 only)")
    
    # Show unmappable cities (TOP 50)
    print("\n" + "-"*80)
    print("❌ TOP 50 CITIES WE CANNOT FIX (Need Manual Mapping)")
    print("-"*80)
    print(f"{'City':<40} {'Orders':>8} {'Customers':>10}")
    print("-"*80)
    
    for row in not_inferable_cities[:50]:
        city = row['city']
        print(f"{city.title():<40} {row['count']:>8,} {row['customers']:>10,}")
    
    if len(not_inferable_cities) > 50:
        print(f"\n... and {len(not_inferable_cities) - 50} more cities (showing top 50 only)")
    
    # Province distribution of what we CAN fix
    print("\n" + "-"*80)
    print("📍 PROVINCE DISTRIBUTION (Orders we can fix)")
    print("-"*80)
    
    province_stats = {}
    for row in inferable_cities:
        province = CITY_TO_PROVINCE[row['city']]
        if province not in province_stats:
            province_stats[province] = {'orders': 0, 'customers': 0, 'cities': 0}
        province_stats[province]['orders'] += row['count']
        province_stats[province]['customers'] += row['customers']
        province_stats[province]['cities'] += 1
    
    for province, stats in sorted(province_stats.items(), key=lambda x: x[1]['orders'], reverse=True):
        print(f"{province:<25} {stats['orders']:>8,} orders  {stats['customers']:>8,} customers  {stats['cities']:>3} cities")
    
    print("\n" + "="*80)
    print(f"✅ FIXABLE: {inferable_orders:,} orders ({inferable_orders/total_null_orders*100:.1f}%)")
    print(f"❌ NEEDS MANUAL MAPPING: {not_inferable_orders:,} orders ({not_inferable_orders/total_null_orders*100:.1f}%)")
    print(f"⚠️  NO CITY DATA: {total_null_orders - inferable_orders - not_inferable_orders:,} orders")
    print("="*80)
    
    cursor.close()
    return all_rows


def preview_updates(conn):
    """Preview what would be updated"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "="*60)
    print("PREVIEW: Proposed Updates")
    print("="*60)
    
    for city, province in CITY_TO_PROVINCE.items():
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM orders
            WHERE province IS NULL 
                AND LOWER(TRIM(customer_city)) = %s
        """, (city,))
        
        result = cursor.fetchone()
        if result['count'] > 0:
            print(f"UPDATE {result['count']:>6} orders: {city.title():<30} → {province}")
    
    cursor.close()


def infer_from_customer_history(conn, dry_run=True):
    """Infer province from customer's other orders"""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("STEP 2: Inferring provinces from customer history")
    print("="*60 + "\n")
    
    # Find customers with NULL province orders who have other orders with known provinces
    if not dry_run:
        cursor.execute("""
            UPDATE orders o1
            SET province = (
                SELECT o2.province
                FROM orders o2
                WHERE o2.unified_customer_id = o1.unified_customer_id
                    AND o2.province IS NOT NULL
                    AND TRIM(o2.province) != ''
                    AND UPPER(TRIM(o2.province)) NOT IN ('UNKNOWN', 'N/A', 'NA', 'NULL', 'NONE')
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
                        AND TRIM(o2.province) != ''
                )
        """)
        updated = cursor.rowcount
        conn.commit()
        print(f"✅ Inferred province for {updated:,} orders from customer history")
        return updated
    else:
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM orders o1
            WHERE o1.province IS NULL
                AND o1.unified_customer_id IS NOT NULL
                AND EXISTS (
                    SELECT 1 FROM orders o2
                    WHERE o2.unified_customer_id = o1.unified_customer_id
                        AND o2.province IS NOT NULL
                        AND TRIM(o2.province) != ''
                )
        """)
        result = cursor.fetchone()
        count = result[0] if result else 0
        print(f"Would infer province for {count:,} orders from customer history")
        return count


def apply_updates(conn, dry_run=True):
    """Apply province updates based on city mapping"""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("STEP 1: Applying city-based province mapping")
    if dry_run:
        print("(DRY RUN MODE)")
    print("="*60 + "\n")
    
    total_updated = 0
    
    for city, province in CITY_TO_PROVINCE.items():
        if not dry_run:
            cursor.execute("""
                UPDATE orders
                SET province = %s
                WHERE province IS NULL 
                    AND LOWER(TRIM(customer_city)) = %s
            """, (province, city))
            
            updated = cursor.rowcount
            total_updated += updated
            
            if updated > 0:
                print(f"✅ Updated {updated:>6} orders: {city.title()} → {province}")
        else:
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM orders
                WHERE province IS NULL 
                    AND LOWER(TRIM(customer_city)) = %s
            """, (city,))
            
            result = cursor.fetchone()
            count = result[0] if result else 0
            
            if count > 0:
                total_updated += count
                print(f"Would update {count:>6} orders: {city.title()} → {province}")
    
    if not dry_run:
        conn.commit()
        print(f"\n✅ STEP 1 COMPLETE: Updated {total_updated:,} orders via city mapping")
    else:
        print(f"\n📋 STEP 1 PREVIEW: Would update {total_updated:,} orders via city mapping")
    
    cursor.close()
    return total_updated


def main():
    parser = argparse.ArgumentParser(description='Clean province data in orders table')
    parser.add_argument('--analyze', action='store_true', help='Analyze NULL provinces')
    parser.add_argument('--preview', action='store_true', help='Preview proposed updates')
    parser.add_argument('--apply', action='store_true', help='Apply updates to database')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Dry run mode (default)')
    
    args = parser.parse_args()
    
    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        database=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD")
    )
    
    try:
        if args.analyze or not (args.preview or args.apply):
            analyze_null_provinces(conn)
        
        if args.preview:
            preview_updates(conn)
        
        if args.apply:
            response = input("\n⚠️  Are you sure you want to update the database? (yes/no): ")
            if response.lower() == 'yes':
                print("\n" + "="*80)
                print("PROVINCE CLEANING PIPELINE - FULL RUN")
                print("="*80)
                
                total_city = apply_updates(conn, dry_run=False)
                total_history = infer_from_customer_history(conn, dry_run=False)
                
                print("\n" + "="*80)
                print("✅ CLEANUP COMPLETE")
                print("="*80)
                print(f"Total orders fixed via city mapping: {total_city:,}")
                print(f"Total orders fixed via customer history: {total_history:,}")
                print(f"Grand total: {total_city + total_history:,} orders cleaned")
                print("="*80)
            else:
                print("❌ Update cancelled")
        elif not args.preview and not args.analyze:
            print("\n" + "="*80)
            print("DRY RUN: Province Cleaning Preview")
            print("="*80)
            
            total_city = apply_updates(conn, dry_run=True)
            total_history = infer_from_customer_history(conn, dry_run=True)
            
            print("\n" + "="*80)
            print("📋 DRY RUN SUMMARY")
            print("="*80)
            print(f"Would fix via city mapping: {total_city:,} orders")
            print(f"Would fix via customer history: {total_history:,} orders")
            print(f"Total coverage: {total_city + total_history:,} orders")
            print("\n💡 Run with --apply to execute these updates")
            print("="*80)
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
