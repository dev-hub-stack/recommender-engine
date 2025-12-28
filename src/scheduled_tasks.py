"""
Scheduled Maintenance Tasks for Data Quality
=============================================

Run this script periodically (weekly) to clean NULL provinces
that slip through the real-time pipeline.

Usage:
    python3 scheduled_tasks.py

Cron example (Sundays at 2 AM):
    0 2 * * 0 cd /opt/mastergroup-ml && /usr/bin/python3 src/scheduled_tasks.py >> /var/log/province_cleanup.log 2>&1
"""

import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime
from province_utils import CITY_TO_PROVINCE

load_dotenv()


def get_db_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        database=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD")
    )


def clean_null_provinces():
    """
    Clean NULL provinces using:
    1. City-to-province mapping
    2. Customer order history
    """
    print(f"\n{'='*80}")
    print(f"PROVINCE CLEANUP TASK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check how many NULL provinces exist
    cursor.execute("SELECT COUNT(*) FROM orders WHERE province IS NULL")
    null_count_before = cursor.fetchone()[0]
    print(f"📊 NULL provinces found: {null_count_before:,}")
    
    if null_count_before == 0:
        print("✅ No cleanup needed!")
        conn.close()
        return 0
    
    # STEP 1: Fix via city mapping
    print(f"\n{'='*60}")
    print("STEP 1: City-based province mapping")
    print(f"{'='*60}\n")
    
    total_city_fixed = 0
    for city, province in CITY_TO_PROVINCE.items():
        cursor.execute("""
            UPDATE orders
            SET province = %s
            WHERE province IS NULL 
                AND LOWER(TRIM(customer_city)) = %s
        """, (province, city))
        
        fixed = cursor.rowcount
        if fixed > 0:
            total_city_fixed += fixed
            print(f"  ✅ {city.title():<30} → {province:<20} ({fixed:>4} orders)")
    
    conn.commit()
    print(f"\n✅ Step 1 Complete: Fixed {total_city_fixed:,} orders via city mapping")
    
    # STEP 2: Fix via customer history
    print(f"\n{'='*60}")
    print("STEP 2: Customer history inference")
    print(f"{'='*60}\n")
    
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
    
    total_history_fixed = cursor.rowcount
    conn.commit()
    print(f"✅ Step 2 Complete: Fixed {total_history_fixed:,} orders via customer history")
    
    # Check final status
    cursor.execute("SELECT COUNT(*) FROM orders WHERE province IS NULL")
    null_count_after = cursor.fetchone()[0]
    
    # Summary
    total_fixed = total_city_fixed + total_history_fixed
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"NULL provinces before: {null_count_before:,}")
    print(f"Fixed via city mapping: {total_city_fixed:,}")
    print(f"Fixed via customer history: {total_history_fixed:,}")
    print(f"Total fixed: {total_fixed:,}")
    print(f"NULL provinces remaining: {null_count_after:,}")
    print(f"Coverage improvement: {(total_fixed / null_count_before * 100):.1f}%")
    print(f"{'='*80}\n")
    
    cursor.close()
    conn.close()
    
    return total_fixed


def check_data_quality():
    """
    Check overall data quality metrics
    """
    print(f"\n{'='*80}")
    print("DATA QUALITY CHECK")
    print(f"{'='*80}\n")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Province coverage
    cursor.execute("""
        SELECT 
            COUNT(*) as total_orders,
            COUNT(*) FILTER (WHERE province IS NOT NULL AND TRIM(province) != '') as with_province,
            COUNT(*) FILTER (WHERE province IS NULL OR TRIM(province) = '') as without_province,
            ROUND(
                COUNT(*) FILTER (WHERE province IS NOT NULL AND TRIM(province) != '')::numeric 
                / COUNT(*) * 100, 
                2
            ) as coverage_pct
        FROM orders
    """)
    
    result = cursor.fetchone()
    total, with_prov, without_prov, coverage = result
    
    print(f"Total orders: {total:,}")
    print(f"With province: {with_prov:,} ({coverage}%)")
    print(f"Without province: {without_prov:,} ({100-coverage:.2f}%)")
    
    # Top unmapped cities
    print(f"\n{'='*60}")
    print("Top 10 Unmapped Cities")
    print(f"{'='*60}")
    
    cursor.execute("""
        SELECT 
            LOWER(TRIM(customer_city)) as city,
            COUNT(*) as count
        FROM orders
        WHERE province IS NULL 
            AND customer_city IS NOT NULL 
            AND TRIM(customer_city) != ''
        GROUP BY LOWER(TRIM(customer_city))
        ORDER BY count DESC
        LIMIT 10
    """)
    
    unmapped = cursor.fetchall()
    if unmapped:
        for city, count in unmapped:
            print(f"  {city.title():<30} {count:>6} orders")
    else:
        print("  ✅ No unmapped cities!")
    
    print(f"{'='*80}\n")
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    try:
        # Run cleanup
        fixed = clean_null_provinces()
        
        # Check data quality
        check_data_quality()
        
        print(f"✅ Cleanup task completed successfully!\n")
        exit(0)
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        exit(1)
