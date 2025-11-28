"""
Export MasterGroup data to AWS Personalize format
Creates 3 CSV files: interactions.csv, items.csv, users.csv
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import csv
import os
from datetime import datetime

# Database connection
DB_CONFIG = {
    'host': os.environ.get('PG_HOST', 'localhost'),
    'port': os.environ.get('PG_PORT', '5432'),
    'dbname': os.environ.get('PG_DB', 'mastergroup_recommendations'),
    'user': os.environ.get('PG_USER', 'postgres'),
    'password': os.environ.get('PG_PASSWORD', 'postgres')
}

OUTPUT_DIR = 'aws_personalize/data'

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def export_interactions():
    """
    Export user-item interactions
    AWS Personalize format: USER_ID, ITEM_ID, TIMESTAMP, EVENT_TYPE
    """
    print("Exporting interactions...")
    
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT 
            o.unified_customer_id as user_id,
            oi.product_id as item_id,
            EXTRACT(EPOCH FROM o.order_date)::bigint as timestamp,
            'purchase' as event_type,
            oi.quantity as event_value
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        WHERE o.unified_customer_id IS NOT NULL 
        AND oi.product_id IS NOT NULL
        AND o.order_date IS NOT NULL
        ORDER BY o.order_date
    """)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    filepath = f"{OUTPUT_DIR}/interactions.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['USER_ID', 'ITEM_ID', 'TIMESTAMP', 'EVENT_TYPE', 'EVENT_VALUE'])
        for row in rows:
            writer.writerow([
                row['user_id'],
                row['item_id'],
                row['timestamp'],
                row['event_type'],
                row['event_value'] or 1
            ])
    
    print(f"  ✅ Exported {len(rows)} interactions to {filepath}")
    return len(rows)

def export_items():
    """
    Export item metadata
    AWS Personalize format: ITEM_ID, CATEGORY, PRICE, etc.
    """
    print("Exporting items...")
    
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT DISTINCT
            oi.product_id as item_id,
            MAX(oi.product_name) as item_name,
            'General' as category,
            AVG(oi.unit_price) as price,
            COUNT(DISTINCT oi.order_id) as purchase_count
        FROM order_items oi
        WHERE oi.product_id IS NOT NULL
        GROUP BY oi.product_id
    """)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    filepath = f"{OUTPUT_DIR}/items.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ITEM_ID', 'ITEM_NAME', 'CATEGORY', 'PRICE', 'PURCHASE_COUNT'])
        for row in rows:
            writer.writerow([
                row['item_id'],
                (row['item_name'] or 'Unknown')[:256],  # Max 256 chars
                (row['category'] or 'General')[:256],
                round(float(row['price'] or 0), 2),
                row['purchase_count']
            ])
    
    print(f"  ✅ Exported {len(rows)} items to {filepath}")
    return len(rows)

def export_users():
    """
    Export user metadata
    AWS Personalize format: USER_ID, plus optional metadata
    """
    print("Exporting users...")
    
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT 
            o.unified_customer_id as user_id,
            MAX(o.customer_city) as city,
            MAX(o.province) as province,
            COUNT(DISTINCT o.id) as order_count,
            COALESCE(SUM(oi.total_price), 0) as total_spend
        FROM orders o
        LEFT JOIN order_items oi ON o.id = oi.order_id
        WHERE o.unified_customer_id IS NOT NULL
        AND TRIM(o.unified_customer_id) != ''
        AND o.unified_customer_id !~ '^[^a-zA-Z0-9]'
        GROUP BY o.unified_customer_id
    """)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    filepath = f"{OUTPUT_DIR}/users.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['USER_ID', 'CITY', 'PROVINCE', 'ORDER_COUNT', 'TOTAL_SPEND'])
        for row in rows:
            writer.writerow([
                row['user_id'],
                (row['city'] or 'Unknown')[:256],
                (row['province'] or 'Unknown')[:256],
                row['order_count'],
                round(float(row['total_spend'] or 0), 2)
            ])
    
    print(f"  ✅ Exported {len(rows)} users to {filepath}")
    return len(rows)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("AWS PERSONALIZE DATA EXPORT")
    print("="*60 + "\n")
    
    interactions = export_interactions()
    items = export_items()
    users = export_users()
    
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"  Interactions: {interactions:,}")
    print(f"  Items:        {items:,}")
    print(f"  Users:        {users:,}")
    print(f"\n  Files saved to: {OUTPUT_DIR}/")
    print("="*60 + "\n")
    
    print("NEXT STEPS:")
    print("1. Upload these files to S3 bucket")
    print("2. Run: python aws_personalize/setup_personalize.py")
    print("")

if __name__ == "__main__":
    main()
