import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    database=os.getenv("PG_DB"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD")
)

cursor = conn.cursor(cursor_factory=RealDictCursor)

# Check what province values exist that would be filtered
cursor.execute("""
    SELECT 
        province,
        COUNT(*) as count,
        COUNT(DISTINCT unified_customer_id) as customer_count
    FROM orders
    WHERE province IS NULL 
        OR TRIM(province) = '' 
        OR UPPER(TRIM(province)) IN ('UNKNOWN', 'N/A', 'NA', 'NULL', 'NONE')
    GROUP BY province
    ORDER BY count DESC
""")

print("=== 'Unknown' Province Values ===")
for row in cursor.fetchall():
    print(f"{row['province']}: {row['count']} orders, {row['customer_count']} customers")

cursor.close()
conn.close()
