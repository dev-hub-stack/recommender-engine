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

# Check cities for NULL provinces
cursor.execute("""
    SELECT 
        COALESCE(city, 'No City') as city,
        COUNT(*) as count
    FROM orders
    WHERE province IS NULL
    GROUP BY city
    ORDER BY count DESC
    LIMIT 20
""")

print("=== Top 20 Cities with NULL Province ===")
for row in cursor.fetchall():
    print(f"{row['city']}: {row['count']} orders")

cursor.close()
conn.close()
