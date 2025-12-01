#!/usr/bin/env python3
"""Data validation script for anomaly detection"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    database=os.getenv("PG_DB"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    sslmode="require"
)

cursor = conn.cursor()

print("=" * 60)
print("DETAILED ANOMALY ANALYSIS")
print("=" * 60)

# Zero revenue sample
print("\n1. ZERO REVENUE ORDERS (Sample 5):")
cursor.execute("SELECT id, order_name, order_type, customer_name, total_price FROM orders WHERE total_price = 0 LIMIT 5")
for row in cursor.fetchall():
    print(f"   {row[1]} | Type: {row[2]} | Customer: {row[3]}")

# Order type breakdown
print("\n2. ORDER TYPE BREAKDOWN:")
cursor.execute("SELECT order_type, COUNT(*) FROM orders GROUP BY order_type ORDER BY 2 DESC")
for row in cursor.fetchall():
    print(f"   {row[0]}: {row[1]} orders")

# Top cities
print("\n3. TOP 10 CITIES:")
cursor.execute("""
    SELECT customer_city, COUNT(*) as cnt 
    FROM orders 
    WHERE customer_city IS NOT NULL 
    GROUP BY customer_city 
    ORDER BY cnt DESC 
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"   {row[0]}: {row[1]} orders")

# Large outliers
print("\n4. LARGEST ORDERS (Top 5):")
cursor.execute("SELECT id, order_name, total_price, customer_name FROM orders ORDER BY total_price DESC LIMIT 5")
for row in cursor.fetchall():
    print(f"   {row[1]} | PKR {row[2]:,.0f} | {row[3]}")

# Check for offline recommendations table
print("\n5. BATCH INFERENCE STATUS:")
try:
    cursor.execute("SELECT COUNT(*) FROM offline_user_recommendations")
    rec_count = cursor.fetchone()[0]
    print(f"   Cached user recommendations: {rec_count}")
    
    cursor.execute("SELECT MAX(updated_at) FROM offline_user_recommendations")
    last_update = cursor.fetchone()[0]
    print(f"   Last updated: {last_update}")
except Exception as e:
    print(f"   Table not found or error: {e}")

# Similar items cache
print("\n6. SIMILAR ITEMS CACHE:")
try:
    cursor.execute("SELECT COUNT(*) FROM offline_similar_items")
    sim_count = cursor.fetchone()[0]
    print(f"   Cached similar items: {sim_count}")
    
    cursor.execute("SELECT MAX(updated_at) FROM offline_similar_items")
    last_sim = cursor.fetchone()[0]
    print(f"   Last updated: {last_sim}")
except Exception as e:
    print(f"   Table not found or error: {e}")

# Product pairs
print("\n7. PRODUCT PAIRS TABLE:")
try:
    cursor.execute("SELECT COUNT(*) FROM product_pairs")
    pairs = cursor.fetchone()[0]
    print(f"   Product pairs: {pairs}")
except Exception as e:
    print(f"   Table not found or error: {e}")

# Customer statistics
print("\n8. CUSTOMER STATISTICS TABLE:")
try:
    cursor.execute("SELECT COUNT(*) FROM customer_statistics")
    stats = cursor.fetchone()[0]
    print(f"   Customer stats records: {stats}")
except Exception as e:
    print(f"   Table not found or error: {e}")

# B2B check - large orders
print("\n9. POTENTIAL B2B ORDERS (>500K PKR):")
cursor.execute("SELECT COUNT(*) FROM orders WHERE total_price > 500000")
b2b = cursor.fetchone()[0]
print(f"   Orders > 500K: {b2b}")

# Date distribution
print("\n10. ORDER DATE DISTRIBUTION:")
cursor.execute("""
    SELECT 
        EXTRACT(YEAR FROM order_date) as year,
        COUNT(*) as cnt
    FROM orders
    GROUP BY 1
    ORDER BY 1
""")
for row in cursor.fetchall():
    print(f"   {int(row[0])}: {row[1]} orders")

print("\n" + "=" * 60)
conn.close()
