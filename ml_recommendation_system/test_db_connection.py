#!/usr/bin/env python3
"""
Test Database Connection and Load Orders
Simple script to verify we can connect to the existing database
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ingestion.simple_db_loader import SimpleDBLoader

# Load environment
load_dotenv()
load_dotenv('../.env')  # Try parent directory too

def main():
    print("="*70)
    print("DATABASE CONNECTION TEST")
    print("="*70)
    
    # Get DATABASE_URL
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("\n❌ DATABASE_URL not found in environment")
        print("\nPlease set DATABASE_URL in your .env file:")
        print("DATABASE_URL=postgresql://user:password@host:port/database")
        return
    
    print(f"\nDatabase URL: {database_url.split('@')[1] if '@' in database_url else 'Not set'}")
    
    # Initialize loader
    loader = SimpleDBLoader(database_url)
    
    # Test connection
    print("\n1. Testing connection...")
    if not loader.test_connection():
        print("❌ Connection failed!")
        return
    
    # Get stats
    print("\n2. Getting database statistics...")
    stats = loader.get_database_stats()
    
    print("\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)
    print(f"Total Orders: {stats.get('total_orders', 0):,}")
    print(f"Unique Customers: {stats.get('unique_customers', 0):,}")
    print(f"Date Range: {stats.get('earliest_order')} to {stats.get('latest_order')}")
    print(f"Total Revenue: PKR {stats.get('total_revenue', 0):,.2f}")
    print(f"Order Types: {stats.get('order_types', {})}")
    print("="*70)
    
    # Load sample orders
    print("\n3. Loading sample orders (limit 5)...")
    orders = loader.load_orders_from_db(limit=5)
    
    if not orders.empty:
        print("\n" + "="*70)
        print("SAMPLE ORDERS")
        print("="*70)
        print(orders[['id', 'customer_id', 'customer_city', 'order_date', 'source']].to_string())
        print("="*70)
    
    print("\n✅ All tests passed!")
    print("\nYou can now use this database for ML training.")
    print("\nNext step: Run training pipeline")
    print("  python run_training.py --source database")


if __name__ == '__main__':
    main()
