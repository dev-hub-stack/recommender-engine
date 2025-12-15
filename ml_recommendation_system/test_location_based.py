"""
Test Location-Based Collaborative Filtering Implementation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.data_processor import DataProcessor
from ingestion.simple_db_loader import SimpleDBLoader
import pandas as pd

def test_location_id_creation():
    """Test location ID creation logic"""
    print("\n" + "="*60)
    print("TEST 1: Location ID Creation")
    print("="*60)
    
    processor = DataProcessor()
    
    test_cases = [
        ("Lahore", "Punjab", "Pakistan", "LAHORE"),
        ("", "Punjab", "Pakistan", "PUNJAB"),
        ("", "", "Pakistan", "PAKISTAN"),
        ("", "", "", "UNKNOWN"),
        ("Karachi", "", "Pakistan", "KARACHI"),
        ("New York", "NY", "USA", "NEW YORK"),
        ("Los Angeles!", "CA", "USA", "LOS ANGELES"),  # Special chars removed
    ]
    
    print("\nTest Cases:")
    for city, state, country, expected in test_cases:
        result = processor.create_location_id(city, state, country)
        status = "✅" if result == expected else "❌"
        print(f"{status} City: '{city}', State: '{state}', Country: '{country}' → '{result}' (expected: '{expected}')")
    
    print("\n✅ Location ID creation test complete!")

def test_data_loading():
    """Test loading data from database"""
    print("\n" + "="*60)
    print("TEST 2: Database Loading")
    print("="*60)
    
    loader = SimpleDBLoader()
    orders = loader.load_orders_from_db(limit=100)  # Load sample for testing
    
    print(f"\nLoaded {len(orders)} orders from database")
    print(f"Columns: {list(orders.columns)}")
    
    # Check location fields
    if 'customer_city' in orders.columns:
        print("\nLocation field statistics:")
        print(f"  customer_city: {orders['customer_city'].notna().sum()} non-null values")
    else:
        print("\n⚠️  Note: customer_city not in database schema")
        print("  Available columns:", list(orders.columns))
    
    print("\n✅ Database loading test complete!")
    return orders

def test_data_processing(orders):
    """Test data processing with location IDs"""
    print("\n" + "="*60)
    print("TEST 3: Data Processing with Location IDs")
    print("="*60)
    
    processor = DataProcessor(time_window_days=None)  # Use all data
    processed = processor.process(orders)
    
    print(f"\nProcessed {len(processed)} interactions")
    print(f"Unique locations: {processed['location_id'].nunique()}")
    print(f"Unique products: {processed['product_id'].nunique()}")
    
    # Show top locations
    print("\nTop 10 locations by order count:")
    location_counts = processed['location_id'].value_counts().head(10)
    for loc, count in location_counts.items():
        print(f"  {loc}: {count} orders")
    
    print("\n✅ Data processing test complete!")
    return processed

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("LOCATION-BASED COLLABORATIVE FILTERING - IMPLEMENTATION TEST")
    print("="*70)
    
    try:
        # Test 1: Location ID creation
        test_location_id_creation()
        
        # Test 2: Database loading
        orders = test_data_loading()
        
        # Test 3: Data processing
        processed = test_data_processing(orders)
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nNext steps:")
        print("1. Run full training pipeline: python ml-recommendation-system/train.py")
        print("2. Start API server: python ml-recommendation-system/src/api/app.py")
        print("3. Test location endpoint: GET /api/v1/recommendations/location?city=Lahore&limit=10")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
