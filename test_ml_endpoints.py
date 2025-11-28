#!/usr/bin/env python3
"""
Test all ML endpoints for frontend components
Run: python3 test_ml_endpoints.py
"""

import requests
import json

BASE_URL = "http://localhost:8001"

def test_endpoint(name, url, key):
    """Test an endpoint and return status"""
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if key in data:
            items = data[key]
            print(f"✅ {name}: {len(items)} items")
            return True, items
        else:
            error = data.get('detail', 'Unknown error')[:80]
            print(f"❌ {name}: {error}")
            return False, error
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: Server not running")
        return False, "Connection failed"
    except Exception as e:
        print(f"❌ {name}: {str(e)[:50]}")
        return False, str(e)

def main():
    print("=" * 50)
    print("   ML ENDPOINTS TEST FOR FRONTEND")
    print("=" * 50)
    print()
    
    # Test ML Status
    print("0️⃣  ML STATUS")
    try:
        r = requests.get(f"{BASE_URL}/api/v1/ml/status", timeout=5)
        status = r.json()
        print(f"   Trained: {status.get('is_trained', False)}")
        algos = status.get('algorithms', {})
        for algo, trained in algos.items():
            icon = "✅" if trained else "❌"
            print(f"   {icon} {algo}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Test each endpoint
    endpoints = [
        ("1️⃣  TOP PRODUCTS", f"{BASE_URL}/api/v1/ml/top-products?limit=5", "products"),
        ("2️⃣  PRODUCT PAIRS", f"{BASE_URL}/api/v1/ml/product-pairs?limit=5", "pairs"),
        ("3️⃣  CUSTOMER SIMILARITY", f"{BASE_URL}/api/v1/ml/customer-similarity?limit=5", "customers"),
        ("4️⃣  COLLABORATIVE PRODUCTS", f"{BASE_URL}/api/v1/ml/collaborative-products?limit=5", "products"),
        ("5️⃣  RFM SEGMENTS", f"{BASE_URL}/api/v1/ml/rfm-segments", "segments"),
    ]
    
    results = {}
    for name, url, key in endpoints:
        success, data = test_endpoint(name, url, key)
        results[name] = success
        
        # Show sample data for successful endpoints
        if success and data:
            if key == "products" and len(data) > 0:
                p = data[0]
                print(f"      Sample: {p.get('product_name', 'N/A')[:30]} - Rs {p.get('total_revenue', 0):,.0f}")
            elif key == "pairs" and len(data) > 0:
                p = data[0]
                print(f"      Sample: {p.get('product_a_name', 'N/A')[:20]} + {p.get('product_b_name', 'N/A')[:20]}")
            elif key == "customers" and len(data) > 0:
                c = data[0]
                print(f"      Sample: {c.get('customer_name', 'N/A')[:30]} - {c.get('total_orders', 0)} orders")
            elif key == "segments" and len(data) > 0:
                s = data[0]
                print(f"      Sample: {s.get('segment_name', 'N/A')} - {s.get('customer_count', 0)} customers")
        print()
    
    # Summary
    print("=" * 50)
    print("   SUMMARY")
    print("=" * 50)
    success_count = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"   {success_count}/{total} endpoints working")
    
    if success_count == total:
        print("   🎉 All endpoints ready for frontend!")
    else:
        print("   ⚠️  Some endpoints need fixes")

if __name__ == "__main__":
    main()
