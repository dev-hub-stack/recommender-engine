#!/usr/bin/env python3
"""
Test ML Endpoints for Frontend Components
Tests all 5 ML endpoints that frontend uses
"""

import requests
import json

BASE_URL = "http://localhost:8001"

def test_endpoint(name, url, data_key):
    """Test an endpoint and return result"""
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            data = response.json()
            items = data.get(data_key, [])
            return True, len(items), data
        else:
            return False, 0, response.text
    except Exception as e:
        return False, 0, str(e)

def main():
    print("=" * 60)
    print("   ML ENDPOINTS TEST FOR FRONTEND")
    print("=" * 60)
    print()
    
    # Check ML Status first
    print("0. ML STATUS:")
    try:
        r = requests.get(f"{BASE_URL}/api/v1/ml/status", timeout=10)
        status = r.json()
        print(f"   Trained: {status.get('is_trained', False)}")
        for algo, trained in status.get('algorithms', {}).items():
            icon = "✅" if trained else "❌"
            print(f"   {icon} {algo}")
        print()
        
        if not status.get('is_trained'):
            print("⚠️  Models not trained! Loading from S3...")
            print()
            r = requests.post(f"{BASE_URL}/api/v1/ml/models/s3/load?time_filter=30days", timeout=300)
            print(f"   S3 Load: {r.json().get('message', 'Unknown')}")
            print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # Test endpoints
    endpoints = [
        ("1. TOP PRODUCTS (TopProductsSection)", 
         f"{BASE_URL}/api/v1/ml/top-products?limit=5&time_filter=30days", 
         "products"),
        
        ("2. PRODUCT PAIRS (CrossSellingSection)", 
         f"{BASE_URL}/api/v1/ml/product-pairs?limit=5&time_filter=30days", 
         "pairs"),
        
        ("3. CUSTOMER SIMILARITY (CustomerSimilaritySection)", 
         f"{BASE_URL}/api/v1/ml/customer-similarity?limit=5&time_filter=30days", 
         "customers"),
        
        ("4. COLLABORATIVE PRODUCTS (TopCollaborativeProductsSection)", 
         f"{BASE_URL}/api/v1/ml/collaborative-products?limit=5&time_filter=30days", 
         "products"),
        
        ("5. RFM SEGMENTS (RFMSegmentationSection)", 
         f"{BASE_URL}/api/v1/ml/rfm-segments?time_filter=30days", 
         "segments"),
    ]
    
    results = []
    
    for name, url, key in endpoints:
        success, count, data = test_endpoint(name, url, key)
        results.append(success)
        
        if success:
            print(f"✅ {name}")
            print(f"   Found: {count} items")
            
            # Show sample data
            items = data.get(key, [])
            if items and len(items) > 0:
                sample = items[0]
                if 'product_name' in sample:
                    print(f"   Sample: {sample.get('product_name', 'N/A')}")
                elif 'segment_name' in sample:
                    print(f"   Sample: {sample.get('segment_name', 'N/A')} - {sample.get('customer_count', 0)} customers")
                elif 'customer_name' in sample:
                    print(f"   Sample: {sample.get('customer_name', 'N/A')}")
                elif 'product_a_name' in sample:
                    print(f"   Sample: {sample.get('product_a_name', 'N/A')} + {sample.get('product_b_name', 'N/A')}")
        else:
            print(f"❌ {name}")
            print(f"   Error: {data[:100]}...")
        
        print()
    
    # Summary
    print("=" * 60)
    print("   SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r)
    print(f"   {success_count}/{len(results)} endpoints working")
    
    if success_count == len(results):
        print("   🎉 All endpoints ready for frontend!")
        print()
        print("   Frontend components mapping:")
        print("   • TopProductsSection       → /ml/top-products")
        print("   • CrossSellingSection      → /ml/product-pairs")
        print("   • CustomerSimilaritySection→ /ml/customer-similarity")
        print("   • TopCollaborativeProducts → /ml/collaborative-products")
        print("   • RFMSegmentationSection   → /ml/rfm-segments")
    else:
        print("   ⚠️  Some endpoints not working")
    
    print()

if __name__ == "__main__":
    main()
