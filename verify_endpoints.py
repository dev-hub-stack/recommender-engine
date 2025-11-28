#!/usr/bin/env python3
"""
Verify all frontend endpoints are working
Run: python3 verify_endpoints.py
"""
import requests
import sys

BASE_URL = "http://localhost:8001"

def test_endpoint(name, url, check_key=None, check_list=False):
    """Test a single endpoint"""
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            if check_list and isinstance(data, list):
                print(f"✅ {name}")
                return True
            elif check_key and check_key in data:
                print(f"✅ {name}")
                return True
            elif not check_key and not check_list:
                print(f"✅ {name}")
                return True
            else:
                print(f"❌ {name} - Missing expected data")
                return False
        else:
            print(f"❌ {name} - Status {resp.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"⏳ {name} - Timeout (slow but may work)")
        return False
    except Exception as e:
        print(f"❌ {name} - {str(e)[:50]}")
        return False

def main():
    print("=" * 60)
    print("   ENDPOINT VERIFICATION TEST")
    print("=" * 60)
    print()
    
    results = {"ml": 0, "analytics": 0, "other": 0}
    totals = {"ml": 6, "analytics": 12, "other": 3}
    
    # ML Endpoints
    print("=== ML ENDPOINTS ===")
    if test_endpoint("ML Status", f"{BASE_URL}/api/v1/ml/status", "is_trained"):
        results["ml"] += 1
    if test_endpoint("ML Top Products", f"{BASE_URL}/api/v1/ml/top-products?limit=3", "products"):
        results["ml"] += 1
    if test_endpoint("ML Product Pairs", f"{BASE_URL}/api/v1/ml/product-pairs?limit=3", "pairs"):
        results["ml"] += 1
    if test_endpoint("ML Customer Similarity", f"{BASE_URL}/api/v1/ml/customer-similarity?limit=3", "success"):
        results["ml"] += 1
    if test_endpoint("ML Collaborative Products", f"{BASE_URL}/api/v1/ml/collaborative-products?limit=3", "products"):
        results["ml"] += 1
    if test_endpoint("ML RFM Segments", f"{BASE_URL}/api/v1/ml/rfm-segments", "segments"):
        results["ml"] += 1
    print(f"   {results['ml']}/{totals['ml']} working")
    print()
    
    # Analytics Endpoints
    print("=== ANALYTICS ENDPOINTS ===")
    if test_endpoint("Dashboard Metrics", f"{BASE_URL}/api/v1/analytics/dashboard", "totalOrders"):
        results["analytics"] += 1
    if test_endpoint("Revenue Trend", f"{BASE_URL}/api/v1/analytics/revenue-trend", "trend"):
        results["analytics"] += 1
    if test_endpoint("Product Analytics", f"{BASE_URL}/api/v1/analytics/products", "products"):
        results["analytics"] += 1
    if test_endpoint("Geographic Provinces", f"{BASE_URL}/api/v1/analytics/geographic/provinces", check_list=True):
        results["analytics"] += 1
    if test_endpoint("Geographic Cities", f"{BASE_URL}/api/v1/analytics/geographic/cities", check_list=True):
        results["analytics"] += 1
    if test_endpoint("RFM Segments", f"{BASE_URL}/api/v1/analytics/customers/rfm-segments", check_list=True):
        results["analytics"] += 1
    if test_endpoint("At-Risk Customers", f"{BASE_URL}/api/v1/analytics/customers/at-risk", check_list=True):
        results["analytics"] += 1
    if test_endpoint("Brand Performance", f"{BASE_URL}/api/v1/analytics/brands/performance", check_list=True):
        results["analytics"] += 1
    if test_endpoint("Collaborative Metrics", f"{BASE_URL}/api/v1/analytics/collaborative-metrics", "totalUsers"):
        results["analytics"] += 1
    if test_endpoint("Collaborative Products", f"{BASE_URL}/api/v1/analytics/collaborative-products", check_list=True):
        results["analytics"] += 1
    if test_endpoint("Collaborative Pairs", f"{BASE_URL}/api/v1/analytics/collaborative-pairs", check_list=True):
        results["analytics"] += 1
    if test_endpoint("Customer Similarity", f"{BASE_URL}/api/v1/analytics/customer-similarity", check_list=True):
        results["analytics"] += 1
    print(f"   {results['analytics']}/{totals['analytics']} working")
    print()
    
    # Other Endpoints
    print("=== OTHER ENDPOINTS ===")
    if test_endpoint("Health Check", f"{BASE_URL}/health", "status"):
        results["other"] += 1
    if test_endpoint("Cache Stats", f"{BASE_URL}/api/v1/cache/stats", "cache_size"):
        results["other"] += 1
    if test_endpoint("System Stats", f"{BASE_URL}/api/v1/stats", "total_orders"):
        results["other"] += 1
    print(f"   {results['other']}/{totals['other']} working")
    print()
    
    # Summary
    total = sum(results.values())
    total_endpoints = sum(totals.values())
    print("=" * 60)
    print(f"   TOTAL: {total}/{total_endpoints} endpoints working")
    print("=" * 60)
    
    if total == total_endpoints:
        print("\n🎉 ALL ENDPOINTS WORKING!")
        return 0
    else:
        print(f"\n⚠️  {total_endpoints - total} endpoints need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
