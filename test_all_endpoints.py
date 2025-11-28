#!/usr/bin/env python3
"""Test ALL frontend endpoints"""
import requests
import json

BASE_URL = "http://localhost:8001"

def test(name, url):
    try:
        r = requests.get(f"{BASE_URL}{url}", timeout=30)
        if r.status_code == 200:
            return "✅", len(str(r.json()))
        return "❌", r.status_code
    except Exception as e:
        return "❌", str(e)[:30]

print("=" * 60)
print("   ALL FRONTEND ENDPOINTS TEST")
print("=" * 60)
print()

# ML Endpoints
ml_endpoints = [
    ("ML Status", "/api/v1/ml/status"),
    ("ML Top Products", "/api/v1/ml/top-products?limit=5"),
    ("ML Product Pairs", "/api/v1/ml/product-pairs?limit=5"),
    ("ML Customer Similarity", "/api/v1/ml/customer-similarity?limit=5"),
    ("ML Collaborative Products", "/api/v1/ml/collaborative-products?limit=5"),
    ("ML RFM Segments", "/api/v1/ml/rfm-segments"),
]

# Analytics Endpoints
analytics_endpoints = [
    ("Dashboard Metrics", "/api/v1/analytics/dashboard?time_filter=30days"),
    ("Revenue Trend", "/api/v1/analytics/revenue-trend?time_filter=30days"),
    ("Product Analytics", "/api/v1/analytics/products?time_filter=30days&limit=10"),
    ("Geographic Provinces", "/api/v1/analytics/geographic/provinces?time_filter=30days"),
    ("Geographic Cities", "/api/v1/analytics/geographic/cities?time_filter=30days&limit=10"),
    ("RFM Segments", "/api/v1/analytics/customers/rfm-segments?time_filter=30days"),
    ("At-Risk Customers", "/api/v1/analytics/customers/at-risk?time_filter=30days"),
    ("Brand Performance", "/api/v1/analytics/brands/performance?time_filter=30days"),
    ("Collaborative Metrics", "/api/v1/analytics/collaborative-metrics?time_filter=30days"),
    ("Collaborative Products", "/api/v1/analytics/collaborative-products?time_filter=30days"),
    ("Collaborative Pairs", "/api/v1/analytics/collaborative-pairs?time_filter=30days"),
    ("Customer Similarity", "/api/v1/analytics/customer-similarity?time_filter=30days"),
]

# Other Endpoints
other_endpoints = [
    ("Health Check", "/health"),
    ("Cache Stats", "/api/v1/cache/stats"),
    ("System Stats", "/api/v1/stats"),
]

print("=== ML ENDPOINTS ===")
ml_ok = 0
for name, url in ml_endpoints:
    status, info = test(name, url)
    print(f"{status} {name}")
    if status == "✅": ml_ok += 1
print(f"   {ml_ok}/{len(ml_endpoints)} working\n")

print("=== ANALYTICS ENDPOINTS ===")
ana_ok = 0
for name, url in analytics_endpoints:
    status, info = test(name, url)
    print(f"{status} {name}")
    if status == "✅": ana_ok += 1
print(f"   {ana_ok}/{len(analytics_endpoints)} working\n")

print("=== OTHER ENDPOINTS ===")
oth_ok = 0
for name, url in other_endpoints:
    status, info = test(name, url)
    print(f"{status} {name}")
    if status == "✅": oth_ok += 1
print(f"   {oth_ok}/{len(other_endpoints)} working\n")

total = ml_ok + ana_ok + oth_ok
total_endpoints = len(ml_endpoints) + len(analytics_endpoints) + len(other_endpoints)
print("=" * 60)
print(f"   TOTAL: {total}/{total_endpoints} endpoints working")
print("=" * 60)
