#!/usr/bin/env python3
"""
Comprehensive Endpoint Verification Script
Tests all recommendation service endpoints and verifies their functionality
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any

# Configuration
BASE_URL = "http://localhost:8001"  # Recommendation service URL
TEST_CUSTOMER_ID = "test_customer_001"
TEST_PRODUCT_ID = "test_product_001"

class EndpointTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
        self.session = requests.Session()
        
    def test_endpoint(self, method: str, endpoint: str, params: Dict = None, 
                     json_data: Dict = None, expected_status: int = 200) -> Dict:
        """Test a single endpoint and return results"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, params=params, json=json_data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = time.time() - start_time
            
            result = {
                "endpoint": endpoint,
                "method": method.upper(),
                "url": url,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_time": round(response_time, 3),
                "success": response.status_code == expected_status,
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to parse JSON response
            try:
                result["response_data"] = response.json()
                result["response_size"] = len(response.content)
            except:
                result["response_data"] = response.text[:500]  # First 500 chars
                result["response_size"] = len(response.content)
            
            # Add error details if request failed
            if not result["success"]:
                result["error"] = f"Expected {expected_status}, got {response.status_code}"
                
            self.results.append(result)
            return result
            
        except Exception as e:
            result = {
                "endpoint": endpoint,
                "method": method.upper(),
                "url": url,
                "status_code": None,
                "expected_status": expected_status,
                "response_time": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(result)
            return result
    
    def print_result(self, result: Dict):
        """Print formatted test result"""
        status_icon = "✅" if result["success"] else "❌"
        print(f"{status_icon} {result['method']} {result['endpoint']}")
        print(f"   Status: {result.get('status_code', 'ERROR')} | Time: {result.get('response_time', 'N/A')}s")
        
        if not result["success"]:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        elif result.get("response_data") and isinstance(result["response_data"], dict):
            # Print summary of response data
            data = result["response_data"]
            if isinstance(data, list):
                print(f"   Response: {len(data)} items")
            elif isinstance(data, dict):
                if "recommendations" in data:
                    recs = data["recommendations"]
                    print(f"   Response: {len(recs)} recommendations")
                elif "data" in data:
                    print(f"   Response: {len(data['data'])} data points")
                else:
                    print(f"   Response: {len(data)} keys")
        print()
    
    def run_all_tests(self):
        """Run all endpoint tests"""
        print("🚀 Starting Comprehensive Endpoint Testing")
        print("=" * 60)
        
        # 1. Health Check
        print("📋 Health & Status Endpoints")
        print("-" * 30)
        self.print_result(self.test_endpoint("GET", "/health"))
        
        # 2. Cache & System Status
        self.print_result(self.test_endpoint("GET", "/api/v1/cache/stats"))
        self.print_result(self.test_endpoint("GET", "/metrics"))
        
        # 3. Sync & Scheduler Endpoints
        print("🔄 Sync & Scheduler Endpoints")
        print("-" * 30)
        self.print_result(self.test_endpoint("GET", "/api/v1/sync/status"))
        self.print_result(self.test_endpoint("GET", "/api/v1/sync/scheduler-status"))
        self.print_result(self.test_endpoint("GET", "/api/v1/sync/history"))
        self.print_result(self.test_endpoint("POST", "/api/v1/sync/trigger"))
        
        # 4. Training Endpoints
        print("🧠 Training Endpoints")
        print("-" * 30)
        self.print_result(self.test_endpoint("GET", "/api/v1/training/status"))
        self.print_result(self.test_endpoint("POST", "/api/v1/training/trigger"))
        
        # 5. Recommendation Endpoints
        print("🎯 Recommendation Endpoints")
        print("-" * 30)
        
        # Collaborative filtering
        self.print_result(self.test_endpoint("GET", "/api/v1/recommendations/collaborative", 
                                           params={"customer_id": TEST_CUSTOMER_ID}))
        
        # Product pairs (cross-selling)
        self.print_result(self.test_endpoint("GET", "/api/v1/recommendations/product-pairs",
                                           params={"product_id": TEST_PRODUCT_ID}))
        
        # Popular products
        self.print_result(self.test_endpoint("GET", "/api/v1/recommendations/popular"))
        
        # Content-based recommendations
        self.print_result(self.test_endpoint("GET", "/api/v1/recommendations/content-based",
                                           params={"customer_id": TEST_CUSTOMER_ID}))
        
        # Matrix factorization
        self.print_result(self.test_endpoint("GET", "/api/v1/recommendations/matrix-factorization",
                                           params={"customer_id": TEST_CUSTOMER_ID}))
        
        # Legacy endpoint
        self.print_result(self.test_endpoint("GET", f"/recommendations/{TEST_CUSTOMER_ID}"))
        
        # 6. Customer Data Endpoints
        print("👤 Customer Data Endpoints")
        print("-" * 30)
        self.print_result(self.test_endpoint("GET", f"/api/v1/customers/{TEST_CUSTOMER_ID}/history"))
        
        # 7. Analytics Endpoints
        print("📊 Analytics Endpoints")
        print("-" * 30)
        self.print_result(self.test_endpoint("GET", "/api/v1/analytics/products"))
        self.print_result(self.test_endpoint("GET", "/api/v1/analytics/dashboard"))
        
        # Test with different time filters
        time_filters = ["today", "7days", "30days", "mtd", "90days", "6months", "1year", "all"]
        print("🕐 Time Filter Testing")
        print("-" * 30)
        for time_filter in time_filters:
            result = self.test_endpoint("GET", "/api/v1/recommendations/popular",
                                      params={"time_filter": time_filter, "limit": 5})
            status_icon = "✅" if result["success"] else "❌"
            print(f"{status_icon} Popular products ({time_filter}): {result.get('status_code', 'ERROR')}")
        
        # 8. Interaction Tracking
        print("\n📝 Interaction Tracking")
        print("-" * 30)
        interaction_data = {
            "user_id": TEST_CUSTOMER_ID,
            "product_id": TEST_PRODUCT_ID,
            "interaction_type": "view"
        }
        self.print_result(self.test_endpoint("POST", "/track-interaction", json_data=interaction_data))
        
        # 9. Performance Tests with Larger Limits
        print("⚡ Performance Tests")
        print("-" * 30)
        
        # Test with large limits
        large_limit_tests = [
            ("/api/v1/recommendations/popular", {"limit": 100}),
            ("/api/v1/analytics/products", {"limit": 50}),
            ("/api/v1/recommendations/collaborative", {"customer_id": TEST_CUSTOMER_ID, "limit": 50})
        ]
        
        for endpoint, params in large_limit_tests:
            result = self.test_endpoint("GET", endpoint, params=params)
            status_icon = "✅" if result["success"] else "❌"
            response_time = result.get("response_time", "N/A")
            print(f"{status_icon} {endpoint} (large limit): {response_time}s")
        
        print("\n" + "=" * 60)
        self.print_summary()
        
    def print_summary(self):
        """Print test summary"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - successful_tests
        
        print("📈 TEST SUMMARY")
        print("-" * 30)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result["success"]:
                    print(f"   - {result['method']} {result['endpoint']}: {result.get('error', 'Unknown error')}")
        
        # Performance stats
        response_times = [r["response_time"] for r in self.results if r.get("response_time")]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            print(f"\n⚡ Performance:")
            print(f"   Average Response Time: {avg_time:.3f}s")
            print(f"   Fastest Response: {min_time:.3f}s")
            print(f"   Slowest Response: {max_time:.3f}s")
    
    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"endpoint_test_results_{timestamp}.json"
        
        test_summary = {
            "test_run_info": {
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url,
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r["success"]),
                "failed_tests": sum(1 for r in self.results if not r["success"])
            },
            "test_results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(test_summary, f, indent=2)
        
        print(f"📝 Results saved to: {filename}")

def check_service_running():
    """Check if the recommendation service is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🔧 Recommendation Engine - Endpoint Verification")
    print("=" * 60)
    
    # Check if service is running
    if not check_service_running():
        print("❌ Recommendation service is not running!")
        print(f"   Please start the service at {BASE_URL}")
        print("   Run: cd recommendation-engine-service && python src/main.py")
        sys.exit(1)
    
    print("✅ Service is running - starting tests...")
    print()
    
    # Run tests
    tester = EndpointTester(BASE_URL)
    tester.run_all_tests()
    
    # Save results
    tester.save_results()
    
    # Exit with appropriate code
    failed_tests = sum(1 for r in tester.results if not r["success"])
    sys.exit(0 if failed_tests == 0 else 1)

if __name__ == "__main__":
    main()
