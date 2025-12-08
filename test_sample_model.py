#!/usr/bin/env python3
"""
Comprehensive Testing Report: Sample Custom Model vs AWS Personalize
Tests all endpoints and functionality to create detailed comparison report
"""

import sys
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime
import os

sys.path.append('src')
import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    """Get production database connection"""
    return psycopg2.connect(
        host='ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com',
        port='5432',
        dbname='mastergroup_recommendations',
        user='postgres',
        password='MasterGroup2024Secure!',
        sslmode='require'
    )

def test_api_endpoints():
    """Test actual API endpoints"""
    base_url = "http://44.201.11.243:8001"
    
    print("🌐 TESTING API ENDPOINTS")
    print("-" * 40)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        health_status = response.status_code == 200
        print(f"   ✅ Health Check: {response.status_code}")
    except Exception as e:
        health_status = False
        print(f"   ❌ Health Check: {str(e)[:50]}...")
    
    # Test recommendation endpoints
    test_user = "03318791419_mian gulam nabi"
    
    endpoints_to_test = [
        f"/api/v1/recommendations/{test_user}",
        f"/api/v1/recommendations/{test_user}/collaborative",
        f"/api/v1/recommendations/{test_user}/popular",
        f"/api/v1/similar-items/1328"
    ]
    
    endpoint_results = {}
    
    for endpoint in endpoints_to_test:
        try:
            start_time = datetime.now()
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                rec_count = len(data.get('recommendations', []))
                
                endpoint_results[endpoint] = {
                    'status': 'success',
                    'status_code': response.status_code,
                    'response_time_ms': response_time,
                    'recommendation_count': rec_count
                }
                
                print(f"   ✅ {endpoint}: {rec_count} recs in {response_time:.1f}ms")
            else:
                endpoint_results[endpoint] = {
                    'status': 'error',
                    'status_code': response.status_code,
                    'response_time_ms': response_time,
                    'recommendation_count': 0
                }
                print(f"   ❌ {endpoint}: HTTP {response.status_code}")
                
        except Exception as e:
            endpoint_results[endpoint] = {
                'status': 'error',
                'error': str(e),
                'response_time_ms': 0,
                'recommendation_count': 0
            }
            print(f"   ❌ {endpoint}: {str(e)[:50]}...")
    
    return endpoint_results

def compare_with_aws_personalize():
    """Compare recommendations with AWS Personalize cache"""
    print("\n📊 COMPARING WITH AWS PERSONALIZE")
    print("-" * 40)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get AWS Personalize cached data
    cursor.execute("""
        SELECT 
            user_id,
            recommendations,
            updated_at
        FROM recommendation_cache
        WHERE recommendations IS NOT NULL
        ORDER BY updated_at DESC
        LIMIT 20
    """)
    
    aws_cache_data = cursor.fetchall()
    print(f"   Found {len(aws_cache_data)} AWS Personalize cached users")
    
    # Get sample of test users
    cursor.execute("""
        SELECT DISTINCT o.unified_customer_id as user_id
        FROM orders o
        WHERE o.unified_customer_id IS NOT NULL
        ORDER BY o.order_date DESC
        LIMIT 10
    """)
    
    test_users = [row['user_id'] for row in cursor.fetchall()]
    
    cursor.close()
    conn.close()
    
    # Compare recommendations
    comparison_results = []
    base_url = "http://44.201.11.243:8001"
    
    for i, test_user in enumerate(test_users[:5]):  # Test 5 users
        print(f"\n   👤 User {i+1}: {test_user}")
        
        # Get custom model recommendations via API
        try:
            response = requests.get(f"{base_url}/api/v1/recommendations/{test_user}", timeout=10)
            if response.status_code == 200:
                custom_data = response.json()
                custom_recs = custom_data.get('recommendations', [])
                custom_items = set([rec.get('product_id', rec.get('item_id', 0)) for rec in custom_recs])
                custom_success = True
            else:
                custom_items = set()
                custom_success = False
        except:
            custom_items = set()
            custom_success = False
        
        # Get AWS Personalize recommendations
        aws_items = set()
        aws_success = False
        
        for aws_data in aws_cache_data:
            if aws_data['user_id'] == test_user:
                try:
                    aws_recs = json.loads(aws_data['recommendations'])
                    aws_items = set([int(rec.get('itemId', 0)) for rec in aws_recs])
                    aws_success = True
                    break
                except:
                    pass
        
        # Calculate similarity
        if custom_items and aws_items:
            overlap = len(custom_items.intersection(aws_items))
            total_unique = len(custom_items.union(aws_items))
            similarity = (overlap / total_unique) * 100 if total_unique > 0 else 0
        else:
            overlap = 0
            similarity = 0
        
        result = {
            'user_id': test_user,
            'custom_success': custom_success,
            'custom_count': len(custom_items),
            'aws_success': aws_success,
            'aws_count': len(aws_items),
            'overlap': overlap,
            'similarity_percent': similarity
        }
        
        comparison_results.append(result)
        
        print(f"      Custom: {len(custom_items)} items ({'✅' if custom_success else '❌'})")
        print(f"      AWS: {len(aws_items)} items ({'✅' if aws_success else '❌'})")
        print(f"      Similarity: {similarity:.1f}% ({overlap} overlapping items)")
    
    return comparison_results

def test_performance():
    """Test performance metrics"""
    print("\n⚡ PERFORMANCE TESTING")
    print("-" * 40)
    
    base_url = "http://44.201.11.243:8001"
    test_user = "03318791419_mian gulam nabi"
    
    # Test response times
    response_times = []
    
    print("   Running 10 performance tests...")
    
    for i in range(10):
        try:
            start_time = datetime.now()
            response = requests.get(f"{base_url}/api/v1/recommendations/{test_user}", timeout=10)
            end_time = datetime.now()
            
            if response.status_code == 200:
                response_time = (end_time - start_time).total_seconds() * 1000
                response_times.append(response_time)
        except:
            pass
    
    if response_times:
        avg_time = np.mean(response_times)
        min_time = np.min(response_times)
        max_time = np.max(response_times)
        
        print(f"   ✅ Average Response Time: {avg_time:.1f}ms")
        print(f"   ✅ Min Response Time: {min_time:.1f}ms")
        print(f"   ✅ Max Response Time: {max_time:.1f}ms")
        
        return {
            'avg_response_time': avg_time,
            'min_response_time': min_time,
            'max_response_time': max_time,
            'successful_requests': len(response_times)
        }
    else:
        print("   ❌ No successful performance tests")
        return None

def analyze_aws_personalize_cache():
    """Analyze AWS Personalize cache data"""
    print("\n🔍 AWS PERSONALIZE CACHE ANALYSIS")
    print("-" * 40)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get cache statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_cached_users,
            COUNT(CASE WHEN recommendations IS NOT NULL THEN 1 END) as users_with_recs,
            AVG(CASE WHEN recommendations IS NOT NULL THEN json_array_length(recommendations::json) END) as avg_recs_per_user
        FROM recommendation_cache
    """)
    
    cache_stats = cursor.fetchone()
    
    # Get sample recommendations to analyze
    cursor.execute("""
        SELECT recommendations
        FROM recommendation_cache
        WHERE recommendations IS NOT NULL
        LIMIT 100
    """)
    
    sample_recs = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    # Analyze recommendation patterns
    all_items = []
    all_scores = []
    
    for rec_data in sample_recs:
        try:
            recs = json.loads(rec_data['recommendations'])
            for rec in recs:
                all_items.append(int(rec.get('itemId', 0)))
                all_scores.append(float(rec.get('score', 0)))
        except:
            pass
    
    unique_items = len(set(all_items))
    avg_score = np.mean(all_scores) if all_scores else 0
    
    print(f"   📊 Total Cached Users: {cache_stats['total_cached_users']:,}")
    print(f"   📊 Users with Recommendations: {cache_stats['users_with_recs']:,}")
    print(f"   📊 Average Recommendations per User: {cache_stats['avg_recs_per_user']:.1f}")
    print(f"   📊 Unique Items in Cache: {unique_items:,}")
    print(f"   📊 Average Recommendation Score: {avg_score:.3f}")
    
    return {
        'total_cached_users': cache_stats['total_cached_users'],
        'users_with_recs': cache_stats['users_with_recs'],
        'avg_recs_per_user': cache_stats['avg_recs_per_user'],
        'unique_items': unique_items,
        'avg_score': avg_score
    }

def main():
    """Main testing function"""
    print("🧪 COMPREHENSIVE MODEL TESTING REPORT")
    print("Sample Custom Model vs AWS Personalize")
    print("=" * 60)
    
    # Test 1: API Endpoints
    endpoint_results = test_api_endpoints()
    
    # Test 2: AWS Personalize Comparison
    comparison_results = compare_with_aws_personalize()
    
    # Test 3: Performance Testing
    performance_results = test_performance()
    
    # Test 4: AWS Personalize Cache Analysis
    cache_analysis = analyze_aws_personalize_cache()
    
    # Generate Summary Report
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE TEST REPORT SUMMARY")
    print("=" * 60)
    
    # Endpoint Summary
    successful_endpoints = sum(1 for result in endpoint_results.values() if result['status'] == 'success')
    total_endpoints = len(endpoint_results)
    
    print(f"\n🌐 API ENDPOINT RESULTS:")
    print(f"   • Success Rate: {successful_endpoints}/{total_endpoints} ({(successful_endpoints/total_endpoints)*100:.1f}%)")
    
    for endpoint, result in endpoint_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"   • {endpoint}: {status} ({result.get('response_time_ms', 0):.1f}ms)")
    
    # Comparison Summary
    if comparison_results:
        successful_comparisons = sum(1 for r in comparison_results if r['custom_success'])
        avg_similarity = np.mean([r['similarity_percent'] for r in comparison_results if r['similarity_percent'] > 0])
        
        print(f"\n📊 AWS PERSONALIZE COMPARISON:")
        print(f"   • Custom Model Success: {successful_comparisons}/{len(comparison_results)} users")
        print(f"   • Average Similarity: {avg_similarity:.1f}%")
        
        for result in comparison_results:
            print(f"   • {result['user_id']}: {result['similarity_percent']:.1f}% similarity")
    
    # Performance Summary
    if performance_results:
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   • Average Response Time: {performance_results['avg_response_time']:.1f}ms")
        print(f"   • Response Time Range: {performance_results['min_response_time']:.1f}ms - {performance_results['max_response_time']:.1f}ms")
        print(f"   • Successful Requests: {performance_results['successful_requests']}/10")
    
    # AWS Cache Summary
    print(f"\n🔍 AWS PERSONALIZE STATUS:")
    print(f"   • Cached Users: {cache_analysis['total_cached_users']:,}")
    print(f"   • Active Recommendations: {cache_analysis['users_with_recs']:,}")
    print(f"   • Average Recs per User: {cache_analysis['avg_recs_per_user']:.1f}")
    
    # Final Assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    
    if successful_endpoints >= total_endpoints * 0.8:
        print(f"   ✅ API ENDPOINTS: Excellent ({successful_endpoints}/{total_endpoints} working)")
    else:
        print(f"   ⚠️ API ENDPOINTS: Needs attention ({successful_endpoints}/{total_endpoints} working)")
    
    if performance_results and performance_results['avg_response_time'] < 100:
        print(f"   ✅ PERFORMANCE: Excellent (<100ms average)")
    elif performance_results and performance_results['avg_response_time'] < 500:
        print(f"   ✅ PERFORMANCE: Good (<500ms average)")
    else:
        print(f"   ⚠️ PERFORMANCE: Needs optimization")
    
    if comparison_results and avg_similarity > 20:
        print(f"   ✅ AWS COMPATIBILITY: Good similarity with AWS Personalize")
    elif comparison_results:
        print(f"   ⚠️ AWS COMPATIBILITY: Different recommendations (may be better)")
    
    print(f"\n🎉 TESTING COMPLETE!")
    print(f"Custom model is {'✅ READY FOR PRODUCTION' if successful_endpoints >= 3 else '⚠️ NEEDS OPTIMIZATION'}")

if __name__ == "__main__":
    main()
