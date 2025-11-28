#!/usr/bin/env python3
"""
Comprehensive ML Model Test Suite
Tests all recommendation algorithms: Collaborative Filtering, Content-Based, 
Matrix Factorization, Popularity-Based, and Hybrid Ensemble

Run with: python test_ml_model_comprehensive.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'algorithms'))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Test Results Storage
class TestResults:
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        
    def add_result(self, test_name: str, passed: bool, details: str = "", metrics: Dict = None):
        self.results.append({
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        })
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
        if details:
            print(f"    Details: {details}")
        if metrics:
            for key, value in metrics.items():
                print(f"    {key}: {value}")
    
    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({100*passed/total:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed: {failed}")
        print(f"Duration: {duration:.2f}s")
        print("=" * 80)
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['test_name']}: {r['details']}")
        
        return passed, failed


# Database Connection
def get_db_connection():
    """Create database connection from environment"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'mastergroup_recommendations'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'postgres')
    )


# =============================================================================
# TEST SUITE 1: COLLABORATIVE FILTERING
# =============================================================================

def test_collaborative_filtering(results: TestResults):
    """Test Collaborative Filtering algorithms"""
    print("\n" + "=" * 80)
    print("TESTING: COLLABORATIVE FILTERING")
    print("=" * 80)
    
    try:
        from algorithms.collaborative_filtering import (
            UserItemMatrix, UserBasedCollaborativeFiltering,
            ItemBasedCollaborativeFiltering, CollaborativeFilteringEngine
        )
        results.add_result("Import Collaborative Filtering", True)
    except ImportError as e:
        results.add_result("Import Collaborative Filtering", False, str(e))
        return
    
    # Test 1: UserItemMatrix
    print("\n--- Testing UserItemMatrix ---")
    try:
        matrix_handler = UserItemMatrix()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u3', 'u3', 'u4', 'u4', 'u5'],
            'item_id': ['p1', 'p2', 'p3', 'p1', 'p4', 'p2', 'p3', 'p1', 'p5', 'p2'],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 2.0, 4.0],
            'timestamp': [datetime.now()] * 10
        })
        
        matrix = matrix_handler.build_matrix(sample_data)
        
        assert matrix is not None, "Matrix should not be None"
        assert matrix_handler.n_users == 5, f"Expected 5 users, got {matrix_handler.n_users}"
        assert matrix_handler.n_items == 5, f"Expected 5 items, got {matrix_handler.n_items}"
        
        # Test sparsity calculation
        sparsity = 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        
        results.add_result("UserItemMatrix Build", True, metrics={
            "n_users": matrix_handler.n_users,
            "n_items": matrix_handler.n_items,
            "sparsity": f"{sparsity:.2%}"
        })
    except Exception as e:
        results.add_result("UserItemMatrix Build", False, str(e))
        return
    
    # Test 2: User-Based CF
    print("\n--- Testing User-Based Collaborative Filtering ---")
    try:
        user_cf = UserBasedCollaborativeFiltering(n_neighbors=10, min_similarity=0.1)
        user_cf.fit(matrix_handler)
        
        # Test similarity matrix
        assert user_cf.user_similarity_matrix is not None, "Similarity matrix should not be None"
        assert user_cf.user_similarity_matrix.shape == (5, 5), "Similarity matrix shape mismatch"
        
        # Test predictions
        prediction = user_cf.predict_rating('u1', 'p4')
        assert isinstance(prediction, float), "Prediction should be float"
        
        # Test recommendations
        recommendations = user_cf.get_recommendations('u1', n_recommendations=3)
        assert isinstance(recommendations, list), "Recommendations should be a list"
        
        results.add_result("User-Based CF Training", True, metrics={
            "similarity_matrix_shape": str(user_cf.user_similarity_matrix.shape),
            "sample_prediction": f"{prediction:.3f}",
            "recommendations_count": len(recommendations)
        })
    except Exception as e:
        results.add_result("User-Based CF Training", False, str(e))
    
    # Test 3: Item-Based CF
    print("\n--- Testing Item-Based Collaborative Filtering ---")
    try:
        item_cf = ItemBasedCollaborativeFiltering(n_neighbors=10, min_similarity=0.1)
        item_cf.fit(matrix_handler)
        
        # Test similarity matrix
        assert item_cf.item_similarity_matrix is not None, "Item similarity matrix should not be None"
        
        # Test recommendations
        recommendations = item_cf.get_recommendations('u1', n_recommendations=3)
        
        results.add_result("Item-Based CF Training", True, metrics={
            "similarity_matrix_shape": str(item_cf.item_similarity_matrix.shape),
            "recommendations_count": len(recommendations)
        })
    except Exception as e:
        results.add_result("Item-Based CF Training", False, str(e))
    
    # Test 4: CollaborativeFilteringEngine
    print("\n--- Testing CollaborativeFilteringEngine ---")
    try:
        cf_engine = CollaborativeFilteringEngine(
            user_cf_weight=0.6,
            item_cf_weight=0.4,
            min_interactions=2
        )
        
        # Train with sample data
        metrics = cf_engine.train(sample_data)
        
        assert cf_engine.is_trained, "Engine should be marked as trained"
        assert "rmse" in metrics or "error" not in metrics, "Training should complete"
        
        results.add_result("CF Engine Training", True, metrics={
            "is_trained": cf_engine.is_trained,
            "rmse": metrics.get("rmse", "N/A"),
            "accuracy": f"{metrics.get('rmse_accuracy', 0):.2f}%"
        })
    except Exception as e:
        results.add_result("CF Engine Training", False, str(e))
    
    # Test 5: CF with Real Database Data
    print("\n--- Testing CF with Real Database ---")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get sample interactions
        cursor.execute("""
            SELECT 
                o.unified_customer_id as user_id,
                oi.product_id as item_id,
                CAST(oi.quantity as FLOAT) as rating,
                o.order_date as timestamp
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            WHERE o.unified_customer_id IS NOT NULL
            AND oi.product_id IS NOT NULL
            ORDER BY o.order_date DESC
            LIMIT 5000
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if len(rows) > 0:
            real_data = pd.DataFrame(rows)
            
            # Train CF engine on real data
            cf_real = CollaborativeFilteringEngine(min_interactions=3)
            real_metrics = cf_real.train(real_data)
            
            results.add_result("CF with Real Data", True, metrics={
                "interactions": len(real_data),
                "unique_users": real_data['user_id'].nunique(),
                "unique_items": real_data['item_id'].nunique(),
                "accuracy": f"{real_metrics.get('rmse_accuracy', 0):.2f}%"
            })
        else:
            results.add_result("CF with Real Data", False, "No data in database")
            
    except Exception as e:
        results.add_result("CF with Real Data", False, str(e))


# =============================================================================
# TEST SUITE 2: CONTENT-BASED FILTERING
# =============================================================================

def test_content_based_filtering(results: TestResults):
    """Test Content-Based Filtering algorithms"""
    print("\n" + "=" * 80)
    print("TESTING: CONTENT-BASED FILTERING")
    print("=" * 80)
    
    try:
        from algorithms.content_based_filtering import ContentBasedFiltering
        results.add_result("Import Content-Based Filtering", True)
    except ImportError as e:
        results.add_result("Import Content-Based Filtering", False, str(e))
        return
    
    # Test with database connection
    print("\n--- Testing Content-Based Filtering ---")
    try:
        conn = get_db_connection()
        cbf = ContentBasedFiltering(pg_conn=conn)
        
        # Build product features
        start_time = time.time()
        cbf.build_product_features()
        build_time = time.time() - start_time
        
        assert cbf.product_vectors is not None, "Product vectors should be built"
        assert len(cbf.product_ids) > 0, "Should have product IDs"
        
        results.add_result("CBF Feature Building", True, metrics={
            "n_products": len(cbf.product_ids),
            "build_time": f"{build_time:.2f}s"
        })
        
        # Test recommendations for a customer
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT DISTINCT o.unified_customer_id
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            WHERE o.unified_customer_id IS NOT NULL
            GROUP BY o.unified_customer_id
            HAVING COUNT(*) > 3
            LIMIT 1
        """)
        test_customer = cursor.fetchone()
        cursor.close()
        
        if test_customer:
            customer_id = test_customer['unified_customer_id']
            recommendations = cbf.get_recommendations(customer_id, limit=10)
            
            results.add_result("CBF Recommendations", True, metrics={
                "customer_id": customer_id[:30] + "...",
                "recommendations_count": len(recommendations),
                "top_score": f"{recommendations[0]['score']:.3f}" if recommendations else "N/A"
            })
        else:
            results.add_result("CBF Recommendations", False, "No test customer found")
        
        conn.close()
        
    except Exception as e:
        results.add_result("Content-Based Filtering", False, str(e))


# =============================================================================
# TEST SUITE 3: MATRIX FACTORIZATION (SVD)
# =============================================================================

def test_matrix_factorization(results: TestResults):
    """Test Matrix Factorization (SVD) algorithms"""
    print("\n" + "=" * 80)
    print("TESTING: MATRIX FACTORIZATION (SVD)")
    print("=" * 80)
    
    try:
        from algorithms.matrix_factorization import MatrixFactorizationSVD
        results.add_result("Import Matrix Factorization", True)
    except ImportError as e:
        results.add_result("Import Matrix Factorization", False, str(e))
        return
    
    # Test with database connection
    print("\n--- Testing Matrix Factorization ---")
    try:
        conn = get_db_connection()
        mf = MatrixFactorizationSVD(pg_conn=conn, n_factors=30)
        
        # Build interaction matrix
        start_time = time.time()
        interaction_matrix, n_users, n_items = mf.build_interaction_matrix()
        build_time = time.time() - start_time
        
        if interaction_matrix is not None:
            results.add_result("MF Matrix Building", True, metrics={
                "n_users": n_users,
                "n_items": n_items,
                "build_time": f"{build_time:.2f}s"
            })
            
            # Train SVD
            start_time = time.time()
            success = mf.train()
            train_time = time.time() - start_time
            
            if success:
                results.add_result("MF SVD Training", True, metrics={
                    "n_factors": mf.n_factors,
                    "user_factors_shape": str(mf.user_factors.shape),
                    "item_factors_shape": str(mf.item_factors.shape),
                    "train_time": f"{train_time:.2f}s"
                })
                
                # Test predictions
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT DISTINCT o.unified_customer_id
                    FROM orders o
                    JOIN order_items oi ON o.id = oi.order_id
                    WHERE o.unified_customer_id IS NOT NULL
                    GROUP BY o.unified_customer_id
                    HAVING COUNT(*) > 5
                    LIMIT 1
                """)
                test_customer = cursor.fetchone()
                cursor.close()
                
                if test_customer:
                    customer_id = test_customer['unified_customer_id']
                    recommendations = mf.get_recommendations(customer_id, limit=10)
                    
                    results.add_result("MF Recommendations", True, metrics={
                        "customer_id": customer_id[:30] + "...",
                        "recommendations_count": len(recommendations),
                        "top_score": f"{recommendations[0]['score']:.3f}" if recommendations else "N/A"
                    })
                else:
                    results.add_result("MF Recommendations", False, "No test customer found")
            else:
                results.add_result("MF SVD Training", False, "Training returned False")
        else:
            results.add_result("MF Matrix Building", False, "No interaction data")
        
        conn.close()
        
    except Exception as e:
        results.add_result("Matrix Factorization", False, str(e))


# =============================================================================
# TEST SUITE 4: POPULARITY-BASED
# =============================================================================

def test_popularity_based(results: TestResults):
    """Test Popularity-Based algorithms"""
    print("\n" + "=" * 80)
    print("TESTING: POPULARITY-BASED RECOMMENDATIONS")
    print("=" * 80)
    
    try:
        from algorithms.popularity_based import (
            PopularityBasedEngine, CustomerSegmentationEngine, TrendingProductAnalyzer
        )
        results.add_result("Import Popularity-Based", True)
    except ImportError as e:
        results.add_result("Import Popularity-Based", False, str(e))
        return
    
    # Test Customer Segmentation
    print("\n--- Testing Customer Segmentation ---")
    try:
        seg_engine = CustomerSegmentationEngine()
        segments = seg_engine.create_income_segments()
        
        assert len(segments) > 0, "Should create segments"
        
        # Test segment assignment
        test_customers = [
            {'city': 'Karachi', 'income': 400000},
            {'city': 'Lahore', 'income': 350000},
            {'city': 'Islamabad', 'income': 450000},
            {'city': 'Multan', 'income': 250000}
        ]
        
        for customer in test_customers:
            segment = seg_engine.assign_customer_segment(customer)
            assert segment is not None, f"Should assign segment for {customer}"
        
        results.add_result("Customer Segmentation", True, metrics={
            "segments_created": len(segments),
            "segment_names": ", ".join(segments.keys())
        })
    except Exception as e:
        results.add_result("Customer Segmentation", False, str(e))
    
    # Test Popularity Engine
    print("\n--- Testing Popularity Engine ---")
    try:
        pop_engine = PopularityBasedEngine(
            min_sales_threshold=5,
            popularity_weight=0.4,
            trend_weight=0.3,
            segment_weight=0.3
        )
        
        # Create sample data for testing
        sample_sales = pd.DataFrame({
            'product_id': ['p1', 'p1', 'p2', 'p2', 'p3'] * 20,
            'sale_date': [datetime.now() - timedelta(days=i) for i in range(100)],
            'quantity': np.random.randint(1, 10, 100),
            'amount': np.random.uniform(1000, 50000, 100)
        })
        
        sample_products = pd.DataFrame({
            'product_id': ['p1', 'p2', 'p3'],
            'category_id': ['cat1', 'cat1', 'cat2']
        })
        
        sample_customers = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c3'],
            'city': ['Karachi', 'Lahore', 'Islamabad'],
            'income_bracket': ['300k-500k PKR', '200k-400k PKR', '400k-600k PKR']
        })
        
        # Add customer_id to sales
        sample_sales['customer_id'] = np.random.choice(['c1', 'c2', 'c3'], 100)
        
        metrics = pop_engine.train(sample_sales, sample_products, sample_customers)
        
        assert pop_engine.is_trained, "Engine should be trained"
        
        results.add_result("Popularity Engine Training", True, metrics={
            "training_time": f"{metrics.get('training_time_seconds', 0):.2f}s",
            "products_analyzed": metrics.get('n_products_analyzed', 0),
            "trending_products": metrics.get('n_trending_products', 0)
        })
    except Exception as e:
        results.add_result("Popularity Engine Training", False, str(e))


# =============================================================================
# TEST SUITE 5: ML RECOMMENDATION SERVICE (HYBRID)
# =============================================================================

def test_ml_recommendation_service(results: TestResults):
    """Test ML Recommendation Service (Hybrid Ensemble)"""
    print("\n" + "=" * 80)
    print("TESTING: ML RECOMMENDATION SERVICE (HYBRID)")
    print("=" * 80)
    
    try:
        from algorithms.ml_recommendation_service import MLRecommendationService, get_ml_service
        results.add_result("Import ML Service", True)
    except ImportError as e:
        results.add_result("Import ML Service", False, str(e))
        return
    
    # Test ML Service Initialization
    print("\n--- Testing ML Service Initialization ---")
    try:
        ml_service = MLRecommendationService()
        
        assert ml_service is not None, "ML Service should initialize"
        assert not ml_service.is_trained, "Should not be trained initially"
        
        results.add_result("ML Service Initialization", True)
    except Exception as e:
        results.add_result("ML Service Initialization", False, str(e))
        return
    
    # Test Data Loading
    print("\n--- Testing Data Loading ---")
    try:
        interactions_df = ml_service.load_interaction_data(time_filter='30days', limit=1000)
        
        if len(interactions_df) > 0:
            results.add_result("ML Service Data Loading", True, metrics={
                "interactions_loaded": len(interactions_df),
                "unique_users": interactions_df['user_id'].nunique(),
                "unique_items": interactions_df['item_id'].nunique()
            })
        else:
            results.add_result("ML Service Data Loading", False, "No interactions loaded")
            return
    except Exception as e:
        results.add_result("ML Service Data Loading", False, str(e))
        return
    
    # Test Full Training
    print("\n--- Testing Full Model Training ---")
    try:
        start_time = time.time()
        training_results = ml_service.train_all_models(time_filter='30days', force_retrain=True)
        train_time = time.time() - start_time
        
        if 'error' not in training_results:
            results.add_result("ML Service Full Training", True, metrics={
                "total_time": f"{train_time:.2f}s",
                "successful_models": training_results.get('successful_models', 0),
                "total_models": training_results.get('total_models', 0)
            })
            
            # Show individual model status
            for model_name, model_info in training_results.get('models', {}).items():
                status = model_info.get('status', 'unknown')
                print(f"    {model_name}: {status}")
        else:
            results.add_result("ML Service Full Training", False, training_results.get('error'))
    except Exception as e:
        results.add_result("ML Service Full Training", False, str(e))
        return
    
    # Test Hybrid Recommendations
    print("\n--- Testing Hybrid Recommendations ---")
    try:
        # Get a test customer
        conn = ml_service.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT o.unified_customer_id
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            WHERE o.unified_customer_id IS NOT NULL
            GROUP BY o.unified_customer_id
            HAVING COUNT(*) > 5
            LIMIT 1
        """)
        test_customer = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if test_customer:
            customer_id = test_customer['unified_customer_id']
            
            start_time = time.time()
            recommendations = ml_service.get_hybrid_recommendations(
                user_id=customer_id,
                n_recommendations=10
            )
            rec_time = time.time() - start_time
            
            results.add_result("Hybrid Recommendations", True, metrics={
                "customer_id": customer_id[:30] + "...",
                "recommendations_count": len(recommendations),
                "inference_time": f"{rec_time*1000:.2f}ms",
                "top_score": f"{recommendations[0]['score']:.3f}" if recommendations else "N/A"
            })
            
            # Display sample recommendations
            print("\n    Sample Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"      {i}. {rec.get('product_name', 'Unknown')[:40]}... (Score: {rec['score']:.3f})")
        else:
            results.add_result("Hybrid Recommendations", False, "No test customer found")
            
    except Exception as e:
        results.add_result("Hybrid Recommendations", False, str(e))


# =============================================================================
# TEST SUITE 6: API ENDPOINTS
# =============================================================================

def test_api_endpoints(results: TestResults):
    """Test API Endpoints"""
    print("\n" + "=" * 80)
    print("TESTING: API ENDPOINTS")
    print("=" * 80)
    
    import requests
    
    base_url = os.getenv('API_BASE_URL', 'http://localhost:8001')
    
    # Test Health Endpoint
    print("\n--- Testing Health Endpoint ---")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            results.add_result("Health Endpoint", True, metrics={
                "status": data.get('status', 'unknown'),
                "postgres": data.get('postgres_connected', False),
                "redis": data.get('redis_connected', False)
            })
        else:
            results.add_result("Health Endpoint", False, f"Status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        results.add_result("Health Endpoint", False, "Server not running (this is OK for unit tests)")
    except Exception as e:
        results.add_result("Health Endpoint", False, str(e))
    
    # Test Recommendations Endpoint
    print("\n--- Testing Recommendations Endpoints ---")
    try:
        # Popular products
        response = requests.get(f"{base_url}/api/v1/recommendations/popular?limit=5", timeout=10)
        if response.status_code == 200:
            data = response.json()
            results.add_result("Popular Products API", True, metrics={
                "count": len(data.get('recommendations', []))
            })
        else:
            results.add_result("Popular Products API", False, f"Status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        results.add_result("Popular Products API", False, "Server not running")
    except Exception as e:
        results.add_result("Popular Products API", False, str(e))


# =============================================================================
# TEST SUITE 7: EDGE CASES & PERFORMANCE
# =============================================================================

def test_edge_cases(results: TestResults):
    """Test Edge Cases and Performance"""
    print("\n" + "=" * 80)
    print("TESTING: EDGE CASES & PERFORMANCE")
    print("=" * 80)
    
    from algorithms.collaborative_filtering import CollaborativeFilteringEngine, UserItemMatrix
    
    # Test 1: Empty Data
    print("\n--- Testing Empty Data Handling ---")
    try:
        empty_data = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
        cf = CollaborativeFilteringEngine()
        metrics = cf.train(empty_data)
        # Should not crash
        results.add_result("Empty Data Handling", True, "Handled gracefully")
    except Exception as e:
        results.add_result("Empty Data Handling", False, str(e))
    
    # Test 2: Single User
    print("\n--- Testing Single User ---")
    try:
        single_user_data = pd.DataFrame({
            'user_id': ['user1'] * 5,
            'item_id': ['item1', 'item2', 'item3', 'item4', 'item5'],
            'rating': [5.0, 4.0, 3.0, 2.0, 1.0],
            'timestamp': [datetime.now()] * 5
        })
        cf = CollaborativeFilteringEngine(min_interactions=1)
        metrics = cf.train(single_user_data)
        results.add_result("Single User Handling", True)
    except Exception as e:
        results.add_result("Single User Handling", False, str(e))
    
    # Test 3: Large Dataset Performance
    print("\n--- Testing Large Dataset Performance ---")
    try:
        n_users = 1000
        n_items = 500
        n_interactions = 10000
        
        large_data = pd.DataFrame({
            'user_id': [f'user_{np.random.randint(0, n_users)}' for _ in range(n_interactions)],
            'item_id': [f'item_{np.random.randint(0, n_items)}' for _ in range(n_interactions)],
            'rating': np.random.uniform(1, 5, n_interactions),
            'timestamp': [datetime.now()] * n_interactions
        })
        
        matrix_handler = UserItemMatrix()
        start_time = time.time()
        matrix = matrix_handler.build_matrix(large_data)
        build_time = time.time() - start_time
        
        results.add_result("Large Dataset Performance", True, metrics={
            "n_interactions": n_interactions,
            "matrix_build_time": f"{build_time:.3f}s",
            "matrix_shape": str(matrix.shape)
        })
    except Exception as e:
        results.add_result("Large Dataset Performance", False, str(e))
    
    # Test 4: Cold Start (New User)
    print("\n--- Testing Cold Start (New User) ---")
    try:
        conn = get_db_connection()
        from algorithms.content_based_filtering import ContentBasedFiltering
        
        cbf = ContentBasedFiltering(pg_conn=conn)
        cbf.build_product_features()
        
        # Request recommendations for non-existent user
        recommendations = cbf.get_recommendations("NEW_USER_DOES_NOT_EXIST", limit=10)
        
        # Should return popular products as fallback
        results.add_result("Cold Start Handling", True, metrics={
            "fallback_recommendations": len(recommendations)
        })
        
        conn.close()
    except Exception as e:
        results.add_result("Cold Start Handling", False, str(e))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_tests():
    """Run all ML model tests"""
    print("\n" + "=" * 80)
    print("🧪 COMPREHENSIVE ML MODEL TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 80)
    
    results = TestResults()
    
    # Run all test suites
    test_collaborative_filtering(results)
    test_content_based_filtering(results)
    test_matrix_factorization(results)
    test_popularity_based(results)
    test_ml_recommendation_service(results)
    test_api_endpoints(results)
    test_edge_cases(results)
    
    # Print summary
    passed, failed = results.summary()
    
    # Save results to file
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results.results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
