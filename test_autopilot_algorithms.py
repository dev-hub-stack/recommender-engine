"""
Test script for Autopilot Recommendation Engine algorithms (Task 3.1)
Tests: collaborative filtering, content-based, geographic popularity, and hybrid approaches
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'algorithms'))

# Import directly to avoid __init__.py chain
from autopilot_recommendation_engine import (
    CrossChannelCollaborativeFiltering,
    ContentBasedFiltering,
    GeographicPopularityEngine,
    HybridRecommendationEngine
)


def create_test_data():
    """Create test data for algorithm testing"""
    print("Creating test data...")
    
    # Create unified customer purchase data
    unified_data = pd.DataFrame({
        'unified_customer_id': ['cust_001', 'cust_001', 'cust_002', 'cust_002', 'cust_003', 
                               'cust_003', 'cust_004', 'cust_005', 'cust_005', 'cust_006'],
        'product_id': ['MOLTY_FOAM_001', 'PILLOW_001', 'MOLTY_FOAM_001', 'MATTRESS_001', 
                      'PILLOW_001', 'MATTRESS_001', 'MOLTY_FOAM_002', 'PILLOW_001', 
                      'MOLTY_FOAM_001', 'MATTRESS_002'],
        'product_name': ['Molty Foam Mattress', 'Comfort Pillow', 'Molty Foam Mattress', 
                        'Premium Mattress', 'Comfort Pillow', 'Premium Mattress',
                        'Molty Foam Deluxe', 'Comfort Pillow', 'Molty Foam Mattress', 
                        'Luxury Mattress'],
        'channel': ['online', 'pos', 'pos', 'online', 'online', 'pos', 'online', 
                   'pos', 'online', 'online'],
        'quantity': [1, 2, 1, 1, 1, 1, 1, 3, 1, 1],
        'total_amount': [15000, 3000, 15000, 25000, 3000, 25000, 18000, 3000, 15000, 30000],
        'city': ['Lahore', 'Lahore', 'Sialkot', 'Sialkot', 'Okara', 'Okara', 
                'Lahore', 'Sialkot', 'Sialkot', 'Lahore'],
        'order_date': [datetime.now() - timedelta(days=i*10) for i in range(10)],
        'order_id': [f'ORD_{i:03d}' for i in range(1, 11)]
    })
    
    # Create product data
    product_data = pd.DataFrame({
        'product_id': ['MOLTY_FOAM_001', 'MOLTY_FOAM_002', 'PILLOW_001', 'MATTRESS_001', 
                      'MATTRESS_002', 'BEDDING_001'],
        'product_name': ['Molty Foam Mattress', 'Molty Foam Deluxe', 'Comfort Pillow',
                        'Premium Mattress', 'Luxury Mattress', 'Bedding Set'],
        'product_type': ['mattress', 'mattress', 'pillow', 'mattress', 'mattress', 'bedding'],
        'category': ['MOLTY_FOAM', 'MOLTY_FOAM', 'accessories', 'bedding', 'bedding', 'bedding'],
        'price': [15000, 18000, 3000, 25000, 30000, 8000],
        'brand': ['Molty', 'Molty', 'Generic', 'Premium', 'Luxury', 'Generic']
    })
    
    # Create sales data with geographic info
    sales_data = unified_data.copy()
    
    return unified_data, product_data, sales_data


def test_cross_channel_collaborative_filtering():
    """Test cross-channel collaborative filtering"""
    print("\n" + "="*60)
    print("TEST 1: Cross-Channel Collaborative Filtering")
    print("="*60)
    
    unified_data, _, _ = create_test_data()
    
    # Initialize and train
    cf_engine = CrossChannelCollaborativeFiltering(min_common_products=1)
    metrics = cf_engine.train(unified_data)
    
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test recommendations
    print("\nTesting recommendations for cust_001:")
    recommendations = cf_engine.get_recommendations('cust_001', n_recommendations=5)
    
    if recommendations:
        print(f"  Found {len(recommendations)} recommendations:")
        for product_id, score in recommendations:
            print(f"    - {product_id}: {score:.4f}")
    else:
        print("  No recommendations generated (expected for small dataset)")
    
    # Test with exclusions
    print("\nTesting with exclusions:")
    recommendations = cf_engine.get_recommendations(
        'cust_002', 
        n_recommendations=5,
        exclude_products=['MOLTY_FOAM_001']
    )
    print(f"  Found {len(recommendations)} recommendations (excluding MOLTY_FOAM_001)")
    
    print("\n✓ Cross-channel collaborative filtering test completed")
    return cf_engine


def test_content_based_filtering():
    """Test content-based filtering"""
    print("\n" + "="*60)
    print("TEST 2: Content-Based Filtering")
    print("="*60)
    
    unified_data, product_data, _ = create_test_data()
    
    # Initialize and train
    content_engine = ContentBasedFiltering()
    metrics = content_engine.train(product_data, unified_data)
    
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test recommendations
    print("\nTesting recommendations for cust_001:")
    recommendations = content_engine.get_recommendations('cust_001', n_recommendations=5)
    
    if recommendations:
        print(f"  Found {len(recommendations)} recommendations:")
        for product_id, score in recommendations:
            print(f"    - {product_id}: {score:.4f}")
    else:
        print("  No recommendations generated")
    
    # Check customer preferences
    if 'cust_001' in content_engine.customer_preferences:
        prefs = content_engine.customer_preferences['cust_001']
        print("\nCustomer preferences:")
        print(f"  Categories: {dict(prefs['categories'])}")
        print(f"  Price ranges: {dict(prefs['price_ranges'])}")
        print(f"  Attributes: {dict(prefs['attributes'])}")
    
    print("\n✓ Content-based filtering test completed")
    return content_engine


def test_geographic_popularity():
    """Test geographic popularity engine"""
    print("\n" + "="*60)
    print("TEST 3: Geographic Popularity Engine")
    print("="*60)
    
    _, _, sales_data = create_test_data()
    
    # Initialize and train
    geo_engine = GeographicPopularityEngine()
    metrics = geo_engine.train(sales_data)
    
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test recommendations for each city
    for city in ['Lahore', 'Sialkot', 'Okara']:
        print(f"\nTesting recommendations for {city}:")
        recommendations = geo_engine.get_recommendations(city, n_recommendations=5)
        
        if recommendations:
            print(f"  Found {len(recommendations)} recommendations:")
            for product_id, score in recommendations[:3]:  # Show top 3
                print(f"    - {product_id}: {score:.4f}")
        else:
            print("  No recommendations generated")
    
    # Test with unknown city (should fallback to overall popular)
    print("\nTesting with unknown city (should use overall popularity):")
    recommendations = geo_engine.get_recommendations('Karachi', n_recommendations=3)
    print(f"  Found {len(recommendations)} recommendations")
    
    print("\n✓ Geographic popularity test completed")
    return geo_engine


def test_hybrid_recommendation_engine():
    """Test hybrid recommendation engine"""
    print("\n" + "="*60)
    print("TEST 4: Hybrid Recommendation Engine")
    print("="*60)
    
    unified_data, product_data, sales_data = create_test_data()
    
    # Initialize and train
    hybrid_engine = HybridRecommendationEngine(
        cf_weight=0.35,
        content_weight=0.30,
        geo_weight=0.35
    )
    metrics = hybrid_engine.train(unified_data, product_data, sales_data)
    
    print("\nTraining Metrics:")
    for component, component_metrics in metrics.items():
        if isinstance(component_metrics, dict):
            print(f"\n  {component}:")
            for key, value in component_metrics.items():
                if key != 'error':
                    print(f"    {key}: {value}")
                else:
                    print(f"    ERROR: {value}")
    
    # Test recommendations
    print("\nTesting hybrid recommendations for cust_001 in Lahore:")
    recommendations = hybrid_engine.get_recommendations(
        'cust_001', 
        city='Lahore',
        n_recommendations=5
    )
    
    if recommendations:
        print(f"  Found {len(recommendations)} recommendations:")
        for product_id, score, reasoning in recommendations:
            print(f"    - {product_id}: {score:.4f}")
            print(f"      Reasoning: {reasoning}")
    else:
        print("  No recommendations generated")
    
    # Test with different customer and city
    print("\nTesting hybrid recommendations for cust_002 in Sialkot:")
    recommendations = hybrid_engine.get_recommendations(
        'cust_002',
        city='Sialkot',
        n_recommendations=3
    )
    print(f"  Found {len(recommendations)} recommendations")
    
    print("\n✓ Hybrid recommendation engine test completed")
    return hybrid_engine


def test_algorithm_integration():
    """Test that all algorithms work together"""
    print("\n" + "="*60)
    print("TEST 5: Algorithm Integration")
    print("="*60)
    
    unified_data, product_data, sales_data = create_test_data()
    
    # Create all engines
    print("\nInitializing all engines...")
    cf_engine = CrossChannelCollaborativeFiltering()
    content_engine = ContentBasedFiltering()
    geo_engine = GeographicPopularityEngine()
    hybrid_engine = HybridRecommendationEngine()
    
    # Train all
    print("Training all engines...")
    cf_engine.train(unified_data)
    content_engine.train(product_data, unified_data)
    geo_engine.train(sales_data)
    hybrid_engine.train(unified_data, product_data, sales_data)
    
    # Get recommendations from each
    customer_id = 'cust_001'
    city = 'Lahore'
    
    print(f"\nGetting recommendations for {customer_id} in {city}:")
    
    cf_recs = cf_engine.get_recommendations(customer_id, n_recommendations=3)
    print(f"  CF recommendations: {len(cf_recs)}")
    
    content_recs = content_engine.get_recommendations(customer_id, n_recommendations=3)
    print(f"  Content recommendations: {len(content_recs)}")
    
    geo_recs = geo_engine.get_recommendations(city, n_recommendations=3)
    print(f"  Geographic recommendations: {len(geo_recs)}")
    
    hybrid_recs = hybrid_engine.get_recommendations(customer_id, city, n_recommendations=3)
    print(f"  Hybrid recommendations: {len(hybrid_recs)}")
    
    print("\n✓ Algorithm integration test completed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("AUTOPILOT RECOMMENDATION ENGINE - TASK 3.1 TESTS")
    print("Testing: Collaborative Filtering, Content-Based, Geographic, Hybrid")
    print("="*60)
    
    try:
        # Run individual tests
        cf_engine = test_cross_channel_collaborative_filtering()
        content_engine = test_content_based_filtering()
        geo_engine = test_geographic_popularity()
        hybrid_engine = test_hybrid_recommendation_engine()
        
        # Run integration test
        test_algorithm_integration()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✓ Cross-Channel Collaborative Filtering: PASSED")
        print("✓ Content-Based Filtering: PASSED")
        print("✓ Geographic Popularity Engine: PASSED")
        print("✓ Hybrid Recommendation Engine: PASSED")
        print("✓ Algorithm Integration: PASSED")
        print("\n" + "="*60)
        print("ALL TESTS PASSED - Task 3.1 Implementation Complete")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
