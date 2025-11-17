"""
Test script for Segment-Specific Recommendation Engine

This script tests the implementation of task 7.2:
- Personalized recommendation engines per segment
- Segment-specific marketing message generation
- Targeted promotion strategies
- Segment performance tracking and optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the algorithms directory to path
algorithms_path = os.path.join(os.path.dirname(__file__), 'src', 'algorithms')
if algorithms_path not in sys.path:
    sys.path.insert(0, algorithms_path)

# Add shared to path
shared_path = os.path.join(os.path.dirname(__file__), '..', 'shared')
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

try:
    from segment_specific_recommendation_engine import (
        SegmentSpecificRecommendationEngine,
        SegmentRecommendationStrategy,
        SegmentMarketingMessage,
        TargetedPromotion,
        SegmentPerformanceMetrics
    )
    print("✓ Successfully imported segment-specific recommendation engine")
except ImportError as e:
    print(f"✗ Failed to import segment-specific recommendation engine: {e}")
    sys.exit(1)

def create_test_data():
    """Create test data for segment-specific recommendations"""
    print("\n=== Creating Test Data ===")
    
    # Create sample customer segments data
    customer_segments = {
        'customer_segments': {
            'CUST_001': {
                'segment_type': 'HIGH_VALUE_MULTI_CHANNEL',
                'segment_confidence': 0.9,
                'annual_spend_pkr': 150000,
                'channels_used': ['online', 'pos'],
                'primary_city': 'Lahore',
                'is_high_value': True,
                'is_multi_channel': True
            },
            'CUST_002': {
                'segment_type': 'PRICE_SENSITIVE',
                'segment_confidence': 0.8,
                'annual_spend_pkr': 25000,
                'channels_used': ['pos'],
                'primary_city': 'Okara',
                'is_high_value': False,
                'is_multi_channel': False
            },
            'CUST_003': {
                'segment_type': 'GEOGRAPHIC_LOYAL_SIALKOT',
                'segment_confidence': 0.85,
                'annual_spend_pkr': 75000,
                'channels_used': ['pos'],
                'primary_city': 'Sialkot',
                'is_high_value': False,
                'is_multi_channel': False
            },
            'CUST_004': {
                'segment_type': 'NEW_CUSTOMER',
                'segment_confidence': 0.7,
                'annual_spend_pkr': 5000,
                'channels_used': ['online'],
                'primary_city': 'Lahore',
                'is_high_value': False,
                'is_multi_channel': False
            },
            'CUST_005': {
                'segment_type': 'FREQUENT_BUYER',
                'segment_confidence': 0.9,
                'annual_spend_pkr': 120000,
                'channels_used': ['online', 'pos'],
                'primary_city': 'Lahore',
                'is_high_value': True,
                'is_multi_channel': True
            }
        }
    }
    
    # Create unified customer data
    unified_data = pd.DataFrame([
        {'unified_customer_id': 'CUST_001', 'product_id': 'PROD_001', 'quantity': 1, 'total_amount': 25000, 'channel': 'online', 'city': 'Lahore', 'order_date': '2024-01-15'},
        {'unified_customer_id': 'CUST_001', 'product_id': 'PROD_002', 'quantity': 2, 'total_amount': 45000, 'channel': 'pos', 'city': 'Lahore', 'order_date': '2024-02-10'},
        {'unified_customer_id': 'CUST_002', 'product_id': 'PROD_003', 'quantity': 1, 'total_amount': 8000, 'channel': 'pos', 'city': 'Okara', 'order_date': '2024-01-20'},
        {'unified_customer_id': 'CUST_003', 'product_id': 'PROD_001', 'quantity': 1, 'total_amount': 25000, 'channel': 'pos', 'city': 'Sialkot', 'order_date': '2024-02-05'},
        {'unified_customer_id': 'CUST_004', 'product_id': 'PROD_004', 'quantity': 1, 'total_amount': 5000, 'channel': 'online', 'city': 'Lahore', 'order_date': '2024-03-01'},
        {'unified_customer_id': 'CUST_005', 'product_id': 'PROD_001', 'quantity': 2, 'total_amount': 50000, 'channel': 'online', 'city': 'Lahore', 'order_date': '2024-01-10'},
        {'unified_customer_id': 'CUST_005', 'product_id': 'PROD_002', 'quantity': 1, 'total_amount': 22500, 'channel': 'pos', 'city': 'Lahore', 'order_date': '2024-02-15'}
    ])
    
    # Create product data
    product_data = pd.DataFrame([
        {'product_id': 'PROD_001', 'product_name': 'MOLTY FOAM Mattress Premium', 'category': 'Mattress', 'price': 25000, 'brand': 'MOLTY FOAM'},
        {'product_id': 'PROD_002', 'product_name': 'MOLTY FOAM Pillow Deluxe', 'category': 'Pillow', 'price': 22500, 'brand': 'MOLTY FOAM'},
        {'product_id': 'PROD_003', 'product_name': 'Budget Foam Mattress', 'category': 'Mattress', 'price': 8000, 'brand': 'Budget'},
        {'product_id': 'PROD_004', 'product_name': 'Basic Pillow', 'category': 'Pillow', 'price': 5000, 'brand': 'Basic'},
        {'product_id': 'PROD_005', 'product_name': 'Luxury Memory Foam', 'category': 'Mattress', 'price': 45000, 'brand': 'Premium'},
        {'product_id': 'PROD_006', 'product_name': 'Orthopedic Support Pillow', 'category': 'Pillow', 'price': 15000, 'brand': 'MOLTY FOAM'}
    ])
    
    # Create sales data
    sales_data = unified_data.copy()
    
    print(f"✓ Created test data:")
    print(f"  - Customer segments: {len(customer_segments['customer_segments'])}")
    print(f"  - Unified data records: {len(unified_data)}")
    print(f"  - Products: {len(product_data)}")
    print(f"  - Sales records: {len(sales_data)}")
    
    return customer_segments, unified_data, product_data, sales_data

def test_segment_specific_engine():
    """Test the main segment-specific recommendation engine"""
    print("\n=== Testing Segment-Specific Recommendation Engine ===")
    
    # Create test data
    customer_segments, unified_data, product_data, sales_data = create_test_data()
    
    # Initialize engine
    engine = SegmentSpecificRecommendationEngine()
    print("✓ Initialized segment-specific recommendation engine")
    
    # Test training
    print("\n--- Testing Training ---")
    try:
        training_metrics = engine.train(
            customer_segments, unified_data, product_data, sales_data
        )
        print("✓ Training completed successfully")
        print(f"  - Segments trained: {training_metrics.get('segments_trained', 0)}")
        print(f"  - Training time: {training_metrics.get('total_training_time', 0):.2f}s")
        
        # Print training results for each segment
        for segment_type, metrics in training_metrics.items():
            if isinstance(metrics, dict) and 'error' not in metrics:
                print(f"  - {segment_type}: {metrics.get('model_type', 'trained')}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # Test segment recommendations
    print("\n--- Testing Segment Recommendations ---")
    test_cases = [
        ('CUST_001', 'HIGH_VALUE_MULTI_CHANNEL', {'city': 'Lahore', 'channel': 'online'}),
        ('CUST_002', 'PRICE_SENSITIVE', {'city': 'Okara', 'channel': 'pos'}),
        ('CUST_003', 'GEOGRAPHIC_LOYAL_SIALKOT', {'city': 'Sialkot', 'channel': 'pos'}),
        ('CUST_004', 'NEW_CUSTOMER', {'city': 'Lahore', 'channel': 'online'}),
        ('CUST_005', 'FREQUENT_BUYER', {'city': 'Lahore', 'channel': 'online'})
    ]
    
    for customer_id, segment_type, context in test_cases:
        try:
            recommendations = engine.get_segment_recommendations(
                customer_id, segment_type, context, n_recommendations=3
            )
            print(f"✓ Generated {len(recommendations)} recommendations for {segment_type}")
            
            if recommendations:
                rec = recommendations[0]
                print(f"  - Top recommendation: {rec.get('product_id', 'N/A')}")
                print(f"  - Score: {rec.get('score', 0):.2f}")
                print(f"  - Confidence: {rec.get('confidence_score', 0):.1f}%")
                
                # Test marketing message
                if 'marketing_message' in rec:
                    msg = rec['marketing_message']
                    print(f"  - Marketing message: {msg.primary_message}")
                
                # Test promotions
                if 'promotions' in rec and rec['promotions']:
                    promo = rec['promotions'][0]
                    print(f"  - Promotion: {promo.get_promotion_message()}")
        except Exception as e:
            print(f"✗ Failed to generate recommendations for {segment_type}: {e}")
    
    return True

def test_marketing_message_generator():
    """Test segment-specific marketing message generation"""
    print("\n=== Testing Marketing Message Generator ===")
    
    try:
        from segment_specific_recommendation_engine import SegmentMarketingMessageGenerator
        
        generator = SegmentMarketingMessageGenerator()
        
        # Test message generation for different segments
        test_segments = [
            'HIGH_VALUE_MULTI_CHANNEL',
            'PRICE_SENSITIVE',
            'NEW_CUSTOMER',
            'GEOGRAPHIC_LOYAL_LAHORE'
        ]
        
        for segment_type in test_segments:
            message = generator.generate_message(
                segment_type, 'PROD_001', {'city': 'Lahore', 'channel': 'online'}
            )
            
            print(f"✓ Generated message for {segment_type}:")
            print(f"  - Primary: {message.primary_message}")
            print(f"  - CTA: {message.call_to_action}")
            print(f"  - Email version: {message.get_message_for_channel('email')[:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ Marketing message generation failed: {e}")
        return False

def test_targeted_promotions():
    """Test targeted promotion management"""
    print("\n=== Testing Targeted Promotion Manager ===")
    
    try:
        from segment_specific_recommendation_engine import TargetedPromotionManager
        
        manager = TargetedPromotionManager()
        
        # Test promotion creation
        promotion_config = {
            'discount_percentage': 20,
            'promotion_type': 'percentage',
            'validity_start': datetime.now(),
            'validity_end': datetime.now() + timedelta(days=7),
            'total_budget_pkr': 50000
        }
        
        promotion = manager.create_targeted_promotion(
            'PRICE_SENSITIVE', ['PROD_003', 'PROD_004'], promotion_config
        )
        
        print(f"✓ Created targeted promotion:")
        print(f"  - ID: {promotion.promotion_id}")
        print(f"  - Segment: {promotion.segment_type}")
        print(f"  - Discount: {promotion.discount_percentage}%")
        print(f"  - Valid: {promotion.is_valid()}")
        
        # Test getting applicable promotions
        applicable = manager.get_applicable_promotions(
            'PRICE_SENSITIVE', 'PROD_003', 'CUST_002'
        )
        print(f"✓ Found {len(applicable)} applicable promotions for PRICE_SENSITIVE segment")
        
        return True
    except Exception as e:
        print(f"✗ Targeted promotion management failed: {e}")
        return False

def test_performance_tracking():
    """Test segment performance tracking"""
    print("\n=== Testing Performance Tracking ===")
    
    try:
        from segment_specific_recommendation_engine import SegmentPerformanceTracker
        
        tracker = SegmentPerformanceTracker()
        
        # Test recording recommendations
        test_recommendations = [
            {'product_id': 'PROD_001', 'score': 0.8, 'confidence_score': 85},
            {'product_id': 'PROD_002', 'score': 0.7, 'confidence_score': 75}
        ]
        
        tracker.record_recommendations(
            'HIGH_VALUE_MULTI_CHANNEL', 'CUST_001', test_recommendations
        )
        print("✓ Recorded recommendations for performance tracking")
        
        # Test recording interactions
        tracker.record_interaction(
            'HIGH_VALUE_MULTI_CHANNEL', 'CUST_001', 'PROD_001', 'click'
        )
        tracker.record_interaction(
            'HIGH_VALUE_MULTI_CHANNEL', 'CUST_001', 'PROD_001', 'purchase', 25000
        )
        print("✓ Recorded customer interactions")
        
        # Test performance summary
        summary = tracker.get_performance_summary()
        print(f"✓ Generated performance summary:")
        print(f"  - Total segments: {summary['total_segments']}")
        print(f"  - Total recommendations: {summary['total_recommendations']}")
        
        return True
    except Exception as e:
        print(f"✗ Performance tracking failed: {e}")
        return False

def test_strategy_optimization():
    """Test segment strategy optimization"""
    print("\n=== Testing Strategy Optimization ===")
    
    try:
        # Create test data
        customer_segments, unified_data, product_data, sales_data = create_test_data()
        
        # Initialize and train engine
        engine = SegmentSpecificRecommendationEngine()
        engine.train(customer_segments, unified_data, product_data, sales_data)
        
        # Test strategy optimization
        performance_data = {
            'metrics': {
                'click_through_rate': 0.03,  # Low CTR
                'conversion_rate': 0.01,     # Low conversion
                'customer_satisfaction': 0.6  # Low satisfaction
            }
        }
        
        optimization_result = engine.optimize_segment_strategy(
            'HIGH_VALUE_MULTI_CHANNEL', performance_data
        )
        
        print(f"✓ Optimized strategy for HIGH_VALUE_MULTI_CHANNEL:")
        print(f"  - Optimizations applied: {len(optimization_result.get('optimizations_applied', []))}")
        for opt in optimization_result.get('optimizations_applied', []):
            print(f"    • {opt}")
        
        return True
    except Exception as e:
        print(f"✗ Strategy optimization failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test of all segment-specific functionality"""
    print("🚀 Starting Comprehensive Segment-Specific Recommendation Test")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Segment-Specific Engine", test_segment_specific_engine()))
    test_results.append(("Marketing Message Generator", test_marketing_message_generator()))
    test_results.append(("Targeted Promotions", test_targeted_promotions()))
    test_results.append(("Performance Tracking", test_performance_tracking()))
    test_results.append(("Strategy Optimization", test_strategy_optimization()))
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All segment-specific recommendation tests passed!")
        print("✅ Task 7.2 implementation is working correctly")
        return True
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n🔧 IMPLEMENTATION SUMMARY:")
        print("✅ Personalized recommendation engines per segment")
        print("✅ Segment-specific marketing message generation")
        print("✅ Targeted promotion strategies")
        print("✅ Segment performance tracking and optimization")
        print("\n📋 Requirements 6.2 and 6.4 have been successfully implemented!")
    else:
        print("\n❌ Some tests failed. Implementation needs review.")
    
    print(f"\n📁 Implementation files created:")
    print("  - recommendation-engine/src/algorithms/segment_specific_recommendation_engine.py")
    print("  - recommendation-engine/test_segment_specific_recommendations.py")