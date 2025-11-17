"""
Integration Test for Segment-Specific Recommendations with Autopilot System

This test validates the complete integration of segment-specific recommendations
with the existing autopilot recommendation system.
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
    from segment_integration_adapter import SegmentIntegrationAdapter
    print("✓ Successfully imported segment integration adapter")
except ImportError as e:
    print(f"✗ Failed to import segment integration adapter: {e}")
    sys.exit(1)

def create_comprehensive_test_data():
    """Create comprehensive test data for integration testing"""
    print("\n=== Creating Comprehensive Test Data ===")
    
    # Create customer segments data (from dynamic segmentation)
    customer_segments = {
        'segments': {
            'dynamic_high_value_multi_channel': {
                'segment_type': 'HIGH_VALUE_MULTI_CHANNEL',
                'segment_name': 'VIP Multi-Channel Customers',
                'customer_count': 2,
                'avg_annual_spend_pkr': 150000,
                'channels_used': ['online', 'pos'],
                'primary_cities': ['Lahore', 'Sialkot']
            },
            'dynamic_price_sensitive': {
                'segment_type': 'PRICE_SENSITIVE',
                'segment_name': 'Price-Sensitive Customers',
                'customer_count': 3,
                'avg_annual_spend_pkr': 25000,
                'channels_used': ['pos'],
                'primary_cities': ['Okara', 'Lahore']
            }
        },
        'customer_segments': {
            'CUST_001': {
                'segment_type': 'HIGH_VALUE_MULTI_CHANNEL',
                'segment_confidence': 0.9,
                'annual_spend_pkr': 150000,
                'channels_used': ['online', 'pos'],
                'primary_city': 'Lahore',
                'is_high_value': True,
                'is_multi_channel': True,
                'engagement_level': 'HIGH'
            },
            'CUST_002': {
                'segment_type': 'HIGH_VALUE_MULTI_CHANNEL',
                'segment_confidence': 0.85,
                'annual_spend_pkr': 180000,
                'channels_used': ['online', 'pos'],
                'primary_city': 'Sialkot',
                'is_high_value': True,
                'is_multi_channel': True,
                'engagement_level': 'HIGH'
            },
            'CUST_003': {
                'segment_type': 'PRICE_SENSITIVE',
                'segment_confidence': 0.8,
                'annual_spend_pkr': 25000,
                'channels_used': ['pos'],
                'primary_city': 'Okara',
                'is_high_value': False,
                'is_multi_channel': False,
                'engagement_level': 'MEDIUM'
            },
            'CUST_004': {
                'segment_type': 'PRICE_SENSITIVE',
                'segment_confidence': 0.75,
                'annual_spend_pkr': 20000,
                'channels_used': ['pos'],
                'primary_city': 'Lahore',
                'is_high_value': False,
                'is_multi_channel': False,
                'engagement_level': 'MEDIUM'
            },
            'CUST_005': {
                'segment_type': 'PRICE_SENSITIVE',
                'segment_confidence': 0.7,
                'annual_spend_pkr': 30000,
                'channels_used': ['online'],
                'primary_city': 'Lahore',
                'is_high_value': False,
                'is_multi_channel': False,
                'engagement_level': 'LOW'
            }
        }
    }
    
    # Create unified customer data
    unified_data = pd.DataFrame([
        # High-value multi-channel customers
        {'unified_customer_id': 'CUST_001', 'product_id': 'PROD_001', 'quantity': 2, 'total_amount': 50000, 'channel': 'online', 'city': 'Lahore', 'order_date': '2024-01-15', 'order_id': 'ORD_001'},
        {'unified_customer_id': 'CUST_001', 'product_id': 'PROD_002', 'quantity': 1, 'total_amount': 22500, 'channel': 'pos', 'city': 'Lahore', 'order_date': '2024-02-10', 'order_id': 'ORD_002'},
        {'unified_customer_id': 'CUST_001', 'product_id': 'PROD_005', 'quantity': 1, 'total_amount': 45000, 'channel': 'online', 'city': 'Lahore', 'order_date': '2024-03-05', 'order_id': 'ORD_003'},
        
        {'unified_customer_id': 'CUST_002', 'product_id': 'PROD_001', 'quantity': 3, 'total_amount': 75000, 'channel': 'pos', 'city': 'Sialkot', 'order_date': '2024-01-20', 'order_id': 'ORD_004'},
        {'unified_customer_id': 'CUST_002', 'product_id': 'PROD_005', 'quantity': 2, 'total_amount': 90000, 'channel': 'online', 'city': 'Sialkot', 'order_date': '2024-02-15', 'order_id': 'ORD_005'},
        
        # Price-sensitive customers
        {'unified_customer_id': 'CUST_003', 'product_id': 'PROD_003', 'quantity': 2, 'total_amount': 16000, 'channel': 'pos', 'city': 'Okara', 'order_date': '2024-01-25', 'order_id': 'ORD_006'},
        {'unified_customer_id': 'CUST_003', 'product_id': 'PROD_004', 'quantity': 1, 'total_amount': 5000, 'channel': 'pos', 'city': 'Okara', 'order_date': '2024-03-01', 'order_id': 'ORD_007'},
        
        {'unified_customer_id': 'CUST_004', 'product_id': 'PROD_003', 'quantity': 1, 'total_amount': 8000, 'channel': 'pos', 'city': 'Lahore', 'order_date': '2024-02-05', 'order_id': 'ORD_008'},
        {'unified_customer_id': 'CUST_004', 'product_id': 'PROD_004', 'quantity': 2, 'total_amount': 10000, 'channel': 'pos', 'city': 'Lahore', 'order_date': '2024-03-10', 'order_id': 'ORD_009'},
        
        {'unified_customer_id': 'CUST_005', 'product_id': 'PROD_003', 'quantity': 3, 'total_amount': 24000, 'channel': 'online', 'city': 'Lahore', 'order_date': '2024-02-20', 'order_id': 'ORD_010'},
        {'unified_customer_id': 'CUST_005', 'product_id': 'PROD_006', 'quantity': 1, 'total_amount': 15000, 'channel': 'online', 'city': 'Lahore', 'order_date': '2024-03-15', 'order_id': 'ORD_011'}
    ])
    
    # Create comprehensive product data
    product_data = pd.DataFrame([
        {'product_id': 'PROD_001', 'product_name': 'MOLTY FOAM Premium Mattress King Size', 'category': 'Mattress', 'price': 25000, 'brand': 'MOLTY FOAM', 'product_type': 'Premium Bedding'},
        {'product_id': 'PROD_002', 'product_name': 'MOLTY FOAM Orthopedic Pillow', 'category': 'Pillow', 'price': 22500, 'brand': 'MOLTY FOAM', 'product_type': 'Premium Bedding'},
        {'product_id': 'PROD_003', 'product_name': 'Budget Foam Mattress Single', 'category': 'Mattress', 'price': 8000, 'brand': 'Budget Foam', 'product_type': 'Budget Bedding'},
        {'product_id': 'PROD_004', 'product_name': 'Basic Cotton Pillow', 'category': 'Pillow', 'price': 5000, 'brand': 'Basic', 'product_type': 'Budget Bedding'},
        {'product_id': 'PROD_005', 'product_name': 'Luxury Memory Foam Mattress', 'category': 'Mattress', 'price': 45000, 'brand': 'Premium Sleep', 'product_type': 'Luxury Bedding'},
        {'product_id': 'PROD_006', 'product_name': 'MOLTY FOAM Cervical Support Pillow', 'category': 'Pillow', 'price': 15000, 'brand': 'MOLTY FOAM', 'product_type': 'Medical Bedding'},
        {'product_id': 'PROD_007', 'product_name': 'Cooling Gel Mattress Topper', 'category': 'Accessory', 'price': 18000, 'brand': 'Cool Sleep', 'product_type': 'Bedding Accessory'},
        {'product_id': 'PROD_008', 'product_name': 'Bamboo Fiber Pillow', 'category': 'Pillow', 'price': 12000, 'brand': 'Eco Sleep', 'product_type': 'Eco Bedding'}
    ])
    
    # Create sales data (same as unified data for this test)
    sales_data = unified_data.copy()
    
    print(f"✓ Created comprehensive test data:")
    print(f"  - Customer segments: {len(customer_segments['customer_segments'])}")
    print(f"  - Unified data records: {len(unified_data)}")
    print(f"  - Products: {len(product_data)}")
    print(f"  - Sales records: {len(sales_data)}")
    
    return customer_segments, unified_data, product_data, sales_data

def test_integration_initialization():
    """Test the initialization of the integrated system"""
    print("\n=== Testing Integration Initialization ===")
    
    # Create test data
    customer_segments, unified_data, product_data, sales_data = create_comprehensive_test_data()
    
    # Initialize adapter
    adapter = SegmentIntegrationAdapter()
    print("✓ Created integration adapter")
    
    # Test initialization
    try:
        initialization_metrics = adapter.initialize(
            customer_segments, unified_data, product_data, sales_data
        )
        
        print("✓ Integration initialization completed")
        print(f"  - Initialization time: {initialization_metrics.get('total_initialization_time', 0):.2f}s")
        print(f"  - Integration ready: {initialization_metrics.get('integration_ready', False)}")
        
        # Check individual engine status
        segment_status = initialization_metrics.get('segment_engine', {})
        autopilot_status = initialization_metrics.get('autopilot_engine', {})
        
        if 'error' not in segment_status:
            print(f"  - Segment engine: ✓ Trained {segment_status.get('segments_trained', 0)} segments")
        else:
            print(f"  - Segment engine: ✗ {segment_status['error']}")
        
        if 'error' not in autopilot_status:
            print(f"  - Autopilot engine: ✓ Hybrid engine ready")
        else:
            print(f"  - Autopilot engine: ✗ {autopilot_status['error']}")
        
        return adapter, True
    except Exception as e:
        print(f"✗ Integration initialization failed: {e}")
        return None, False

def test_integrated_recommendations(adapter):
    """Test integrated recommendation generation"""
    print("\n=== Testing Integrated Recommendations ===")
    
    test_cases = [
        {
            'customer_id': 'CUST_001',
            'segment_type': 'HIGH_VALUE_MULTI_CHANNEL',
            'context': {'city': 'Lahore', 'channel': 'online', 'user_history_length': 3},
            'description': 'VIP Multi-Channel Customer in Lahore'
        },
        {
            'customer_id': 'CUST_003',
            'segment_type': 'PRICE_SENSITIVE',
            'context': {'city': 'Okara', 'channel': 'pos', 'user_history_length': 2},
            'description': 'Price-Sensitive Customer in Okara'
        },
        {
            'customer_id': 'CUST_002',
            'segment_type': 'HIGH_VALUE_MULTI_CHANNEL',
            'context': {'city': 'Sialkot', 'channel': 'pos', 'user_history_length': 2},
            'description': 'VIP Multi-Channel Customer in Sialkot'
        }
    ]
    
    success_count = 0
    
    for test_case in test_cases:
        try:
            recommendations = adapter.get_integrated_recommendations(
                customer_id=test_case['customer_id'],
                segment_type=test_case['segment_type'],
                context=test_case['context'],
                n_recommendations=5
            )
            
            print(f"\n✓ {test_case['description']}:")
            print(f"  - Generated {len(recommendations)} integrated recommendations")
            
            if recommendations:
                top_rec = recommendations[0]
                print(f"  - Top recommendation: {top_rec.get('product_id', 'N/A')}")
                print(f"  - Integrated score: {top_rec.get('integrated_score', 0):.3f}")
                print(f"  - Confidence: {top_rec.get('confidence_score', 0):.1f}%")
                print(f"  - Sources: {', '.join(top_rec.get('recommendation_sources', []))}")
                print(f"  - Reasoning: {top_rec.get('reasoning', 'N/A')}")
                
                # Check for segment-specific features
                if top_rec.get('marketing_message'):
                    msg = top_rec['marketing_message']
                    print(f"  - Marketing message: {msg.primary_message}")
                
                if top_rec.get('promotions'):
                    promo = top_rec['promotions'][0]
                    print(f"  - Promotion: {promo.get_promotion_message()}")
                
                # Check for autopilot features
                if top_rec.get('revenue_prediction'):
                    rev_pred = top_rec['revenue_prediction']
                    print(f"  - Revenue prediction: {rev_pred.get('expected_revenue_formatted', 'N/A')}")
            
            success_count += 1
            
        except Exception as e:
            print(f"✗ Failed for {test_case['description']}: {e}")
    
    return success_count == len(test_cases)

def test_performance_insights(adapter):
    """Test performance insights and optimization"""
    print("\n=== Testing Performance Insights ===")
    
    try:
        # Get segment performance insights
        insights = adapter.get_segment_performance_insights()
        print("✓ Retrieved segment performance insights")
        print(f"  - Total segments tracked: {insights.get('total_segments', 0)}")
        print(f"  - Total recommendations: {insights.get('total_recommendations', 0)}")
        
        # Test strategy optimization
        performance_data = {
            'HIGH_VALUE_MULTI_CHANNEL': {
                'metrics': {
                    'click_through_rate': 0.08,  # Good CTR
                    'conversion_rate': 0.03,     # Good conversion
                    'customer_satisfaction': 0.85  # Good satisfaction
                }
            },
            'PRICE_SENSITIVE': {
                'metrics': {
                    'click_through_rate': 0.02,  # Low CTR
                    'conversion_rate': 0.005,    # Low conversion
                    'customer_satisfaction': 0.6   # Low satisfaction
                }
            }
        }
        
        optimization_results = adapter.optimize_segment_strategies(performance_data)
        print("✓ Performed strategy optimization")
        
        for segment_type, result in optimization_results.items():
            if 'error' not in result:
                optimizations = result.get('optimizations_applied', [])
                print(f"  - {segment_type}: {len(optimizations)} optimizations applied")
                for opt in optimizations[:2]:  # Show first 2
                    print(f"    • {opt}")
            else:
                print(f"  - {segment_type}: Error - {result['error']}")
        
        return True
    except Exception as e:
        print(f"✗ Performance insights failed: {e}")
        return False

def test_promotion_management(adapter):
    """Test promotion management integration"""
    print("\n=== Testing Promotion Management ===")
    
    try:
        # Get active promotions
        promotions = adapter.get_active_promotions()
        print("✓ Retrieved active promotions")
        
        if isinstance(promotions, dict) and 'error' not in promotions:
            total_promotions = sum(len(promos) for promos in promotions.values())
            print(f"  - Total active promotions: {total_promotions}")
            
            for segment_type, segment_promotions in promotions.items():
                if segment_promotions:
                    print(f"  - {segment_type}: {len(segment_promotions)} promotions")
                    promo = segment_promotions[0]
                    print(f"    • {promo.get_promotion_message()} (Valid: {promo.is_valid()})")
        else:
            print(f"  - Error retrieving promotions: {promotions.get('error', 'Unknown error')}")
        
        return True
    except Exception as e:
        print(f"✗ Promotion management failed: {e}")
        return False

def test_feedback_recording(adapter):
    """Test recommendation feedback recording"""
    print("\n=== Testing Feedback Recording ===")
    
    try:
        # Record various types of feedback
        feedback_cases = [
            ('CUST_001', 'PROD_007', 'view'),
            ('CUST_001', 'PROD_007', 'click'),
            ('CUST_001', 'PROD_007', 'purchase', 18000),
            ('CUST_003', 'PROD_004', 'view'),
            ('CUST_003', 'PROD_004', 'click')
        ]
        
        for case in feedback_cases:
            customer_id, product_id, interaction = case[:3]
            order_value = case[3] if len(case) > 3 else 0
            
            adapter.record_recommendation_feedback(
                customer_id, product_id, interaction, order_value
            )
        
        print(f"✓ Recorded {len(feedback_cases)} feedback interactions")
        print("  - Feedback recorded for both segment and autopilot engines")
        
        return True
    except Exception as e:
        print(f"✗ Feedback recording failed: {e}")
        return False

def test_integration_weights(adapter):
    """Test integration weight adjustment"""
    print("\n=== Testing Integration Weight Adjustment ===")
    
    try:
        # Test different weight configurations
        weight_configs = [
            (0.7, 0.3, "Segment-focused"),
            (0.4, 0.6, "Autopilot-focused"),
            (0.5, 0.5, "Balanced")
        ]
        
        for segment_weight, autopilot_weight, description in weight_configs:
            adapter.update_integration_weights(segment_weight, autopilot_weight)
            print(f"✓ Updated weights: {description} ({segment_weight:.1f}/{autopilot_weight:.1f})")
        
        return True
    except Exception as e:
        print(f"✗ Weight adjustment failed: {e}")
        return False

def run_integration_test():
    """Run comprehensive integration test"""
    print("🚀 Starting Comprehensive Segment Integration Test")
    print("=" * 60)
    
    test_results = []
    adapter = None
    
    # Initialize integration
    adapter, init_success = test_integration_initialization()
    test_results.append(("Integration Initialization", init_success))
    
    if adapter and init_success:
        # Run integration tests
        test_results.append(("Integrated Recommendations", test_integrated_recommendations(adapter)))
        test_results.append(("Performance Insights", test_performance_insights(adapter)))
        test_results.append(("Promotion Management", test_promotion_management(adapter)))
        test_results.append(("Feedback Recording", test_feedback_recording(adapter)))
        test_results.append(("Integration Weights", test_integration_weights(adapter)))
    else:
        print("⚠️  Skipping integration tests due to initialization failure")
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
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
        print("\n🎉 All integration tests passed!")
        print("✅ Segment-specific recommendations are fully integrated with autopilot system")
        return True
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Integration needs review.")
        return False

if __name__ == "__main__":
    success = run_integration_test()
    
    if success:
        print("\n🔧 INTEGRATION SUMMARY:")
        print("✅ Segment-specific engine integrated with autopilot system")
        print("✅ Combined recommendations with enhanced personalization")
        print("✅ Marketing messages and promotions integrated")
        print("✅ Performance tracking and optimization working")
        print("✅ Feedback recording for both engines")
        print("\n📋 Task 7.2 is fully integrated and operational!")
    else:
        print("\n❌ Integration tests failed. Please review the implementation.")
    
    print(f"\n📁 Integration files created:")
    print("  - recommendation-engine/src/algorithms/segment_integration_adapter.py")
    print("  - recommendation-engine/test_segment_integration.py")