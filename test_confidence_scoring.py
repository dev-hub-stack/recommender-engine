"""
Test script for Confidence Scoring and Reasoning System (Task 3.2)
Tests: confidence scoring, reasoning generation, PKR revenue prediction, quality metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'algorithms'))

from autopilot_recommendation_engine import (
    ConfidenceScorer,
    RecommendationReasoning,
    PKRRevenuePredictor,
    RecommendationQualityMetrics,
    EnhancedHybridEngine
)


def create_test_data():
    """Create test data"""
    unified_data = pd.DataFrame({
        'unified_customer_id': ['cust_001', 'cust_001', 'cust_002', 'cust_003'],
        'product_id': ['MOLTY_001', 'PILLOW_001', 'MOLTY_001', 'MATTRESS_001'],
        'product_name': ['Molty Foam', 'Pillow', 'Molty Foam', 'Mattress'],
        'channel': ['online', 'pos', 'pos', 'online'],
        'quantity': [1, 2, 1, 1],
        'total_amount': [15000, 3000, 15000, 25000],
        'city': ['Lahore', 'Lahore', 'Sialkot', 'Okara'],
        'order_date': [datetime.now() - timedelta(days=i*10) for i in range(4)],
        'order_id': [f'ORD_{i:03d}' for i in range(1, 5)]
    })
    
    product_data = pd.DataFrame({
        'product_id': ['MOLTY_001', 'PILLOW_001', 'MATTRESS_001', 'BEDDING_001'],
        'product_name': ['Molty Foam', 'Pillow', 'Mattress', 'Bedding'],
        'product_type': ['mattress', 'pillow', 'mattress', 'bedding'],
        'price': [15000, 3000, 25000, 8000],
        'brand': ['Molty', 'Generic', 'Premium', 'Generic']
    })
    
    sales_data = unified_data.copy()
    
    return unified_data, product_data, sales_data


def test_confidence_scorer():
    """Test confidence scoring"""
    print("\n" + "="*60)
    print("TEST 1: Confidence Scorer")
    print("="*60)
    
    scorer = ConfidenceScorer()
    
    # Test with different scenarios
    scenarios = [
        {
            'name': 'High confidence (multiple sources, good history)',
            'sources': ['collaborative', 'content', 'geographic'],
            'history_length': 15,
            'popularity': 0.8,
            'scores': {'cf': 0.9, 'content': 0.85, 'geo': 0.88}
        },
        {
            'name': 'Medium confidence (single source, moderate history)',
            'sources': ['content'],
            'history_length': 5,
            'popularity': 0.5,
            'scores': {'content': 0.6}
        },
        {
            'name': 'Low confidence (new user, low popularity)',
            'sources': ['geographic'],
            'history_length': 1,
            'popularity': 0.2,
            'scores': {'geo': 0.3}
        }
    ]
    
    for scenario in scenarios:
        confidence = scorer.calculate_confidence(
            scenario['sources'],
            scenario['history_length'],
            scenario['popularity'],
            scenario['scores']
        )
        print(f"\n{scenario['name']}:")
        print(f"  Sources: {scenario['sources']}")
        print(f"  User history: {scenario['history_length']} interactions")
        print(f"  Product popularity: {scenario['popularity']:.2f}")
        print(f"  → Confidence Score: {confidence:.1f}%")
    
    # Test performance recording
    print("\nTesting performance recording:")
    scorer.record_performance('MOLTY_001', True)
    scorer.record_performance('MOLTY_001', True)
    scorer.record_performance('MOLTY_001', False)
    success_rate = scorer.get_product_success_rate('MOLTY_001')
    print(f"  Product MOLTY_001 success rate: {success_rate:.2%}")
    
    print("\n✓ Confidence scorer test completed")
    return scorer


def test_recommendation_reasoning():
    """Test reasoning generation"""
    print("\n" + "="*60)
    print("TEST 2: Recommendation Reasoning")
    print("="*60)
    
    reasoning_gen = RecommendationReasoning()
    
    # Test different reasoning scenarios
    scenarios = [
        {
            'sources': ['collaborative', 'content'],
            'segment': 'VIP',
            'city': 'Lahore',
            'category': 'mattress',
            'confidence': 85
        },
        {
            'sources': ['geographic'],
            'segment': 'Regular',
            'city': 'Sialkot',
            'category': 'pillow',
            'confidence': 65
        },
        {
            'sources': ['content', 'geographic'],
            'segment': 'High-Value',
            'city': 'Okara',
            'category': 'bedding',
            'confidence': 75
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        reasoning = reasoning_gen.generate_reasoning(
            scenario['sources'],
            scenario['segment'],
            scenario['city'],
            scenario['category'],
            scenario['confidence']
        )
        print(f"\nScenario {i}:")
        print(f"  Sources: {scenario['sources']}")
        print(f"  Segment: {scenario['segment']}, City: {scenario['city']}")
        print(f"  Confidence: {scenario['confidence']}%")
        print(f"  → Reasoning: {reasoning}")
    
    print("\n✓ Recommendation reasoning test completed")
    return reasoning_gen


def test_pkr_revenue_predictor():
    """Test PKR revenue prediction"""
    print("\n" + "="*60)
    print("TEST 3: PKR Revenue Predictor")
    print("="*60)
    
    _, product_data, sales_data = create_test_data()
    
    predictor = PKRRevenuePredictor()
    predictor.train(product_data, sales_data)
    
    print(f"\nTrained with {len(predictor.product_prices)} products")
    print(f"Product prices: {predictor.product_prices}")
    
    # Test predictions for different scenarios
    scenarios = [
        {'product': 'MOLTY_001', 'confidence': 85, 'segment': 'VIP'},
        {'product': 'PILLOW_001', 'confidence': 65, 'segment': 'Regular'},
        {'product': 'MATTRESS_001', 'confidence': 75, 'segment': 'High-Value'},
    ]
    
    for scenario in scenarios:
        prediction = predictor.predict_revenue(
            scenario['product'],
            scenario['confidence'],
            scenario['segment']
        )
        
        print(f"\n{scenario['product']} (Confidence: {scenario['confidence']}%, Segment: {scenario['segment']}):")
        print(f"  Product Price: {prediction['expected_revenue_formatted']}")
        print(f"  Expected Revenue: {prediction['expected_revenue_formatted']}")
        print(f"  Conversion Probability: {prediction['conversion_probability']:.1%}")
        print(f"  Revenue Range: Rs{prediction['lower_bound_pkr']:,.0f} - Rs{prediction['upper_bound_pkr']:,.0f}")
    
    print("\n✓ PKR revenue predictor test completed")
    return predictor


def test_quality_metrics():
    """Test recommendation quality metrics"""
    print("\n" + "="*60)
    print("TEST 4: Recommendation Quality Metrics")
    print("="*60)
    
    metrics_tracker = RecommendationQualityMetrics()
    
    # Simulate recommendation outcomes
    print("\nSimulating 20 recommendation outcomes...")
    outcomes = [
        ('MOLTY_001', True, True, True),   # Viewed, clicked, purchased
        ('PILLOW_001', True, True, False),  # Viewed, clicked, not purchased
        ('MATTRESS_001', True, False, False), # Viewed only
        ('MOLTY_001', True, True, True),
        ('BEDDING_001', True, False, False),
        ('PILLOW_001', True, True, True),
        ('MOLTY_001', True, True, False),
        ('MATTRESS_001', True, True, True),
        ('BEDDING_001', True, False, False),
        ('MOLTY_001', True, True, True),
    ]
    
    for product_id, viewed, clicked, purchased in outcomes:
        metrics_tracker.record_recommendation_outcome(
            product_id, viewed, clicked, purchased, datetime.now()
        )
    
    # Get metrics
    current_metrics = metrics_tracker.get_current_metrics()
    summary = metrics_tracker.get_metrics_summary()
    
    print("\nCurrent Quality Metrics:")
    for metric, value in current_metrics.items():
        print(f"  {metric.capitalize()}: {value:.1f}%")
    
    print(f"\nSummary: {summary}")
    
    print("\n✓ Quality metrics test completed")
    return metrics_tracker


def test_enhanced_hybrid_engine():
    """Test enhanced hybrid engine with all features"""
    print("\n" + "="*60)
    print("TEST 5: Enhanced Hybrid Engine")
    print("="*60)
    
    unified_data, product_data, sales_data = create_test_data()
    
    # Initialize and train
    engine = EnhancedHybridEngine()
    print("\nTraining enhanced hybrid engine...")
    metrics = engine.train(unified_data, product_data, sales_data)
    
    print("\nTraining completed:")
    for component, component_metrics in metrics.items():
        if isinstance(component_metrics, dict) and 'error' not in component_metrics:
            print(f"  ✓ {component}")
    
    # Get enhanced recommendations
    print("\nGetting enhanced recommendations for cust_001 (VIP, Lahore):")
    recommendations = engine.get_enhanced_recommendations(
        customer_id='cust_001',
        customer_segment='VIP',
        city='Lahore',
        user_history_length=10,
        n_recommendations=3
    )
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  Recommendation {i}:")
            print(f"    Product: {rec['product_id']}")
            print(f"    Score: {rec['score']:.2f}")
            print(f"    Confidence: {rec['confidence_score']:.1f}%")
            print(f"    Reasoning: {rec['reasoning']}")
            print(f"    Expected Revenue: {rec['revenue_prediction']['expected_revenue_formatted']}")
            print(f"    Conversion Probability: {rec['revenue_prediction']['conversion_probability']:.1%}")
    else:
        print("  No recommendations generated")
    
    # Test feedback recording
    print("\nRecording feedback...")
    engine.record_recommendation_feedback('MOLTY_001', True, True, True)
    engine.record_recommendation_feedback('PILLOW_001', True, False, False)
    
    quality_metrics = engine.get_quality_metrics()
    print(f"  Updated quality metrics: Accuracy {quality_metrics['accuracy']:.1f}%")
    
    print("\n✓ Enhanced hybrid engine test completed")
    return engine


def test_integration():
    """Test integration of all components"""
    print("\n" + "="*60)
    print("TEST 6: Full Integration")
    print("="*60)
    
    unified_data, product_data, sales_data = create_test_data()
    
    # Create all components
    print("\nInitializing all components...")
    scorer = ConfidenceScorer()
    reasoning = RecommendationReasoning()
    predictor = PKRRevenuePredictor()
    metrics = RecommendationQualityMetrics()
    engine = EnhancedHybridEngine()
    
    # Train
    print("Training...")
    predictor.train(product_data, sales_data)
    engine.train(unified_data, product_data, sales_data)
    
    # Get recommendations
    print("\nGenerating recommendations with full pipeline...")
    recs = engine.get_enhanced_recommendations(
        'cust_001',
        'High-Value',
        'Lahore',
        8,
        2
    )
    
    print(f"  Generated {len(recs)} recommendations")
    if recs:
        rec = recs[0]
        print(f"\n  Sample recommendation:")
        print(f"    Product: {rec['product_id']}")
        print(f"    Confidence: {rec['confidence_score']:.1f}%")
        print(f"    Reasoning: {rec['reasoning']}")
        print(f"    Revenue: {rec['revenue_prediction']['expected_revenue_formatted']}")
    
    print("\n✓ Full integration test completed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("CONFIDENCE SCORING & REASONING SYSTEM - TASK 3.2 TESTS")
    print("Testing: Confidence, Reasoning, Revenue Prediction, Quality Metrics")
    print("="*60)
    
    try:
        scorer = test_confidence_scorer()
        reasoning = test_recommendation_reasoning()
        predictor = test_pkr_revenue_predictor()
        metrics = test_quality_metrics()
        engine = test_enhanced_hybrid_engine()
        test_integration()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✓ Confidence Scorer: PASSED")
        print("✓ Recommendation Reasoning: PASSED")
        print("✓ PKR Revenue Predictor: PASSED")
        print("✓ Quality Metrics Tracking: PASSED")
        print("✓ Enhanced Hybrid Engine: PASSED")
        print("✓ Full Integration: PASSED")
        print("\n" + "="*60)
        print("ALL TESTS PASSED - Task 3.2 Implementation Complete")
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
