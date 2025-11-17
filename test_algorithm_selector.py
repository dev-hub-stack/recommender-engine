"""
Test Algorithm Selector
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'algorithms'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from algorithm_selector import AlgorithmSelector, AlgorithmType, SelectionDecision
from customer_context_analyzer import CustomerContext
from datetime import datetime


def create_test_context(purchase_count, complexity_score=0.5, 
                       is_cross_channel=False, total_revenue=50000):
    """Create test customer context"""
    return CustomerContext(
        customer_id=f"TEST_{purchase_count}",
        purchase_count=purchase_count,
        total_revenue_pkr=total_revenue,
        avg_order_value_pkr=total_revenue / purchase_count if purchase_count > 0 else 0,
        channels_used=['Online', 'POS'] if is_cross_channel else ['Online'],
        channel_count=2 if is_cross_channel else 1,
        is_cross_channel=is_cross_channel,
        primary_channel='Online',
        channel_diversity_score=0.5 if is_cross_channel else 0.0,
        cities=['Karachi'],
        city_count=1,
        is_multi_city=False,
        primary_city='Karachi',
        geographic_diversity_score=0.0,
        product_categories=['Electronics', 'Clothing'],
        category_count=2,
        category_diversity_score=0.5,
        top_categories=[('Electronics', 5), ('Clothing', 3)],
        last_purchase_days_ago=10,
        first_purchase_days_ago=100,
        purchase_frequency_days=10.0,
        recency_score=0.8,
        behavioral_complexity_score=complexity_score,
        analyzed_at=datetime.utcnow()
    )


def test_deep_learning_selection():
    """Test deep learning selection for 10+ purchases"""
    print("\n=== Test 1: Deep Learning Selection (10+ purchases) ===")
    
    selector = AlgorithmSelector()
    context = create_test_context(purchase_count=15)
    available_algorithms = ['deep_learning', 'collaborative', 'popularity']
    
    decision = selector.select_algorithm(context, available_algorithms)
    
    print(f"Customer: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Selected Algorithm: {decision.selected_algorithm.value}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Fallback: {decision.fallback_algorithm.value if decision.fallback_algorithm else 'None'}")
    print(f"Use Ensemble: {decision.use_ensemble}")
    
    assert decision.selected_algorithm == AlgorithmType.DEEP_LEARNING
    assert decision.confidence >= 0.8
    assert not decision.use_ensemble
    
    print("✓ Deep learning selection test passed")


def test_collaborative_filtering_selection():
    """Test collaborative filtering selection for 5-9 purchases"""
    print("\n=== Test 2: Collaborative Filtering Selection (5-9 purchases) ===")
    
    selector = AlgorithmSelector()
    context = create_test_context(purchase_count=7)
    available_algorithms = ['deep_learning', 'collaborative', 'popularity']
    
    decision = selector.select_algorithm(context, available_algorithms)
    
    print(f"Customer: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Selected Algorithm: {decision.selected_algorithm.value}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Fallback: {decision.fallback_algorithm.value if decision.fallback_algorithm else 'None'}")
    
    assert decision.selected_algorithm == AlgorithmType.COLLABORATIVE_FILTERING
    assert decision.confidence >= 0.7
    
    print("✓ Collaborative filtering selection test passed")


def test_popularity_based_selection():
    """Test popularity-based selection for <5 purchases"""
    print("\n=== Test 3: Popularity-Based Selection (<5 purchases) ===")
    
    selector = AlgorithmSelector()
    context = create_test_context(purchase_count=3)
    available_algorithms = ['deep_learning', 'collaborative', 'popularity']
    
    decision = selector.select_algorithm(context, available_algorithms)
    
    print(f"Customer: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Selected Algorithm: {decision.selected_algorithm.value}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    
    assert decision.selected_algorithm == AlgorithmType.POPULARITY_BASED
    assert decision.confidence >= 0.8
    
    print("✓ Popularity-based selection test passed")


def test_ensemble_borderline_case():
    """Test ensemble selection for borderline cases (8-12 purchases)"""
    print("\n=== Test 4: Ensemble Selection (Borderline Case) ===")
    
    selector = AlgorithmSelector()
    context = create_test_context(purchase_count=10)
    available_algorithms = ['deep_learning', 'collaborative', 'popularity']
    
    decision = selector.select_algorithm(context, available_algorithms)
    
    print(f"Customer: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Selected Algorithm: {decision.selected_algorithm.value}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Use Ensemble: {decision.use_ensemble}")
    if decision.ensemble_weights:
        print(f"Ensemble Weights: {decision.ensemble_weights}")
    
    assert decision.use_ensemble
    assert decision.selected_algorithm == AlgorithmType.ENSEMBLE
    assert decision.ensemble_weights is not None
    
    print("✓ Ensemble borderline case test passed")


def test_ensemble_high_value_customer():
    """Test ensemble selection for high-value customers"""
    print("\n=== Test 5: Ensemble Selection (High-Value Customer) ===")
    
    selector = AlgorithmSelector()
    context = create_test_context(purchase_count=7, total_revenue=150000)
    available_algorithms = ['deep_learning', 'collaborative', 'popularity']
    
    decision = selector.select_algorithm(context, available_algorithms)
    
    print(f"Customer: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Total Revenue: Rs{context.total_revenue_pkr:,.2f}")
    print(f"Selected Algorithm: {decision.selected_algorithm.value}")
    print(f"Use Ensemble: {decision.use_ensemble}")
    print(f"Reasoning: {decision.reasoning}")
    if decision.ensemble_weights:
        print(f"Ensemble Weights: {decision.ensemble_weights}")
    
    assert decision.use_ensemble
    assert decision.selected_algorithm == AlgorithmType.ENSEMBLE
    
    print("✓ Ensemble high-value customer test passed")


def test_cross_channel_customer():
    """Test collaborative filtering preference for cross-channel customers"""
    print("\n=== Test 6: Cross-Channel Customer ===")
    
    selector = AlgorithmSelector()
    context = create_test_context(purchase_count=6, is_cross_channel=True)
    available_algorithms = ['deep_learning', 'collaborative', 'popularity']
    
    decision = selector.select_algorithm(context, available_algorithms)
    
    print(f"Customer: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Is Cross-Channel: {context.is_cross_channel}")
    print(f"Channels: {context.channel_count}")
    print(f"Selected Algorithm: {decision.selected_algorithm.value}")
    print(f"Reasoning: {decision.reasoning}")
    
    assert decision.selected_algorithm == AlgorithmType.COLLABORATIVE_FILTERING
    
    print("✓ Cross-channel customer test passed")


def test_high_complexity_upgrade():
    """Test algorithm upgrade for high complexity customers"""
    print("\n=== Test 7: High Complexity Upgrade ===")
    
    selector = AlgorithmSelector()
    context = create_test_context(purchase_count=4, complexity_score=0.75)
    available_algorithms = ['deep_learning', 'collaborative', 'popularity']
    
    decision = selector.select_algorithm(context, available_algorithms)
    
    print(f"Customer: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Complexity Score: {context.behavioral_complexity_score:.2f}")
    print(f"Selected Algorithm: {decision.selected_algorithm.value}")
    print(f"Reasoning: {decision.reasoning}")
    
    # Should upgrade from popularity to collaborative due to high complexity
    assert decision.selected_algorithm == AlgorithmType.COLLABORATIVE_FILTERING
    
    print("✓ High complexity upgrade test passed")


def test_update_selection_rules():
    """Test updating selection rules"""
    print("\n=== Test 8: Update Selection Rules ===")
    
    selector = AlgorithmSelector()
    
    # Get initial rules
    initial_rules = selector.get_selection_rules()
    print(f"Initial Rules: {initial_rules}")
    
    # Update rules
    selector.update_selection_rules(
        deep_learning_threshold=12,
        collaborative_threshold=6
    )
    
    updated_rules = selector.get_selection_rules()
    print(f"Updated Rules: {updated_rules}")
    
    assert updated_rules['deep_learning_threshold'] == 12
    assert updated_rules['collaborative_threshold'] == 6
    
    # Test with updated rules
    context = create_test_context(purchase_count=7)
    available_algorithms = ['deep_learning', 'collaborative', 'popularity']
    
    decision = selector.select_algorithm(context, available_algorithms)
    print(f"Decision with updated rules: {decision.selected_algorithm.value}")
    
    # With new threshold of 6, 7 purchases should use collaborative
    assert decision.selected_algorithm == AlgorithmType.COLLABORATIVE_FILTERING
    
    print("✓ Update selection rules test passed")


def test_ensemble_weight_calculation():
    """Test ensemble weight calculation"""
    print("\n=== Test 9: Ensemble Weight Calculation ===")
    
    selector = AlgorithmSelector()
    
    # Test different contexts
    contexts = [
        create_test_context(purchase_count=10, complexity_score=0.7),
        create_test_context(purchase_count=8, is_cross_channel=True),
        create_test_context(purchase_count=12, total_revenue=120000)
    ]
    
    available_algorithms = ['deep_learning', 'collaborative', 'popularity']
    
    for i, context in enumerate(contexts, 1):
        decision = selector.select_algorithm(context, available_algorithms)
        if decision.use_ensemble and decision.ensemble_weights:
            print(f"\nContext {i}:")
            print(f"  Purchase Count: {context.purchase_count}")
            print(f"  Complexity: {context.behavioral_complexity_score:.2f}")
            print(f"  Revenue: Rs{context.total_revenue_pkr:,.0f}")
            print(f"  Ensemble Weights: {decision.ensemble_weights}")
            
            # Verify weights sum to 1.0
            total_weight = sum(decision.ensemble_weights.values())
            assert abs(total_weight - 1.0) < 0.01, f"Weights don't sum to 1.0: {total_weight}"
    
    print("\n✓ Ensemble weight calculation test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Algorithm Selector")
    print("=" * 60)
    
    try:
        test_deep_learning_selection()
        test_collaborative_filtering_selection()
        test_popularity_based_selection()
        test_ensemble_borderline_case()
        test_ensemble_high_value_customer()
        test_cross_channel_customer()
        test_high_complexity_upgrade()
        test_update_selection_rules()
        test_ensemble_weight_calculation()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
