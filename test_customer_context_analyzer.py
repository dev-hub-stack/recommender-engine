"""
Test Customer Context Analyzer
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'algorithms'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

# Import directly without going through __init__.py
from customer_context_analyzer import CustomerContextAnalyzer, CustomerContext


def create_sample_purchase_history(customer_type='new'):
    """Create sample purchase history for testing"""
    
    if customer_type == 'new':
        # New customer with no history
        return pd.DataFrame()
    
    elif customer_type == 'simple':
        # Simple customer: few purchases, single channel, single city
        return pd.DataFrame({
            'order_date': [
                datetime.now() - timedelta(days=10),
                datetime.now() - timedelta(days=5),
            ],
            'total_amount': [1500.0, 2000.0],
            'channel': ['Online', 'Online'],
            'city': ['Karachi', 'Karachi'],
            'product_category': ['Electronics', 'Electronics']
        })
    
    elif customer_type == 'moderate':
        # Moderate customer: 5-9 purchases, some diversity
        dates = [datetime.now() - timedelta(days=i*10) for i in range(7)]
        return pd.DataFrame({
            'order_date': dates,
            'total_amount': [1500, 2000, 1800, 2200, 1600, 1900, 2100],
            'channel': ['Online', 'POS', 'Online', 'Online', 'POS', 'Online', 'POS'],
            'city': ['Karachi', 'Karachi', 'Lahore', 'Karachi', 'Karachi', 'Lahore', 'Karachi'],
            'product_category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing', 'Electronics', 'Home']
        })
    
    elif customer_type == 'complex':
        # Complex customer: 10+ purchases, high diversity
        dates = [datetime.now() - timedelta(days=i*7) for i in range(15)]
        return pd.DataFrame({
            'order_date': dates,
            'total_amount': np.random.uniform(1000, 5000, 15),
            'channel': np.random.choice(['Online', 'POS', 'Mobile'], 15),
            'city': np.random.choice(['Karachi', 'Lahore', 'Islamabad'], 15),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'], 15)
        })
    
    return pd.DataFrame()


def test_new_customer():
    """Test analysis of new customer with no history"""
    print("\n=== Test 1: New Customer ===")
    
    analyzer = CustomerContextAnalyzer()
    purchase_history = create_sample_purchase_history('new')
    
    context = analyzer.analyze_customer('CUST001', purchase_history)
    
    print(f"Customer ID: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Complexity Score: {context.behavioral_complexity_score:.3f}")
    print(f"Is Cross-Channel: {context.is_cross_channel}")
    print(f"Is Multi-City: {context.is_multi_city}")
    
    assert context.purchase_count == 0
    assert context.behavioral_complexity_score == 0.0
    assert not context.is_cross_channel
    assert not context.is_multi_city
    
    print("✓ New customer test passed")


def test_simple_customer():
    """Test analysis of simple customer"""
    print("\n=== Test 2: Simple Customer ===")
    
    analyzer = CustomerContextAnalyzer()
    purchase_history = create_sample_purchase_history('simple')
    
    context = analyzer.analyze_customer('CUST002', purchase_history)
    
    print(f"Customer ID: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Total Revenue PKR: Rs{context.total_revenue_pkr:,.2f}")
    print(f"Avg Order Value PKR: Rs{context.avg_order_value_pkr:,.2f}")
    print(f"Channels Used: {context.channels_used}")
    print(f"Channel Count: {context.channel_count}")
    print(f"Is Cross-Channel: {context.is_cross_channel}")
    print(f"Cities: {context.cities}")
    print(f"Is Multi-City: {context.is_multi_city}")
    print(f"Categories: {context.product_categories}")
    print(f"Category Count: {context.category_count}")
    print(f"Last Purchase Days Ago: {context.last_purchase_days_ago}")
    print(f"Recency Score: {context.recency_score:.3f}")
    print(f"Complexity Score: {context.behavioral_complexity_score:.3f}")
    
    assert context.purchase_count == 2
    assert context.channel_count == 1
    assert not context.is_cross_channel
    assert context.city_count == 1
    assert not context.is_multi_city
    assert context.category_count == 1
    assert context.last_purchase_days_ago <= 10
    assert context.recency_score > 0.5
    
    print("✓ Simple customer test passed")


def test_moderate_customer():
    """Test analysis of moderate customer"""
    print("\n=== Test 3: Moderate Customer ===")
    
    analyzer = CustomerContextAnalyzer()
    purchase_history = create_sample_purchase_history('moderate')
    
    context = analyzer.analyze_customer('CUST003', purchase_history)
    
    print(f"Customer ID: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Total Revenue PKR: Rs{context.total_revenue_pkr:,.2f}")
    print(f"Channels Used: {context.channels_used}")
    print(f"Channel Count: {context.channel_count}")
    print(f"Is Cross-Channel: {context.is_cross_channel}")
    print(f"Channel Diversity Score: {context.channel_diversity_score:.3f}")
    print(f"Cities: {context.cities}")
    print(f"City Count: {context.city_count}")
    print(f"Is Multi-City: {context.is_multi_city}")
    print(f"Geographic Diversity Score: {context.geographic_diversity_score:.3f}")
    print(f"Categories: {context.product_categories}")
    print(f"Category Count: {context.category_count}")
    print(f"Category Diversity Score: {context.category_diversity_score:.3f}")
    print(f"Top Categories: {context.top_categories}")
    print(f"Purchase Frequency Days: {context.purchase_frequency_days:.1f}")
    print(f"Complexity Score: {context.behavioral_complexity_score:.3f}")
    
    assert context.purchase_count == 7
    assert context.is_cross_channel
    assert context.channel_count >= 2
    assert context.is_multi_city
    assert context.city_count >= 2
    assert context.category_count >= 3
    assert context.behavioral_complexity_score > 0.2
    
    print("✓ Moderate customer test passed")


def test_complex_customer():
    """Test analysis of complex customer"""
    print("\n=== Test 4: Complex Customer ===")
    
    analyzer = CustomerContextAnalyzer()
    purchase_history = create_sample_purchase_history('complex')
    
    context = analyzer.analyze_customer('CUST004', purchase_history)
    
    print(f"Customer ID: {context.customer_id}")
    print(f"Purchase Count: {context.purchase_count}")
    print(f"Total Revenue PKR: Rs{context.total_revenue_pkr:,.2f}")
    print(f"Avg Order Value PKR: Rs{context.avg_order_value_pkr:,.2f}")
    print(f"Channels Used: {context.channels_used}")
    print(f"Channel Count: {context.channel_count}")
    print(f"Channel Diversity Score: {context.channel_diversity_score:.3f}")
    print(f"Cities: {context.cities}")
    print(f"City Count: {context.city_count}")
    print(f"Geographic Diversity Score: {context.geographic_diversity_score:.3f}")
    print(f"Categories: {context.product_categories}")
    print(f"Category Count: {context.category_count}")
    print(f"Category Diversity Score: {context.category_diversity_score:.3f}")
    print(f"Top Categories: {context.top_categories[:3]}")
    print(f"Complexity Score: {context.behavioral_complexity_score:.3f}")
    
    assert context.purchase_count >= 10
    assert context.is_cross_channel
    assert context.is_multi_city
    assert context.category_count >= 3
    assert context.behavioral_complexity_score > 0.3
    
    print("✓ Complex customer test passed")


def test_caching():
    """Test context caching"""
    print("\n=== Test 5: Context Caching ===")
    
    analyzer = CustomerContextAnalyzer()
    purchase_history = create_sample_purchase_history('moderate')
    
    # First analysis
    context1 = analyzer.analyze_customer('CUST005', purchase_history)
    
    # Second analysis (should be cached)
    context2 = analyzer.analyze_customer('CUST005', purchase_history)
    
    # Check cache stats
    cache_stats = analyzer.get_cache_stats()
    print(f"Cache Size: {cache_stats['cache_size']}")
    print(f"Cache TTL Hours: {cache_stats['cache_ttl_hours']}")
    
    # Get cached context directly
    cached = analyzer.get_cached_context('CUST005')
    
    assert cached is not None
    assert cached.customer_id == 'CUST005'
    assert cache_stats['cache_size'] >= 1
    
    print("✓ Caching test passed")


def test_context_to_dict():
    """Test context serialization"""
    print("\n=== Test 6: Context Serialization ===")
    
    analyzer = CustomerContextAnalyzer()
    purchase_history = create_sample_purchase_history('moderate')
    
    context = analyzer.analyze_customer('CUST006', purchase_history)
    context_dict = context.to_dict()
    
    print(f"Context Dictionary Keys: {list(context_dict.keys())[:5]}...")
    print(f"Purchase Count: {context_dict['purchase_count']}")
    print(f"Complexity Score: {context_dict['behavioral_complexity_score']:.3f}")
    
    assert isinstance(context_dict, dict)
    assert 'customer_id' in context_dict
    assert 'purchase_count' in context_dict
    assert 'behavioral_complexity_score' in context_dict
    
    print("✓ Serialization test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Customer Context Analyzer")
    print("=" * 60)
    
    try:
        test_new_customer()
        test_simple_customer()
        test_moderate_customer()
        test_complex_customer()
        test_caching()
        test_context_to_dict()
        
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
