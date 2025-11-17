"""
Test Algorithm Performance Dashboard
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'algorithms'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

from algorithm_performance_dashboard import AlgorithmPerformanceDashboard


def test_track_algorithm_usage():
    """Test tracking algorithm usage"""
    print("\n=== Test 1: Track Algorithm Usage ===")
    
    dashboard = AlgorithmPerformanceDashboard(tracking_window_days=30)
    
    # Track some usage
    dashboard.track_algorithm_usage(
        customer_id="CUST001",
        algorithm_used="deep_learning",
        recommendation_ids=["PROD1", "PROD2", "PROD3"],
        response_time_ms=150.5
    )
    
    dashboard.track_algorithm_usage(
        customer_id="CUST002",
        algorithm_used="collaborative_filtering",
        recommendation_ids=["PROD4", "PROD5"],
        response_time_ms=85.2
    )
    
    print(f"Total usage records: {len(dashboard.usage_records)}")
    print(f"Algorithms tracked: {set(r.algorithm_used for r in dashboard.usage_records)}")
    
    assert len(dashboard.usage_records) == 2
    assert len(dashboard.response_times['deep_learning']) == 1
    assert len(dashboard.response_times['collaborative_filtering']) == 1
    
    print("✓ Algorithm usage tracking test passed")


def test_track_revenue():
    """Test tracking revenue by algorithm"""
    print("\n=== Test 2: Track Revenue ===")
    
    dashboard = AlgorithmPerformanceDashboard()
    
    # Track revenue
    dashboard.track_revenue(
        customer_id="CUST001",
        algorithm_used="deep_learning",
        product_id="PROD1",
        revenue_pkr=5000.0
    )
    
    dashboard.track_revenue(
        customer_id="CUST002",
        algorithm_used="deep_learning",
        product_id="PROD2",
        revenue_pkr=3500.0
    )
    
    dashboard.track_revenue(
        customer_id="CUST003",
        algorithm_used="collaborative_filtering",
        product_id="PROD3",
        revenue_pkr=7200.0
    )
    
    print(f"Total revenue records: {len(dashboard.revenue_records)}")
    
    # Calculate revenue by algorithm
    dl_revenue = dashboard.calculate_revenue_by_algorithm("deep_learning")
    cf_revenue = dashboard.calculate_revenue_by_algorithm("collaborative_filtering")
    
    print(f"Deep Learning Revenue: {dl_revenue['formatted_revenue']}")
    print(f"Collaborative Filtering Revenue: {cf_revenue['formatted_revenue']}")
    
    assert dl_revenue['total_revenue_pkr'] == 8500.0
    assert cf_revenue['total_revenue_pkr'] == 7200.0
    
    print("✓ Revenue tracking test passed")


def test_track_conversions():
    """Test tracking clicks and purchases"""
    print("\n=== Test 3: Track Conversions ===")
    
    dashboard = AlgorithmPerformanceDashboard()
    
    # Track usage first
    for i in range(10):
        dashboard.track_algorithm_usage(
            customer_id=f"CUST{i:03d}",
            algorithm_used="deep_learning",
            recommendation_ids=[f"PROD{i}"],
            response_time_ms=100.0
        )
    
    # Track clicks
    for i in range(5):
        dashboard.track_click(
            customer_id=f"CUST{i:03d}",
            algorithm_used="deep_learning",
            product_id=f"PROD{i}"
        )
    
    # Track purchases
    for i in range(2):
        dashboard.track_purchase(
            customer_id=f"CUST{i:03d}",
            algorithm_used="deep_learning",
            product_id=f"PROD{i}",
            revenue_pkr=5000.0
        )
    
    # Calculate conversion rates
    conversion_rates = dashboard.calculate_conversion_rates("deep_learning")
    
    print(f"Total Recommendations: {conversion_rates['total_recommendations']}")
    print(f"Total Clicks: {conversion_rates['total_clicks']}")
    print(f"Total Purchases: {conversion_rates['total_purchases']}")
    print(f"Click-Through Rate: {conversion_rates['click_through_rate']:.2f}%")
    print(f"Conversion Rate: {conversion_rates['conversion_rate']:.2f}%")
    print(f"Purchase Rate from Clicks: {conversion_rates['purchase_rate_from_clicks']:.2f}%")
    
    assert conversion_rates['total_recommendations'] == 10
    assert conversion_rates['total_clicks'] == 5
    assert conversion_rates['total_purchases'] == 2
    assert conversion_rates['click_through_rate'] == 50.0
    assert conversion_rates['conversion_rate'] == 20.0
    
    print("✓ Conversion tracking test passed")


def test_performance_comparison_dashboard():
    """Test building performance comparison dashboard"""
    print("\n=== Test 4: Performance Comparison Dashboard ===")
    
    dashboard = AlgorithmPerformanceDashboard()
    
    # Simulate data for multiple algorithms
    algorithms = ["deep_learning", "collaborative_filtering", "popularity_based"]
    
    for algo in algorithms:
        # Track usage
        for i in range(20):
            dashboard.track_algorithm_usage(
                customer_id=f"CUST{i:03d}",
                algorithm_used=algo,
                recommendation_ids=[f"PROD{i}"],
                response_time_ms=100.0 + (algorithms.index(algo) * 20)
            )
        
        # Track clicks (different rates)
        click_count = 10 if algo == "deep_learning" else (8 if algo == "collaborative_filtering" else 6)
        for i in range(click_count):
            dashboard.track_click(
                customer_id=f"CUST{i:03d}",
                algorithm_used=algo,
                product_id=f"PROD{i}"
            )
        
        # Track purchases (different rates)
        purchase_count = 5 if algo == "deep_learning" else (4 if algo == "collaborative_filtering" else 3)
        revenue_per_purchase = 6000.0 if algo == "deep_learning" else (5000.0 if algo == "collaborative_filtering" else 4000.0)
        
        for i in range(purchase_count):
            dashboard.track_purchase(
                customer_id=f"CUST{i:03d}",
                algorithm_used=algo,
                product_id=f"PROD{i}",
                revenue_pkr=revenue_per_purchase
            )
    
    # Build dashboard
    comparison_dashboard = dashboard.build_performance_comparison_dashboard()
    
    print(f"\nGenerated At: {comparison_dashboard['generated_at']}")
    print(f"Time Period: {comparison_dashboard['time_period_days']} days")
    print(f"Number of Algorithms: {len(comparison_dashboard['algorithms'])}")
    
    print("\n--- Summary ---")
    summary = comparison_dashboard['summary']
    print(f"Total Revenue: {summary['formatted_total_revenue']}")
    print(f"Total Recommendations: {summary['total_recommendations']}")
    print(f"Total Purchases: {summary['total_purchases']}")
    print(f"Overall Conversion Rate: {summary['overall_conversion_rate']:.2f}%")
    print(f"Best by Revenue: {summary['best_by_revenue']['algorithm']} ({summary['best_by_revenue']['formatted']})")
    print(f"Best by Conversion: {summary['best_by_conversion']['algorithm']} ({summary['best_by_conversion']['conversion_rate']:.2f}%)")
    print(f"Fastest: {summary['fastest']['algorithm']} ({summary['fastest']['avg_response_time_ms']:.2f}ms)")
    
    print("\n--- Algorithm Performance ---")
    for algo_data in comparison_dashboard['algorithms']:
        print(f"\n{algo_data['algorithm']}:")
        print(f"  Revenue: Rs{algo_data['total_revenue_pkr']:,.2f} ({algo_data['revenue_share_percentage']:.1f}%)")
        print(f"  Conversion Rate: {algo_data['conversion_rate']:.2f}%")
        print(f"  Rank by Revenue: #{algo_data['rank_by_revenue']}")
        print(f"  Rank by Conversion: #{algo_data['rank_by_conversion']}")
    
    print("\n--- Rankings ---")
    print(f"By Revenue: {comparison_dashboard['rankings']['by_revenue']}")
    print(f"By Conversion: {comparison_dashboard['rankings']['by_conversion']}")
    
    assert len(comparison_dashboard['algorithms']) == 3
    assert summary['total_recommendations'] == 60
    assert summary['total_purchases'] == 12
    assert 'deep_learning' in comparison_dashboard['rankings']['by_revenue']
    
    print("\n✓ Performance comparison dashboard test passed")


def test_export_to_dataframe():
    """Test exporting to pandas DataFrame"""
    print("\n=== Test 5: Export to DataFrame ===")
    
    dashboard = AlgorithmPerformanceDashboard()
    
    # Add some data
    for algo in ["deep_learning", "collaborative_filtering"]:
        for i in range(5):
            dashboard.track_algorithm_usage(
                customer_id=f"CUST{i:03d}",
                algorithm_used=algo,
                recommendation_ids=[f"PROD{i}"],
                response_time_ms=100.0
            )
            
            dashboard.track_purchase(
                customer_id=f"CUST{i:03d}",
                algorithm_used=algo,
                product_id=f"PROD{i}",
                revenue_pkr=5000.0
            )
    
    # Export to DataFrame
    df = dashboard.export_to_dataframe()
    
    print(f"DataFrame Shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:5]}...")
    print(f"\nFirst Algorithm:")
    print(df.iloc[0][['algorithm', 'total_revenue_pkr', 'conversion_rate']])
    
    assert len(df) == 2
    assert 'algorithm' in df.columns
    assert 'total_revenue_pkr' in df.columns
    assert 'conversion_rate' in df.columns
    
    print("\n✓ DataFrame export test passed")


def test_get_algorithm_performance():
    """Test getting performance for specific algorithm"""
    print("\n=== Test 6: Get Algorithm Performance ===")
    
    dashboard = AlgorithmPerformanceDashboard()
    
    # Add data
    for i in range(10):
        dashboard.track_algorithm_usage(
            customer_id=f"CUST{i:03d}",
            algorithm_used="deep_learning",
            recommendation_ids=[f"PROD{i}"],
            response_time_ms=120.0
        )
    
    for i in range(3):
        dashboard.track_purchase(
            customer_id=f"CUST{i:03d}",
            algorithm_used="deep_learning",
            product_id=f"PROD{i}",
            revenue_pkr=6000.0
        )
    
    # Get performance
    performance = dashboard.get_algorithm_performance("deep_learning")
    
    print(f"Algorithm: {performance.algorithm}")
    print(f"Total Recommendations: {performance.total_recommendations}")
    print(f"Total Revenue: Rs{performance.total_revenue_pkr:,.2f}")
    print(f"Conversion Rate: {performance.conversion_rate:.2f}%")
    print(f"Avg Response Time: {performance.avg_response_time_ms:.2f}ms")
    
    assert performance.algorithm == "deep_learning"
    assert performance.total_recommendations == 10
    assert performance.total_revenue_pkr == 18000.0
    assert performance.conversion_rate == 30.0
    
    print("✓ Get algorithm performance test passed")


def test_clear_old_records():
    """Test clearing old records"""
    print("\n=== Test 7: Clear Old Records ===")
    
    dashboard = AlgorithmPerformanceDashboard()
    
    # Add current records
    for i in range(5):
        dashboard.track_algorithm_usage(
            customer_id=f"CUST{i:03d}",
            algorithm_used="deep_learning",
            recommendation_ids=[f"PROD{i}"],
            response_time_ms=100.0
        )
    
    initial_count = len(dashboard.usage_records)
    print(f"Initial records: {initial_count}")
    
    # Clear old records (should keep all since they're recent)
    dashboard.clear_old_records(days_to_keep=90)
    
    after_clear_count = len(dashboard.usage_records)
    print(f"After clearing (90 days): {after_clear_count}")
    
    assert after_clear_count == initial_count
    
    print("✓ Clear old records test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing Algorithm Performance Dashboard")
    print("=" * 60)
    
    try:
        test_track_algorithm_usage()
        test_track_revenue()
        test_track_conversions()
        test_performance_comparison_dashboard()
        test_export_to_dataframe()
        test_get_algorithm_performance()
        test_clear_old_records()
        
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
