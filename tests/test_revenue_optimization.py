"""
Tests for Revenue Optimization Engine
Tests all three subtasks: 5.1, 5.2, and 5.3
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithms'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from revenue_optimization_engine import (
    ABTestingFramework, ABTestVariant, PerformanceBasedOptimizer,
    ConversionRateOptimizer, RevenueMaximizationEngine, OptimizationStrategy
)
from dynamic_pricing_engine import (
    Promotion, PromotionAwareRecommendationEngine, DynamicPricingEngine,
    SeasonalPromotionOptimizer, InventoryAwareRecommendationAdjuster
)
from performance_monitoring_engine import (
    RealTimePerformanceTracker, AutomatedAlertSystem, ROICalculator,
    AutomatedOptimizationSuggester, AlertSeverity
)


# ============================================================================
# SUBTASK 5.1 TESTS: Automated Recommendation Optimization
# ============================================================================

class TestABTestingFramework:
    """Test A/B testing framework for recommendation strategies"""
    
    def test_create_ab_test(self):
        """Test creating an A/B test"""
        framework = ABTestingFramework()
        
        variants = [
            ABTestVariant(
                variant_id="control",
                variant_name="Current Algorithm",
                algorithm_config={"type": "collaborative_filtering"},
                traffic_allocation=0.5,
                is_control=True,
                created_at=datetime.now()
            ),
            ABTestVariant(
                variant_id="test",
                variant_name="New Algorithm",
                algorithm_config={"type": "deep_learning"},
                traffic_allocation=0.5,
                is_control=False,
                created_at=datetime.now()
            )
        ]
        
        test_id = framework.create_ab_test(
            test_name="algorithm_comparison",
            variants=variants,
            duration_days=14
        )
        
        assert test_id is not None
        assert test_id in framework.active_tests
        assert len(framework.active_tests[test_id]['variants']) == 2
    
    def test_variant_assignment(self):
        """Test user assignment to variants"""
        framework = ABTestingFramework()
        
        variants = [
            ABTestVariant("control", "Control", {}, 0.5, True, datetime.now()),
            ABTestVariant("test", "Test", {}, 0.5, False, datetime.now())
        ]
        
        test_id = framework.create_ab_test("test", variants)
        
        # Assign multiple users
        assignments = {}
        for i in range(100):
            user_id = f"user_{i}"
            variant = framework.assign_variant(test_id, user_id)
            assignments[variant] = assignments.get(variant, 0) + 1
        
        # Check distribution is roughly 50/50
        assert 30 <= assignments.get("control", 0) <= 70
        assert 30 <= assignments.get("test", 0) <= 70
    
    def test_record_and_calculate_results(self):
        """Test recording events and calculating results"""
        framework = ABTestingFramework(min_sample_size=10)
        
        variants = [
            ABTestVariant("control", "Control", {}, 0.5, True, datetime.now()),
            ABTestVariant("test", "Test", {}, 0.5, False, datetime.now())
        ]
        
        test_id = framework.create_ab_test("test", variants)
        
        # Record events for control
        for i in range(100):
            framework.record_impression(test_id, "control")
            if i < 10:
                framework.record_click(test_id, "control")
            if i < 5:
                framework.record_conversion(test_id, "control", 5000.0)
        
        # Record events for test (better performance)
        for i in range(100):
            framework.record_impression(test_id, "test")
            if i < 20:
                framework.record_click(test_id, "test")
            if i < 10:
                framework.record_conversion(test_id, "test", 5000.0)
        
        # Calculate results
        results = framework.calculate_test_results(test_id)
        
        assert "control" in results
        assert "test" in results
        assert results["test"].conversion_rate > results["control"].conversion_rate
        assert results["test"].revenue_pkr > results["control"].revenue_pkr
    
    def test_winning_variant(self):
        """Test determining winning variant"""
        framework = ABTestingFramework(min_sample_size=10)
        
        variants = [
            ABTestVariant("control", "Control", {}, 0.5, True, datetime.now()),
            ABTestVariant("test", "Test", {}, 0.5, False, datetime.now())
        ]
        
        test_id = framework.create_ab_test("test", variants, target_metric="conversion_rate")
        
        # Make test variant clearly better
        for i in range(100):
            framework.record_impression(test_id, "control")
            framework.record_impression(test_id, "test")
            
            if i < 5:
                framework.record_conversion(test_id, "control", 5000.0)
            if i < 15:
                framework.record_conversion(test_id, "test", 5000.0)
        
        winner = framework.get_winning_variant(test_id)
        assert winner == "test"


class TestPerformanceBasedOptimizer:
    """Test performance-based algorithm adjustment"""
    
    def test_record_and_calculate_scores(self):
        """Test recording performance and calculating scores"""
        optimizer = PerformanceBasedOptimizer()
        
        # Record performance for multiple algorithms
        optimizer.record_algorithm_performance(
            "algo_1", impressions=1000, clicks=50, conversions=10, 
            revenue_pkr=50000, timestamp=datetime.now()
        )
        
        optimizer.record_algorithm_performance(
            "algo_2", impressions=1000, clicks=80, conversions=20,
            revenue_pkr=100000, timestamp=datetime.now()
        )
        
        scores = optimizer.calculate_algorithm_scores()
        
        assert "algo_1" in scores
        assert "algo_2" in scores
        assert scores["algo_2"] > scores["algo_1"]  # algo_2 performs better
    
    def test_optimize_weights(self):
        """Test optimizing algorithm weights"""
        optimizer = PerformanceBasedOptimizer()
        
        # Record performance
        optimizer.record_algorithm_performance(
            "algo_1", 1000, 50, 10, 50000, datetime.now()
        )
        optimizer.record_algorithm_performance(
            "algo_2", 1000, 80, 20, 100000, datetime.now()
        )
        
        # Optimize weights
        weights = optimizer.optimize_algorithm_weights(OptimizationStrategy.BALANCED)
        
        assert "algo_1" in weights
        assert "algo_2" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Weights sum to 1
        assert weights["algo_2"] > weights["algo_1"]  # Better algorithm gets higher weight


class TestConversionRateOptimizer:
    """Test conversion rate optimization"""
    
    def test_analyze_conversion_funnel(self):
        """Test conversion funnel analysis"""
        optimizer = ConversionRateOptimizer()
        
        # Create funnel data
        funnel_data = pd.DataFrame({
            'stage': ['impression'] * 1000 + ['click'] * 100 + ['add_to_cart'] * 50 + ['purchase'] * 25
        })
        
        analysis = optimizer.analyze_conversion_funnel(funnel_data)
        
        assert 'stages' in analysis
        assert 'drop_off_points' in analysis
        assert 'optimization_opportunities' in analysis
        assert len(analysis['stages']) == 4


class TestRevenueMaximizationEngine:
    """Test revenue maximization algorithms"""
    
    def test_calculate_expected_revenue(self):
        """Test expected revenue calculation"""
        engine = RevenueMaximizationEngine()
        
        # Create historical data
        historical_data = pd.DataFrame({
            'product_id': ['prod_1'] * 100,
            'customer_segment': ['high_value'] * 100,
            'converted': [True] * 20 + [False] * 80,
            'revenue': [5000] * 20 + [0] * 80
        })
        
        expected_revenue = engine.calculate_expected_revenue(
            'prod_1', 'high_value', historical_data
        )
        
        assert expected_revenue > 0
        assert expected_revenue == 1000.0  # 20% conversion * 5000 avg
    
    def test_optimize_product_mix(self):
        """Test product mix optimization"""
        engine = RevenueMaximizationEngine()
        
        historical_data = pd.DataFrame({
            'product_id': ['prod_1'] * 50 + ['prod_2'] * 50 + ['prod_3'] * 50,
            'customer_segment': ['high_value'] * 150,
            'converted': [True] * 10 + [False] * 40 + [True] * 20 + [False] * 30 + [True] * 5 + [False] * 45,
            'revenue': [5000] * 10 + [0] * 40 + [6000] * 20 + [0] * 30 + [4000] * 5 + [0] * 45
        })
        
        optimized_mix = engine.optimize_product_mix(
            ['prod_1', 'prod_2', 'prod_3'],
            'high_value',
            historical_data,
            num_recommendations=2
        )
        
        assert len(optimized_mix) == 2
        assert optimized_mix[0][0] == 'prod_2'  # Best performing product


# ============================================================================
# SUBTASK 5.2 TESTS: Dynamic Pricing and Promotion Integration
# ============================================================================

class TestPromotionAwareRecommendationEngine:
    """Test promotion-aware recommendation engine"""
    
    def test_register_and_get_promotions(self):
        """Test registering and retrieving promotions"""
        engine = PromotionAwareRecommendationEngine()
        
        promotion = Promotion(
            promotion_id="promo_1",
            promotion_name="Summer Sale",
            discount_percentage=20.0,
            discount_amount_pkr=0.0,
            applicable_products=["prod_1", "prod_2"],
            applicable_categories=["bedding"],
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now() + timedelta(days=30),
            is_active=True,
            priority=10
        )
        
        engine.register_promotion(promotion)
        
        active_promos = engine.get_active_promotions("prod_1", "bedding")
        assert len(active_promos) == 1
        assert active_promos[0].promotion_id == "promo_1"
    
    def test_adjust_recommendations_for_promotions(self):
        """Test adjusting recommendations based on promotions"""
        engine = PromotionAwareRecommendationEngine()
        
        promotion = Promotion(
            "promo_1", "Sale", 20.0, 0.0, ["prod_2"], ["bedding"],
            datetime.now() - timedelta(days=1), datetime.now() + timedelta(days=30),
            True, 10
        )
        engine.register_promotion(promotion)
        
        base_recs = [("prod_1", 0.8), ("prod_2", 0.7), ("prod_3", 0.6)]
        product_metadata = {
            "prod_1": {"category": "furniture"},
            "prod_2": {"category": "bedding"},
            "prod_3": {"category": "accessories"}
        }
        
        adjusted_recs = engine.adjust_recommendations_for_promotions(base_recs, product_metadata)
        
        # prod_2 should be boosted due to promotion (0.7 * 1.1 = 0.77 < 0.8, so prod_1 still first)
        # But prod_2 should have a promotion attached
        prod_2_rec = next((r for r in adjusted_recs if r[0] == "prod_2"), None)
        assert prod_2_rec is not None
        assert prod_2_rec[2] is not None  # Has promotion
        assert prod_2_rec[2].promotion_id == "promo_1"


class TestDynamicPricingEngine:
    """Test dynamic pricing consideration in PKR"""
    
    def test_calculate_optimal_price(self):
        """Test optimal price calculation"""
        engine = DynamicPricingEngine()
        
        base_price = 10000.0  # Rs 10,000
        
        # High demand, low inventory
        optimal_price = engine.calculate_optimal_price(
            "prod_1", base_price, current_demand=0.9, inventory_level=5
        )
        
        assert optimal_price > base_price  # Should increase price
        
        # Low demand, high inventory
        optimal_price = engine.calculate_optimal_price(
            "prod_2", base_price, current_demand=0.2, inventory_level=150
        )
        
        assert optimal_price < base_price  # Should decrease price
    
    def test_recommend_price_adjustment(self):
        """Test price adjustment recommendations"""
        engine = DynamicPricingEngine()
        engine.demand_elasticity["prod_1"] = -1.5
        
        recommendation = engine.recommend_price_adjustment(
            "prod_1", 10000.0, sales_velocity=0.5, target_margin=0.3
        )
        
        assert 'action' in recommendation
        assert 'recommended_price_pkr' in recommendation
        assert recommendation['action'] == 'decrease'  # Slow moving product


class TestSeasonalPromotionOptimizer:
    """Test seasonal promotion optimization"""
    
    def test_get_current_season(self):
        """Test getting current season"""
        optimizer = SeasonalPromotionOptimizer()
        
        # This will depend on current month
        season = optimizer.get_current_season()
        # Just check it returns a valid result
        assert season is None or hasattr(season, 'season_name')
    
    def test_optimize_seasonal_promotions(self):
        """Test creating seasonal promotions"""
        optimizer = SeasonalPromotionOptimizer()
        
        product_catalog = pd.DataFrame({
            'product_id': ['prod_1', 'prod_2', 'prod_3'],
            'category': ['winter_bedding', 'summer_bedding', 'accessories']
        })
        
        promotions = optimizer.optimize_seasonal_promotions(product_catalog)
        
        # May or may not have promotions depending on current season
        assert isinstance(promotions, list)


class TestInventoryAwareRecommendationAdjuster:
    """Test inventory-aware recommendation adjustments"""
    
    def test_update_and_get_inventory_status(self):
        """Test inventory status tracking"""
        adjuster = InventoryAwareRecommendationAdjuster(low_stock_threshold=10, high_stock_threshold=100)
        
        adjuster.update_inventory("prod_1", 5)
        adjuster.update_inventory("prod_2", 50)
        adjuster.update_inventory("prod_3", 150)
        
        assert adjuster.get_inventory_status("prod_1") == "low_stock"
        assert adjuster.get_inventory_status("prod_2") == "normal"
        assert adjuster.get_inventory_status("prod_3") == "high_stock"
    
    def test_adjust_recommendations_for_inventory(self):
        """Test adjusting recommendations based on inventory"""
        adjuster = InventoryAwareRecommendationAdjuster()
        
        adjuster.update_inventory("prod_1", 5)   # Low stock
        adjuster.update_inventory("prod_2", 50)  # Normal
        adjuster.update_inventory("prod_3", 150) # High stock
        adjuster.update_inventory("prod_4", 0)   # Out of stock
        
        base_recs = [
            ("prod_1", 0.9),
            ("prod_2", 0.8),
            ("prod_3", 0.7),
            ("prod_4", 0.6)
        ]
        
        adjusted_recs = adjuster.adjust_recommendations_for_inventory(base_recs)
        
        # prod_4 should be removed (out of stock)
        product_ids = [r[0] for r in adjusted_recs]
        assert "prod_4" not in product_ids
        
        # prod_3 should be boosted (0.7 * 1.15 = 0.805) but prod_1 reduced (0.9 * 0.9 = 0.81)
        # So prod_1 should still be first, but prod_3 should be second
        assert adjusted_recs[0][0] == "prod_1"
        assert adjusted_recs[1][0] == "prod_3"  # Boosted from third to second


# ============================================================================
# SUBTASK 5.3 TESTS: Automated Performance Monitoring
# ============================================================================

class TestRealTimePerformanceTracker:
    """Test real-time performance tracking"""
    
    def test_record_events_and_calculate_metrics(self):
        """Test recording events and calculating metrics"""
        tracker = RealTimePerformanceTracker()
        
        # Record events
        for i in range(100):
            tracker.record_event("impression", f"prod_{i % 10}", f"user_{i}")
            if i < 20:
                tracker.record_event("click", f"prod_{i % 10}", f"user_{i}")
            if i < 10:
                tracker.record_event("conversion", f"prod_{i % 10}", f"user_{i}", revenue_pkr=5000.0)
        
        metrics = tracker.calculate_current_metrics()
        
        assert metrics.impressions == 100
        assert metrics.clicks == 20
        assert metrics.conversions == 10
        assert metrics.revenue_pkr == 50000.0
        assert metrics.click_through_rate == 0.2
        assert metrics.conversion_rate == 0.1
    
    def test_performance_by_product(self):
        """Test getting performance by product"""
        tracker = RealTimePerformanceTracker()
        
        # Record events for different products
        for i in range(50):
            tracker.record_event("impression", "prod_1", f"user_{i}")
            if i < 10:
                tracker.record_event("conversion", "prod_1", f"user_{i}", revenue_pkr=5000.0)
        
        for i in range(50):
            tracker.record_event("impression", "prod_2", f"user_{i}")
            if i < 5:
                tracker.record_event("conversion", "prod_2", f"user_{i}", revenue_pkr=3000.0)
        
        product_metrics = tracker.get_performance_by_product()
        
        assert "prod_1" in product_metrics
        assert "prod_2" in product_metrics
        assert product_metrics["prod_1"].conversion_rate > product_metrics["prod_2"].conversion_rate


class TestAutomatedAlertSystem:
    """Test automated alert system"""
    
    def test_check_performance_issues(self):
        """Test checking for performance issues"""
        alert_system = AutomatedAlertSystem()
        
        # Create metrics with low CTR
        from performance_monitoring_engine import PerformanceMetrics
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            impressions=1000,
            clicks=10,  # 1% CTR (below 2% threshold)
            conversions=2,
            revenue_pkr=10000.0,
            click_through_rate=0.01,
            conversion_rate=0.002,
            average_order_value_pkr=5000.0,
            revenue_per_impression_pkr=10.0
        )
        
        alerts = alert_system.check_performance_issues(metrics)
        
        assert len(alerts) > 0
        assert any(a.metric_name == "click_through_rate" for a in alerts)
    
    def test_revenue_drop_alert(self):
        """Test revenue drop alert"""
        alert_system = AutomatedAlertSystem()
        
        from performance_monitoring_engine import PerformanceMetrics
        
        # Set baseline
        baseline = PerformanceMetrics(
            datetime.now(), 1000, 50, 20, 100000.0, 0.05, 0.02, 5000.0, 100.0
        )
        alert_system.set_baseline(baseline)
        
        # Create metrics with significant revenue drop
        current = PerformanceMetrics(
            datetime.now(), 1000, 50, 20, 70000.0, 0.05, 0.02, 3500.0, 70.0
        )
        
        alerts = alert_system.check_performance_issues(current)
        
        revenue_alerts = [a for a in alerts if a.metric_name == "revenue_pkr"]
        assert len(revenue_alerts) > 0
        assert revenue_alerts[0].severity == AlertSeverity.CRITICAL


class TestROICalculator:
    """Test ROI calculation and tracking in PKR"""
    
    def test_record_costs_and_calculate_roi(self):
        """Test recording costs and calculating ROI"""
        calculator = ROICalculator()
        
        # Record costs
        calculator.record_cost("infrastructure", 50000.0)
        calculator.record_cost("marketing", 30000.0)
        calculator.record_cost("development", 20000.0)
        
        # Calculate ROI
        roi_result = calculator.calculate_roi(revenue_pkr=200000.0, period_days=30)
        
        assert roi_result['total_revenue_pkr'] == 200000.0
        assert roi_result['total_costs_pkr'] == 100000.0
        assert roi_result['net_profit_pkr'] == 100000.0
        assert roi_result['roi'] == 1.0  # 100% ROI
        assert roi_result['roi_percentage'] == 100.0
    
    def test_customer_acquisition_cost(self):
        """Test CAC calculation"""
        calculator = ROICalculator()
        
        cac = calculator.calculate_customer_acquisition_cost(
            new_customers=100,
            marketing_costs_pkr=50000.0
        )
        
        assert cac == 500.0  # Rs 500 per customer


class TestAutomatedOptimizationSuggester:
    """Test automated optimization suggestions"""
    
    def test_analyze_and_suggest(self):
        """Test generating optimization suggestions"""
        suggester = AutomatedOptimizationSuggester()
        
        from performance_monitoring_engine import PerformanceMetrics
        
        # Create poor performance metrics
        metrics = PerformanceMetrics(
            datetime.now(), 1000, 15, 3, 15000.0, 0.015, 0.003, 5000.0, 15.0
        )
        
        suggestions = suggester.analyze_and_suggest(metrics, {}, [])
        
        assert len(suggestions) > 0
        assert any(s['category'] == 'recommendation_relevance' for s in suggestions)
        assert any(s['category'] == 'pricing_strategy' for s in suggestions)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRevenueOptimizationIntegration:
    """Integration tests for complete revenue optimization workflow"""
    
    def test_complete_optimization_workflow(self):
        """Test complete optimization workflow"""
        # Initialize all components
        ab_framework = ABTestingFramework()
        performance_tracker = RealTimePerformanceTracker()
        alert_system = AutomatedAlertSystem()
        roi_calculator = ROICalculator()
        
        # Create A/B test
        variants = [
            ABTestVariant("control", "Control", {}, 0.5, True, datetime.now()),
            ABTestVariant("test", "Test", {}, 0.5, False, datetime.now())
        ]
        test_id = ab_framework.create_ab_test("optimization_test", variants)
        
        # Simulate traffic
        for i in range(100):
            user_id = f"user_{i}"
            variant = ab_framework.assign_variant(test_id, user_id)
            
            # Record events
            ab_framework.record_impression(test_id, variant)
            performance_tracker.record_event("impression", f"prod_{i % 10}", user_id)
            
            if i < 20:
                ab_framework.record_click(test_id, variant)
                performance_tracker.record_event("click", f"prod_{i % 10}", user_id)
            
            if i < 10:
                revenue = 5000.0
                ab_framework.record_conversion(test_id, variant, revenue)
                performance_tracker.record_event("conversion", f"prod_{i % 10}", user_id, revenue)
        
        # Calculate metrics
        metrics = performance_tracker.calculate_current_metrics()
        assert metrics.impressions == 100
        
        # Check for alerts
        alerts = alert_system.check_performance_issues(metrics)
        
        # Calculate ROI
        roi_calculator.record_cost("infrastructure", 10000.0)
        roi_result = roi_calculator.calculate_roi(metrics.revenue_pkr)
        
        assert roi_result['total_revenue_pkr'] == metrics.revenue_pkr
        
        # Get A/B test results
        ab_results = ab_framework.calculate_test_results(test_id)
        assert len(ab_results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
