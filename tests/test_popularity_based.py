"""
Unit tests for popularity-based recommendation engine
Tests demographic segmentation, trending analysis, and recommendation accuracy
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from algorithms.popularity_based import (
    CustomerSegmentationEngine, TrendingProductAnalyzer, PopularityBasedEngine
)
from models.recommendation import (
    RecommendationRequest, RecommendationContext, RecommendationAlgorithm
)


class TestCustomerSegmentationEngine:
    """Test cases for customer segmentation"""
    
    def setup_method(self):
        """Setup test data"""
        self.segmentation_engine = CustomerSegmentationEngine()
        self.segmentation_engine.create_income_segments()
    
    def test_create_income_segments(self):
        """Test creation of income-based segments"""
        segments = self.segmentation_engine.create_income_segments()
        
        assert len(segments) == 4
        assert 'premium_karachi' in segments
        assert 'premium_lahore' in segments
        assert 'premium_islamabad' in segments
        assert 'general_metro' in segments
        
        # Check segment properties
        karachi_segment = segments['premium_karachi']
        assert karachi_segment.target_income_range == (300000, 500000)
        assert 'Karachi' in karachi_segment.geographic_focus
    
    def test_assign_customer_segment_premium_karachi(self):
        """Test assignment to premium Karachi segment"""
        customer_data = {
            'city': 'Karachi',
            'income': 400000
        }
        
        segment = self.segmentation_engine.assign_customer_segment(customer_data)
        assert segment == 'premium_karachi'
    
    def test_assign_customer_segment_premium_lahore(self):
        """Test assignment to premium Lahore segment"""
        customer_data = {
            'city': 'Lahore',
            'income': 350000
        }
        
        segment = self.segmentation_engine.assign_customer_segment(customer_data)
        assert segment == 'premium_lahore'
    
    def test_assign_customer_segment_premium_islamabad(self):
        """Test assignment to premium Islamabad segment"""
        customer_data = {
            'city': 'Islamabad',
            'income': 450000
        }
        
        segment = self.segmentation_engine.assign_customer_segment(customer_data)
        assert segment == 'premium_islamabad'
    
    def test_assign_customer_segment_general_metro_low_income(self):
        """Test assignment to general metro for low income"""
        customer_data = {
            'city': 'Karachi',
            'income': 250000  # Below premium threshold
        }
        
        segment = self.segmentation_engine.assign_customer_segment(customer_data)
        assert segment == 'general_metro'
    
    def test_assign_customer_segment_general_metro_high_income(self):
        """Test assignment to general metro for high income"""
        customer_data = {
            'city': 'Karachi',
            'income': 600000  # Above premium threshold
        }
        
        segment = self.segmentation_engine.assign_customer_segment(customer_data)
        assert segment == 'general_metro'
    
    def test_assign_customer_segment_unknown_city(self):
        """Test assignment for unknown city"""
        customer_data = {
            'city': 'Faisalabad',
            'income': 400000
        }
        
        segment = self.segmentation_engine.assign_customer_segment(customer_data)
        assert segment == 'general_metro'


class TestTrendingProductAnalyzer:
    """Test cases for trending product analysis"""
    
    def setup_method(self):
        """Setup test data"""
        self.analyzer = TrendingProductAnalyzer(trend_window_days=30)
        
        # Create sample sales data
        np.random.seed(42)
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=60),
            end=datetime.now(),
            freq='D'
        )
        
        sales_data = []
        products = [f'product_{i}' for i in range(10)]
        
        for date in dates:
            for product in products:
                # Some products are more popular
                if 'product_1' in product or 'product_2' in product:
                    n_sales = np.random.poisson(5)  # High sales
                else:
                    n_sales = np.random.poisson(1)  # Low sales
                
                for _ in range(n_sales):
                    sales_data.append({
                        'product_id': product,
                        'sale_date': date,
                        'quantity': np.random.randint(1, 5),
                        'amount': np.random.uniform(100, 1000)
                    })
        
        self.sales_data = pd.DataFrame(sales_data)
        
        # Create product data
        self.product_data = pd.DataFrame({
            'product_id': products,
            'category_id': [f'category_{i%3}' for i in range(10)],
            'name': [f'Product {i}' for i in range(10)]
        })
    
    def test_calculate_sales_velocity(self):
        """Test sales velocity calculation"""
        velocity_data = self.analyzer.calculate_sales_velocity(self.sales_data)
        
        assert isinstance(velocity_data, dict)
        assert len(velocity_data) > 0
        
        # Check velocity data structure
        for product_id, data in velocity_data.items():
            assert 'sales_velocity' in data
            assert 'revenue_velocity' in data
            assert 'total_sales' in data
            assert 'total_revenue' in data
            assert 'days_active' in data
            
            assert data['sales_velocity'] >= 0
            assert data['revenue_velocity'] >= 0
            assert data['total_sales'] >= 0
            assert data['days_active'] > 0
        
        # Product_1 should have higher velocity than others
        if 'product_1' in velocity_data and 'product_5' in velocity_data:
            assert velocity_data['product_1']['sales_velocity'] > velocity_data['product_5']['sales_velocity']
    
    def test_identify_trending_products(self):
        """Test trending product identification"""
        trending = self.analyzer.identify_trending_products(
            self.sales_data, self.product_data, top_n=5
        )
        
        assert isinstance(trending, list)
        assert len(trending) <= 5
        assert len(trending) > 0
        
        # Check trending product structure
        for product in trending:
            assert 'product_id' in product
            assert 'trending_score' in product
            assert 'sales_velocity' in product
            assert 'revenue_velocity' in product
            assert 'category' in product
            
            assert product['trending_score'] > 0
        
        # Check that products are sorted by trending score
        scores = [p['trending_score'] for p in trending]
        assert scores == sorted(scores, reverse=True)
    
    def test_analyze_category_trends(self):
        """Test category trend analysis"""
        category_trends = self.analyzer.analyze_category_trends(
            self.sales_data, self.product_data
        )
        
        assert isinstance(category_trends, dict)
        assert len(category_trends) > 0
        
        # Check category trend structure
        for category_id, trends in category_trends.items():
            assert 'growth_rate' in trends
            assert 'recent_revenue' in trends
            assert 'previous_revenue' in trends
            assert 'total_products' in trends
            assert 'avg_sales_velocity' in trends
            
            assert trends['growth_rate'] >= 0
            assert trends['recent_revenue'] >= 0
            assert trends['previous_revenue'] >= 0
            assert trends['total_products'] > 0


class TestPopularityBasedEngine:
    """Test cases for main popularity-based engine"""
    
    def setup_method(self):
        """Setup test data and engine"""
        self.engine = PopularityBasedEngine(
            min_sales_threshold=3,
            popularity_weight=0.4,
            trend_weight=0.3,
            segment_weight=0.3
        )
        
        # Create comprehensive test data
        np.random.seed(42)
        
        # Sales data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=90),
            end=datetime.now(),
            freq='D'
        )
        
        sales_data = []
        products = [f'product_{i}' for i in range(20)]
        customers = [f'customer_{i}' for i in range(50)]
        
        for date in dates:
            n_transactions = np.random.poisson(10)
            for _ in range(n_transactions):
                sales_data.append({
                    'product_id': np.random.choice(products),
                    'customer_id': np.random.choice(customers),
                    'sale_date': date,
                    'quantity': np.random.randint(1, 5),
                    'amount': np.random.uniform(100, 2000)
                })
        
        self.sales_data = pd.DataFrame(sales_data)
        
        # Product data
        self.product_data = pd.DataFrame({
            'product_id': products,
            'category_id': [f'category_{i%5}' for i in range(20)],
            'name': [f'Product {i}' for i in range(20)]
        })
        
        # Customer data
        cities = ['Karachi', 'Lahore', 'Islamabad', 'Faisalabad']
        income_brackets = ['200k-300k PKR', '300k-500k PKR', '500k-700k PKR']
        
        customer_data = []
        for customer in customers:
            customer_data.append({
                'customer_id': customer,
                'city': np.random.choice(cities),
                'income_bracket': np.random.choice(income_brackets)
            })
        
        self.customer_data = pd.DataFrame(customer_data)
    
    def test_train_success(self):
        """Test successful model training"""
        metrics = self.engine.train(
            self.sales_data, 
            self.product_data, 
            self.customer_data
        )
        
        assert self.engine.is_trained
        assert isinstance(metrics, dict)
        
        # Check metrics
        assert 'training_time_seconds' in metrics
        assert 'n_products_analyzed' in metrics
        assert 'n_trending_products' in metrics
        assert 'n_segments' in metrics
        assert 'avg_popularity_score' in metrics
        
        assert metrics['training_time_seconds'] > 0
        assert metrics['n_products_analyzed'] > 0
        assert metrics['n_segments'] == 4  # Should have 4 segments
        
        # Check that internal data structures are populated
        assert len(self.engine.popularity_scores) > 0
        assert len(self.engine.segment_preferences) > 0
    
    def test_calculate_popularity_scores(self):
        """Test popularity score calculation"""
        popularity_scores = self.engine._calculate_popularity_scores(
            self.sales_data, self.product_data
        )
        
        assert isinstance(popularity_scores, dict)
        assert len(popularity_scores) > 0
        
        # Check that all scores are positive
        for product_id, score in popularity_scores.items():
            assert score > 0
            assert isinstance(score, (int, float))
    
    def test_calculate_segment_preferences(self):
        """Test segment preference calculation"""
        # First create segments
        self.engine.segmentation_engine.create_income_segments()
        
        segment_prefs = self.engine._calculate_segment_preferences(
            self.sales_data, self.product_data, self.customer_data
        )
        
        assert isinstance(segment_prefs, dict)
        assert len(segment_prefs) > 0
        
        # Check segment preference structure
        for segment_id, preferences in segment_prefs.items():
            assert isinstance(preferences, dict)
            
            # Check that preferences sum to approximately 1 (normalized)
            if preferences:
                total_pref = sum(preferences.values())
                assert abs(total_pref - 1.0) < 0.01  # Allow small floating point errors
    
    def test_parse_income_bracket(self):
        """Test income bracket parsing"""
        # Test various income bracket formats
        assert self.engine._parse_income_bracket('300k-500k PKR') == 400000
        assert self.engine._parse_income_bracket('200k-400k') == 300000
        assert self.engine._parse_income_bracket('500k') == 500000
        assert self.engine._parse_income_bracket('') == 350000  # Default
        assert self.engine._parse_income_bracket(None) == 350000  # Default
    
    def test_get_recommendations_trained_model(self):
        """Test recommendation generation with trained model"""
        self.engine.train(self.sales_data, self.product_data, self.customer_data)
        
        # Test with customer data
        customer_data = {
            'city': 'Karachi',
            'income': 400000
        }
        
        request = RecommendationRequest(
            user_id='test_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        response = self.engine.get_recommendations(request, customer_data)
        
        assert response.total_count <= 5
        assert response.algorithm_used == RecommendationAlgorithm.POPULARITY_BASED
        assert not response.fallback_applied
        assert response.processing_time_ms > 0
        
        # Check recommendation objects
        for rec in response.recommendations:
            assert rec.user_id == 'test_user'
            assert isinstance(rec.product_id, str)
            assert rec.score > 0
            assert rec.algorithm == RecommendationAlgorithm.POPULARITY_BASED
            assert 0 <= rec.metadata.confidence_score <= 1
            assert 'segment' in rec.metadata.explanation
    
    def test_get_recommendations_untrained_model(self):
        """Test recommendation generation with untrained model"""
        request = RecommendationRequest(
            user_id='test_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        response = self.engine.get_recommendations(request)
        
        assert response.total_count == 0
        assert response.fallback_applied
        assert len(response.recommendations) == 0
    
    def test_get_recommendations_with_exclusions(self):
        """Test recommendations with excluded products"""
        self.engine.train(self.sales_data, self.product_data, self.customer_data)
        
        # Get initial recommendations
        request1 = RecommendationRequest(
            user_id='test_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        response1 = self.engine.get_recommendations(request1)
        initial_products = [rec.product_id for rec in response1.recommendations]
        
        # Get recommendations with exclusions
        request2 = RecommendationRequest(
            user_id='test_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5,
            exclude_products=initial_products[:2]  # Exclude first 2 products
        )
        
        response2 = self.engine.get_recommendations(request2)
        excluded_products = [rec.product_id for rec in response2.recommendations]
        
        # Check that excluded products are not in new recommendations
        for excluded_product in request2.exclude_products:
            assert excluded_product not in excluded_products
    
    def test_get_segment_recommendations(self):
        """Test segment-specific recommendations"""
        self.engine.train(self.sales_data, self.product_data, self.customer_data)
        
        segment_recs = self.engine.get_segment_recommendations(
            'premium_karachi', num_recommendations=5
        )
        
        assert isinstance(segment_recs, list)
        assert len(segment_recs) <= 5
        
        # Check recommendation format
        for product_id, score in segment_recs:
            assert isinstance(product_id, str)
            assert isinstance(score, (int, float))
            assert score >= 0
        
        # Check that recommendations are sorted by score
        scores = [score for _, score in segment_recs]
        assert scores == sorted(scores, reverse=True)
    
    def test_get_segment_recommendations_invalid_segment(self):
        """Test segment recommendations for invalid segment"""
        self.engine.train(self.sales_data, self.product_data, self.customer_data)
        
        segment_recs = self.engine.get_segment_recommendations('invalid_segment')
        assert segment_recs == []
    
    def test_get_trending_recommendations(self):
        """Test trending product recommendations"""
        self.engine.train(self.sales_data, self.product_data, self.customer_data)
        
        trending_recs = self.engine.get_trending_recommendations(num_recommendations=5)
        
        assert isinstance(trending_recs, list)
        assert len(trending_recs) <= 5
        
        # Check trending recommendation structure
        for trending_product in trending_recs:
            assert 'product_id' in trending_product
            assert 'trending_score' in trending_product
            assert 'sales_velocity' in trending_product
            assert trending_product['trending_score'] > 0
    
    def test_get_trending_recommendations_untrained(self):
        """Test trending recommendations with untrained model"""
        trending_recs = self.engine.get_trending_recommendations()
        assert trending_recs == []


class TestSegmentAccuracy:
    """Test accuracy of customer segmentation for 300k-500k PKR bracket"""
    
    def setup_method(self):
        """Setup segmentation engine"""
        self.engine = CustomerSegmentationEngine()
        self.engine.create_income_segments()
    
    def test_target_income_bracket_accuracy(self):
        """Test accuracy of targeting 300k-500k PKR income bracket"""
        test_cases = [
            # Should be assigned to premium segments
            ({'city': 'Karachi', 'income': 300000}, 'premium_karachi'),
            ({'city': 'Karachi', 'income': 400000}, 'premium_karachi'),
            ({'city': 'Karachi', 'income': 500000}, 'premium_karachi'),
            ({'city': 'Lahore', 'income': 350000}, 'premium_lahore'),
            ({'city': 'Islamabad', 'income': 450000}, 'premium_islamabad'),
            
            # Should be assigned to general metro
            ({'city': 'Karachi', 'income': 250000}, 'general_metro'),
            ({'city': 'Karachi', 'income': 600000}, 'general_metro'),
            ({'city': 'Faisalabad', 'income': 400000}, 'general_metro'),
        ]
        
        correct_assignments = 0
        total_assignments = len(test_cases)
        
        for customer_data, expected_segment in test_cases:
            actual_segment = self.engine.assign_customer_segment(customer_data)
            if actual_segment == expected_segment:
                correct_assignments += 1
        
        accuracy = correct_assignments / total_assignments
        assert accuracy >= 0.85  # Should have at least 85% accuracy
        
        print(f"Segmentation accuracy: {accuracy:.2%}")


class TestPerformanceBenchmarks:
    """Performance benchmark tests for popularity-based engine"""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        # Create large dataset
        np.random.seed(42)
        
        # Large sales data
        n_products = 1000
        n_customers = 5000
        n_transactions = 100000
        
        products = [f'product_{i}' for i in range(n_products)]
        customers = [f'customer_{i}' for i in range(n_customers)]
        
        sales_data = []
        for _ in range(n_transactions):
            sales_data.append({
                'product_id': np.random.choice(products),
                'customer_id': np.random.choice(customers),
                'sale_date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'quantity': np.random.randint(1, 5),
                'amount': np.random.uniform(100, 2000)
            })
        
        sales_df = pd.DataFrame(sales_data)
        
        # Product data
        product_df = pd.DataFrame({
            'product_id': products,
            'category_id': [f'category_{i%20}' for i in range(n_products)],
            'name': [f'Product {i}' for i in range(n_products)]
        })
        
        # Customer data
        cities = ['Karachi', 'Lahore', 'Islamabad']
        income_brackets = ['300k-500k PKR', '400k-600k PKR', '200k-400k PKR']
        
        customer_df = pd.DataFrame({
            'customer_id': customers,
            'city': np.random.choice(cities, n_customers),
            'income_bracket': np.random.choice(income_brackets, n_customers)
        })
        
        # Test training performance
        engine = PopularityBasedEngine(min_sales_threshold=10)
        start_time = datetime.now()
        
        metrics = engine.train(sales_df, product_df, customer_df)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert training_time < 30  # Should complete within 30 seconds
        assert engine.is_trained
        assert metrics['n_products_analyzed'] > 0
        
        # Test recommendation generation performance
        request = RecommendationRequest(
            user_id='test_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=10
        )
        
        customer_data = {'city': 'Karachi', 'income': 400000}
        
        start_time = datetime.now()
        response = engine.get_recommendations(request, customer_data)
        recommendation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Should generate recommendations within 200ms (requirement)
        assert recommendation_time < 200
        assert response.processing_time_ms < 200
        assert len(response.recommendations) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])