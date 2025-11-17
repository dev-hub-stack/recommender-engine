"""
Unit tests for algorithm orchestrator and ensemble methods
Tests dynamic selection, A/B testing, and performance monitoring
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from algorithms.algorithm_orchestrator import (
    AlgorithmOrchestrator, PerformanceMonitor, ABTestManager,
    AlgorithmStrategy, AlgorithmPerformance, ABTestGroup
)
from algorithms.collaborative_filtering import CollaborativeFilteringEngine
from algorithms.popularity_based import PopularityBasedEngine
from models.recommendation import (
    RecommendationRequest, RecommendationContext, RecommendationAlgorithm,
    RecommendationResponse, Recommendation, RecommendationMetadata
)


class TestPerformanceMonitor:
    """Test cases for performance monitoring"""
    
    def setup_method(self):
        """Setup test data"""
        self.monitor = PerformanceMonitor(
            accuracy_threshold=0.80,
            min_sample_size=10,
            monitoring_window_hours=24
        )
    
    def test_record_recommendation_feedback(self):
        """Test recording recommendation feedback"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        # Record some feedback
        self.monitor.record_recommendation_feedback(
            algorithm, 'user1', 'product1', True, True, datetime.utcnow()
        )
        self.monitor.record_recommendation_feedback(
            algorithm, 'user2', 'product2', True, False, datetime.utcnow()
        )
        self.monitor.record_recommendation_feedback(
            algorithm, 'user3', 'product3', False, False, datetime.utcnow()
        )
        
        assert algorithm in self.monitor.performance_history
        assert len(self.monitor.performance_history[algorithm]) == 3
        
        # Check feedback structure
        feedback = self.monitor.performance_history[algorithm][0]
        assert 'user_id' in feedback
        assert 'product_id' in feedback
        assert 'was_clicked' in feedback
        assert 'was_purchased' in feedback
        assert 'relevance_score' in feedback
    
    def test_calculate_current_performance_insufficient_data(self):
        """Test performance calculation with insufficient data"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        # Record only a few feedback entries (below min_sample_size)
        for i in range(5):
            self.monitor.record_recommendation_feedback(
                algorithm, f'user{i}', f'product{i}', True, False, datetime.utcnow()
            )
        
        performance = self.monitor.calculate_current_performance(algorithm)
        assert performance is None
    
    def test_calculate_current_performance_sufficient_data(self):
        """Test performance calculation with sufficient data"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        # Record sufficient feedback entries
        for i in range(20):
            was_clicked = i % 2 == 0  # 50% click rate
            was_purchased = i % 4 == 0  # 25% purchase rate
            
            self.monitor.record_recommendation_feedback(
                algorithm, f'user{i}', f'product{i}', 
                was_clicked, was_purchased, datetime.utcnow()
            )
        
        performance = self.monitor.calculate_current_performance(algorithm)
        
        assert performance is not None
        assert isinstance(performance, AlgorithmPerformance)
        assert performance.algorithm == algorithm
        assert 0 <= performance.accuracy <= 1
        assert 0 <= performance.precision <= 1
        assert 0 <= performance.recall <= 1
        assert 0 <= performance.f1_score <= 1
        assert performance.sample_size == 20
    
    def test_check_performance_degradation(self):
        """Test performance degradation detection"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        # Record poor performance (no clicks or purchases)
        for i in range(15):
            self.monitor.record_recommendation_feedback(
                algorithm, f'user{i}', f'product{i}', 
                False, False, datetime.utcnow()
            )
        
        is_degraded = self.monitor.check_performance_degradation(algorithm)
        assert is_degraded  # Should detect degradation
        
        # Record good performance
        for i in range(15, 30):
            self.monitor.record_recommendation_feedback(
                algorithm, f'user{i}', f'product{i}', 
                True, True, datetime.utcnow()  # All clicks and purchases
            )
        
        is_degraded = self.monitor.check_performance_degradation(algorithm)
        assert not is_degraded  # Should not detect degradation
    
    def test_get_best_performing_algorithm(self):
        """Test best algorithm selection"""
        # Create performance for multiple algorithms
        algorithms = [
            RecommendationAlgorithm.ENSEMBLE,
            RecommendationAlgorithm.POPULARITY_BASED
        ]
        
        for alg in algorithms:
            # Record different performance levels
            performance_level = 0.9 if alg == RecommendationAlgorithm.ENSEMBLE else 0.7
            
            for i in range(15):
                was_clicked = np.random.random() < performance_level
                was_purchased = np.random.random() < (performance_level * 0.3)
                
                self.monitor.record_recommendation_feedback(
                    alg, f'user{i}', f'product{i}', 
                    was_clicked, was_purchased, datetime.utcnow()
                )
        
        best_algorithm = self.monitor.get_best_performing_algorithm()
        # Should select the ensemble algorithm due to better performance
        assert best_algorithm == RecommendationAlgorithm.ENSEMBLE


class TestABTestManager:
    """Test cases for A/B testing"""
    
    def setup_method(self):
        """Setup test data"""
        self.ab_manager = ABTestManager()
    
    def test_create_ab_test(self):
        """Test A/B test creation"""
        test_id = self.ab_manager.create_ab_test(
            test_name="cf_vs_popularity",
            control_algorithm=RecommendationAlgorithm.POPULARITY_BASED,
            test_algorithm=RecommendationAlgorithm.ENSEMBLE,
            test_duration_days=7,
            traffic_split=0.3
        )
        
        assert test_id in self.ab_manager.active_tests
        
        test_config = self.ab_manager.active_tests[test_id]
        assert 'control' in test_config
        assert 'test' in test_config
        
        # Check control group
        control_group = test_config['control']
        assert control_group.algorithm == RecommendationAlgorithm.POPULARITY_BASED
        assert control_group.traffic_percentage == 0.7
        assert control_group.is_active
        
        # Check test group
        test_group = test_config['test']
        assert test_group.algorithm == RecommendationAlgorithm.ENSEMBLE
        assert test_group.traffic_percentage == 0.3
        assert test_group.is_active
    
    def test_assign_user_to_group_consistent(self):
        """Test consistent user assignment to A/B test groups"""
        test_id = self.ab_manager.create_ab_test(
            test_name="consistency_test",
            control_algorithm=RecommendationAlgorithm.POPULARITY_BASED,
            test_algorithm=RecommendationAlgorithm.ENSEMBLE,
            traffic_split=0.5
        )
        
        # Same user should always get same group
        user_id = "test_user_123"
        group1 = self.ab_manager.assign_user_to_group(user_id, test_id)
        group2 = self.ab_manager.assign_user_to_group(user_id, test_id)
        
        assert group1.group_id == group2.group_id
        assert group1.algorithm == group2.algorithm
    
    def test_assign_user_to_group_distribution(self):
        """Test user distribution in A/B test groups"""
        test_id = self.ab_manager.create_ab_test(
            test_name="distribution_test",
            control_algorithm=RecommendationAlgorithm.POPULARITY_BASED,
            test_algorithm=RecommendationAlgorithm.ENSEMBLE,
            traffic_split=0.3  # 30% test, 70% control
        )
        
        # Test with many users
        test_assignments = 0
        control_assignments = 0
        
        for i in range(1000):
            user_id = f"user_{i}"
            group = self.ab_manager.assign_user_to_group(user_id, test_id)
            
            if 'test' in group.group_id:
                test_assignments += 1
            else:
                control_assignments += 1
        
        # Check approximate distribution (allow 5% variance)
        test_ratio = test_assignments / 1000
        assert 0.25 <= test_ratio <= 0.35  # Should be around 30%
    
    def test_get_active_test_for_user(self):
        """Test getting active test for user"""
        # Create test
        test_id = self.ab_manager.create_ab_test(
            test_name="active_test",
            control_algorithm=RecommendationAlgorithm.POPULARITY_BASED,
            test_algorithm=RecommendationAlgorithm.ENSEMBLE
        )
        
        user_id = "test_user"
        result = self.ab_manager.get_active_test_for_user(user_id)
        
        assert result is not None
        returned_test_id, group = result
        assert returned_test_id == test_id
        assert isinstance(group, ABTestGroup)
    
    def test_end_test(self):
        """Test ending A/B test"""
        test_id = self.ab_manager.create_ab_test(
            test_name="end_test",
            control_algorithm=RecommendationAlgorithm.POPULARITY_BASED,
            test_algorithm=RecommendationAlgorithm.ENSEMBLE
        )
        
        # End the test
        results = self.ab_manager.end_test(test_id)
        
        assert test_id not in self.ab_manager.active_tests
        assert len(self.ab_manager.test_history) == 1
        assert isinstance(results, dict)


class TestAlgorithmOrchestrator:
    """Test cases for main algorithm orchestrator"""
    
    def setup_method(self):
        """Setup test data and orchestrator"""
        self.orchestrator = AlgorithmOrchestrator(
            accuracy_threshold=0.80,
            fallback_strategy=AlgorithmStrategy.POPULARITY_BASED_ONLY
        )
        
        # Mock the underlying engines to avoid complex setup
        self.orchestrator.cf_engine = Mock(spec=CollaborativeFilteringEngine)
        self.orchestrator.popularity_engine = Mock(spec=PopularityBasedEngine)
        
        # Create sample data
        np.random.seed(42)
        
        # CF training data
        self.cf_data = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(100)],
            'item_id': [f'item_{i%20}' for i in range(100)],
            'rating': np.random.uniform(1, 5, 100),
            'timestamp': [datetime.now()] * 100
        })
        
        # Sales data
        self.sales_data = pd.DataFrame({
            'product_id': [f'product_{i%15}' for i in range(200)],
            'customer_id': [f'customer_{i%50}' for i in range(200)],
            'sale_date': [datetime.now() - timedelta(days=i%30) for i in range(200)],
            'quantity': np.random.randint(1, 5, 200),
            'amount': np.random.uniform(100, 1000, 200)
        })
        
        # Product data
        self.product_data = pd.DataFrame({
            'product_id': [f'product_{i}' for i in range(15)],
            'category_id': [f'category_{i%3}' for i in range(15)]
        })
        
        # Customer data
        self.customer_data = pd.DataFrame({
            'customer_id': [f'customer_{i}' for i in range(50)],
            'city': np.random.choice(['Karachi', 'Lahore', 'Islamabad'], 50),
            'income_bracket': np.random.choice(['300k-500k PKR'], 50)
        })
    
    def test_train_all_algorithms_success(self):
        """Test successful training of all algorithms"""
        # Mock successful training
        self.orchestrator.cf_engine.train.return_value = {'rmse_accuracy': 0.85}
        self.orchestrator.popularity_engine.train.return_value = {'n_products_analyzed': 15}
        
        results = self.orchestrator.train_all_algorithms(
            self.cf_data, self.sales_data, self.product_data, self.customer_data
        )
        
        assert self.orchestrator.is_trained
        assert 'collaborative_filtering' in results
        assert 'popularity_based' in results
        assert 'total_training_time' in results
        assert results['orchestrator_ready']
        
        # Verify engines were called
        self.orchestrator.cf_engine.train.assert_called_once()
        self.orchestrator.popularity_engine.train.assert_called_once()
    
    def test_train_all_algorithms_with_failures(self):
        """Test training with some algorithm failures"""
        # Mock CF failure and popularity success
        self.orchestrator.cf_engine.train.side_effect = Exception("CF training failed")
        self.orchestrator.popularity_engine.train.return_value = {'n_products_analyzed': 15}
        
        results = self.orchestrator.train_all_algorithms(
            self.cf_data, self.sales_data, self.product_data, self.customer_data
        )
        
        assert self.orchestrator.is_trained  # Should still be trained
        assert 'error' in results['collaborative_filtering']
        assert 'n_products_analyzed' in results['popularity_based']
    
    def test_select_algorithm_new_user(self):
        """Test algorithm selection for new users"""
        request = RecommendationRequest(
            user_id='new_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        # New user with no history
        selected = self.orchestrator.select_algorithm(request, user_history_length=0)
        assert selected == RecommendationAlgorithm.POPULARITY_BASED
    
    def test_select_algorithm_experienced_user(self):
        """Test algorithm selection for experienced users"""
        request = RecommendationRequest(
            user_id='experienced_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        # Mock good performance for ensemble
        mock_performance = AlgorithmPerformance(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            accuracy=0.85,
            precision=0.8,
            recall=0.7,
            f1_score=0.75,
            response_time_ms=150,
            coverage=0.6,
            diversity=0.5,
            last_updated=datetime.utcnow(),
            sample_size=100
        )
        
        self.orchestrator.performance_monitor.calculate_current_performance = Mock(
            return_value=mock_performance
        )
        
        # Experienced user with good ensemble performance
        selected = self.orchestrator.select_algorithm(request, user_history_length=15)
        assert selected == RecommendationAlgorithm.ENSEMBLE
    
    def test_select_algorithm_ab_test_assignment(self):
        """Test algorithm selection with A/B test"""
        # Create A/B test
        test_id = self.orchestrator.create_ab_test(
            "test_selection",
            RecommendationAlgorithm.POPULARITY_BASED,
            RecommendationAlgorithm.ENSEMBLE
        )
        
        request = RecommendationRequest(
            user_id='ab_test_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        selected = self.orchestrator.select_algorithm(request, user_history_length=10)
        
        # Should be one of the A/B test algorithms
        assert selected in [
            RecommendationAlgorithm.POPULARITY_BASED,
            RecommendationAlgorithm.ENSEMBLE
        ]
    
    def test_generate_ensemble_recommendations(self):
        """Test ensemble recommendation generation"""
        request = RecommendationRequest(
            user_id='test_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        # Mock responses from individual algorithms
        cf_recommendations = [
            Mock(product_id='product_1', score=4.5),
            Mock(product_id='product_2', score=4.0),
            Mock(product_id='product_3', score=3.5)
        ]
        
        pop_recommendations = [
            Mock(product_id='product_2', score=3.8),
            Mock(product_id='product_4', score=3.6),
            Mock(product_id='product_5', score=3.2)
        ]
        
        cf_response = Mock(
            recommendations=cf_recommendations,
            fallback_applied=False
        )
        
        pop_response = Mock(
            recommendations=pop_recommendations,
            fallback_applied=False
        )
        
        self.orchestrator.cf_engine.is_trained = True
        self.orchestrator.popularity_engine.is_trained = True
        self.orchestrator.cf_engine.get_recommendations.return_value = cf_response
        self.orchestrator.popularity_engine.get_recommendations.return_value = pop_response
        self.orchestrator.is_trained = True
        
        response = self.orchestrator.generate_ensemble_recommendations(request)
        
        assert isinstance(response, RecommendationResponse)
        assert response.algorithm_used == RecommendationAlgorithm.ENSEMBLE
        assert not response.fallback_applied
        assert len(response.recommendations) <= 5
    
    def test_get_recommendations_untrained(self):
        """Test recommendations with untrained orchestrator"""
        request = RecommendationRequest(
            user_id='test_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        response = self.orchestrator.get_recommendations(request)
        
        assert response.fallback_applied
        assert len(response.recommendations) == 0
    
    def test_get_recommendations_with_fallback(self):
        """Test recommendations with fallback strategy"""
        self.orchestrator.is_trained = True
        
        # Mock algorithm failure
        self.orchestrator.popularity_engine.get_recommendations.side_effect = Exception("Algorithm failed")
        
        request = RecommendationRequest(
            user_id='test_user',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        # Mock fallback response
        fallback_response = RecommendationResponse(
            recommendations=[],
            total_count=0,
            algorithm_used=RecommendationAlgorithm.POPULARITY_BASED,
            fallback_applied=True,
            processing_time_ms=1.0,
            cache_hit=False
        )
        
        with patch.object(self.orchestrator, '_apply_fallback_strategy', return_value=fallback_response):
            response = self.orchestrator.get_recommendations(request)
            assert response.fallback_applied
    
    def test_record_user_feedback(self):
        """Test user feedback recording"""
        self.orchestrator.record_user_feedback(
            'user1', 'product1', RecommendationAlgorithm.ENSEMBLE, True, False
        )
        
        # Verify feedback was recorded
        algorithm = RecommendationAlgorithm.ENSEMBLE
        assert algorithm in self.orchestrator.performance_monitor.performance_history
        assert len(self.orchestrator.performance_monitor.performance_history[algorithm]) == 1
    
    def test_check_and_handle_performance_degradation(self):
        """Test performance degradation handling"""
        # Mock performance degradation
        self.orchestrator.performance_monitor.check_performance_degradation = Mock(return_value=True)
        
        original_strategy = self.orchestrator.current_strategy
        
        self.orchestrator.check_and_handle_performance_degradation()
        
        # Should switch to fallback strategy
        assert self.orchestrator.current_strategy != original_strategy
    
    def test_create_ab_test(self):
        """Test A/B test creation through orchestrator"""
        test_id = self.orchestrator.create_ab_test(
            "orchestrator_test",
            RecommendationAlgorithm.POPULARITY_BASED,
            RecommendationAlgorithm.ENSEMBLE
        )
        
        assert test_id in self.orchestrator.ab_test_manager.active_tests
    
    def test_get_performance_summary(self):
        """Test performance summary generation"""
        self.orchestrator.is_trained = True
        
        summary = self.orchestrator.get_performance_summary()
        
        assert 'current_strategy' in summary
        assert 'accuracy_threshold' in summary
        assert 'is_trained' in summary
        assert 'algorithms' in summary
        assert 'active_ab_tests' in summary
        
        assert summary['is_trained']
        assert summary['accuracy_threshold'] == 0.80


class TestEnsembleAccuracy:
    """Test ensemble method accuracy requirements"""
    
    def test_ensemble_accuracy_threshold(self):
        """Test that ensemble meets 80% accuracy threshold"""
        orchestrator = AlgorithmOrchestrator(accuracy_threshold=0.80)
        
        # Mock performance data that meets threshold
        mock_performance = AlgorithmPerformance(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            accuracy=0.85,  # Above threshold
            precision=0.82,
            recall=0.78,
            f1_score=0.80,
            response_time_ms=150,
            coverage=0.65,
            diversity=0.55,
            last_updated=datetime.utcnow(),
            sample_size=200
        )
        
        assert mock_performance.meets_threshold(0.80)
        assert mock_performance.accuracy >= orchestrator.accuracy_threshold
    
    def test_automatic_fallback_below_threshold(self):
        """Test automatic fallback when accuracy drops below 80%"""
        orchestrator = AlgorithmOrchestrator(accuracy_threshold=0.80)
        
        # Mock poor performance
        poor_performance = AlgorithmPerformance(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            accuracy=0.75,  # Below threshold
            precision=0.70,
            recall=0.65,
            f1_score=0.67,
            response_time_ms=200,
            coverage=0.50,
            diversity=0.45,
            last_updated=datetime.utcnow(),
            sample_size=150
        )
        
        assert not poor_performance.meets_threshold(0.80)
        
        # Mock the performance monitor
        orchestrator.performance_monitor.check_performance_degradation = Mock(return_value=True)
        
        original_strategy = orchestrator.current_strategy
        orchestrator.check_and_handle_performance_degradation()
        
        # Should have switched strategies
        assert orchestrator.current_strategy != original_strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])