"""
Integration Tests for Deep Learning Engine with Algorithm Orchestrator
Tests the integration of deep learning models with the existing recommendation system
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.algorithm_orchestrator import AlgorithmOrchestrator, AlgorithmStrategy
from algorithms.deep_learning_engine import ContextualData
from shared.models.recommendation import (
    RecommendationRequest, RecommendationContext, RecommendationAlgorithm,
    RecommendationResponse, Recommendation, RecommendationMetadata
)


class TestDeepLearningIntegration(unittest.TestCase):
    """Test deep learning integration with algorithm orchestrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = AlgorithmOrchestrator(
            accuracy_threshold=0.80,
            fallback_strategy=AlgorithmStrategy.POPULARITY_BASED_ONLY
        )
        
        # Create mock training data
        self.cf_data = pd.DataFrame({
            'user_id': ['USER_001', 'USER_002'] * 50,
            'product_id': ['PROD_001', 'PROD_002'] * 50,
            'rating': [4.5, 3.8] * 50
        })
        
        self.sales_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'] * 30,
            'product_id': ['PROD_001', 'PROD_002'] * 30,
            'amount': [25000, 15000] * 30,
            'order_date': [datetime.utcnow() - timedelta(days=i) for i in range(60)]
        })
        
        self.product_data = pd.DataFrame({
            'product_id': ['PROD_001', 'PROD_002', 'PROD_003'],
            'category_id': ['CAT_001', 'CAT_002', 'CAT_001'],
            'brand_id': ['BRAND_A', 'BRAND_B', 'BRAND_A'],
            'price': [25000, 15000, 35000]
        })
        
        self.customer_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'city': ['Karachi', 'Lahore', 'Islamabad'],
            'income_bracket': ['300k-500k', '300k-500k', '500k+']
        })
        
        self.interaction_data = pd.DataFrame({
            'user_id': ['USER_001', 'USER_002', 'USER_001'] * 20,
            'product_id': ['PROD_001', 'PROD_002', 'PROD_003'] * 20,
            'category_id': ['CAT_001', 'CAT_002', 'CAT_001'] * 20,
            'brand_id': ['BRAND_A', 'BRAND_B', 'BRAND_A'] * 20,
            'timestamp': [datetime.utcnow() - timedelta(hours=i) for i in range(60)],
            'purchased': [True, False, True] * 20,
            'clicked': [True, True, False] * 20,
            'price': [25000, 15000, 35000] * 20
        })
    
    @patch('algorithms.collaborative_filtering.CollaborativeFilteringEngine.train')
    @patch('algorithms.popularity_based.PopularityBasedEngine.train')
    @patch('algorithms.deep_learning_engine.DeepLearningRecommendationEngine.train')
    def test_orchestrator_trains_all_algorithms(self, mock_dl_train, mock_pop_train, mock_cf_train):
        """Test that orchestrator trains all algorithms including deep learning"""
        # Mock training results
        mock_cf_train.return_value = {'rmse_accuracy': 0.85, 'model_ready': True}
        mock_pop_train.return_value = {'accuracy': 0.78, 'model_ready': True}
        mock_dl_train.return_value = {'final_accuracy': 0.87, 'model_ready': True}
        
        # Train all algorithms
        results = self.orchestrator.train_all_algorithms(
            self.cf_data, self.sales_data, self.product_data, 
            self.customer_data, self.interaction_data
        )
        
        # Check that all algorithms were trained
        self.assertIn('collaborative_filtering', results)
        self.assertIn('popularity_based', results)
        self.assertIn('deep_learning', results)
        
        # Check training results
        self.assertEqual(results['deep_learning']['final_accuracy'], 0.87)
        self.assertTrue(results['orchestrator_ready'])
        
        # Verify training methods were called
        mock_cf_train.assert_called_once_with(self.cf_data)
        mock_pop_train.assert_called_once_with(
            self.sales_data, self.product_data, self.customer_data
        )
        mock_dl_train.assert_called_once_with(self.interaction_data)
    
    def test_orchestrator_handles_missing_interaction_data(self):
        """Test orchestrator handles missing interaction data gracefully"""
        with patch('algorithms.collaborative_filtering.CollaborativeFilteringEngine.train') as mock_cf_train, \
             patch('algorithms.popularity_based.PopularityBasedEngine.train') as mock_pop_train:
            
            mock_cf_train.return_value = {'rmse_accuracy': 0.85}
            mock_pop_train.return_value = {'accuracy': 0.78}
            
            # Train without interaction data
            results = self.orchestrator.train_all_algorithms(
                self.cf_data, self.sales_data, self.product_data, self.customer_data
            )
            
            # Check that deep learning training was skipped
            self.assertIn('deep_learning', results)
            self.assertIn('error', results['deep_learning'])
            self.assertIn('No interaction data provided', results['deep_learning']['error'])
    
    def test_algorithm_selection_includes_deep_learning(self):
        """Test that algorithm selection includes deep learning for appropriate users"""
        # Mock trained deep learning engine
        self.orchestrator.deep_learning_engine.is_trained = True
        self.orchestrator.is_trained = True
        
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.PRODUCT_PAGE,
            num_recommendations=5
        )
        
        # Test selection for different user history lengths
        
        # New user (< 3 interactions) -> Popularity-based
        algorithm = self.orchestrator.select_algorithm(request, user_history_length=2)
        self.assertEqual(algorithm, RecommendationAlgorithm.POPULARITY_BASED)
        
        # Moderate history (3-10 interactions) -> Deep learning
        algorithm = self.orchestrator.select_algorithm(request, user_history_length=5)
        self.assertEqual(algorithm, RecommendationAlgorithm.CROSS_DOMAIN)
        
        # High history (>10 interactions) -> Ensemble (if performing well)
        with patch.object(self.orchestrator.performance_monitor, 'calculate_current_performance') as mock_perf:
            mock_performance = Mock()
            mock_performance.meets_threshold.return_value = True
            mock_perf.return_value = mock_performance
            
            algorithm = self.orchestrator.select_algorithm(request, user_history_length=15)
            self.assertEqual(algorithm, RecommendationAlgorithm.ENSEMBLE)
    
    def test_algorithm_selection_fallback_when_dl_not_trained(self):
        """Test algorithm selection falls back when deep learning not trained"""
        # Deep learning engine not trained
        self.orchestrator.deep_learning_engine.is_trained = False
        self.orchestrator.is_trained = True
        
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        # Moderate history should fall back to popularity-based
        algorithm = self.orchestrator.select_algorithm(request, user_history_length=5)
        self.assertEqual(algorithm, RecommendationAlgorithm.POPULARITY_BASED)
    
    @patch('algorithms.deep_learning_engine.DeepLearningRecommendationEngine.get_recommendations')
    def test_contextual_recommendations(self, mock_dl_recommendations):
        """Test contextual recommendations using deep learning"""
        # Mock deep learning response
        mock_recommendations = [
            Recommendation(
                user_id="USER_001",
                product_id="PROD_001",
                score=0.85,
                algorithm=RecommendationAlgorithm.CROSS_DOMAIN,
                context=RecommendationContext.PRODUCT_PAGE,
                metadata=RecommendationMetadata(
                    confidence_score=0.85,
                    explanation="Personalized recommendation",
                    fallback_used=False,
                    model_version="deep_learning_v1.0.0",
                    processing_time_ms=150.0,
                    ab_test_group=None
                ),
                timestamp=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=2)
            )
        ]
        
        mock_response = RecommendationResponse(
            recommendations=mock_recommendations,
            total_count=1,
            algorithm_used=RecommendationAlgorithm.CROSS_DOMAIN,
            fallback_applied=False,
            processing_time_ms=150.0,
            cache_hit=False
        )
        mock_dl_recommendations.return_value = mock_response
        
        # Set up trained orchestrator
        self.orchestrator.deep_learning_engine.is_trained = True
        self.orchestrator.is_trained = True
        
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.PRODUCT_PAGE,
            num_recommendations=5
        )
        
        contextual_data = ContextualData(
            timestamp=datetime.utcnow(),
            location="Karachi",
            device_type="mobile",
            session_duration=1800.0,
            pages_viewed=10
        )
        
        # Get contextual recommendations
        response = self.orchestrator.get_contextual_recommendations(
            request, contextual_data
        )
        
        # Check response
        self.assertEqual(response.algorithm_used, RecommendationAlgorithm.CROSS_DOMAIN)
        self.assertFalse(response.fallback_applied)
        self.assertEqual(len(response.recommendations), 1)
        
        # Verify deep learning engine was called with contextual data
        mock_dl_recommendations.assert_called_once_with(request, contextual_data, None)
    
    def test_customer_preference_updates(self):
        """Test customer preference updates through orchestrator"""
        # Set up trained deep learning engine
        self.orchestrator.deep_learning_engine.is_trained = True
        
        user_id = "USER_001"
        product_id = "PROD_001"
        interaction_type = "purchase"
        product_metadata = {
            'category': 'FUR',
            'brand': 'BRAND_A',
            'price': 30000.0
        }
        
        # Update preferences
        self.orchestrator.update_customer_preferences(
            user_id, product_id, interaction_type, product_metadata
        )
        
        # Check that preferences were updated in deep learning engine
        self.assertIn(user_id, self.orchestrator.deep_learning_engine.customer_preferences)
        
        preferences = self.orchestrator.deep_learning_engine.customer_preferences[user_id]
        self.assertEqual(preferences.customer_id, user_id)
    
    @patch('algorithms.deep_learning_engine.DeepLearningRecommendationEngine.get_recommendations')
    @patch('algorithms.popularity_based.PopularityBasedEngine.get_recommendations')
    def test_fallback_to_deep_learning(self, mock_pop_recommendations, mock_dl_recommendations):
        """Test fallback to deep learning when other algorithms fail"""
        # Mock popularity engine failure
        mock_pop_recommendations.side_effect = Exception("Popularity engine failed")
        
        # Mock successful deep learning response
        mock_dl_response = RecommendationResponse(
            recommendations=[],
            total_count=0,
            algorithm_used=RecommendationAlgorithm.CROSS_DOMAIN,
            fallback_applied=False,
            processing_time_ms=100.0,
            cache_hit=False
        )
        mock_dl_recommendations.return_value = mock_dl_response
        
        # Set up orchestrator
        self.orchestrator.deep_learning_engine.is_trained = True
        self.orchestrator.is_trained = True
        self.orchestrator.fallback_strategy = AlgorithmStrategy.POPULARITY_BASED_ONLY
        
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        # This should trigger fallback strategy
        response = self.orchestrator._apply_fallback_strategy(request)
        
        # Should have tried deep learning as final fallback
        mock_dl_recommendations.assert_called_once()
        self.assertEqual(response.algorithm_used, RecommendationAlgorithm.CROSS_DOMAIN)
    
    def test_performance_summary_includes_deep_learning(self):
        """Test performance summary includes deep learning metrics"""
        # Set up trained engines
        self.orchestrator.deep_learning_engine.is_trained = True
        self.orchestrator.deep_learning_engine.personalization_model.training_history = {
            'loss': [0.8, 0.6, 0.4],
            'val_accuracy': [0.7, 0.8, 0.85]
        }
        self.orchestrator.is_trained = True
        
        # Get performance summary
        summary = self.orchestrator.get_performance_summary()
        
        # Check that deep learning metrics are included
        self.assertIn('deep_learning_metrics', summary)
        self.assertTrue(summary['deep_learning_metrics']['is_trained'])
        self.assertIn('training_history', summary['deep_learning_metrics'])
    
    @patch('algorithms.deep_learning_engine.DeepLearningRecommendationEngine.get_recommendations')
    def test_recommendation_with_contextual_data(self, mock_dl_recommendations):
        """Test recommendations with contextual data integration"""
        # Mock deep learning response
        mock_response = RecommendationResponse(
            recommendations=[],
            total_count=0,
            algorithm_used=RecommendationAlgorithm.CROSS_DOMAIN,
            fallback_applied=False,
            processing_time_ms=120.0,
            cache_hit=False
        )
        mock_dl_recommendations.return_value = mock_response
        
        # Set up orchestrator to select deep learning
        self.orchestrator.deep_learning_engine.is_trained = True
        self.orchestrator.is_trained = True
        
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.PRODUCT_PAGE,
            num_recommendations=5
        )
        
        contextual_data = ContextualData(
            timestamp=datetime(2024, 1, 15, 18, 30, 0),
            location="Lahore",
            device_type="desktop",
            session_duration=2400.0,
            pages_viewed=20,
            current_cart_value=35000.0
        )
        
        # Get recommendations with contextual data
        response = self.orchestrator.get_recommendations(
            request, user_history_length=5, contextual_data=contextual_data
        )
        
        # Verify deep learning was called with contextual data
        mock_dl_recommendations.assert_called_once_with(request, contextual_data, None)
        self.assertEqual(response.algorithm_used, RecommendationAlgorithm.CROSS_DOMAIN)


class TestDeepLearningPerformanceIntegration(unittest.TestCase):
    """Test performance aspects of deep learning integration"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.orchestrator = AlgorithmOrchestrator()
        
        # Create larger datasets for performance testing
        np.random.seed(42)
        
        self.large_interaction_data = pd.DataFrame({
            'user_id': [f"USER_{i:03d}" for i in np.random.randint(0, 100, 1000)],
            'product_id': [f"PROD_{i:04d}" for i in np.random.randint(0, 500, 1000)],
            'category_id': [f"CAT_{i:02d}" for i in np.random.randint(0, 20, 1000)],
            'brand_id': [f"BRAND_{i:02d}" for i in np.random.randint(0, 50, 1000)],
            'timestamp': [datetime.utcnow() - timedelta(hours=i) for i in range(1000)],
            'purchased': np.random.choice([True, False], 1000, p=[0.3, 0.7]),
            'clicked': np.random.choice([True, False], 1000, p=[0.6, 0.4]),
            'price': np.random.uniform(5000, 100000, 1000)
        })
    
    @patch('tensorflow.keras.Model.fit')
    @patch('tensorflow.keras.Model.predict')
    def test_training_performance_with_large_dataset(self, mock_predict, mock_fit):
        """Test training performance with larger dataset"""
        # Mock training components
        mock_history = Mock()
        mock_history.history = {'loss': [0.8, 0.6, 0.4], 'val_accuracy': [0.7, 0.8, 0.85]}
        mock_fit.return_value = mock_history
        
        # Mock predictions
        mock_predict.return_value = [
            np.random.random((2000, 1)),  # recommendation_score
            np.random.random((2000, 1)),  # cross_domain_probability
            np.random.random((2000, 10))  # explanation_features
        ]
        
        # Mock other engines
        with patch('algorithms.collaborative_filtering.CollaborativeFilteringEngine.train') as mock_cf, \
             patch('algorithms.popularity_based.PopularityBasedEngine.train') as mock_pop:
            
            mock_cf.return_value = {'rmse_accuracy': 0.85}
            mock_pop.return_value = {'accuracy': 0.78}
            
            # Measure training time
            start_time = datetime.now()
            
            results = self.orchestrator.train_all_algorithms(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 
                pd.DataFrame(), self.large_interaction_data
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Check performance (should complete reasonably quickly)
            self.assertLess(training_time, 60)  # Less than 1 minute for test
            self.assertIn('deep_learning', results)
            self.assertTrue(results['orchestrator_ready'])
    
    @patch('algorithms.deep_learning_engine.DeepPersonalizationModel.predict_recommendation_score')
    def test_recommendation_response_time_integration(self, mock_predict):
        """Test integrated recommendation response time"""
        # Mock fast predictions
        mock_predict.return_value = (0.8, 0.3, Mock())
        
        # Set up trained orchestrator
        self.orchestrator.deep_learning_engine.is_trained = True
        self.orchestrator.is_trained = True
        
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=10
        )
        
        contextual_data = ContextualData(timestamp=datetime.utcnow())
        
        # Measure response time for multiple requests
        response_times = []
        
        for _ in range(10):
            start_time = datetime.now()
            response = self.orchestrator.get_recommendations(
                request, user_history_length=5, contextual_data=contextual_data
            )
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            response_times.append(response_time)
        
        # Check average response time
        avg_response_time = sum(response_times) / len(response_times)
        self.assertLess(avg_response_time, 200)  # Less than 200ms average
        
        # Check that all responses were successful
        self.assertEqual(response.algorithm_used, RecommendationAlgorithm.CROSS_DOMAIN)
    
    def test_concurrent_preference_updates(self):
        """Test concurrent customer preference updates"""
        # Set up trained engine
        self.orchestrator.deep_learning_engine.is_trained = True
        
        # Simulate concurrent updates for multiple users
        start_time = datetime.now()
        
        for user_idx in range(50):
            for interaction_idx in range(10):
                user_id = f"USER_{user_idx:03d}"
                product_id = f"PROD_{interaction_idx:04d}"
                
                product_metadata = {
                    'category': f'CAT_{interaction_idx % 5}',
                    'brand': f'BRAND_{interaction_idx % 10}',
                    'price': 10000 + (interaction_idx * 1000)
                }
                
                self.orchestrator.update_customer_preferences(
                    user_id, product_id, 'click', product_metadata
                )
        
        update_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Should handle 500 updates efficiently
        self.assertLess(update_time, 1000)  # Less than 1 second
        
        # Check that preferences were created for all users
        self.assertEqual(
            len(self.orchestrator.deep_learning_engine.customer_preferences), 50
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)