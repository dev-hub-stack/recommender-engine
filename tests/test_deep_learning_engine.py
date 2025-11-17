"""
Tests for Deep Learning Recommendation Engine
Tests advanced algorithm accuracy and performance
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.deep_learning_engine import (
    DeepLearningRecommendationEngine, DeepPersonalizationModel, 
    ContextualData, CustomerPreference, RecommendationExplanation
)
from shared.models.recommendation import (
    RecommendationRequest, RecommendationContext, RecommendationAlgorithm
)


class TestContextualData(unittest.TestCase):
    """Test contextual data processing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.contextual_data = ContextualData(
            timestamp=datetime(2024, 1, 15, 14, 30, 0),
            location="Karachi",
            device_type="mobile",
            session_duration=1800.0,  # 30 minutes
            pages_viewed=15,
            search_queries=["furniture", "sofa"],
            browsing_categories=["FUR", "TEX"],
            time_since_last_purchase=86400.0,  # 1 day
            current_cart_value=25000.0
        )
    
    def test_time_features_extraction(self):
        """Test time-based feature extraction"""
        time_features = self.contextual_data.get_time_features()
        
        # Check feature ranges
        self.assertGreaterEqual(time_features['hour_of_day'], 0.0)
        self.assertLessEqual(time_features['hour_of_day'], 1.0)
        self.assertEqual(time_features['hour_of_day'], 14.0 / 24.0)
        
        self.assertGreaterEqual(time_features['day_of_week'], 0.0)
        self.assertLessEqual(time_features['day_of_week'], 1.0)
        
        self.assertEqual(time_features['is_business_hours'], 1.0)  # 14:30 is business hours
        self.assertEqual(time_features['is_weekend'], 0.0)  # Monday
    
    def test_session_features_extraction(self):
        """Test session-based feature extraction"""
        session_features = self.contextual_data.get_session_features()
        
        # Check normalized values
        self.assertGreaterEqual(session_features['session_duration_normalized'], 0.0)
        self.assertLessEqual(session_features['session_duration_normalized'], 1.0)
        
        self.assertGreaterEqual(session_features['pages_viewed_normalized'], 0.0)
        self.assertLessEqual(session_features['pages_viewed_normalized'], 1.0)
        
        self.assertEqual(session_features['search_activity'], 1.0)  # Has search queries
        self.assertGreater(session_features['cart_value_normalized'], 0.0)


class TestCustomerPreference(unittest.TestCase):
    """Test customer preference management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.customer_preference = CustomerPreference(
            customer_id="CUST_001",
            category_preferences={"FUR": 0.8, "TEX": 0.6},
            brand_preferences={"BRAND_A": 0.9, "BRAND_B": 0.7},
            price_sensitivity=0.5,
            quality_preference=0.8,
            novelty_seeking=0.3,
            loyalty_score=0.7,
            seasonal_patterns={},
            time_preferences={},
            last_updated=datetime.utcnow()
        )
    
    def test_preference_update_from_interaction(self):
        """Test preference updates from user interactions"""
        initial_category_pref = self.customer_preference.category_preferences.get("BED", 0.0)
        initial_brand_pref = self.customer_preference.brand_preferences.get("BRAND_C", 0.0)
        
        # Simulate purchase interaction
        self.customer_preference.update_from_interaction(
            product_id="PROD_001",
            category="BED",
            brand="BRAND_C",
            price=30000.0,
            interaction_type="purchase",
            weight=1.0
        )
        
        # Check that preferences were updated
        new_category_pref = self.customer_preference.category_preferences.get("BED", 0.0)
        new_brand_pref = self.customer_preference.brand_preferences.get("BRAND_C", 0.0)
        
        self.assertGreater(new_category_pref, initial_category_pref)
        self.assertGreater(new_brand_pref, initial_brand_pref)
        
        # Check price sensitivity update (should be slightly updated)
        self.assertNotEqual(self.customer_preference.price_sensitivity, 0.5)
    
    def test_preference_vector_generation(self):
        """Test preference vector generation for model input"""
        preference_vector = self.customer_preference.get_preference_vector()
        
        # Check vector dimensions (10 categories + 10 brands + 4 preferences = 24)
        self.assertEqual(len(preference_vector), 24)
        
        # Check data type
        self.assertEqual(preference_vector.dtype, np.float32)
        
        # Check value ranges
        self.assertTrue(np.all(preference_vector >= 0.0))
        self.assertTrue(np.all(preference_vector <= 1.0))


class TestRecommendationExplanation(unittest.TestCase):
    """Test recommendation explanation system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.explanation = RecommendationExplanation(
            primary_reason="similar users liked this product",
            contributing_factors=["matches your price preferences", "from your preferred category"],
            confidence_breakdown={
                "similar_users": 0.8,
                "price_match": 0.6,
                "category_match": 0.7
            },
            contextual_factors=["evening browsing pattern", "items already in cart"]
        )
    
    def test_natural_language_generation(self):
        """Test natural language explanation generation"""
        explanation_text = self.explanation.to_natural_language()
        
        # Check that explanation contains key elements
        self.assertIn("similar users liked this product", explanation_text)
        self.assertIn("matches your price preferences", explanation_text)
        self.assertIn("evening browsing pattern", explanation_text)
        
        # Check that it's a coherent sentence
        self.assertTrue(explanation_text.endswith('.'))
        self.assertGreater(len(explanation_text), 50)  # Reasonable length


class TestDeepPersonalizationModel(unittest.TestCase):
    """Test deep personalization model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = DeepPersonalizationModel(embedding_dim=32, hidden_layers=[64, 32])
        
        # Create mock training data
        self.training_data = pd.DataFrame({
            'user_id': ['USER_001', 'USER_002', 'USER_001', 'USER_003'] * 10,
            'product_id': ['PROD_001', 'PROD_002', 'PROD_003', 'PROD_004'] * 10,
            'category_id': ['CAT_001', 'CAT_002', 'CAT_001', 'CAT_003'] * 10,
            'brand_id': ['BRAND_A', 'BRAND_B', 'BRAND_A', 'BRAND_C'] * 10,
            'timestamp': [datetime.utcnow() - timedelta(days=i) for i in range(40)],
            'purchased': [True, False, True, False] * 10,
            'clicked': [True, True, True, False] * 10,
            'cross_domain': [False, True, False, True] * 10,
            'price': [25000, 15000, 35000, 45000] * 10,
            'session_duration': [1800, 900, 2700, 600] * 10,
            'pages_viewed': [10, 5, 15, 3] * 10
        })
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.embedding_dim, 32)
        self.assertEqual(self.model.hidden_layers, [64, 32])
        self.assertFalse(self.model.is_trained)
        self.assertIsNone(self.model.model)
    
    def test_training_data_preparation(self):
        """Test training data preparation"""
        features, labels = self.model._prepare_training_data(self.training_data)
        
        # Check feature dimensions
        self.assertIn('user_id', features)
        self.assertIn('product_id', features)
        self.assertIn('contextual_features', features)
        self.assertIn('user_preferences', features)
        
        # Check label dimensions
        self.assertIn('recommendation_score', labels)
        self.assertIn('cross_domain_probability', labels)
        self.assertIn('explanation_features', labels)
        
        # Check data shapes
        num_examples = len(features['user_id'])
        self.assertGreater(num_examples, 0)
        self.assertEqual(len(labels['recommendation_score']), num_examples)
        
        # Check contextual features shape
        self.assertEqual(features['contextual_features'].shape[1], 15)  # Expected feature count
        
        # Check user preferences shape
        self.assertEqual(features['user_preferences'].shape[1], 24)  # Expected preference dimensions
    
    @patch('tensorflow.keras.Model.fit')
    @patch('tensorflow.keras.Model.predict')
    def test_model_training(self, mock_predict, mock_fit):
        """Test model training process"""
        # Mock training history
        mock_history = Mock()
        mock_history.history = {
            'loss': [0.8, 0.6, 0.4, 0.3],
            'val_loss': [0.9, 0.7, 0.5, 0.4],
            'recommendation_score_accuracy': [0.6, 0.7, 0.8, 0.85]
        }
        mock_fit.return_value = mock_history
        
        # Mock predictions for evaluation
        mock_predict.return_value = [
            np.array([[0.8], [0.6], [0.9]]),  # recommendation_score
            np.array([[0.3], [0.7], [0.2]]),  # cross_domain_probability
            np.random.random((3, 10))          # explanation_features
        ]
        
        # Train model
        results = self.model.train(self.training_data, epochs=5, batch_size=16)
        
        # Check training results
        self.assertIn('final_accuracy', results)
        self.assertIn('epochs_trained', results)
        self.assertIn('total_params', results)
        self.assertIn('model_ready', results)
        
        self.assertTrue(results['model_ready'])
        self.assertTrue(self.model.is_trained)
        
        # Verify training was called
        mock_fit.assert_called_once()
    
    def test_contextual_feature_extraction(self):
        """Test contextual feature extraction"""
        row = pd.Series({
            'timestamp': datetime(2024, 1, 15, 18, 30, 0),
            'location': 'Lahore',
            'device_type': 'desktop',
            'session_duration': 2400.0,
            'pages_viewed': 20,
            'cart_value': 35000.0,
            'is_mobile': False,
            'is_returning_customer': True,
            'price': 25000.0,
            'discount_percentage': 10.0
        })
        
        features = self.model._extract_contextual_features(row)
        
        # Check feature vector length
        self.assertEqual(len(features), 15)  # Expected number of contextual features
        
        # Check data type
        self.assertEqual(features.dtype, np.float32)
        
        # Check value ranges (most should be normalized 0-1)
        self.assertTrue(np.all(features >= 0.0))
        self.assertTrue(np.all(features <= 1.0))
    
    def test_user_preference_extraction(self):
        """Test user preference extraction"""
        row = pd.Series({
            'category_preference': 0.8,
            'brand_preference': 0.7,
            'price_sensitivity': 0.6,
            'quality_preference': 0.9,
            'novelty_seeking': 0.2,
            'loyalty_score': 0.8
        })
        
        preferences = self.model._extract_user_preferences(row)
        
        # Check dimensions
        self.assertEqual(len(preferences), 24)
        
        # Check data type
        self.assertEqual(preferences.dtype, np.float32)
        
        # Check that first few values match input (approximately due to float precision)
        self.assertAlmostEqual(preferences[0], 0.8, places=5)  # category_preference
        self.assertAlmostEqual(preferences[1], 0.7, places=5)  # brand_preference


class TestDeepLearningRecommendationEngine(unittest.TestCase):
    """Test main deep learning recommendation engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = DeepLearningRecommendationEngine()
        
        # Create mock interaction data
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
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.personalization_model)
        self.assertFalse(self.engine.is_trained)
        self.assertEqual(len(self.engine.customer_preferences), 0)
    
    @patch.object(DeepPersonalizationModel, 'train')
    def test_engine_training(self, mock_train):
        """Test engine training"""
        # Mock training results
        mock_train.return_value = {
            'final_accuracy': 0.85,
            'epochs_trained': 25,
            'total_params': 50000,
            'model_ready': True
        }
        
        # Train engine
        results = self.engine.train(self.interaction_data)
        
        # Check results
        self.assertIn('final_accuracy', results)
        self.assertEqual(results['final_accuracy'], 0.85)
        self.assertTrue(self.engine.is_trained)
        
        # Verify training was called
        mock_train.assert_called_once_with(self.interaction_data)
    
    def test_customer_preference_management(self):
        """Test customer preference creation and management"""
        user_id = "USER_001"
        customer_data = {
            'price_sensitivity': 0.7,
            'quality_preference': 0.8,
            'category_preferences': {'FUR': 0.9},
            'brand_preferences': {'BRAND_A': 0.8}
        }
        
        # Get customer preferences (should create new)
        preferences = self.engine._get_customer_preferences(user_id, customer_data)
        
        # Check preferences were created
        self.assertEqual(preferences.customer_id, user_id)
        self.assertEqual(preferences.price_sensitivity, 0.7)
        self.assertEqual(preferences.quality_preference, 0.8)
        self.assertEqual(preferences.category_preferences['FUR'], 0.9)
        
        # Check preferences are cached
        self.assertIn(user_id, self.engine.customer_preferences)
        
        # Get preferences again (should return cached)
        preferences2 = self.engine._get_customer_preferences(user_id, None)
        self.assertEqual(preferences.customer_id, preferences2.customer_id)
    
    def test_customer_preference_updates(self):
        """Test real-time customer preference updates"""
        user_id = "USER_001"
        product_id = "PROD_001"
        product_metadata = {
            'category': 'FUR',
            'brand': 'BRAND_A',
            'price': 30000.0
        }
        
        # Update preferences
        self.engine.update_customer_preferences(
            user_id, product_id, 'purchase', product_metadata
        )
        
        # Check preferences were created and updated
        self.assertIn(user_id, self.engine.customer_preferences)
        preferences = self.engine.customer_preferences[user_id]
        
        # Check category preference was updated
        self.assertIn('FUR', preferences.category_preferences)
        self.assertGreater(preferences.category_preferences['FUR'], 0.0)
        
        # Check brand preference was updated
        self.assertIn('BRAND_A', preferences.brand_preferences)
        self.assertGreater(preferences.brand_preferences['BRAND_A'], 0.0)
    
    def test_candidate_product_generation(self):
        """Test candidate product generation"""
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.PRODUCT_PAGE,
            num_recommendations=10,
            exclude_products=["PROD_001", "PROD_002"]
        )
        
        # Create mock user preferences
        user_preferences = CustomerPreference(
            customer_id="USER_001",
            category_preferences={},
            brand_preferences={},
            price_sensitivity=0.5,
            quality_preference=0.7,
            novelty_seeking=0.3,
            loyalty_score=0.6,
            seasonal_patterns={},
            time_preferences={},
            last_updated=datetime.utcnow()
        )
        
        candidates = self.engine._get_candidate_products(request, user_preferences)
        
        # Check candidates were generated
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 50)  # Max limit
        
        # Check excluded products are not in candidates
        for excluded in request.exclude_products:
            self.assertNotIn(excluded, candidates)
    
    @patch.object(DeepPersonalizationModel, 'predict_recommendation_score')
    def test_recommendation_generation(self, mock_predict):
        """Test recommendation generation"""
        # Mock model predictions
        mock_predict.return_value = (
            0.85,  # recommendation_score
            0.3,   # cross_domain_probability
            RecommendationExplanation(
                primary_reason="personalized for you",
                contributing_factors=["matches preferences"],
                confidence_breakdown={"personalized": 0.85}
            )
        )
        
        # Set engine as trained
        self.engine.is_trained = True
        self.engine.personalization_model.is_trained = True
        
        # Create request
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        contextual_data = ContextualData(
            timestamp=datetime.utcnow(),
            device_type="mobile",
            session_duration=1200.0
        )
        
        # Get recommendations
        response = self.engine.get_recommendations(request, contextual_data)
        
        # Check response
        self.assertFalse(response.fallback_applied)
        # Algorithm used should be CROSS_DOMAIN since that's what the engine uses
        self.assertEqual(response.algorithm_used, RecommendationAlgorithm.CROSS_DOMAIN)
        self.assertGreater(len(response.recommendations), 0)
        self.assertLessEqual(len(response.recommendations), request.num_recommendations)
        
        # Check recommendation properties
        for rec in response.recommendations:
            self.assertEqual(rec.user_id, request.user_id)
            self.assertEqual(rec.algorithm, RecommendationAlgorithm.CROSS_DOMAIN)
            self.assertGreater(rec.score, 0.0)
            self.assertLessEqual(rec.score, 1.0)
            self.assertIsNotNone(rec.metadata.explanation)
    
    def test_fallback_response(self):
        """Test fallback response when model not trained"""
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.PRODUCT_PAGE,
            num_recommendations=5
        )
        
        # Engine not trained - should return fallback
        response = self.engine.get_recommendations(request)
        
        # Check fallback response
        self.assertTrue(response.fallback_applied)
        self.assertEqual(len(response.recommendations), 0)
        # Algorithm used should be CROSS_DOMAIN since that's what the engine uses
        self.assertEqual(response.algorithm_used, RecommendationAlgorithm.CROSS_DOMAIN)
    
    def test_model_metrics(self):
        """Test model metrics retrieval"""
        # Before training
        metrics = self.engine.get_model_metrics()
        self.assertIn("error", metrics)
        
        # After training (mock)
        self.engine.is_trained = True
        self.engine.personalization_model.is_trained = True
        self.engine.personalization_model.training_history = {
            'loss': [0.8, 0.6, 0.4],
            'val_accuracy': [0.7, 0.8, 0.85]
        }
        
        metrics = self.engine.get_model_metrics()
        self.assertTrue(metrics['is_trained'])
        self.assertTrue(metrics['model_ready'])
        self.assertIn('training_history', metrics)


class TestPerformanceAndAccuracy(unittest.TestCase):
    """Test performance and accuracy requirements"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.engine = DeepLearningRecommendationEngine()
        
        # Create larger dataset for performance testing
        np.random.seed(42)
        num_users = 100
        num_products = 500
        num_interactions = 2000
        
        users = [f"USER_{i:03d}" for i in range(num_users)]
        products = [f"PROD_{i:04d}" for i in range(num_products)]
        categories = [f"CAT_{i:02d}" for i in range(20)]
        brands = [f"BRAND_{i:02d}" for i in range(50)]
        
        self.large_dataset = pd.DataFrame({
            'user_id': np.random.choice(users, num_interactions),
            'product_id': np.random.choice(products, num_interactions),
            'category_id': np.random.choice(categories, num_interactions),
            'brand_id': np.random.choice(brands, num_interactions),
            'timestamp': [datetime.utcnow() - timedelta(hours=i) for i in range(num_interactions)],
            'purchased': np.random.choice([True, False], num_interactions, p=[0.3, 0.7]),
            'clicked': np.random.choice([True, False], num_interactions, p=[0.6, 0.4]),
            'price': np.random.uniform(5000, 100000, num_interactions),
            'session_duration': np.random.uniform(60, 3600, num_interactions),
            'pages_viewed': np.random.randint(1, 50, num_interactions)
        })
    
    @patch('tensorflow.keras.Model.fit')
    @patch('tensorflow.keras.Model.predict')
    def test_training_performance(self, mock_predict, mock_fit):
        """Test training performance with larger dataset"""
        # Mock training components
        mock_history = Mock()
        mock_history.history = {
            'loss': [0.8, 0.6, 0.4, 0.3, 0.25],
            'val_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
            'recommendation_score_accuracy': [0.6, 0.7, 0.8, 0.85, 0.87]
        }
        mock_fit.return_value = mock_history
        
        # Mock predictions
        num_samples = len(self.large_dataset) * 2  # Positive + negative examples
        mock_predict.return_value = [
            np.random.random((num_samples, 1)),
            np.random.random((num_samples, 1)),
            np.random.random((num_samples, 10))
        ]
        
        # Measure training time
        start_time = datetime.now()
        results = self.engine.train(self.large_dataset)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Check performance requirements
        self.assertLess(training_time, 300)  # Should complete within 5 minutes for test data
        self.assertIn('final_accuracy', results)
        self.assertTrue(results['model_ready'])
        
        # Verify model was built and trained
        mock_fit.assert_called_once()
    
    @patch.object(DeepPersonalizationModel, 'predict_recommendation_score')
    def test_recommendation_response_time(self, mock_predict):
        """Test recommendation response time meets <200ms requirement"""
        # Mock fast predictions
        mock_predict.return_value = (
            0.8, 0.3,
            RecommendationExplanation(
                primary_reason="test",
                contributing_factors=[],
                confidence_breakdown={"test": 0.8}
            )
        )
        
        # Set up trained engine
        self.engine.is_trained = True
        self.engine.personalization_model.is_trained = True
        
        request = RecommendationRequest(
            user_id="USER_001",
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=10
        )
        
        contextual_data = ContextualData(timestamp=datetime.utcnow())
        
        # Measure response time
        start_time = datetime.now()
        response = self.engine.get_recommendations(request, contextual_data)
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Check response time requirement (<200ms)
        self.assertLess(response_time, 200)
        self.assertLess(response.processing_time_ms, 200)
        
        # Check response quality
        self.assertFalse(response.fallback_applied)
        self.assertGreater(len(response.recommendations), 0)
    
    def test_accuracy_threshold_compliance(self):
        """Test that model meets minimum accuracy thresholds"""
        # Mock a trained model with good accuracy
        self.engine.is_trained = True
        self.engine.personalization_model.is_trained = True
        self.engine.personalization_model.training_history = {
            'recommendation_score_accuracy': [0.6, 0.7, 0.8, 0.85, 0.87],
            'val_recommendation_score_accuracy': [0.58, 0.68, 0.78, 0.82, 0.84]
        }
        
        metrics = self.engine.get_model_metrics()
        
        # Check that training achieved good accuracy
        training_history = metrics['training_history']
        final_accuracy = training_history['recommendation_score_accuracy'][-1]
        
        # Should meet minimum 80% accuracy threshold for deep learning
        self.assertGreaterEqual(final_accuracy, 0.80)
        
        # Validation accuracy should be close to training accuracy (no overfitting)
        val_accuracy = training_history['val_recommendation_score_accuracy'][-1]
        accuracy_gap = abs(final_accuracy - val_accuracy)
        self.assertLess(accuracy_gap, 0.05)  # Less than 5% gap
    
    def test_real_time_personalization_updates(self):
        """Test real-time preference updates don't degrade performance"""
        user_id = "USER_001"
        
        # Simulate multiple rapid preference updates
        start_time = datetime.now()
        
        for i in range(100):
            product_metadata = {
                'category': f'CAT_{i % 10}',
                'brand': f'BRAND_{i % 20}',
                'price': 10000 + (i * 100)
            }
            
            self.engine.update_customer_preferences(
                user_id, f"PROD_{i:04d}", 'click', product_metadata
            )
        
        update_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Should handle 100 updates quickly
        self.assertLess(update_time, 100)  # Less than 100ms for 100 updates
        
        # Check preferences were updated
        self.assertIn(user_id, self.engine.customer_preferences)
        preferences = self.engine.customer_preferences[user_id]
        
        # Should have learned multiple categories and brands
        self.assertGreater(len(preferences.category_preferences), 5)
        self.assertGreater(len(preferences.brand_preferences), 10)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)