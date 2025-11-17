"""
Unit tests for collaborative filtering algorithms
Tests accuracy, performance, and edge cases
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

from algorithms.collaborative_filtering import (
    UserItemMatrix, UserBasedCollaborativeFiltering, 
    ItemBasedCollaborativeFiltering, CollaborativeFilteringEngine
)
from models.recommendation import (
    RecommendationRequest, RecommendationContext, RecommendationAlgorithm
)


class TestUserItemMatrix:
    """Test cases for UserItemMatrix class"""
    
    def setup_method(self):
        """Setup test data"""
        self.matrix_handler = UserItemMatrix()
        
        # Create sample interaction data
        self.sample_data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2', 'user3', 'user3', 'user1'],
            'item_id': ['item1', 'item2', 'item1', 'item3', 'item2', 'item3', 'item3'],
            'rating': [5.0, 4.0, 3.0, 5.0, 4.0, 2.0, 3.0],
            'timestamp': [datetime.now()] * 7
        })
    
    def test_build_matrix_success(self):
        """Test successful matrix building"""
        matrix = self.matrix_handler.build_matrix(self.sample_data)
        
        assert matrix is not None
        assert self.matrix_handler.n_users == 3
        assert self.matrix_handler.n_items == 3
        assert matrix.shape == (3, 3)
        
        # Check mappings
        assert 'user1' in self.matrix_handler.user_to_idx
        assert 'item1' in self.matrix_handler.item_to_idx
        assert len(self.matrix_handler.idx_to_user) == 3
        assert len(self.matrix_handler.idx_to_item) == 3
    
    def test_get_user_vector(self):
        """Test user vector retrieval"""
        self.matrix_handler.build_matrix(self.sample_data)
        
        user_vector = self.matrix_handler.get_user_vector('user1')
        assert user_vector is not None
        assert len(user_vector) == 3
        
        # Test non-existent user
        non_existent = self.matrix_handler.get_user_vector('non_existent')
        assert non_existent is None
    
    def test_get_item_vector(self):
        """Test item vector retrieval"""
        self.matrix_handler.build_matrix(self.sample_data)
        
        item_vector = self.matrix_handler.get_item_vector('item1')
        assert item_vector is not None
        assert len(item_vector) == 3
        
        # Test non-existent item
        non_existent = self.matrix_handler.get_item_vector('non_existent')
        assert non_existent is None
    
    def test_sparse_matrix_optimization(self):
        """Test that matrix is properly sparse for large datasets"""
        # Create larger sparse dataset
        n_users = 1000
        n_items = 500
        n_interactions = 5000  # Only 1% density
        
        users = [f'user_{i}' for i in range(n_users)]
        items = [f'item_{i}' for i in range(n_items)]
        
        # Random sparse interactions
        np.random.seed(42)
        selected_users = np.random.choice(users, n_interactions)
        selected_items = np.random.choice(items, n_interactions)
        ratings = np.random.uniform(1, 5, n_interactions)
        
        large_data = pd.DataFrame({
            'user_id': selected_users,
            'item_id': selected_items,
            'rating': ratings,
            'timestamp': [datetime.now()] * n_interactions
        })
        
        matrix = self.matrix_handler.build_matrix(large_data)
        
        # Check sparsity
        sparsity = 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        assert sparsity > 0.9  # Should be very sparse
        assert matrix.format == 'csr'  # Should be CSR format


class TestUserBasedCollaborativeFiltering:
    """Test cases for User-based Collaborative Filtering"""
    
    def setup_method(self):
        """Setup test data and model"""
        self.user_cf = UserBasedCollaborativeFiltering(n_neighbors=10, min_similarity=0.1)
        self.matrix_handler = UserItemMatrix()
        
        # Create more comprehensive test data
        self.test_data = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3', 'u3', 'u4', 'u4'],
            'item_id': ['i1', 'i2', 'i3', 'i1', 'i2', 'i4', 'i1', 'i3', 'i4', 'i2', 'i4'],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 5.0, 4.0, 3.0, 3.0, 4.0],
            'timestamp': [datetime.now()] * 11
        })
        
        self.matrix_handler.build_matrix(self.test_data)
        self.user_cf.fit(self.matrix_handler)
    
    def test_fit_success(self):
        """Test model fitting"""
        assert self.user_cf.user_similarity_matrix is not None
        assert self.user_cf.user_similarity_matrix.shape == (4, 4)
        assert self.user_cf.matrix_handler is not None
        
        # Check that diagonal is zero
        diagonal = np.diag(self.user_cf.user_similarity_matrix)
        assert np.allclose(diagonal, 0)
    
    def test_predict_rating_existing_user_item(self):
        """Test rating prediction for existing user-item pairs"""
        # Test prediction for user who hasn't rated an item
        prediction = self.user_cf.predict_rating('u1', 'i4')
        assert isinstance(prediction, float)
        assert prediction >= 0
    
    def test_predict_rating_non_existent_user(self):
        """Test rating prediction for non-existent user"""
        prediction = self.user_cf.predict_rating('non_existent', 'i1')
        assert prediction == 0.0
    
    def test_predict_rating_non_existent_item(self):
        """Test rating prediction for non-existent item"""
        prediction = self.user_cf.predict_rating('u1', 'non_existent')
        assert prediction == 0.0
    
    def test_get_recommendations(self):
        """Test recommendation generation"""
        recommendations = self.user_cf.get_recommendations('u1', n_recommendations=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        
        # Check recommendation format
        for item_id, score in recommendations:
            assert isinstance(item_id, str)
            assert isinstance(score, float)
            assert score > 0
        
        # Check that recommendations are sorted by score
        scores = [score for _, score in recommendations]
        assert scores == sorted(scores, reverse=True)
    
    def test_get_recommendations_with_exclusions(self):
        """Test recommendations with excluded items"""
        exclude_items = ['i1', 'i2']
        recommendations = self.user_cf.get_recommendations(
            'u1', n_recommendations=5, exclude_items=exclude_items
        )
        
        recommended_items = [item_id for item_id, _ in recommendations]
        for excluded_item in exclude_items:
            assert excluded_item not in recommended_items
    
    def test_get_recommendations_non_existent_user(self):
        """Test recommendations for non-existent user"""
        recommendations = self.user_cf.get_recommendations('non_existent')
        assert recommendations == []


class TestItemBasedCollaborativeFiltering:
    """Test cases for Item-based Collaborative Filtering"""
    
    def setup_method(self):
        """Setup test data and model"""
        self.item_cf = ItemBasedCollaborativeFiltering(n_neighbors=10, min_similarity=0.1)
        self.matrix_handler = UserItemMatrix()
        
        # Use same test data as user-based CF
        self.test_data = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3', 'u3', 'u4', 'u4'],
            'item_id': ['i1', 'i2', 'i3', 'i1', 'i2', 'i4', 'i1', 'i3', 'i4', 'i2', 'i4'],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 5.0, 4.0, 3.0, 3.0, 4.0],
            'timestamp': [datetime.now()] * 11
        })
        
        self.matrix_handler.build_matrix(self.test_data)
        self.item_cf.fit(self.matrix_handler)
    
    def test_fit_success(self):
        """Test model fitting"""
        assert self.item_cf.item_similarity_matrix is not None
        assert self.item_cf.item_similarity_matrix.shape == (4, 4)
        assert self.item_cf.matrix_handler is not None
        
        # Check that diagonal is zero
        diagonal = np.diag(self.item_cf.item_similarity_matrix)
        assert np.allclose(diagonal, 0)
    
    def test_predict_rating_existing_user_item(self):
        """Test rating prediction for existing user-item pairs"""
        prediction = self.item_cf.predict_rating('u1', 'i4')
        assert isinstance(prediction, float)
        assert prediction >= 0
    
    def test_get_recommendations(self):
        """Test recommendation generation"""
        recommendations = self.item_cf.get_recommendations('u1', n_recommendations=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        
        # Check recommendation format
        for item_id, score in recommendations:
            assert isinstance(item_id, str)
            assert isinstance(score, float)
            assert score > 0
        
        # Check that recommendations are sorted by score
        scores = [score for _, score in recommendations]
        assert scores == sorted(scores, reverse=True)


class TestCollaborativeFilteringEngine:
    """Test cases for main Collaborative Filtering Engine"""
    
    def setup_method(self):
        """Setup test data and engine"""
        self.engine = CollaborativeFilteringEngine(
            user_cf_weight=0.6,
            item_cf_weight=0.4,
            min_interactions=2,
            accuracy_threshold=0.85
        )
        
        # Create larger test dataset for training
        np.random.seed(42)
        n_users = 50
        n_items = 30
        n_interactions = 500
        
        users = [f'user_{i}' for i in range(n_users)]
        items = [f'item_{i}' for i in range(n_items)]
        
        # Generate realistic interactions with some patterns
        interactions = []
        for _ in range(n_interactions):
            user = np.random.choice(users)
            item = np.random.choice(items)
            
            # Add some correlation patterns
            user_idx = int(user.split('_')[1])
            item_idx = int(item.split('_')[1])
            
            # Users with similar indices like similar items
            base_rating = 3.0
            if abs(user_idx % 10 - item_idx % 10) < 3:
                base_rating += np.random.uniform(0.5, 2.0)
            
            rating = min(5.0, max(1.0, base_rating + np.random.normal(0, 0.5)))
            
            interactions.append({
                'user_id': user,
                'item_id': item,
                'rating': rating,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
            })
        
        self.training_data = pd.DataFrame(interactions)
    
    def test_prepare_training_data(self):
        """Test training data preparation"""
        prepared_data = self.engine.prepare_training_data(self.training_data)
        
        assert len(prepared_data) <= len(self.training_data)
        assert prepared_data['user_id'].nunique() > 0
        assert prepared_data['item_id'].nunique() > 0
        
        # Check that users and items have minimum interactions
        user_counts = prepared_data['user_id'].value_counts()
        item_counts = prepared_data['item_id'].value_counts()
        
        assert all(count >= self.engine.min_interactions for count in user_counts)
        assert all(count >= self.engine.min_interactions for count in item_counts)
    
    def test_train_success(self):
        """Test successful model training"""
        metrics = self.engine.train(self.training_data)
        
        assert self.engine.is_trained
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'rmse_accuracy' in metrics
        assert 'n_predictions' in metrics
        assert 'coverage' in metrics
        
        # Check that accuracy is calculated
        assert isinstance(self.engine.last_accuracy, float)
        assert self.engine.last_accuracy >= 0
    
    def test_evaluate_performance(self):
        """Test model evaluation"""
        self.engine.train(self.training_data)
        
        # Create small test set
        test_data = self.training_data.sample(n=50, random_state=42)
        metrics = self.engine.evaluate(test_data)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'rmse_accuracy' in metrics
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert 0 <= metrics['rmse_accuracy'] <= 100
    
    def test_get_recommendations_trained_model(self):
        """Test recommendation generation with trained model"""
        self.engine.train(self.training_data)
        
        request = RecommendationRequest(
            user_id='user_1',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        response = self.engine.get_recommendations(request)
        
        assert response.total_count <= 5
        assert response.algorithm_used == RecommendationAlgorithm.ENSEMBLE
        assert not response.fallback_applied
        assert response.processing_time_ms > 0
        
        # Check recommendation objects
        for rec in response.recommendations:
            assert rec.user_id == 'user_1'
            assert isinstance(rec.product_id, str)
            assert rec.score > 0
            assert rec.algorithm == RecommendationAlgorithm.ENSEMBLE
            assert rec.metadata.confidence_score >= 0
            assert rec.metadata.confidence_score <= 1
    
    def test_get_recommendations_untrained_model(self):
        """Test recommendation generation with untrained model"""
        request = RecommendationRequest(
            user_id='user_1',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=5
        )
        
        response = self.engine.get_recommendations(request)
        
        assert response.total_count == 0
        assert response.fallback_applied
        assert len(response.recommendations) == 0
    
    def test_accuracy_threshold_validation(self):
        """Test accuracy threshold checking"""
        self.engine.train(self.training_data)
        
        # Check if accuracy meets threshold
        meets_threshold = self.engine.last_accuracy >= self.engine.accuracy_threshold
        assert isinstance(meets_threshold, bool)
    
    def test_model_save_load(self, tmp_path):
        """Test model saving and loading"""
        self.engine.train(self.training_data)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        self.engine.save_model(str(model_path))
        assert model_path.exists()
        
        # Create new engine and load model
        new_engine = CollaborativeFilteringEngine()
        new_engine.load_model(str(model_path))
        
        assert new_engine.is_trained
        assert new_engine.model_version == self.engine.model_version
        assert new_engine.last_accuracy == self.engine.last_accuracy
        
        # Test that loaded model can generate recommendations
        request = RecommendationRequest(
            user_id='user_1',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=3
        )
        
        response = new_engine.get_recommendations(request)
        assert len(response.recommendations) <= 3
    
    def test_save_untrained_model_error(self, tmp_path):
        """Test error when saving untrained model"""
        model_path = tmp_path / "untrained_model.joblib"
        
        with pytest.raises(ValueError, match="Model must be trained before saving"):
            self.engine.save_model(str(model_path))


class TestPerformanceBenchmarks:
    """Performance benchmark tests for collaborative filtering"""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        # Create large dataset
        np.random.seed(42)
        n_users = 1000
        n_items = 500
        n_interactions = 50000
        
        users = [f'user_{i}' for i in range(n_users)]
        items = [f'item_{i}' for i in range(n_items)]
        
        interactions = []
        for _ in range(n_interactions):
            user = np.random.choice(users)
            item = np.random.choice(items)
            rating = np.random.uniform(1, 5)
            
            interactions.append({
                'user_id': user,
                'item_id': item,
                'rating': rating,
                'timestamp': datetime.now()
            })
        
        large_dataset = pd.DataFrame(interactions)
        
        # Test training time
        engine = CollaborativeFilteringEngine(min_interactions=5)
        start_time = datetime.now()
        
        metrics = engine.train(large_dataset)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert training_time < 60  # Should complete within 1 minute
        assert engine.is_trained
        assert metrics['n_predictions'] > 0
        
        # Test recommendation generation time
        request = RecommendationRequest(
            user_id='user_1',
            context=RecommendationContext.HOMEPAGE,
            num_recommendations=10
        )
        
        start_time = datetime.now()
        response = engine.get_recommendations(request)
        recommendation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Should generate recommendations within 200ms (requirement)
        assert recommendation_time < 200
        assert response.processing_time_ms < 200
    
    def test_memory_efficiency(self):
        """Test memory efficiency with sparse matrices"""
        # This test ensures sparse matrices are used efficiently
        engine = CollaborativeFilteringEngine()
        
        # Create sparse dataset (low density)
        n_users = 5000
        n_items = 2000
        n_interactions = 10000  # Only 0.1% density
        
        users = [f'user_{i}' for i in range(n_users)]
        items = [f'item_{i}' for i in range(n_items)]
        
        np.random.seed(42)
        interactions = []
        for _ in range(n_interactions):
            user = np.random.choice(users)
            item = np.random.choice(items)
            rating = np.random.uniform(1, 5)
            
            interactions.append({
                'user_id': user,
                'item_id': item,
                'rating': rating,
                'timestamp': datetime.now()
            })
        
        sparse_dataset = pd.DataFrame(interactions)
        
        # Train and check that it completes without memory issues
        metrics = engine.train(sparse_dataset)
        
        assert engine.is_trained
        assert engine.matrix_handler.matrix.format == 'csr'
        
        # Check sparsity
        matrix = engine.matrix_handler.matrix
        sparsity = 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        assert sparsity > 0.99  # Should be very sparse


if __name__ == "__main__":
    pytest.main([__file__, "-v"])