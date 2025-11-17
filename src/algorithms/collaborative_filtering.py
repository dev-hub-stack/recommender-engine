"""
Collaborative Filtering Algorithms Implementation
Implements both user-based and item-based collaborative filtering
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import structlog
from datetime import datetime, timedelta

import sys
import os
# Removed shared dependency for local execution

from models.recommendation import (
    Recommendation, RecommendationAlgorithm, RecommendationContext, 
    RecommendationMetadata, RecommendationRequest, RecommendationResponse
)

logger = structlog.get_logger()


class UserItemMatrix:
    """Manages user-item interaction matrix with sparse optimization"""
    
    def __init__(self):
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        self.matrix = None
        self.n_users = 0
        self.n_items = 0
    
    def build_matrix(self, interactions_df: pd.DataFrame) -> csr_matrix:
        """
        Build sparse user-item matrix from interactions dataframe
        
        Args:
            interactions_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        
        Returns:
            Sparse CSR matrix of user-item interactions
        """
        logger.info("Building user-item matrix", 
                   n_interactions=len(interactions_df))
        
        # Create user and item mappings
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        # Create sparse matrix
        row_indices = [self.user_to_idx[user] for user in interactions_df['user_id']]
        col_indices = [self.item_to_idx[item] for item in interactions_df['item_id']]
        ratings = interactions_df['rating'].values
        
        self.matrix = csr_matrix(
            (ratings, (row_indices, col_indices)), 
            shape=(self.n_users, self.n_items)
        )
        
        logger.info("User-item matrix built successfully",
                   n_users=self.n_users,
                   n_items=self.n_items,
                   sparsity=1 - (self.matrix.nnz / (self.n_users * self.n_items)))
        
        return self.matrix
    
    def get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """Get user's rating vector"""
        if user_id not in self.user_to_idx:
            return None
        user_idx = self.user_to_idx[user_id]
        return self.matrix[user_idx].toarray().flatten()
    
    def get_item_vector(self, item_id: str) -> Optional[np.ndarray]:
        """Get item's rating vector"""
        if item_id not in self.item_to_idx:
            return None
        item_idx = self.item_to_idx[item_id]
        return self.matrix[:, item_idx].toarray().flatten()


class UserBasedCollaborativeFiltering:
    """User-based collaborative filtering implementation"""
    
    def __init__(self, n_neighbors: int = 50, min_similarity: float = 0.1):
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.user_similarity_matrix = None
        self.matrix_handler = None
        
    def fit(self, matrix_handler: UserItemMatrix):
        """
        Train user-based collaborative filtering model
        
        Args:
            matrix_handler: UserItemMatrix instance with built matrix
        """
        logger.info("Training user-based collaborative filtering",
                   n_users=matrix_handler.n_users,
                   n_items=matrix_handler.n_items)
        
        self.matrix_handler = matrix_handler
        
        # Calculate user similarity matrix
        user_matrix = matrix_handler.matrix
        self.user_similarity_matrix = cosine_similarity(user_matrix)
        
        # Set diagonal to 0 (user shouldn't be similar to themselves for recommendations)
        np.fill_diagonal(self.user_similarity_matrix, 0)
        
        logger.info("User similarity matrix computed successfully")
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for user-item pair
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating (0 if prediction not possible)
        """
        if (user_id not in self.matrix_handler.user_to_idx or 
            item_id not in self.matrix_handler.item_to_idx):
            return 0.0
        
        user_idx = self.matrix_handler.user_to_idx[user_id]
        item_idx = self.matrix_handler.item_to_idx[item_id]
        
        # Get similar users
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Find users who rated this item
        item_ratings = self.matrix_handler.matrix[:, item_idx].toarray().flatten()
        rated_users = np.where(item_ratings > 0)[0]
        
        if len(rated_users) == 0:
            return 0.0
        
        # Get similarities for users who rated this item
        relevant_similarities = user_similarities[rated_users]
        relevant_ratings = item_ratings[rated_users]
        
        # Filter by minimum similarity
        valid_mask = relevant_similarities >= self.min_similarity
        if not np.any(valid_mask):
            return 0.0
        
        valid_similarities = relevant_similarities[valid_mask]
        valid_ratings = relevant_ratings[valid_mask]
        
        # Calculate weighted average
        if np.sum(valid_similarities) == 0:
            return 0.0
        
        predicted_rating = np.sum(valid_similarities * valid_ratings) / np.sum(valid_similarities)
        return predicted_rating
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10,
                          exclude_items: List[str] = None) -> List[Tuple[str, float]]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            exclude_items: Items to exclude from recommendations
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if exclude_items is None:
            exclude_items = []
        
        if user_id not in self.matrix_handler.user_to_idx:
            return []
        
        user_idx = self.matrix_handler.user_to_idx[user_id]
        user_ratings = self.matrix_handler.matrix[user_idx].toarray().flatten()
        
        # Get items user hasn't rated
        unrated_items = np.where(user_ratings == 0)[0]
        
        recommendations = []
        for item_idx in unrated_items:
            item_id = self.matrix_handler.idx_to_item[item_idx]
            
            if item_id in exclude_items:
                continue
                
            predicted_rating = self.predict_rating(user_id, item_id)
            if predicted_rating > 0:
                recommendations.append((item_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]


class ItemBasedCollaborativeFiltering:
    """Item-based collaborative filtering implementation"""
    
    def __init__(self, n_neighbors: int = 50, min_similarity: float = 0.1):
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.item_similarity_matrix = None
        self.matrix_handler = None
        
    def fit(self, matrix_handler: UserItemMatrix):
        """
        Train item-based collaborative filtering model
        
        Args:
            matrix_handler: UserItemMatrix instance with built matrix
        """
        logger.info("Training item-based collaborative filtering",
                   n_users=matrix_handler.n_users,
                   n_items=matrix_handler.n_items)
        
        self.matrix_handler = matrix_handler
        
        # Calculate item similarity matrix
        item_matrix = matrix_handler.matrix.T  # Transpose to get items x users
        self.item_similarity_matrix = cosine_similarity(item_matrix)
        
        # Set diagonal to 0
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        logger.info("Item similarity matrix computed successfully")
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for user-item pair
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating (0 if prediction not possible)
        """
        if (user_id not in self.matrix_handler.user_to_idx or 
            item_id not in self.matrix_handler.item_to_idx):
            return 0.0
        
        user_idx = self.matrix_handler.user_to_idx[user_id]
        item_idx = self.matrix_handler.item_to_idx[item_id]
        
        # Get user's ratings
        user_ratings = self.matrix_handler.matrix[user_idx].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return 0.0
        
        # Get similarities for this item with user's rated items
        item_similarities = self.item_similarity_matrix[item_idx]
        relevant_similarities = item_similarities[rated_items]
        relevant_ratings = user_ratings[rated_items]
        
        # Filter by minimum similarity
        valid_mask = relevant_similarities >= self.min_similarity
        if not np.any(valid_mask):
            return 0.0
        
        valid_similarities = relevant_similarities[valid_mask]
        valid_ratings = relevant_ratings[valid_mask]
        
        # Calculate weighted average
        if np.sum(valid_similarities) == 0:
            return 0.0
        
        predicted_rating = np.sum(valid_similarities * valid_ratings) / np.sum(valid_similarities)
        return predicted_rating
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10,
                          exclude_items: List[str] = None) -> List[Tuple[str, float]]:
        """
        Get recommendations for a user
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            exclude_items: Items to exclude from recommendations
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if exclude_items is None:
            exclude_items = []
        
        if user_id not in self.matrix_handler.user_to_idx:
            return []
        
        user_idx = self.matrix_handler.user_to_idx[user_id]
        user_ratings = self.matrix_handler.matrix[user_idx].toarray().flatten()
        
        # Get items user hasn't rated
        unrated_items = np.where(user_ratings == 0)[0]
        
        recommendations = []
        for item_idx in unrated_items:
            item_id = self.matrix_handler.idx_to_item[item_idx]
            
            if item_id in exclude_items:
                continue
                
            predicted_rating = self.predict_rating(user_id, item_id)
            if predicted_rating > 0:
                recommendations.append((item_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]


class CollaborativeFilteringEngine:
    """Main collaborative filtering engine combining user and item-based approaches"""
    
    def __init__(self, user_cf_weight: float = 0.6, item_cf_weight: float = 0.4,
                 min_interactions: int = 5, accuracy_threshold: float = 0.85):
        self.user_cf_weight = user_cf_weight
        self.item_cf_weight = item_cf_weight
        self.min_interactions = min_interactions
        self.accuracy_threshold = accuracy_threshold
        
        self.user_cf = UserBasedCollaborativeFiltering()
        self.item_cf = ItemBasedCollaborativeFiltering()
        self.matrix_handler = UserItemMatrix()
        
        self.is_trained = False
        self.last_accuracy = 0.0
        self.model_version = "1.0.0"
        
    def prepare_training_data(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean training data
        
        Args:
            interactions_df: Raw interactions with columns ['user_id', 'item_id', 'rating', 'timestamp']
            
        Returns:
            Cleaned interactions dataframe
        """
        logger.info("Preparing training data", n_interactions=len(interactions_df))
        
        # Remove users with too few interactions
        user_counts = interactions_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        filtered_df = interactions_df[interactions_df['user_id'].isin(valid_users)]
        
        # Remove items with too few interactions
        item_counts = filtered_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.min_interactions].index
        filtered_df = filtered_df[filtered_df['item_id'].isin(valid_items)]
        
        logger.info("Training data prepared",
                   original_interactions=len(interactions_df),
                   filtered_interactions=len(filtered_df),
                   n_users=filtered_df['user_id'].nunique(),
                   n_items=filtered_df['item_id'].nunique())
        
        return filtered_df
    
    def train(self, interactions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train collaborative filtering models
        
        Args:
            interactions_df: Training data with user-item interactions
            
        Returns:
            Training metrics dictionary
        """
        start_time = datetime.now()
        logger.info("Starting collaborative filtering training")
        
        # Prepare data
        clean_data = self.prepare_training_data(interactions_df)
        
        # Split for validation
        train_data, test_data = train_test_split(clean_data, test_size=0.2, random_state=42)
        
        # Build matrix
        self.matrix_handler.build_matrix(train_data)
        
        # Train models
        self.user_cf.fit(self.matrix_handler)
        self.item_cf.fit(self.matrix_handler)
        
        # Evaluate on test set
        metrics = self.evaluate(test_data)
        self.last_accuracy = metrics.get('rmse_accuracy', 0.0)
        
        self.is_trained = True
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("Collaborative filtering training completed",
                   training_time_seconds=training_time,
                   accuracy=self.last_accuracy,
                   meets_threshold=self.last_accuracy >= self.accuracy_threshold)
        
        return metrics
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test interactions dataframe
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            # Get predictions from both models
            user_pred = self.user_cf.predict_rating(user_id, item_id)
            item_pred = self.item_cf.predict_rating(user_id, item_id)
            
            # Ensemble prediction
            if user_pred > 0 and item_pred > 0:
                ensemble_pred = (self.user_cf_weight * user_pred + 
                               self.item_cf_weight * item_pred)
            elif user_pred > 0:
                ensemble_pred = user_pred
            elif item_pred > 0:
                ensemble_pred = item_pred
            else:
                continue  # Skip if no prediction possible
            
            predictions.append(ensemble_pred)
            actuals.append(actual_rating)
        
        if len(predictions) == 0:
            return {"error": "No predictions generated"}
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        # Convert RMSE to accuracy percentage (assuming rating scale 1-5)
        rmse_accuracy = max(0, (1 - rmse / 4) * 100)  # 4 is max possible RMSE for 1-5 scale
        
        return {
            "rmse": rmse,
            "mae": mae,
            "rmse_accuracy": rmse_accuracy,
            "n_predictions": len(predictions),
            "coverage": len(predictions) / len(test_data) * 100
        }
    
    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Generate recommendations using collaborative filtering
        
        Args:
            request: Recommendation request
            
        Returns:
            Recommendation response
        """
        start_time = datetime.now()
        
        if not self.is_trained:
            return RecommendationResponse(
                recommendations=[],
                total_count=0,
                algorithm_used=RecommendationAlgorithm.COLLABORATIVE_FILTERING_USER,
                fallback_applied=True,
                processing_time_ms=0,
                cache_hit=False
            )
        
        # Get recommendations from both models
        user_recs = self.user_cf.get_recommendations(
            request.user_id, 
            request.num_recommendations * 2,  # Get more to allow for ensemble
            request.exclude_products
        )
        
        item_recs = self.item_cf.get_recommendations(
            request.user_id,
            request.num_recommendations * 2,
            request.exclude_products
        )
        
        # Combine recommendations using ensemble approach
        combined_scores = {}
        
        # Add user-based recommendations
        for item_id, score in user_recs:
            combined_scores[item_id] = self.user_cf_weight * score
        
        # Add item-based recommendations
        for item_id, score in item_recs:
            if item_id in combined_scores:
                combined_scores[item_id] += self.item_cf_weight * score
            else:
                combined_scores[item_id] = self.item_cf_weight * score
        
        # Sort and select top recommendations
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_recs = sorted_recs[:request.num_recommendations]
        
        # Create recommendation objects
        recommendations = []
        for item_id, score in top_recs:
            metadata = RecommendationMetadata(
                confidence_score=min(score / 5.0, 1.0),  # Normalize to 0-1
                explanation=f"Based on similar users and items (CF ensemble)",
                fallback_used=False,
                model_version=self.model_version,
                processing_time_ms=0,  # Will be set below
                ab_test_group=None
            )
            
            rec = Recommendation(
                user_id=request.user_id,
                product_id=item_id,
                score=score,
                algorithm=RecommendationAlgorithm.ENSEMBLE,
                context=request.context,
                metadata=metadata,
                timestamp=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            recommendations.append(rec)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update processing time in metadata
        for rec in recommendations:
            rec.metadata.processing_time_ms = processing_time
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            algorithm_used=RecommendationAlgorithm.ENSEMBLE,
            fallback_applied=False,
            processing_time_ms=processing_time,
            cache_hit=False
        )
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'user_cf': self.user_cf,
            'item_cf': self.item_cf,
            'matrix_handler': self.matrix_handler,
            'model_version': self.model_version,
            'last_accuracy': self.last_accuracy,
            'training_params': {
                'user_cf_weight': self.user_cf_weight,
                'item_cf_weight': self.item_cf_weight,
                'min_interactions': self.min_interactions,
                'accuracy_threshold': self.accuracy_threshold
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info("Model saved successfully", filepath=filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.user_cf = model_data['user_cf']
        self.item_cf = model_data['item_cf']
        self.matrix_handler = model_data['matrix_handler']
        self.model_version = model_data['model_version']
        self.last_accuracy = model_data['last_accuracy']
        
        # Load training params
        params = model_data['training_params']
        self.user_cf_weight = params['user_cf_weight']
        self.item_cf_weight = params['item_cf_weight']
        self.min_interactions = params['min_interactions']
        self.accuracy_threshold = params['accuracy_threshold']
        
        self.is_trained = True
        logger.info("Model loaded successfully", 
                   filepath=filepath,
                   model_version=self.model_version,
                   last_accuracy=self.last_accuracy)