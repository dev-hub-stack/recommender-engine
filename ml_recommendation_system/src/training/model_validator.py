"""
Model Validator Module
Validates trained model on test data
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Validate collaborative filtering model"""
    
    def __init__(self, n_recommendations: int = 10):
        """
        Initialize ModelValidator
        
        Args:
            n_recommendations: Number of recommendations to generate
        """
        self.n_recommendations = n_recommendations
        self.metrics = {}
    
    def precision_at_k(self, recommended: List, actual: List, k: int = 10) -> float:
        """
        Calculate precision@k
        
        Args:
            recommended: List of recommended items
            actual: List of actual purchased items
            k: Number of recommendations to consider
            
        Returns:
            Precision score
        """
        if not recommended or not actual:
            return 0.0
        
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(actual))
        return hits / k if k > 0 else 0.0
    
    def recall_at_k(self, recommended: List, actual: List, k: int = 10) -> float:
        """
        Calculate recall@k
        
        Args:
            recommended: List of recommended items
            actual: List of actual purchased items
            k: Number of recommendations to consider
            
        Returns:
            Recall score
        """
        if not recommended or not actual:
            return 0.0
        
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(actual))
        return hits / len(actual) if len(actual) > 0 else 0.0
    
    def hit_rate(self, all_recommended: List[List], all_actual: List[List]) -> float:
        """
        Calculate hit rate (% of users with at least 1 correct recommendation)
        
        Args:
            all_recommended: List of recommendation lists
            all_actual: List of actual purchase lists
            
        Returns:
            Hit rate
        """
        if not all_recommended or not all_actual:
            return 0.0
        
        hits = 0
        for recommended, actual in zip(all_recommended, all_actual):
            if len(set(recommended) & set(actual)) > 0:
                hits += 1
        
        return hits / len(all_recommended) if len(all_recommended) > 0 else 0.0
    
    def coverage(self, all_recommended: List[List], total_items: int) -> float:
        """
        Calculate catalog coverage (% of items that can be recommended)
        
        Args:
            all_recommended: List of recommendation lists
            total_items: Total number of items in catalog
            
        Returns:
            Coverage score
        """
        if not all_recommended or total_items == 0:
            return 0.0
        
        unique_recommended = set()
        for recommended in all_recommended:
            unique_recommended.update(recommended)
        
        return len(unique_recommended) / total_items
    
    def validate(self, 
                 test_data: pd.DataFrame,
                 interaction_matrix: csr_matrix,
                 item_similarity: csr_matrix,
                 matrix_builder) -> Dict:
        """
        Validate model on test data
        
        Args:
            test_data: Test dataset
            interaction_matrix: Training interaction matrix
            item_similarity: Item similarity matrix
            matrix_builder: MatrixBuilder instance for ID mapping
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info("\n" + "="*60)
        logger.info("MODEL VALIDATION")
        logger.info("="*60)
        
        # Group test data by location
        test_by_location = test_data.groupby('location_id')['product_id'].apply(list).to_dict()
        
        logger.info(f"Validating on {len(test_by_location):,} test locations...")
        
        all_precisions = []
        all_recalls = []
        all_recommended = []
        all_actual = []
        
        validated_locations = 0
        
        for location_id, actual_products in test_by_location.items():
            # Get location index
            location_idx = matrix_builder.get_location_index(location_id)
            
            if location_idx == -1:
                # Location not in training data (cold start)
                continue
            
            # Get location's purchase history from training
            location_purchases = interaction_matrix[location_idx].toarray().flatten()
            purchased_indices = np.where(location_purchases > 0)[0]
            
            if len(purchased_indices) == 0:
                continue
            
            # Generate recommendations using item-based CF
            recommendations = self._generate_item_based_recommendations(
                purchased_indices,
                item_similarity,
                location_purchases,
                matrix_builder,
                self.n_recommendations
            )
            
            if not recommendations:
                continue
            
            # Calculate metrics
            precision = self.precision_at_k(recommendations, actual_products, self.n_recommendations)
            recall = self.recall_at_k(recommendations, actual_products, self.n_recommendations)
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_recommended.append(recommendations)
            all_actual.append(actual_products)
            
            validated_locations += 1
        
        # Calculate aggregate metrics
        self.metrics = {
            'precision@10': np.mean(all_precisions) if all_precisions else 0.0,
            'recall@10': np.mean(all_recalls) if all_recalls else 0.0,
            'hit_rate': self.hit_rate(all_recommended, all_actual),
            'coverage': self.coverage(all_recommended, matrix_builder.n_products),
            'validated_locations': validated_locations,
            'total_test_locations': len(test_by_location)
        }
        
        # Log results
        logger.info(f"\nValidation Results:")
        logger.info(f"  Locations validated: {validated_locations:,} / {len(test_by_location):,}")
        logger.info(f"  Precision@10: {self.metrics['precision@10']:.4f} ({self.metrics['precision@10']*100:.2f}%)")
        logger.info(f"  Recall@10: {self.metrics['recall@10']:.4f} ({self.metrics['recall@10']*100:.2f}%)")
        logger.info(f"  Hit Rate: {self.metrics['hit_rate']:.4f} ({self.metrics['hit_rate']*100:.2f}%)")
        logger.info(f"  Coverage: {self.metrics['coverage']:.4f} ({self.metrics['coverage']*100:.2f}%)")
        
        # Interpret results
        logger.info(f"\nInterpretation:")
        if self.metrics['precision@10'] >= 0.15:
            logger.info(f"  ✅ Precision is GOOD (>= 15%)")
        else:
            logger.info(f"  ⚠️  Precision is below target (< 15%)")
        
        if self.metrics['hit_rate'] >= 0.40:
            logger.info(f"  ✅ Hit rate is GOOD (>= 40%)")
        else:
            logger.info(f"  ⚠️  Hit rate is below target (< 40%)")
        
        logger.info("="*60)
        
        return self.metrics
    
    def _generate_item_based_recommendations(self,
                                            purchased_indices: np.ndarray,
                                            item_similarity: csr_matrix,
                                            customer_purchases: np.ndarray,
                                            matrix_builder,
                                            n_recommendations: int) -> List[str]:
        """
        Generate recommendations using item-based collaborative filtering
        
        Args:
            purchased_indices: Indices of items customer purchased
            item_similarity: Item similarity matrix
            customer_purchases: Customer's purchase vector
            matrix_builder: MatrixBuilder for ID mapping
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended product IDs
        """
        # Calculate scores for all items
        scores = np.zeros(item_similarity.shape[0])
        
        for item_idx in purchased_indices:
            # Get similar items
            similar_items = item_similarity[item_idx].toarray().flatten()
            scores += similar_items
        
        # Remove already purchased items
        scores[purchased_indices] = -np.inf
        
        # Get top-N items
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        # Convert indices to product IDs
        recommendations = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include items with positive scores
                product_id = matrix_builder.get_product_id(idx)
                recommendations.append(product_id)
        
        return recommendations
