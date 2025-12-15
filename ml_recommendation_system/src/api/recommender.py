"""
Recommender Module
Generates recommendations using trained models
"""
import numpy as np
import logging
from typing import List, Dict, Optional
from .model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Recommender:
    """Generate recommendations using collaborative filtering"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        
        if not self.model_loader.is_loaded():
            raise RuntimeError("Models not loaded. Call ModelLoader().load_models() first.")
    
    def get_location_recommendations(self, 
                                     location_id: str, 
                                     n_recommendations: int = 10,
                                     exclude_purchased: bool = True,
                                     use_hybrid: bool = True,
                                     popularity_weight: float = 0.3) -> List[Dict]:
        """
        Get personalized recommendations for a location
        
        Args:
            location_id: Location identifier
            n_recommendations: Number of recommendations to return
            exclude_purchased: Whether to exclude already purchased products
            use_hybrid: Whether to blend collaborative filtering with popularity
            popularity_weight: Weight for popularity (0.3 = 30% popular, 70% collaborative)
            
        Returns:
            List of recommended products with scores
        """
        # Get location index
        location_idx = self.model_loader.get_location_index(location_id)
        
        if location_idx is None:
            # Location not in training data - return popular products
            logger.info(f"Location {location_id} not found, returning popular products")
            return self.get_popular_products(n_recommendations)
        
        # Get location's purchase history
        location_purchases = self.model_loader.interaction_matrix[location_idx].toarray().flatten()
        purchased_indices = np.where(location_purchases > 0)[0]
        
        if len(purchased_indices) == 0:
            # No purchase history - return popular products
            logger.info(f"Location {location_id} has no purchases, returning popular products")
            return self.get_popular_products(n_recommendations)
        
        # Calculate recommendation scores using item-based CF
        cf_scores = np.zeros(self.model_loader.item_similarity.shape[0])
        
        for item_idx in purchased_indices:
            # Get similar items and their scores
            similar_items = self.model_loader.item_similarity[item_idx].toarray().flatten()
            # Weight by purchase quantity
            cf_scores += similar_items * location_purchases[item_idx]
        
        # Normalize CF scores to 0-1 range
        if cf_scores.max() > 0:
            cf_scores = cf_scores / cf_scores.max()
        
        # Blend with popularity if hybrid mode enabled
        if use_hybrid:
            # Get popularity scores
            popularity_scores = self.model_loader.interaction_matrix.sum(axis=0).A1
            # Normalize popularity scores to 0-1 range
            if popularity_scores.max() > 0:
                popularity_scores = popularity_scores / popularity_scores.max()
            
            # Blend: (1-weight) * CF + weight * popularity
            scores = (1 - popularity_weight) * cf_scores + popularity_weight * popularity_scores
        else:
            scores = cf_scores
        
        # Exclude already purchased items
        if exclude_purchased:
            scores[purchased_indices] = -np.inf
        
        # Get top-N items
        top_indices = np.argsort(scores)[::-1][:n_recommendations * 2]  # Get extra for filtering
        
        # Build recommendations list
        recommendations = []
        for idx in top_indices:
            if scores[idx] > 0 and len(recommendations) < n_recommendations:
                product_id = self.model_loader.get_product_id(idx)
                recommendations.append({
                    'product_id': product_id,
                    'product_name': product_id,  # Same as ID for now
                    'score': float(scores[idx]),
                    'cf_score': float(cf_scores[idx]),
                    'popularity_score': float(popularity_scores[idx]) if use_hybrid else 0.0,
                    'rank': len(recommendations) + 1
                })
        
        # If not enough recommendations, fill with popular products
        if len(recommendations) < n_recommendations:
            popular = self.get_popular_products(n_recommendations - len(recommendations))
            # Add popular products that aren't already in recommendations
            existing_ids = {r['product_id'] for r in recommendations}
            for prod in popular:
                if prod['product_id'] not in existing_ids:
                    prod['rank'] = len(recommendations) + 1
                    recommendations.append(prod)
                    if len(recommendations) >= n_recommendations:
                        break
        
        return recommendations
    
    def get_user_recommendations(self, 
                                 customer_id: str, 
                                 n_recommendations: int = 10,
                                 exclude_purchased: bool = True) -> List[Dict]:
        """
        DEPRECATED: Use get_location_recommendations() instead
        Kept for backward compatibility with old API endpoints
        """
        logger.warning("get_user_recommendations() is deprecated. Use get_location_recommendations() instead.")
        return self.get_location_recommendations(customer_id, n_recommendations, exclude_purchased)
    
    def get_similar_products(self, 
                            product_id: str, 
                            n_recommendations: int = 10) -> List[Dict]:
        """
        Get products similar to a given product
        
        Args:
            product_id: Product identifier
            n_recommendations: Number of similar products to return
            
        Returns:
            List of similar products with similarity scores
        """
        # Get product index
        product_idx = self.model_loader.get_product_index(product_id)
        
        if product_idx is None:
            logger.warning(f"Product {product_id} not found")
            return []
        
        # Get similarity scores
        similarity_scores = self.model_loader.item_similarity[product_idx].toarray().flatten()
        
        # Get top-N similar products (excluding self)
        top_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations + 1]
        
        # Build recommendations list
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            if similarity_scores[idx] > 0:
                similar_product_id = self.model_loader.get_product_id(idx)
                recommendations.append({
                    'product_id': similar_product_id,
                    'product_name': similar_product_id,
                    'similarity_score': float(similarity_scores[idx]),
                    'rank': rank
                })
        
        return recommendations
    
    def get_popular_products(self, n_recommendations: int = 10) -> List[Dict]:
        """
        Get most popular products (fallback for new customers)
        
        Args:
            n_recommendations: Number of products to return
            
        Returns:
            List of popular products
        """
        # Sum purchases across all customers
        product_popularity = self.model_loader.interaction_matrix.sum(axis=0).A1
        
        # Get top-N products
        top_indices = np.argsort(product_popularity)[::-1][:n_recommendations]
        
        # Build recommendations list
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            product_id = self.model_loader.get_product_id(idx)
            recommendations.append({
                'product_id': product_id,
                'product_name': product_id,
                'popularity_score': float(product_popularity[idx]),
                'rank': rank
            })
        
        return recommendations
    
    def get_location_purchase_history(self, location_id: str) -> List[Dict]:
        """
        Get location's purchase history
        
        Args:
            location_id: Location identifier
            
        Returns:
            List of purchased products with quantities
        """
        location_idx = self.model_loader.get_location_index(location_id)
        
        if location_idx is None:
            return []
        
        # Get location's purchases
        location_purchases = self.model_loader.interaction_matrix[location_idx].toarray().flatten()
        purchased_indices = np.where(location_purchases > 0)[0]
        
        # Build purchase history
        history = []
        for idx in purchased_indices:
            product_id = self.model_loader.get_product_id(idx)
            history.append({
                'product_id': product_id,
                'product_name': product_id,
                'quantity': float(location_purchases[idx])
            })
        
        # Sort by quantity (most purchased first)
        history.sort(key=lambda x: x['quantity'], reverse=True)
        
        return history
    
    def get_customer_purchase_history(self, customer_id: str) -> List[Dict]:
        """
        DEPRECATED: Use get_location_purchase_history() instead
        Kept for backward compatibility
        """
        logger.warning("get_customer_purchase_history() is deprecated. Use get_location_purchase_history() instead.")
        return self.get_location_purchase_history(customer_id)
    
    def batch_recommendations(self, 
                             location_ids: List[str], 
                             n_recommendations: int = 10) -> Dict[str, List[Dict]]:
        """
        Get recommendations for multiple locations (batch processing)
        
        Args:
            location_ids: List of location identifiers
            n_recommendations: Number of recommendations per location
            
        Returns:
            Dictionary mapping location_id to recommendations
        """
        results = {}
        
        for location_id in location_ids:
            try:
                results[location_id] = self.get_location_recommendations(
                    location_id, 
                    n_recommendations
                )
            except Exception as e:
                logger.error(f"Error generating recommendations for {location_id}: {e}")
                results[location_id] = []
        
        return results
    
    def get_cart_recommendations(self,
                                location_id: str,
                                cart_skus: List[str],
                                n_recommendations: int = 5,
                                cart_weight: float = 0.5,
                                location_weight: float = 0.3,
                                popularity_weight: float = 0.2) -> List[Dict]:
        """
        Get recommendations based on cart contents and location
        "Frequently Bought Together" recommendations for checkout page
        
        Args:
            location_id: Location identifier (city-state-country)
            cart_skus: List of product SKUs currently in cart
            n_recommendations: Number of recommendations to return
            cart_weight: Weight for cart-based similarity (default: 0.5 = 50%)
            location_weight: Weight for location preferences (default: 0.3 = 30%)
            popularity_weight: Weight for popularity (default: 0.2 = 20%)
            
        Returns:
            List of recommended products with scores
        """
        logger.info(f"Generating cart recommendations for location={location_id}, cart_items={len(cart_skus)}")
        
        # Initialize scores
        n_products = self.model_loader.item_similarity.shape[0]
        cart_scores = np.zeros(n_products)
        location_scores = np.zeros(n_products)
        
        # Get cart product indices
        cart_indices = []
        for sku in cart_skus:
            idx = self.model_loader.get_product_index(sku)
            if idx is not None:
                cart_indices.append(idx)
            else:
                logger.warning(f"Product {sku} not found in trained model")
        
        if len(cart_indices) == 0:
            logger.warning("No cart items found in model, falling back to popular products")
            return self.get_popular_products(n_recommendations)
        
        # 1. Calculate cart-based similarity scores
        # For each item in cart, get similar items and aggregate
        for cart_idx in cart_indices:
            similar_items = self.model_loader.item_similarity[cart_idx].toarray().flatten()
            cart_scores += similar_items
        
        # Average across cart items
        if len(cart_indices) > 0:
            cart_scores = cart_scores / len(cart_indices)
        
        # Normalize to 0-1 range
        if cart_scores.max() > 0:
            cart_scores = cart_scores / cart_scores.max()
        
        # 2. Get location-based preferences
        location_idx = self.model_loader.get_location_index(location_id)
        
        if location_idx is not None:
            # Get location's purchase history
            location_purchases = self.model_loader.interaction_matrix[location_idx].toarray().flatten()
            purchased_indices = np.where(location_purchases > 0)[0]
            
            if len(purchased_indices) > 0:
                # Calculate location preference scores using item-based CF
                for item_idx in purchased_indices:
                    similar_items = self.model_loader.item_similarity[item_idx].toarray().flatten()
                    location_scores += similar_items * location_purchases[item_idx]
                
                # Normalize to 0-1 range
                if location_scores.max() > 0:
                    location_scores = location_scores / location_scores.max()
        
        # 3. Get popularity scores
        popularity_scores = self.model_loader.interaction_matrix.sum(axis=0).A1
        if popularity_scores.max() > 0:
            popularity_scores = popularity_scores / popularity_scores.max()
        
        # 4. Combine all signals
        combined_scores = (
            cart_weight * cart_scores +
            location_weight * location_scores +
            popularity_weight * popularity_scores
        )
        
        # Exclude items already in cart
        for cart_idx in cart_indices:
            combined_scores[cart_idx] = -np.inf
        
        # Get top-N items
        top_indices = np.argsort(combined_scores)[::-1][:n_recommendations * 2]
        
        # Build recommendations list
        recommendations = []
        for idx in top_indices:
            if combined_scores[idx] > 0 and len(recommendations) < n_recommendations:
                product_id = self.model_loader.get_product_id(idx)
                recommendations.append({
                    'product_id': product_id,
                    'product_name': product_id,
                    'score': float(combined_scores[idx]),
                    'cart_similarity_score': float(cart_scores[idx]),
                    'location_preference_score': float(location_scores[idx]),
                    'popularity_score': float(popularity_scores[idx]),
                    'rank': len(recommendations) + 1,
                    'reason': self._get_recommendation_reason(
                        cart_scores[idx],
                        location_scores[idx],
                        popularity_scores[idx]
                    )
                })
        
        # If not enough recommendations, fill with popular products
        if len(recommendations) < n_recommendations:
            popular = self.get_popular_products(n_recommendations - len(recommendations))
            existing_ids = {r['product_id'] for r in recommendations}
            for prod in popular:
                if prod['product_id'] not in existing_ids:
                    prod['rank'] = len(recommendations) + 1
                    prod['reason'] = 'Popular product'
                    recommendations.append(prod)
                    if len(recommendations) >= n_recommendations:
                        break
        
        logger.info(f"Generated {len(recommendations)} cart recommendations")
        return recommendations
    
    def _get_recommendation_reason(self, cart_score: float, location_score: float, popularity_score: float) -> str:
        """
        Generate human-readable reason for recommendation
        
        Args:
            cart_score: Cart similarity score
            location_score: Location preference score
            popularity_score: Popularity score
            
        Returns:
            Reason string
        """
        scores = {
            'Frequently bought with cart items': cart_score,
            'Popular in your area': location_score,
            'Trending product': popularity_score
        }
        
        # Get top reason
        top_reason = max(scores.items(), key=lambda x: x[1])
        return top_reason[0]
