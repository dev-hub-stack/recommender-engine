"""
ML Recommendation Service
Wrapper for collaborative filtering ML model
"""
import sys
import os
import logging
from typing import List, Dict, Optional

# Add ml_recommendation_system to path
ml_system_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_recommendation_system')
sys.path.insert(0, os.path.dirname(ml_system_path))

# Now import ML modules with absolute imports
from ml_recommendation_system.src.api.model_loader import ModelLoader
from ml_recommendation_system.src.api.recommender import Recommender  
from ml_recommendation_system.src.pipeline.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class MLRecommendationService:
    """Service for ML-based collaborative filtering recommendations"""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize ML recommendation service
        
        Args:
            model_dir: Path to trained models directory
        """
        if model_dir is None:
            # ml_system_path is already set above
            model_dir = os.path.join(ml_system_path, 'data', 'models', 'production')
        
        self.model_dir = model_dir
        self.model_loader = None
        self.recommender = None
        self.data_processor = DataProcessor()
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Load ML models
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Loading ML recommendation models...")
            
            # Load models
            self.model_loader = ModelLoader()
            if not self.model_loader.load_models(self.model_dir):
                logger.error("Failed to load ML models")
                return False
            
            # Initialize recommender
            self.recommender = Recommender()
            self._initialized = True
            
            logger.info("✅ ML recommendation models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML recommendation service: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._initialized and self.recommender is not None
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.is_ready():
            return {"error": "ML models not loaded"}
        
        return self.model_loader.get_model_info()
    
    def get_location_recommendations(
        self,
        city: str = "",
        state: str = "",
        country: str = "",
        limit: int = 10,
        exclude_purchased: bool = True
    ) -> List[Dict]:
        """
        Get recommendations for a location
        
        Args:
            city: Customer city
            state: Customer state
            country: Customer country
            limit: Number of recommendations
            exclude_purchased: Exclude already purchased products
            
        Returns:
            List of recommendations
        """
        if not self.is_ready():
            raise RuntimeError("ML models not loaded")
        
        # Create location ID
        location_id = self.data_processor.create_location_id(city, state, country)
        
        # Get recommendations
        return self.recommender.get_location_recommendations(
            location_id=location_id,
            n_recommendations=limit,
            exclude_purchased=exclude_purchased
        )
    
    def get_cart_recommendations(
        self,
        city: str = "",
        state: str = "",
        country: str = "",
        cart_skus: List[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Get cart-based recommendations
        
        Args:
            city: Customer city
            state: Customer state
            country: Customer country
            cart_skus: List of SKUs in cart
            limit: Number of recommendations
            
        Returns:
            List of recommendations
        """
        if not self.is_ready():
            raise RuntimeError("ML models not loaded")
        
        if cart_skus is None:
            cart_skus = []
        
        # Create location ID
        location_id = self.data_processor.create_location_id(city, state, country)
        
        # Get recommendations
        return self.recommender.get_cart_recommendations(
            location_id=location_id,
            cart_skus=cart_skus,
            n_recommendations=limit
        )
    
    def get_similar_products(
        self,
        product_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get similar products
        
        Args:
            product_id: Product SKU
            limit: Number of similar products
            
        Returns:
            List of similar products
        """
        if not self.is_ready():
            raise RuntimeError("ML models not loaded")
        
        return self.recommender.get_similar_products(
            product_id=product_id,
            n_recommendations=limit
        )
    
    def get_popular_products(self, limit: int = 10) -> List[Dict]:
        """
        Get popular products
        
        Args:
            limit: Number of products
            
        Returns:
            List of popular products
        """
        if not self.is_ready():
            raise RuntimeError("ML models not loaded")
        
        return self.recommender.get_popular_products(n_recommendations=limit)
    
    def batch_recommendations(
        self,
        locations: List[Dict],
        limit: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Get recommendations for multiple locations
        
        Args:
            locations: List of location dicts with city, state, country
            limit: Number of recommendations per location
            
        Returns:
            Dict mapping location_id to recommendations
        """
        if not self.is_ready():
            raise RuntimeError("ML models not loaded")
        
        # Create location IDs
        location_ids = []
        for loc in locations:
            location_id = self.data_processor.create_location_id(
                loc.get('city', ''),
                loc.get('state', ''),
                loc.get('country', '')
            )
            location_ids.append(location_id)
        
        # Get batch recommendations
        return self.recommender.batch_recommendations(location_ids, limit)


# Global instance (initialized at startup)
ml_service = MLRecommendationService()
