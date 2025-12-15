"""
Model Loader Module
Loads trained models and provides singleton access
"""
import pickle
import json
import os
import logging
from typing import Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton class to load and cache trained models"""
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelLoader._models_loaded:
            self.interaction_matrix = None
            self.item_similarity = None
            self.user_similarity = None
            self.location_encoder = None  # Changed from customer_encoder
            self.product_encoder = None
            self.metadata = None
            self.validation_report = None
            self.model_dir = None
            self.load_time = None
    
    def load_models(self, model_dir: str = 'data/models/production') -> bool:
        """
        Load all model artifacts
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            True if successful, False otherwise
        """
        if ModelLoader._models_loaded and self.model_dir == model_dir:
            logger.info("Models already loaded")
            return True
        
        logger.info(f"Loading models from {model_dir}...")
        start_time = datetime.now()
        
        try:
            # Check if directory exists
            if not os.path.exists(model_dir):
                logger.error(f"Model directory not found: {model_dir}")
                return False
            
            # Load interaction matrix
            logger.info("  Loading interaction matrix...")
            with open(os.path.join(model_dir, 'interaction_matrix.pkl'), 'rb') as f:
                self.interaction_matrix = pickle.load(f)
            
            # Load item similarity
            logger.info("  Loading item similarity matrix...")
            with open(os.path.join(model_dir, 'item_similarity.pkl'), 'rb') as f:
                self.item_similarity = pickle.load(f)
            
            # Load user similarity
            logger.info("  Loading user similarity matrix...")
            with open(os.path.join(model_dir, 'user_similarity.pkl'), 'rb') as f:
                self.user_similarity = pickle.load(f)
            
            # Load encoders
            logger.info("  Loading location encoder...")
            # Try new location_encoder.pkl first, fallback to customer_encoder.pkl for backward compatibility
            location_encoder_path = os.path.join(model_dir, 'location_encoder.pkl')
            customer_encoder_path = os.path.join(model_dir, 'customer_encoder.pkl')
            
            if os.path.exists(location_encoder_path):
                with open(location_encoder_path, 'rb') as f:
                    self.location_encoder = pickle.load(f)
            elif os.path.exists(customer_encoder_path):
                logger.warning("  Using deprecated customer_encoder.pkl - retrain model to use location_encoder.pkl")
                with open(customer_encoder_path, 'rb') as f:
                    self.location_encoder = pickle.load(f)
            else:
                raise FileNotFoundError("Neither location_encoder.pkl nor customer_encoder.pkl found")
            
            logger.info("  Loading product encoder...")
            with open(os.path.join(model_dir, 'product_encoder.pkl'), 'rb') as f:
                self.product_encoder = pickle.load(f)
            
            # Load metadata
            logger.info("  Loading metadata...")
            with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)
            
            # Load validation report
            logger.info("  Loading validation report...")
            with open(os.path.join(model_dir, 'validation_report.json'), 'r') as f:
                self.validation_report = json.load(f)
            
            self.model_dir = model_dir
            self.load_time = datetime.now()
            ModelLoader._models_loaded = True
            
            load_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Models loaded successfully in {load_duration:.2f} seconds")
            logger.info(f"  Locations: {len(self.location_encoder.classes_):,}")
            logger.info(f"  Products: {len(self.product_encoder.classes_):,}")
            logger.info(f"  Training date: {self.metadata['training_date']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def get_location_index(self, location_id: str) -> Optional[int]:
        """Get matrix index for location ID"""
        try:
            return self.location_encoder.transform([str(location_id)])[0]
        except:
            return None
    
    def get_product_index(self, product_id: str) -> Optional[int]:
        """Get matrix index for product ID"""
        try:
            return self.product_encoder.transform([str(product_id)])[0]
        except:
            return None
    
    def get_location_id(self, location_idx: int) -> str:
        """Get location ID from matrix index"""
        return self.location_encoder.inverse_transform([location_idx])[0]
    
    def get_product_id(self, product_idx: int) -> str:
        """Get product ID from matrix index"""
        return self.product_encoder.inverse_transform([product_idx])[0]
    
    # Deprecated methods for backward compatibility
    def get_customer_index(self, customer_id: str) -> Optional[int]:
        """DEPRECATED: Use get_location_index() instead"""
        return self.get_location_index(customer_id)
    
    def get_customer_id(self, customer_idx: int) -> str:
        """DEPRECATED: Use get_location_id() instead"""
        return self.get_location_id(customer_idx)
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return ModelLoader._models_loaded
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if not self.is_loaded():
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_dir': self.model_dir,
            'load_time': self.load_time.isoformat() if self.load_time else None,
            'n_locations': len(self.location_encoder.classes_),
            'n_products': len(self.product_encoder.classes_),
            'training_date': self.metadata.get('training_date'),
            'training_duration_seconds': self.metadata.get('training_duration_seconds'),
            'validation_metrics': self.validation_report
        }
