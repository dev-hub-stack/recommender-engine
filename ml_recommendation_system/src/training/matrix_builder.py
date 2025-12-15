"""
Matrix Builder Module
Builds user-item interaction matrix for collaborative filtering
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatrixBuilder:
    """Build sparse location-item interaction matrix"""
    
    def __init__(self, min_location_interactions: int = 1, min_product_interactions: int = 2):
        """
        Initialize MatrixBuilder
        
        Args:
            min_location_interactions: Minimum purchases per location
            min_product_interactions: Minimum purchases per product
        """
        self.min_location_interactions = min_location_interactions
        self.min_product_interactions = min_product_interactions
        self.location_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.interaction_matrix = None
        self.n_locations = 0
        self.n_products = 0
    
    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out locations/products with too few interactions
        
        Args:
            data: DataFrame with location_id, product_id, quantity
            
        Returns:
            Filtered DataFrame
        """
        logger.info("Filtering data...")
        original_len = len(data)
        
        # Filter locations
        if self.min_location_interactions > 1:
            location_counts = data['location_id'].value_counts()
            valid_locations = location_counts[location_counts >= self.min_location_interactions].index
            data = data[data['location_id'].isin(valid_locations)]
            logger.info(f"  Locations with >= {self.min_location_interactions} interactions: {len(valid_locations):,}")
        
        # Filter products
        if self.min_product_interactions > 1:
            product_counts = data['product_id'].value_counts()
            valid_products = product_counts[product_counts >= self.min_product_interactions].index
            data = data[data['product_id'].isin(valid_products)]
            logger.info(f"  Products with >= {self.min_product_interactions} interactions: {len(valid_products):,}")
        
        filtered_len = len(data)
        logger.info(f"  Interactions retained: {filtered_len:,} / {original_len:,} ({filtered_len/original_len*100:.1f}%)")
        
        return data
    
    def build_matrix(self, data: pd.DataFrame) -> csr_matrix:
        """
        Build sparse location-item interaction matrix
        
        Args:
            data: DataFrame with location_id, product_id, quantity
            
        Returns:
            Sparse interaction matrix (locations × products)
        """
        logger.info("\n" + "="*60)
        logger.info("BUILDING INTERACTION MATRIX")
        logger.info("="*60)
        
        # Filter data
        data = self.filter_data(data)
        
        # Encode location and product IDs
        logger.info("Encoding location and product IDs...")
        # Convert to string to ensure uniform type
        data['location_id'] = data['location_id'].astype(str)
        data['product_id'] = data['product_id'].astype(str)
        
        data['location_idx'] = self.location_encoder.fit_transform(data['location_id'])
        data['product_idx'] = self.product_encoder.fit_transform(data['product_id'])
        
        self.n_locations = data['location_idx'].nunique()
        self.n_products = data['product_idx'].nunique()
        
        logger.info(f"  Locations: {self.n_locations:,}")
        logger.info(f"  Products: {self.n_products:,}")
        
        # Build sparse matrix
        logger.info("Creating sparse matrix...")
        self.interaction_matrix = csr_matrix(
            (data['quantity'], (data['location_idx'], data['product_idx'])),
            shape=(self.n_locations, self.n_products),
            dtype=np.float32
        )
        
        # Calculate sparsity
        n_interactions = self.interaction_matrix.nnz
        total_possible = self.n_locations * self.n_products
        sparsity = 100 * (1 - n_interactions / total_possible)
        
        logger.info(f"  Matrix shape: {self.n_locations:,} × {self.n_products:,}")
        logger.info(f"  Non-zero interactions: {n_interactions:,}")
        logger.info(f"  Sparsity: {sparsity:.4f}%")
        logger.info(f"  Memory (approx): {self.interaction_matrix.data.nbytes / 1024 / 1024:.1f} MB")
        logger.info("="*60)
        
        return self.interaction_matrix
    
    def get_location_index(self, location_id: str) -> int:
        """Get matrix index for location ID"""
        try:
            return self.location_encoder.transform([location_id])[0]
        except:
            return -1
    
    def get_product_index(self, product_id: str) -> int:
        """Get matrix index for product ID"""
        try:
            return self.product_encoder.transform([product_id])[0]
        except:
            return -1
    
    def get_location_id(self, location_idx: int) -> str:
        """Get location ID from matrix index"""
        return self.location_encoder.inverse_transform([location_idx])[0]
    
    def get_product_id(self, product_idx: int) -> str:
        """Get product ID from matrix index"""
        return self.product_encoder.inverse_transform([product_idx])[0]
