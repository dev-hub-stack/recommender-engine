"""
Data Splitter Module
Handles train/test split for model training
"""
import pandas as pd
import logging
from datetime import timedelta
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train and test sets"""
    
    def __init__(self, method: str = 'time_based', test_ratio: float = 0.1, test_days: int = 60):
        """
        Initialize DataSplitter
        
        Args:
            method: 'time_based' or 'random'
            test_ratio: Ratio of test data (0.1 = 10%)
            test_days: Number of days for test set (if time_based)
        """
        self.method = method
        self.test_ratio = test_ratio
        self.test_days = test_days
    
    def split_time_based(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by time (most recent data for testing)
        
        Args:
            data: DataFrame with order_date column
            
        Returns:
            Tuple of (train_data, test_data)
        """
        logger.info(f"Performing time-based split (last {self.test_days} days for testing)")
        
        # Convert to datetime
        data['order_date'] = pd.to_datetime(data['order_date'])
        
        # Calculate split date
        max_date = data['order_date'].max()
        split_date = max_date - timedelta(days=self.test_days)
        
        # Split
        train_data = data[data['order_date'] < split_date].copy()
        test_data = data[data['order_date'] >= split_date].copy()
        
        logger.info(f"Split date: {split_date.date()}")
        logger.info(f"Train data: {len(train_data):,} interactions ({len(train_data)/len(data)*100:.1f}%)")
        logger.info(f"  Date range: {train_data['order_date'].min().date()} to {train_data['order_date'].max().date()}")
        logger.info(f"  Unique locations: {train_data['location_id'].nunique():,}")
        logger.info(f"  Unique products: {train_data['product_id'].nunique():,}")
        
        logger.info(f"Test data: {len(test_data):,} interactions ({len(test_data)/len(data)*100:.1f}%)")
        logger.info(f"  Date range: {test_data['order_date'].min().date()} to {test_data['order_date'].max().date()}")
        logger.info(f"  Unique locations: {test_data['location_id'].nunique():,}")
        logger.info(f"  Unique products: {test_data['product_id'].nunique():,}")
        
        return train_data, test_data
    
    def split_random(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data randomly
        
        Args:
            data: DataFrame
            
        Returns:
            Tuple of (train_data, test_data)
        """
        logger.info(f"Performing random split ({(1-self.test_ratio)*100:.0f}% train, {self.test_ratio*100:.0f}% test)")
        
        # Shuffle data
        data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split index
        split_idx = int(len(data_shuffled) * (1 - self.test_ratio))
        
        # Split
        train_data = data_shuffled[:split_idx].copy()
        test_data = data_shuffled[split_idx:].copy()
        
        logger.info(f"Train data: {len(train_data):,} interactions ({len(train_data)/len(data)*100:.1f}%)")
        logger.info(f"  Unique locations: {train_data['location_id'].nunique():,}")
        logger.info(f"  Unique products: {train_data['product_id'].nunique():,}")
        
        logger.info(f"Test data: {len(test_data):,} interactions ({len(test_data)/len(data)*100:.1f}%)")
        logger.info(f"  Unique locations: {test_data['location_id'].nunique():,}")
        logger.info(f"  Unique products: {test_data['product_id'].nunique():,}")
        
        return train_data, test_data
    
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data based on configured method
        
        Args:
            data: DataFrame to split
            
        Returns:
            Tuple of (train_data, test_data)
        """
        logger.info("\n" + "="*60)
        logger.info("TRAIN/TEST SPLIT")
        logger.info("="*60)
        logger.info(f"Total data: {len(data):,} interactions")
        logger.info(f"Split method: {self.method}")
        
        if self.method == 'time_based':
            train, test = self.split_time_based(data)
        elif self.method == 'random':
            train, test = self.split_random(data)
        else:
            raise ValueError(f"Unknown split method: {self.method}")
        
        logger.info("="*60)
        
        return train, test
