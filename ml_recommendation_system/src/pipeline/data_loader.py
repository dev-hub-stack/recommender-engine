"""
Data Loader Module
Handles loading data from CSV files, API endpoints, or database
"""
import pandas as pd
import requests
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load order data from CSV files, API, or database"""
    
    def __init__(self, source_type: str = 'database'):
        """
        Initialize DataLoader
        
        Args:
            source_type: 'csv', 'api', or 'database' (default: database)
        """
        self.source_type = source_type
        
    def load_from_csv(self, pos_file: str, oe_file: str) -> pd.DataFrame:
        """
        Load data from CSV files
        
        Args:
            pos_file: Path to POS orders CSV
            oe_file: Path to OE orders CSV
            
        Returns:
            Combined DataFrame with all orders
        """
        logger.info(f"Loading POS orders from {pos_file}")
        pos_orders = pd.read_csv(pos_file)
        logger.info(f"Loaded {len(pos_orders)} POS orders")
        
        logger.info(f"Loading OE orders from {oe_file}")
        oe_orders = pd.read_csv(oe_file)
        logger.info(f"Loaded {len(oe_orders)} OE orders")
        
        # Select only needed columns
        needed_columns = ['customer_phone', 'customer_email', 'has_items', 'order_date', 'id']
        
        pos_clean = pos_orders[needed_columns].copy()
        oe_clean = oe_orders[needed_columns].copy()
        
        # Add source column to track origin
        pos_clean['source'] = 'POS'
        oe_clean['source'] = 'OE'
        
        # Combine datasets
        all_orders = pd.concat([pos_clean, oe_clean], ignore_index=True)
        logger.info(f"Combined total: {len(all_orders)} orders")
        
        return all_orders
    
    def load_from_api(self, api_config: Dict[str, str]) -> pd.DataFrame:
        """
        Load data from API endpoints (for production)
        
        Args:
            api_config: Dictionary with API configuration
                {
                    'pos_endpoint': 'https://api.example.com/pos-orders',
                    'oe_endpoint': 'https://api.example.com/oe-orders',
                    'auth_token': 'Bearer xxx',
                    'headers': {...}
                }
                
        Returns:
            Combined DataFrame with all orders
        """
        logger.info("Loading data from API endpoints")
        
        headers = api_config.get('headers', {})
        if 'auth_token' in api_config:
            headers['Authorization'] = api_config['auth_token']
        
        # Load POS orders
        logger.info(f"Fetching POS orders from {api_config['pos_endpoint']}")
        pos_response = requests.get(api_config['pos_endpoint'], headers=headers)
        pos_response.raise_for_status()
        pos_data = pos_response.json()
        pos_orders = pd.DataFrame(pos_data)
        logger.info(f"Fetched {len(pos_orders)} POS orders")
        
        # Load OE orders
        logger.info(f"Fetching OE orders from {api_config['oe_endpoint']}")
        oe_response = requests.get(api_config['oe_endpoint'], headers=headers)
        oe_response.raise_for_status()
        oe_data = oe_response.json()
        oe_orders = pd.DataFrame(oe_data)
        logger.info(f"Fetched {len(oe_orders)} OE orders")
        
        # Select only needed columns
        needed_columns = ['customer_phone', 'customer_email', 'has_items', 'order_date', 'id']
        
        pos_clean = pos_orders[needed_columns].copy()
        oe_clean = oe_orders[needed_columns].copy()
        
        # Add source column
        pos_clean['source'] = 'POS'
        oe_clean['source'] = 'OE'
        
        # Combine datasets
        all_orders = pd.concat([pos_clean, oe_clean], ignore_index=True)
        logger.info(f"Combined total: {len(all_orders)} orders")
        
        return all_orders
    
    def load_from_database(self, database_url: str, **filters) -> pd.DataFrame:
        """
        Load data directly from existing database
        
        Args:
            database_url: PostgreSQL connection string
            **filters: Optional filters (start_date, end_date, limit)
            
        Returns:
            DataFrame with all orders
        """
        import sys
        import os
        
        # Add src to path if not already there
        src_path = os.path.join(os.path.dirname(__file__), '..')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from ingestion.simple_db_loader import SimpleDBLoader
        
        logger.info("Loading data from database...")
        
        loader = SimpleDBLoader(database_url)
        
        # Load orders
        orders = loader.load_orders_from_db(
            start_date=filters.get('start_date'),
            end_date=filters.get('end_date'),
            limit=filters.get('limit')
        )
        
        logger.info(f"Loaded {len(orders)} orders from database")
        
        return orders
    
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Load data based on source_type
        
        Args:
            **kwargs: Arguments for specific loader
                For CSV: pos_file, oe_file
                For API: api_config
                For Database: database_url, start_date, end_date, limit
                
        Returns:
            Combined DataFrame with all orders
        """
        if self.source_type == 'csv':
            return self.load_from_csv(kwargs['pos_file'], kwargs['oe_file'])
        elif self.source_type == 'api':
            return self.load_from_api(kwargs['api_config'])
        elif self.source_type == 'database':
            return self.load_from_database(
                database_url=kwargs['database_url'],
                start_date=kwargs.get('start_date'),
                end_date=kwargs.get('end_date'),
                limit=kwargs.get('limit')
            )
        else:
            raise ValueError(f"Unknown source_type: {self.source_type}")
