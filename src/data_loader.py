"""
Data Loader for Recommendation Engine
Provides unified data loading interface using UnifiedDataLayer for API-based data access.

This module replaces CSV-based data loading with API-based loading while maintaining
backward compatibility with existing algorithm interfaces.

Requirements: 5.1, 5.2 (API data loading for recommendation engine)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog

import sys
import os

# Removed shared dependency for local execution
# from data_processing.unified_data_layer import UnifiedDataLayer

logger = structlog.get_logger(__name__)


class RecommendationDataLoader:
    """
    Data loader for recommendation engine that uses UnifiedDataLayer for API-based data access.
    
    Features:
    - Loads data from API via UnifiedDataLayer
    - Uses 6-month lookback for training data (as per requirements)
    - Prepares data in formats expected by recommendation algorithms
    - Maintains backward compatibility with existing algorithm interfaces
    - Tracks data freshness
    
    Requirements: 5.1, 5.2
    """
    
    def __init__(self, mode: str = 'api', lookback_months: int = 6):
        """
        Initialize the data loader.
        
        Args:
            mode: Data source mode - 'api' or 'csv' (default: 'api')
            lookback_months: Number of months to look back for training data (default: 6)
        
        Requirements: 5.1 (API data loading), 5.2 (6-month lookback)
        """
        self.mode = mode
        self.lookback_months = lookback_months
        self.data_layer = UnifiedDataLayer(mode=mode)
        self.logger = logger.bind(component="RecommendationDataLoader", mode=mode)
        
        # Data freshness tracking
        self.last_load_time: Optional[datetime] = None
        self.data_timestamp: Optional[datetime] = None
        
        self.logger.info(
            "Recommendation data loader initialized",
            mode=mode,
            lookback_months=lookback_months
        )
    
    def load_training_data(
        self,
        use_lookback: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training data for recommendation algorithms.
        
        Uses 6-month lookback for training to focus on recent patterns.
        
        Args:
            use_lookback: If True, use 6-month lookback; if False, load all data
        
        Returns:
            Tuple of (sales_data, product_data, customer_data)
        
        Requirements: 5.1, 5.2 (6-month lookback for training)
        """
        start_time = datetime.now()
        self.logger.info(
            "Loading training data",
            use_lookback=use_lookback,
            lookback_months=self.lookback_months if use_lookback else None
        )
        
        try:
            # Determine time range
            time_range = '6M' if use_lookback else 'ALL'
            
            # Load all orders (POS + OE)
            all_orders = self.data_layer.load_all_orders(time_range=time_range)
            
            if all_orders is None or all_orders.empty:
                self.logger.error("No orders loaded from data layer")
                raise ValueError("Failed to load orders data")
            
            self.logger.info(
                "Orders loaded successfully",
                n_orders=len(all_orders),
                time_range=time_range
            )
            
            # Prepare data for algorithms
            sales_data = self._prepare_sales_data(all_orders)
            product_data = self._extract_product_data(all_orders)
            customer_data = self._extract_customer_data(all_orders)
            
            # Update freshness tracking
            self.last_load_time = datetime.now()
            self.data_timestamp = all_orders['order_date'].max() if 'order_date' in all_orders.columns else datetime.now()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                "Training data loaded successfully",
                n_sales=len(sales_data),
                n_products=len(product_data),
                n_customers=len(customer_data),
                elapsed_seconds=elapsed,
                data_timestamp=self.data_timestamp
            )
            
            return sales_data, product_data, customer_data
        
        except Exception as e:
            self.logger.error("Failed to load training data", error=str(e))
            raise
    
    def _prepare_sales_data(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare sales data in format expected by recommendation algorithms.
        
        Args:
            orders_df: Raw orders DataFrame
        
        Returns:
            Sales DataFrame with required columns
        """
        self.logger.debug("Preparing sales data")
        
        # Create sales data with required columns
        sales_data = orders_df.copy()
        
        # Ensure required columns exist
        required_columns = {
            'order_id': 'id',
            'product_id': 'product_name',  # Use product_name as product_id if not available
            'customer_id': 'unified_customer_id',
            'order_date': 'order_date',
            'quantity': 'quantity',
            'total_amount': 'total_price',
            'city': 'customer_city',
            'order_type': 'order_type'
        }
        
        # Map columns
        for target_col, source_col in required_columns.items():
            if target_col not in sales_data.columns and source_col in sales_data.columns:
                sales_data[target_col] = sales_data[source_col]
        
        # Add derived columns
        if 'order_date' in sales_data.columns:
            sales_data['sale_date'] = pd.to_datetime(sales_data['order_date'])
        
        # Handle missing quantity (default to 1)
        if 'quantity' not in sales_data.columns:
            sales_data['quantity'] = 1
        
        # Handle missing total_amount
        if 'total_amount' not in sales_data.columns and 'total_price' in sales_data.columns:
            sales_data['total_amount'] = sales_data['total_price']
        
        self.logger.debug(
            "Sales data prepared",
            n_rows=len(sales_data),
            columns=list(sales_data.columns)
        )
        
        return sales_data
    
    def _extract_product_data(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract product data from orders.
        
        Args:
            orders_df: Raw orders DataFrame
        
        Returns:
            Product DataFrame with unique products
        """
        self.logger.debug("Extracting product data")
        
        # Get unique products
        product_columns = ['product_name', 'product_type', 'brand_name']
        available_columns = [col for col in product_columns if col in orders_df.columns]
        
        if not available_columns:
            self.logger.warning("No product columns found, creating minimal product data")
            # Create minimal product data
            unique_products = orders_df['product_name'].unique() if 'product_name' in orders_df.columns else []
            product_data = pd.DataFrame({
                'product_id': unique_products,
                'product_name': unique_products,
                'category': 'unknown',
                'price': 10000  # Default price
            })
        else:
            product_data = orders_df[available_columns].drop_duplicates()
            
            # Add product_id if not present
            if 'product_id' not in product_data.columns:
                if 'product_name' in product_data.columns:
                    product_data['product_id'] = product_data['product_name']
            
            # Add category if not present
            if 'category' not in product_data.columns:
                if 'product_type' in product_data.columns:
                    product_data['category'] = product_data['product_type']
                else:
                    product_data['category'] = 'unknown'
            
            # Estimate price from orders
            if 'price' not in product_data.columns:
                if 'total_price' in orders_df.columns and 'product_name' in orders_df.columns:
                    avg_prices = orders_df.groupby('product_name')['total_price'].mean()
                    product_data['price'] = product_data['product_name'].map(avg_prices).fillna(10000)
                else:
                    product_data['price'] = 10000  # Default
        
        self.logger.debug(
            "Product data extracted",
            n_products=len(product_data),
            columns=list(product_data.columns)
        )
        
        return product_data
    
    def _extract_customer_data(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract customer data from orders.
        
        Args:
            orders_df: Raw orders DataFrame
        
        Returns:
            Customer DataFrame with unique customers
        """
        self.logger.debug("Extracting customer data")
        
        # Get unique customers
        customer_columns = [
            'unified_customer_id', 'customer_name', 'customer_email',
            'customer_phone', 'customer_city', 'customer_address'
        ]
        available_columns = [col for col in customer_columns if col in orders_df.columns]
        
        if not available_columns:
            self.logger.warning("No customer columns found, creating minimal customer data")
            # Create minimal customer data
            unique_customers = orders_df['customer_name'].unique() if 'customer_name' in orders_df.columns else []
            customer_data = pd.DataFrame({
                'customer_id': unique_customers,
                'customer_name': unique_customers,
                'city': 'unknown'
            })
        else:
            customer_data = orders_df[available_columns].drop_duplicates()
            
            # Add customer_id if not present
            if 'customer_id' not in customer_data.columns:
                if 'unified_customer_id' in customer_data.columns:
                    customer_data['customer_id'] = customer_data['unified_customer_id']
                elif 'customer_name' in customer_data.columns:
                    customer_data['customer_id'] = customer_data['customer_name']
            
            # Add city if not present
            if 'city' not in customer_data.columns:
                if 'customer_city' in customer_data.columns:
                    customer_data['city'] = customer_data['customer_city']
                else:
                    customer_data['city'] = 'unknown'
        
        self.logger.debug(
            "Customer data extracted",
            n_customers=len(customer_data),
            columns=list(customer_data.columns)
        )
        
        return customer_data
    
    def load_collaborative_filtering_data(
        self,
        use_lookback: bool = True
    ) -> pd.DataFrame:
        """
        Load data formatted for collaborative filtering algorithms.
        
        Args:
            use_lookback: If True, use 6-month lookback
        
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        
        Requirements: 5.1, 5.2
        """
        self.logger.info("Loading collaborative filtering data")
        
        try:
            # Load orders
            time_range = '6M' if use_lookback else 'ALL'
            all_orders = self.data_layer.load_all_orders(time_range=time_range)
            
            if all_orders is None or all_orders.empty:
                raise ValueError("No orders loaded")
            
            # Prepare CF data
            cf_data = pd.DataFrame()
            
            # Map to CF format
            cf_data['user_id'] = all_orders.get('unified_customer_id', all_orders.get('customer_name', ''))
            cf_data['item_id'] = all_orders.get('product_name', '')
            
            # Create implicit rating from quantity and price
            quantity = all_orders.get('quantity', 1)
            price = all_orders.get('total_price', 0)
            
            # Rating = normalized score based on quantity and price
            # Higher quantity and price = higher rating
            cf_data['rating'] = np.clip(
                (quantity / quantity.max() * 2.5) + (price / price.max() * 2.5),
                1.0,
                5.0
            )
            
            cf_data['timestamp'] = pd.to_datetime(all_orders.get('order_date', datetime.now()))
            
            # Remove rows with missing critical data
            cf_data = cf_data.dropna(subset=['user_id', 'item_id'])
            
            self.logger.info(
                "Collaborative filtering data loaded",
                n_interactions=len(cf_data),
                n_users=cf_data['user_id'].nunique(),
                n_items=cf_data['item_id'].nunique()
            )
            
            return cf_data
        
        except Exception as e:
            self.logger.error("Failed to load CF data", error=str(e))
            raise
    
    def get_data_freshness(self) -> Dict[str, any]:
        """
        Get data freshness information.
        
        Returns:
            Dictionary with freshness metrics
        
        Requirements: 10.4 (data freshness tracking)
        """
        if self.last_load_time is None:
            return {
                'status': 'not_loaded',
                'last_load_time': None,
                'data_timestamp': None,
                'age_hours': None
            }
        
        age_hours = (datetime.now() - self.data_timestamp).total_seconds() / 3600 if self.data_timestamp else None
        
        return {
            'status': 'loaded',
            'last_load_time': self.last_load_time.isoformat(),
            'data_timestamp': self.data_timestamp.isoformat() if self.data_timestamp else None,
            'age_hours': age_hours,
            'mode': self.mode,
            'lookback_months': self.lookback_months
        }
    
    def load_latest_orders(
        self,
        since_hours: int = 1
    ) -> pd.DataFrame:
        """
        Load latest orders for real-time recommendation updates.
        
        Fetches orders from the last N hours to enable real-time recommendations
        based on the most recent customer behavior.
        
        Args:
            since_hours: Number of hours to look back (default: 1)
        
        Returns:
            DataFrame with recent orders
        
        Requirements: 5.3 (real-time recommendations with latest orders)
        """
        self.logger.info("Loading latest orders for real-time updates", since_hours=since_hours)
        
        try:
            # For real-time updates, we want fresh data from API
            # Force cache refresh by using a very short time range
            all_orders = self.data_layer.load_all_orders(time_range='1M')
            
            if all_orders is None or all_orders.empty:
                self.logger.warning("No orders loaded for real-time update")
                return pd.DataFrame()
            
            # Filter to recent orders
            if 'order_date' in all_orders.columns:
                cutoff_time = datetime.now() - timedelta(hours=since_hours)
                all_orders['order_date'] = pd.to_datetime(all_orders['order_date'])
                recent_orders = all_orders[all_orders['order_date'] >= cutoff_time]
            else:
                # If no date column, return all (fallback)
                recent_orders = all_orders
            
            self.logger.info(
                "Latest orders loaded",
                n_orders=len(recent_orders),
                since_hours=since_hours
            )
            
            return recent_orders
        
        except Exception as e:
            self.logger.error("Failed to load latest orders", error=str(e))
            return pd.DataFrame()
    
    def get_real_time_product_trends(
        self,
        since_hours: int = 24
    ) -> Dict[str, Dict[str, any]]:
        """
        Get real-time product trends for recommendation updates.
        
        Analyzes recent orders to identify trending products and patterns.
        
        Args:
            since_hours: Number of hours to analyze (default: 24)
        
        Returns:
            Dictionary mapping product_id to trend metrics
        
        Requirements: 5.3 (real-time recommendation updates)
        """
        self.logger.info("Calculating real-time product trends", since_hours=since_hours)
        
        try:
            recent_orders = self.load_latest_orders(since_hours=since_hours)
            
            if recent_orders.empty:
                return {}
            
            # Calculate trend metrics
            product_trends = {}
            
            # Group by product
            if 'product_name' in recent_orders.columns:
                product_stats = recent_orders.groupby('product_name').agg({
                    'quantity': 'sum' if 'quantity' in recent_orders.columns else 'count',
                    'total_price': 'sum' if 'total_price' in recent_orders.columns else 'count',
                    'order_date': 'count'
                }).reset_index()
                
                product_stats.columns = ['product_id', 'total_quantity', 'total_revenue', 'order_count']
                
                for _, row in product_stats.iterrows():
                    product_id = row['product_id']
                    product_trends[product_id] = {
                        'total_quantity': int(row['total_quantity']),
                        'total_revenue': float(row['total_revenue']),
                        'order_count': int(row['order_count']),
                        'velocity': row['total_quantity'] / since_hours,  # Sales per hour
                        'timestamp': datetime.now().isoformat()
                    }
            
            self.logger.info(
                "Real-time trends calculated",
                n_trending_products=len(product_trends)
            )
            
            return product_trends
        
        except Exception as e:
            self.logger.error("Failed to calculate real-time trends", error=str(e))
            return {}
    
    def invalidate_cache(self):
        """
        Invalidate data layer cache to force fresh data fetch.
        
        Useful after sync operations to ensure recommendations use latest data.
        
        Requirements: 5.3 (update recommendation cache after sync)
        """
        self.logger.info("Invalidating data layer cache")
        
        try:
            if hasattr(self.data_layer, 'cache_manager') and self.data_layer.cache_manager:
                self.data_layer.cache_manager.invalidate_cache()
                self.logger.info("Cache invalidated successfully")
            else:
                self.logger.warning("No cache manager available to invalidate")
        
        except Exception as e:
            self.logger.error("Failed to invalidate cache", error=str(e))
    
    def switch_mode(self, mode: str):
        """
        Switch data source mode.
        
        Args:
            mode: 'api' or 'csv'
        """
        if mode not in ['api', 'csv']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'api' or 'csv'")
        
        self.mode = mode
        self.data_layer = UnifiedDataLayer(mode=mode)
        self.logger.info("Data source mode switched", new_mode=mode)
