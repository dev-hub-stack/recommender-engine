"""
Real-Time Recommendation Updater
Handles real-time updates to recommendations based on latest order data.

Requirements: 5.3 (real-time recommendation updates)
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import structlog
from collections import defaultdict

import sys
import os

# Removed shared dependency for local execution

from data_loader import RecommendationDataLoader

logger = structlog.get_logger(__name__)


class RealTimeRecommendationUpdater:
    """
    Manages real-time updates to recommendation cache and trending products.
    
    Features:
    - Fetches latest orders for real-time recommendations
    - Updates recommendation cache after each sync
    - Tracks trending products in real-time
    - Adds data freshness to recommendation responses
    
    Requirements: 5.3
    """
    
    def __init__(self, data_loader: RecommendationDataLoader):
        """
        Initialize the real-time updater.
        
        Args:
            data_loader: RecommendationDataLoader instance
        """
        self.data_loader = data_loader
        self.logger = logger.bind(component="RealTimeRecommendationUpdater")
        
        # Cache for real-time data
        self.trending_products: Dict[str, Dict] = {}
        self.recent_purchases: Dict[str, List[str]] = defaultdict(list)  # customer_id -> product_ids
        self.last_update_time: Optional[datetime] = None
        
        self.logger.info("Real-time recommendation updater initialized")
    
    def update_from_latest_orders(self, since_hours: int = 1) -> Dict[str, any]:
        """
        Update recommendations based on latest orders.
        
        Fetches recent orders and updates trending products and customer preferences.
        
        Args:
            since_hours: Number of hours to look back
        
        Returns:
            Update metrics
        
        Requirements: 5.3 (fetch latest orders for real-time recommendations)
        """
        start_time = datetime.now()
        self.logger.info("Updating recommendations from latest orders", since_hours=since_hours)
        
        try:
            # Load latest orders
            recent_orders = self.data_loader.load_latest_orders(since_hours=since_hours)
            
            if recent_orders.empty:
                self.logger.warning("No recent orders found")
                return {
                    'status': 'no_data',
                    'n_orders': 0,
                    'n_trending_products': 0,
                    'n_customers_updated': 0
                }
            
            # Update trending products
            n_trending = self._update_trending_products(recent_orders)
            
            # Update recent purchases
            n_customers = self._update_recent_purchases(recent_orders)
            
            self.last_update_time = datetime.now()
            elapsed = (self.last_update_time - start_time).total_seconds()
            
            metrics = {
                'status': 'success',
                'n_orders': len(recent_orders),
                'n_trending_products': n_trending,
                'n_customers_updated': n_customers,
                'update_time': self.last_update_time.isoformat(),
                'elapsed_seconds': elapsed
            }
            
            self.logger.info("Real-time update completed", **metrics)
            return metrics
        
        except Exception as e:
            self.logger.error("Failed to update from latest orders", error=str(e))
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _update_trending_products(self, orders_df: pd.DataFrame) -> int:
        """
        Update trending products from recent orders.
        
        Args:
            orders_df: Recent orders DataFrame
        
        Returns:
            Number of trending products updated
        """
        try:
            if 'product_name' not in orders_df.columns:
                return 0
            
            # Calculate product metrics
            product_stats = orders_df.groupby('product_name').agg({
                'quantity': 'sum' if 'quantity' in orders_df.columns else 'count',
                'total_price': 'sum' if 'total_price' in orders_df.columns else 'count',
                'order_date': 'count'
            }).reset_index()
            
            product_stats.columns = ['product_id', 'quantity', 'revenue', 'order_count']
            
            # Update trending products cache
            for _, row in product_stats.iterrows():
                product_id = row['product_id']
                self.trending_products[product_id] = {
                    'quantity': int(row['quantity']),
                    'revenue': float(row['revenue']),
                    'order_count': int(row['order_count']),
                    'trending_score': float(row['quantity'] * 0.5 + row['order_count'] * 0.5),
                    'last_updated': datetime.now().isoformat()
                }
            
            return len(self.trending_products)
        
        except Exception as e:
            self.logger.error("Failed to update trending products", error=str(e))
            return 0
    
    def _update_recent_purchases(self, orders_df: pd.DataFrame) -> int:
        """
        Update recent purchases by customer.
        
        Args:
            orders_df: Recent orders DataFrame
        
        Returns:
            Number of customers updated
        """
        try:
            if 'unified_customer_id' not in orders_df.columns or 'product_name' not in orders_df.columns:
                return 0
            
            # Group by customer
            for customer_id in orders_df['unified_customer_id'].unique():
                customer_orders = orders_df[orders_df['unified_customer_id'] == customer_id]
                products = customer_orders['product_name'].tolist()
                
                # Keep only recent purchases (last 100)
                self.recent_purchases[customer_id] = (
                    self.recent_purchases[customer_id] + products
                )[-100:]
            
            return len(self.recent_purchases)
        
        except Exception as e:
            self.logger.error("Failed to update recent purchases", error=str(e))
            return 0
    
    def get_trending_products(self, top_n: int = 10) -> List[Dict]:
        """
        Get current trending products.
        
        Args:
            top_n: Number of top trending products to return
        
        Returns:
            List of trending product dictionaries
        
        Requirements: 5.3 (real-time trending products)
        """
        if not self.trending_products:
            return []
        
        # Sort by trending score
        sorted_products = sorted(
            self.trending_products.items(),
            key=lambda x: x[1]['trending_score'],
            reverse=True
        )
        
        return [
            {
                'product_id': product_id,
                **metrics
            }
            for product_id, metrics in sorted_products[:top_n]
        ]
    
    def get_customer_recent_purchases(self, customer_id: str, limit: int = 10) -> List[str]:
        """
        Get customer's recent purchases.
        
        Args:
            customer_id: Customer identifier
            limit: Maximum number of products to return
        
        Returns:
            List of recently purchased product IDs
        """
        if customer_id not in self.recent_purchases:
            return []
        
        return self.recent_purchases[customer_id][-limit:]
    
    def add_freshness_to_response(
        self,
        recommendations: List[Dict]
    ) -> List[Dict]:
        """
        Add data freshness information to recommendation responses.
        
        Args:
            recommendations: List of recommendation dictionaries
        
        Returns:
            Recommendations with freshness information added
        
        Requirements: 5.3 (add data freshness to recommendation responses)
        """
        freshness_info = self.data_loader.get_data_freshness()
        
        # Add freshness to each recommendation
        for rec in recommendations:
            rec['data_freshness'] = {
                'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
                'data_age_hours': freshness_info.get('age_hours'),
                'data_timestamp': freshness_info.get('data_timestamp'),
                'mode': freshness_info.get('mode')
            }
        
        return recommendations
    
    def on_sync_complete(self):
        """
        Callback to be called after ETL sync completes.
        
        Invalidates cache and triggers real-time update.
        
        Requirements: 5.3 (update recommendation cache after each sync)
        """
        self.logger.info("Sync complete callback triggered")
        
        try:
            # Invalidate cache to force fresh data
            self.data_loader.invalidate_cache()
            
            # Update from latest orders
            metrics = self.update_from_latest_orders(since_hours=1)
            
            self.logger.info("Post-sync update completed", **metrics)
        
        except Exception as e:
            self.logger.error("Failed to update after sync", error=str(e))
    
    def get_update_status(self) -> Dict[str, any]:
        """
        Get current update status.
        
        Returns:
            Status dictionary with metrics
        """
        return {
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'n_trending_products': len(self.trending_products),
            'n_customers_tracked': len(self.recent_purchases),
            'data_freshness': self.data_loader.get_data_freshness()
        }
