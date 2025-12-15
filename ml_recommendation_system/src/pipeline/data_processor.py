"""
Data Processor Module
Handles data cleaning, transformation, and feature engineering
"""
import pandas as pd
import ast
import logging
import re
from typing import List, Dict, Any
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean order data for ML training"""
    
    def __init__(self, time_window_days: int = 730):
        """
        Initialize DataProcessor
        
        Args:
            time_window_days: Number of days of historical data to use (default: 2 years)
        """
        self.time_window_days = time_window_days
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for location ID creation
        
        Args:
            text: Input text
            
        Returns:
            Normalized text (uppercase, no special chars)
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and uppercase
        text = str(text).upper().strip()
        
        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^A-Z0-9\s]', '', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def create_location_id(self, city: str, state: str, country: str) -> str:
        """
        Create location ID using city-first approach
        
        Strategy:
        1. If city exists → use city
        2. Else if state exists → use state
        3. Else if country exists → use country
        4. Else → use "UNKNOWN"
        
        Args:
            city: Customer city
            state: Customer state
            country: Customer country
            
        Returns:
            Location ID string
        """
        # Normalize inputs
        city = self.normalize_text(city)
        state = self.normalize_text(state)
        country = self.normalize_text(country)
        
        # City-first approach
        if city:
            return city
        elif state:
            return state
        elif country:
            return country
        else:
            return "UNKNOWN"
    
    def parse_product_items(self, has_items_str: Any) -> List[Dict[str, Any]]:
        """
        Parse product information from has_items JSON string
        
        Args:
            has_items_str: JSON string or dict containing product info
            
        Returns:
            List of product dictionaries
        """
        if pd.isna(has_items_str) or has_items_str == '':
            return []
        
        try:
            # Convert string to dict if needed
            if isinstance(has_items_str, str):
                # Try JSON first (more reliable for database data)
                import json
                try:
                    items = json.loads(has_items_str)
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for Python literals
                    items = ast.literal_eval(has_items_str)
            else:
                items = has_items_str
            
            # Handle single item or list of items
            if isinstance(items, dict):
                items = [items]
            elif not isinstance(items, list):
                return []
            
            # Extract relevant product info
            products = []
            for item in items:
                # Skip if item is not a dict
                if not isinstance(item, dict):
                    continue
                    
                # Use SKU as product_id, fallback to normalized title if SKU doesn't exist
                # (POS orders don't have SKU, OE orders do)
                # Handle None values by converting to empty string first
                sku = str(item.get('sku') or '').strip()
                title = str(item.get('title') or '').strip()
                
                # Prefer SKU, but create SKU-like ID from title if SKU is missing
                if sku:
                    product_id = sku
                elif title:
                    # Normalize title to SKU format: "JET FOAM 78-72-6" → "JET-FOAM-78-72-6"
                    product_id = self.normalize_text(title).replace(' ', '-')
                else:
                    product_id = ''
                
                product = {
                    'product_id': product_id,  # SKU or normalized title
                    'product_name': title,  # Keep original title for display
                    'quantity': item.get('quantity', 1),
                    'price': item.get('price', 0)
                }
                if product['product_id']:  # Only add if we have some identifier
                    products.append(product)
            
            return products
        except Exception as e:
            logger.warning(f"Failed to parse has_items: {e}")
            return []
    
    def apply_time_filter(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Filter orders to recent time window
        
        Args:
            orders: DataFrame with order_date column
            
        Returns:
            Filtered DataFrame
        """
        # Convert to datetime
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        
        if self.time_window_days is None:
            logger.info("No time filter applied - using ALL available data")
            logger.info(f"Total orders: {len(orders)}")
            logger.info(f"Date range: {orders['order_date'].min()} to {orders['order_date'].max()}")
            return orders.copy()
        
        logger.info(f"Applying time filter: last {self.time_window_days} days")
        
        # Calculate cutoff date
        max_date = orders['order_date'].max()
        cutoff_date = max_date - timedelta(days=self.time_window_days)
        
        # Filter
        filtered_orders = orders[orders['order_date'] >= cutoff_date].copy()
        
        logger.info(f"Orders before filter: {len(orders)}")
        logger.info(f"Orders after filter: {len(filtered_orders)}")
        logger.info(f"Date range: {filtered_orders['order_date'].min()} to {filtered_orders['order_date'].max()}")
        
        return filtered_orders
    
    def process(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw orders data
        
        Steps:
        1. Apply time filter
        2. Parse product information
        3. Create customer ID
        4. Explode products (one row per product)
        5. Clean and validate data
        
        Args:
            orders: Raw orders DataFrame
            
        Returns:
            Processed DataFrame ready for ML training
        """
        logger.info("Starting data processing...")
        
        # Step 1: Apply time filter
        orders = self.apply_time_filter(orders)
        
        # Step 2: Create location IDs
        logger.info("\n[STEP 2.1] Creating location IDs...")
        orders['location_id'] = orders.apply(
            lambda row: self.create_location_id(
                row.get('customer_city', ''),
                row.get('customer_state', ''),
                row.get('customer_country', '')
            ),
            axis=1
        )
        
        # Log location ID distribution
        location_counts = orders['location_id'].value_counts()
        logger.info(f"  Total unique locations: {len(location_counts)}")
        logger.info(f"  Top 5 locations:")
        for loc, count in location_counts.head(5).items():
            logger.info(f"    {loc}: {count} orders")
        
        # Step 3: Parse product information
        logger.info("\n[STEP 2.2] Parsing product information from has_items...")
        orders['products'] = orders['has_items'].apply(self.parse_product_items)
        
        # Step 4: Explode products (one row per product)
        logger.info("\n[STEP 2.3] Exploding products...")
        orders_exploded = orders.explode('products')
        
        # Extract product details
        orders_exploded['product_id'] = orders_exploded['products'].apply(
            lambda x: x['product_id'] if isinstance(x, dict) else None
        )
        orders_exploded['product_name'] = orders_exploded['products'].apply(
            lambda x: x['product_name'] if isinstance(x, dict) else None
        )
        orders_exploded['quantity'] = orders_exploded['products'].apply(
            lambda x: x['quantity'] if isinstance(x, dict) else 0
        )
        orders_exploded['price'] = orders_exploded['products'].apply(
            lambda x: x['price'] if isinstance(x, dict) else 0
        )
        
        # Convert price to numeric (handle string values)
        orders_exploded['price'] = pd.to_numeric(orders_exploded['price'], errors='coerce').fillna(0)
        
        # Step 5: Clean and validate
        logger.info("\n[STEP 2.4] Cleaning and validating data...")
        
        # Select final columns
        processed = orders_exploded[[
            'location_id', 
            'product_id', 
            'product_name',
            'quantity', 
            'price',
            'order_date',
            'id',
            'source'
        ]].copy()
        
        # Remove rows with missing critical data
        processed = processed.dropna(subset=['location_id', 'product_id'])
        
        # Remove duplicates
        processed = processed.drop_duplicates()
        
        # Ensure quantity is positive
        processed = processed[processed['quantity'] > 0]
        
        logger.info(f"Processing complete: {len(processed)} location-product interactions")
        logger.info(f"Unique locations: {processed['location_id'].nunique()}")
        logger.info(f"Unique products: {processed['product_id'].nunique()}")
        
        return processed
    
    def aggregate_interactions(self, processed: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple purchases of same product by same location
        
        Args:
            processed: Processed DataFrame
            
        Returns:
            Aggregated DataFrame
        """
        logger.info("Aggregating location-product interactions...")
        
        aggregated = processed.groupby(['location_id', 'product_id', 'product_name']).agg({
            'quantity': 'sum',
            'price': 'mean',
            'order_date': 'max',
            'source': 'first'
        }).reset_index()
        
        logger.info(f"After aggregation: {len(aggregated)} unique location-product pairs")
        
        return aggregated
