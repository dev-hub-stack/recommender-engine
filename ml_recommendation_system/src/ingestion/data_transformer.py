"""
Data Transformer
Transforms API response data to database model format
"""
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransformer:
    """Transform API data to database model format"""
    
    @staticmethod
    def transform_pos_order(api_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform POS API response to POSOrder model format
        
        Args:
            api_data: Raw data from POS API
            
        Returns:
            Dictionary matching POSOrder model fields
        """
        try:
            # Convert has_items to JSON string if it's a dict/list
            has_items = api_data.get('has_items', [])
            if isinstance(has_items, (dict, list)):
                has_items = json.dumps(has_items)
            
            # Parse order_date
            order_date = api_data.get('order_date')
            if isinstance(order_date, str):
                try:
                    order_date = datetime.strptime(order_date, '%Y-%m-%d')
                except ValueError:
                    try:
                        order_date = datetime.strptime(order_date, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        order_date = datetime.now()
            
            # Build transformed data
            transformed = {
                'id': str(api_data.get('id')),
                'customer_phone': api_data.get('customer_phone'),
                'customer_email': api_data.get('customer_email'),
                'customer_name': api_data.get('customer_name'),
                'customer_address': api_data.get('customer_address'),
                'customer_city': api_data.get('customer_city'),
                'customer_state': api_data.get('customer_state'),
                'customer_country': api_data.get('customer_country'),
                'order_date': order_date,
                'order_source': api_data.get('order_source', 'POS System'),
                'order_status': api_data.get('order_status'),
                'order_status_id': api_data.get('order_status_id'),
                'has_items': has_items,
                'dealer_id': api_data.get('dealer_id') or api_data.get('dealer', {}).get('id'),
                'dealer_name': api_data.get('dealer_name') or api_data.get('dealer', {}).get('name'),
                'dealership_id': api_data.get('dealership_id'),
                'total_price': float(api_data.get('total_price', 0)) if api_data.get('total_price') else None,
                'discount': float(api_data.get('discount', 0)) if api_data.get('discount') else None,
                'dealer_discount': float(api_data.get('dealer_discount', 0)) if api_data.get('dealer_discount') else None,
                'payment_mode': api_data.get('payment_mode'),
                'brand_name': api_data.get('brand_name'),
                'courier_id': api_data.get('courier_id'),
                'is_split': bool(api_data.get('is_split', False))
            }
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming POS order {api_data.get('id')}: {str(e)}")
            raise
    
    @staticmethod
    def transform_oe_order(api_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform OE API response to OEOrder model format
        
        Args:
            api_data: Raw data from OE API
            
        Returns:
            Dictionary matching OEOrder model fields
        """
        try:
            # Convert has_items to JSON string if it's a dict/list
            has_items = api_data.get('has_items', [])
            if isinstance(has_items, (dict, list)):
                has_items = json.dumps(has_items)
            
            # Convert assigned_tags to JSON string
            assigned_tags = api_data.get('assigned_tags', [])
            if isinstance(assigned_tags, (dict, list)):
                assigned_tags = json.dumps(assigned_tags)
            
            # Convert order_comments to JSON string
            order_comments = api_data.get('order_comments', [])
            if isinstance(order_comments, (dict, list)):
                order_comments = json.dumps(order_comments)
            
            # Parse order_date
            order_date = api_data.get('order_date')
            if isinstance(order_date, str):
                try:
                    order_date = datetime.strptime(order_date, '%Y-%m-%d')
                except ValueError:
                    try:
                        order_date = datetime.strptime(order_date, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        order_date = datetime.now()
            
            # Build transformed data
            transformed = {
                'id': str(api_data.get('id')),
                'customer_phone': api_data.get('customer_phone'),
                'customer_email': api_data.get('customer_email'),
                'customer_name': api_data.get('customer_name'),
                'customer_address': api_data.get('customer_address'),
                'customer_city': api_data.get('customer_city'),
                'customer_state': api_data.get('customer_state'),
                'customer_country': api_data.get('customer_country'),
                'order_date': order_date,
                'order_name': api_data.get('order_name'),
                'order_status': api_data.get('order_status'),
                'order_status_id': api_data.get('order_status_id'),
                'order_comments': order_comments,
                'has_items': has_items,
                'total_price': float(api_data.get('total_price', 0)) if api_data.get('total_price') else None,
                'discount': float(api_data.get('discount', 0)) if api_data.get('discount') else None,
                'payment_mode': api_data.get('payment_mode'),
                'brand_name': api_data.get('brand_name'),
                'courier_id': api_data.get('courier_id'),
                'is_split': bool(api_data.get('is_split', False)),
                'assigned_tags': assigned_tags
            }
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming OE order {api_data.get('id')}: {str(e)}")
            raise
    
    @staticmethod
    def validate_order_data(data: Dict[str, Any], order_type: str) -> bool:
        """
        Validate that order data has required fields
        
        Args:
            data: Transformed order data
            order_type: 'pos' or 'oe'
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['id', 'order_date', 'has_items']
        
        for field in required_fields:
            if field not in data or data[field] is None:
                logger.warning(f"Missing required field '{field}' in {order_type} order {data.get('id')}")
                return False
        
        return True
