"""
Master Group API Sync Service
Handles incremental synchronization of orders from Master Group APIs
"""
import requests
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime, timedelta
import time
import logging
import json
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.master_group_api import (
    MASTER_GROUP_CONFIG, SYNC_CONFIG, PG_CONFIG,
    get_api_url, get_auth_headers
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyncService:
    """Service for synchronizing orders from Master Group APIs"""
    
    def __init__(self):
        self.pg_conn = None
        self.sync_status = {
            'is_running': False,
            'last_sync': None,
            'last_success': None,
            'last_error': None,
            'total_synced': 0,
            'errors_count': 0
        }
    
    def connect_db(self):
        """Connect to PostgreSQL database"""
        if not self.pg_conn or self.pg_conn.closed:
            self.pg_conn = psycopg2.connect(**PG_CONFIG)
            logger.info("Connected to PostgreSQL")
    
    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the last successful sync timestamp"""
        try:
            self.connect_db()
            cursor = self.pg_conn.cursor()
            cursor.execute("""
                SELECT last_sync_timestamp 
                FROM sync_metadata 
                WHERE sync_type IN ('pos', 'oe', 'full')
                ORDER BY last_sync_timestamp DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Error getting last sync time: {e}")
            return None
    
    def update_sync_metadata(self, orders_synced: int, errors_count: int, status: str = 'success'):
        """Update sync metadata in database"""
        try:
            self.connect_db()
            cursor = self.pg_conn.cursor()
            
            cursor.execute("""
                INSERT INTO sync_metadata 
                (sync_type, last_sync_timestamp, sync_status, orders_synced, error_message)
                VALUES (%s, %s, %s, %s, %s)
            """, ('full', datetime.now(), status, orders_synced, None if errors_count == 0 else f'{errors_count} errors'))
            
            self.pg_conn.commit()
            cursor.close()
            logger.info(f"Sync metadata updated: {orders_synced} orders, {errors_count} errors")
        except Exception as e:
            logger.error(f"Error updating sync metadata: {e}")
            if self.pg_conn:
                self.pg_conn.rollback()
    
    def fetch_pos_orders(self, start_date: str, end_date: str, limit: Optional[int] = None) -> List[Dict]:
        """Fetch POS orders from Master Group API"""
        try:
            url = get_api_url('pos_orders')
            headers = get_auth_headers()
            params = {
                'start_date': start_date,
                'end_date': end_date
            }
            if limit:
                params['limit'] = limit
            
            logger.info(f"Fetching POS orders from {start_date} to {end_date}")
            
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=MASTER_GROUP_CONFIG['timeout']
            )
            
            if response.status_code == 200:
                data = response.json()
                orders = data if isinstance(data, list) else data.get('data', [])
                logger.info(f"Fetched {len(orders)} POS orders")
                return orders
            else:
                logger.error(f"POS API error: {response.status_code} - {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching POS orders: {e}")
            return []
    
    def fetch_oe_orders(self, days: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """Fetch OE orders from Master Group API"""
        try:
            url = get_api_url('oe_orders')
            headers = get_auth_headers()
            params = {'days': days}
            if limit:
                params['limit'] = limit
            
            logger.info(f"Fetching OE orders for last {days} days")
            
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=600  # 10 minutes for OE API (it's slower)
            )
            
            if response.status_code == 200:
                data = response.json()
                orders = data if isinstance(data, list) else data.get('data', [])
                logger.info(f"Fetched {len(orders)} OE orders")
                return orders
            else:
                logger.error(f"OE API error: {response.status_code} - {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching OE orders: {e}")
            return []
    
    def transform_order_data(self, order: Dict, source: str) -> Dict:
        """Transform API order data to database format"""
        try:
            # Extract customer ID (phone + name combination)
            customer_phone = order.get('customer_phone', order.get('phone', ''))
            customer_name = order.get('customer_name', order.get('name', ''))
            unified_customer_id = f"{customer_phone}_{customer_name}".strip('_')
            
            # Extract items
            items = order.get('has_items', order.get('items', []))
            if not isinstance(items, list):
                items = []
            
            # Create items JSON
            items_json = []
            for item in items:
                items_json.append({
                    'product_id': str(item.get('id', item.get('product_id', ''))),  # API uses 'id' for product
                    'product_name': item.get('title', item.get('product_name', item.get('name', ''))),  # API uses 'title'
                    'quantity': int(item.get('quantity', 1)),
                    'price': float(item.get('price', 0)),
                    'unit_price': float(item.get('base_price', item.get('unit_price', item.get('price', 0))))
                })
            
            return {
                'id': str(order.get('id', order.get('order_id', ''))),
                'order_type': source,
                'order_date': order.get('order_date', order.get('created_at', datetime.now())),
                'unified_customer_id': unified_customer_id,
                'customer_name': customer_name,
                'customer_phone': customer_phone,
                'customer_city': order.get('customer_city', order.get('city', '')),
                'customer_address': order.get('customer_address', order.get('address', '')),
                'customer_email': order.get('customer_email', order.get('email', '')),
                'total_price': float(order.get('total_price', order.get('total', 0))),
                'payment_mode': order.get('payment_mode', order.get('payment', 'Cash')),
                'brand_name': order.get('brand_name', order.get('brand', '')),
                'order_status': order.get('order_status', order.get('status', 'completed')),
                'order_name': order.get('order_name', order.get('name', '')),
                'items_json': json.dumps(items_json),
                'items': items_json
            }
        except Exception as e:
            logger.error(f"Error transforming order: {e}")
            return None
    
    def insert_orders(self, orders: List[Dict]) -> Tuple[int, int]:
        """Insert orders into database"""
        orders_inserted = 0
        items_inserted = 0
        
        try:
            self.connect_db()
            cursor = self.pg_conn.cursor()
            
            for order in orders:
                try:
                    # Use SAVEPOINT to handle individual order errors without aborting entire transaction
                    cursor.execute("SAVEPOINT order_insert")
                    
                    # Insert order
                    cursor.execute("""
                        INSERT INTO orders (
                            id, order_type, order_date, unified_customer_id,
                            customer_name, customer_phone, customer_city, customer_address,
                            customer_email, total_price, payment_mode, brand_name,
                            order_status, order_name, items_json
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            updated_at = CURRENT_TIMESTAMP,
                            synced_at = CURRENT_TIMESTAMP
                    """, (
                        order['id'], order['order_type'], order['order_date'],
                        order['unified_customer_id'], order['customer_name'],
                        order['customer_phone'], order['customer_city'],
                        order['customer_address'], order['customer_email'],
                        order['total_price'], order['payment_mode'],
                        order['brand_name'], order['order_status'],
                        order['order_name'], order['items_json']
                    ))
                    
                    if cursor.rowcount > 0:
                        orders_inserted += 1
                    
                    # Insert order items
                    for item in order.get('items', []):
                        cursor.execute("""
                            INSERT INTO order_items (
                                order_id, product_id, product_name,
                                quantity, unit_price, total_price
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (
                            order['id'],
                            item['product_id'],
                            item['product_name'],
                            item['quantity'],
                            item['unit_price'],
                            item['price']
                        ))
                        
                        if cursor.rowcount > 0:
                            items_inserted += 1
                    
                    # Release savepoint if successful
                    cursor.execute("RELEASE SAVEPOINT order_insert")
                
                except Exception as e:
                    # Rollback to savepoint on error, transaction continues
                    cursor.execute("ROLLBACK TO SAVEPOINT order_insert")
                    logger.error(f"Error inserting order {order.get('id')}: {e}")
                    continue
            
            self.pg_conn.commit()
            cursor.close()
            logger.info(f"Inserted {orders_inserted} orders, {items_inserted} items")
            
        except Exception as e:
            logger.error(f"Error in batch insert: {e}")
            if self.pg_conn:
                self.pg_conn.rollback()
        
        return orders_inserted, items_inserted
    
    def rebuild_recommendation_tables(self):
        """Rebuild recommendation tables after sync"""
        try:
            self.connect_db()
            cursor = self.pg_conn.cursor()
            
            logger.info("Rebuilding recommendation tables...")
            
            # Rebuild customer purchases
            cursor.execute("SELECT rebuild_customer_purchases()")
            customer_purchases = cursor.fetchone()[0]
            logger.info(f"Rebuilt {customer_purchases} customer purchase records")
            
            # Rebuild product pairs
            cursor.execute("SELECT rebuild_product_pairs()")
            product_pairs = cursor.fetchone()[0]
            logger.info(f"Rebuilt {product_pairs} product pair records")
            
            # Rebuild product statistics
            cursor.execute("SELECT rebuild_product_statistics()")
            product_stats = cursor.fetchone()[0]
            logger.info(f"Rebuilt {product_stats} product statistics records")
            
            # Rebuild customer statistics
            cursor.execute("SELECT rebuild_customer_statistics()")
            customer_stats = cursor.fetchone()[0]
            logger.info(f"Rebuilt {customer_stats} customer statistics records")
            
            self.pg_conn.commit()
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error rebuilding recommendation tables: {e}")
            if self.pg_conn:
                self.pg_conn.rollback()
    
    def sync_incremental(self) -> Dict:
        """Perform incremental sync (only new orders since last sync)"""
        if self.sync_status['is_running']:
            return {'status': 'already_running', 'message': 'Sync is already in progress'}
        
        self.sync_status['is_running'] = True
        self.sync_status['last_sync'] = datetime.now()
        
        try:
            # Calculate time range for incremental sync
            last_sync = self.get_last_sync_time()
            lookback_minutes = SYNC_CONFIG['lookback_minutes']
            
            if last_sync:
                # Sync from last sync time minus lookback (safety margin)
                start_time = last_sync - timedelta(minutes=lookback_minutes)
            else:
                # First sync: get last 30 days
                start_time = datetime.now() - timedelta(days=30)
            
            end_time = datetime.now()
            
            start_date = start_time.strftime('%Y-%m-%d')
            end_date = end_time.strftime('%Y-%m-%d')
            
            logger.info(f"Starting incremental sync from {start_date} to {end_date}")
            
            # Fetch orders
            pos_orders = self.fetch_pos_orders(start_date, end_date, SYNC_CONFIG['batch_size'])
            oe_orders = self.fetch_oe_orders(days=1, limit=SYNC_CONFIG['batch_size'])
            
            # Transform orders
            transformed_orders = []
            for order in pos_orders:
                transformed = self.transform_order_data(order, 'POS')
                if transformed:
                    transformed_orders.append(transformed)
            
            for order in oe_orders:
                transformed = self.transform_order_data(order, 'OE')
                if transformed:
                    transformed_orders.append(transformed)
            
            # Insert orders
            orders_inserted, items_inserted = self.insert_orders(transformed_orders)
            
            # Rebuild recommendation tables
            if orders_inserted > 0:
                self.rebuild_recommendation_tables()
            
            # Update metadata
            self.update_sync_metadata(orders_inserted, 0, 'completed')
            
            self.sync_status['last_success'] = datetime.now()
            self.sync_status['total_synced'] += orders_inserted
            
            result = {
                'status': 'success',
                'start_date': start_date,
                'end_date': end_date,
                'pos_orders_fetched': len(pos_orders),
                'oe_orders_fetched': len(oe_orders),
                'orders_inserted': orders_inserted,
                'items_inserted': items_inserted,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Sync completed successfully: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.sync_status['last_error'] = str(e)
            self.sync_status['errors_count'] += 1
            self.update_sync_metadata(0, 1, 'failed')
            
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            self.sync_status['is_running'] = False
    
    def get_sync_status(self) -> Dict:
        """Get current sync status"""
        return {
            'is_running': self.sync_status['is_running'],
            'last_sync': self.sync_status['last_sync'].isoformat() if self.sync_status['last_sync'] else None,
            'last_success': self.sync_status['last_success'].isoformat() if self.sync_status['last_success'] else None,
            'last_error': self.sync_status['last_error'],
            'total_synced': self.sync_status['total_synced'],
            'errors_count': self.sync_status['errors_count'],
            'sync_interval_minutes': SYNC_CONFIG['interval_minutes'],
            'auto_sync_enabled': SYNC_CONFIG['enable_auto_sync']
        }
    
    def close(self):
        """Close database connection"""
        if self.pg_conn and not self.pg_conn.closed:
            self.pg_conn.close()
            logger.info("Database connection closed")


# Global sync service instance
_sync_service = None

def get_sync_service() -> SyncService:
    """Get or create sync service instance"""
    global _sync_service
    if _sync_service is None:
        _sync_service = SyncService()
    return _sync_service
