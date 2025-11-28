"""
AWS Personalize Real-Time Sync
Sends new orders/interactions to AWS Personalize as they come in from Master Group APIs
"""

import boto3
import os
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
AWS_PROFILE = os.environ.get('AWS_PROFILE', 'mastergroup')
TRACKING_ID = os.environ.get('PERSONALIZE_TRACKING_ID', '')
DATASET_GROUP_ARN = os.environ.get(
    'PERSONALIZE_DATASET_GROUP_ARN',
    'arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations'
)


class PersonalizeSyncService:
    """
    Service to sync real-time events to AWS Personalize
    
    Two sync modes:
    1. Real-time events: Send immediately as orders come in
    2. Batch sync: Periodic full data export and model retrain
    """
    
    def __init__(self):
        self.session = boto3.Session(
            profile_name=AWS_PROFILE,
            region_name=AWS_REGION
        )
        self.personalize_events = self.session.client('personalize-events')
        self.personalize = self.session.client('personalize')
        self.s3 = self.session.client('s3')
        
        self.tracking_id = TRACKING_ID
        self.s3_bucket = 'mastergroup-personalize-data'
        
        # Sync statistics
        self.stats = {
            'events_sent': 0,
            'events_failed': 0,
            'last_sync': None,
            'last_batch_sync': None
        }
        
        logger.info("PersonalizeSyncService initialized")
    
    def send_purchase_event(
        self,
        user_id: str,
        item_id: str,
        quantity: int = 1,
        price: float = 0,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Send a single purchase event to AWS Personalize
        Called after each order is inserted into PostgreSQL
        """
        if not self.tracking_id:
            logger.debug("Tracking ID not configured, skipping event")
            return False
        
        try:
            event_time = timestamp or datetime.now()
            session_id = f"session-{user_id}-{int(event_time.timestamp())}"
            
            self.personalize_events.put_events(
                trackingId=self.tracking_id,
                userId=str(user_id),
                sessionId=session_id,
                eventList=[{
                    'eventType': 'purchase',
                    'eventValue': float(quantity),
                    'itemId': str(item_id),
                    'sentAt': event_time,
                    'properties': json.dumps({
                        'price': price,
                        'quantity': quantity
                    })
                }]
            )
            
            self.stats['events_sent'] += 1
            logger.debug(f"Sent purchase event: user={user_id}, item={item_id}")
            return True
            
        except Exception as e:
            self.stats['events_failed'] += 1
            logger.error(f"Failed to send purchase event: {e}")
            return False
    
    def send_order_events(self, order: Dict) -> int:
        """
        Send all events from a single order
        Called from sync_service after inserting order
        
        Args:
            order: Order dict with 'unified_customer_id', 'order_date', 'items'
            
        Returns:
            Number of events sent successfully
        """
        events_sent = 0
        user_id = order.get('unified_customer_id')
        
        if not user_id:
            return 0
        
        order_date = order.get('order_date')
        if isinstance(order_date, str):
            try:
                order_date = datetime.fromisoformat(order_date.replace('Z', '+00:00'))
            except:
                order_date = datetime.now()
        
        for item in order.get('items', []):
            product_id = item.get('product_id')
            if product_id:
                if self.send_purchase_event(
                    user_id=user_id,
                    item_id=product_id,
                    quantity=item.get('quantity', 1),
                    price=item.get('unit_price', 0),
                    timestamp=order_date
                ):
                    events_sent += 1
        
        return events_sent
    
    def send_batch_events(self, orders: List[Dict]) -> Dict:
        """
        Send events for a batch of orders
        Called after sync_service.insert_orders()
        
        Args:
            orders: List of order dicts
            
        Returns:
            Stats dict with counts
        """
        total_events = 0
        failed_orders = 0
        
        for order in orders:
            try:
                events = self.send_order_events(order)
                total_events += events
            except Exception as e:
                failed_orders += 1
                logger.error(f"Failed to send events for order {order.get('id')}: {e}")
        
        self.stats['last_sync'] = datetime.now().isoformat()
        
        return {
            'events_sent': total_events,
            'orders_processed': len(orders),
            'failed_orders': failed_orders
        }
    
    def create_event_tracker(self) -> Optional[str]:
        """
        Create an event tracker for real-time events
        Only needs to be done once
        """
        try:
            response = self.personalize.create_event_tracker(
                name='mastergroup-event-tracker',
                datasetGroupArn=DATASET_GROUP_ARN
            )
            tracking_id = response['trackingId']
            logger.info(f"Created event tracker: {tracking_id}")
            return tracking_id
        except self.personalize.exceptions.ResourceAlreadyExistsException:
            # Get existing tracker
            response = self.personalize.list_event_trackers(
                datasetGroupArn=DATASET_GROUP_ARN
            )
            for tracker in response.get('eventTrackers', []):
                if tracker['name'] == 'mastergroup-event-tracker':
                    logger.info(f"Using existing event tracker: {tracker['eventTrackerArn']}")
                    return tracker.get('trackingId')
            return None
        except Exception as e:
            logger.error(f"Failed to create event tracker: {e}")
            return None
    
    def get_sync_stats(self) -> Dict:
        """Get current sync statistics"""
        return {
            **self.stats,
            'tracking_configured': bool(self.tracking_id)
        }


# Global instance
_personalize_sync = None

def get_personalize_sync() -> PersonalizeSyncService:
    """Get or create PersonalizeSyncService instance"""
    global _personalize_sync
    if _personalize_sync is None:
        _personalize_sync = PersonalizeSyncService()
    return _personalize_sync


def sync_orders_to_personalize(orders: List[Dict]) -> Dict:
    """
    Convenience function to sync orders to Personalize
    Call this from sync_service.py after inserting orders
    
    Usage in sync_service.py:
        from aws_personalize.personalize_sync import sync_orders_to_personalize
        
        # After inserting orders
        orders_inserted, items_inserted = self.insert_orders(orders)
        personalize_result = sync_orders_to_personalize(orders)
    """
    try:
        service = get_personalize_sync()
        return service.send_batch_events(orders)
    except Exception as e:
        logger.error(f"Personalize sync failed: {e}")
        return {'error': str(e), 'events_sent': 0}
