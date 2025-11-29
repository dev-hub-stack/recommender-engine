"""
AWS Personalize Service
Replaces local ML models with AWS Personalize API calls
"""

import boto3
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)

class AWSPersonalizeService:
    """
    Service to get recommendations from AWS Personalize BATCH INFERENCE
    Reads from offline cache tables populated by batch jobs (cost-optimized approach)
    """
    
    def __init__(self, pg_conn=None):
        self.region = os.environ.get('AWS_REGION', 'us-east-1')
        self.pg_conn = pg_conn
        
        # Database connection for offline cache
        if not self.pg_conn:
            try:
                self.pg_conn = psycopg2.connect(
                    host=os.environ.get('PG_HOST', 'localhost'),
                    port=int(os.environ.get('PG_PORT', 5432)),
                    database=os.environ.get('PG_DATABASE', 'mastergroup_recommendations'),
                    user=os.environ.get('PG_USER', 'postgres'),
                    password=os.environ.get('PG_PASSWORD', '')
                )
                logger.info("Connected to PostgreSQL for offline recommendations")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.pg_conn = None
        
        self.is_configured = bool(self.pg_conn)
        
        if not self.is_configured:
            logger.warning("AWS Personalize offline cache not available. Check database connection.")
    
    def get_recommendations_for_user(
        self, 
        user_id: str, 
        num_results: int = 10,
        filter_arn: Optional[str] = None
    ) -> List[Dict]:
        """
        Get personalized recommendations for a user from OFFLINE CACHE
        Reads from batch inference results stored in PostgreSQL
        
        Args:
            user_id: Customer ID
            num_results: Number of recommendations to return
            filter_arn: Not used (batch inference doesn't support filters)
            
        Returns:
            List of recommended items with scores
        """
        if not self.is_configured or not self.pg_conn:
            logger.error("Offline cache not available")
            return []
        
        try:
            cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            
            # Read from offline cache
            cursor.execute(
                """SELECT recommendations 
                   FROM offline_user_recommendations 
                   WHERE user_id = %s
                   LIMIT 1""",
                (str(user_id),)
            )
            
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                logger.debug(f"No cached recommendations for user {user_id}")
                return []
            
            # Parse JSON recommendations
            recs = result['recommendations']
            if isinstance(recs, str):
                recs = json.loads(recs)
            
            # Limit results
            recommendations = []
            for item in recs[:num_results]:
                recommendations.append({
                    'product_id': str(item['product_id']),
                    'score': float(item.get('score', 0)),
                    'algorithm': 'aws_personalize_batch'
                })
            
            logger.debug(f"Got {len(recommendations)} cached recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error reading cached recommendations: {e}")
            return []
    
    def get_similar_items(
        self, 
        item_id: str, 
        num_results: int = 10
    ) -> List[Dict]:
        """
        Get similar items for cross-selling from OFFLINE CACHE
        Reads from batch inference results stored in PostgreSQL
        
        Args:
            item_id: Product ID
            num_results: Number of similar items
            
        Returns:
            List of similar items with scores
        """
        if not self.is_configured or not self.pg_conn:
            logger.error("Offline cache not available")
            return []
        
        try:
            cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            
            # Read from offline similar items cache
            cursor.execute(
                """SELECT similar_items 
                   FROM offline_similar_items 
                   WHERE product_id = %s
                   LIMIT 1""",
                (str(item_id),)
            )
            
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                logger.debug(f"No cached similar items for product {item_id}")
                return []
            
            # Parse JSON similar items
            items = result['similar_items']
            if isinstance(items, str):
                items = json.loads(items)
            
            # Limit results
            similar_items = []
            for item in items[:num_results]:
                similar_items.append({
                    'product_id': str(item['product_id']),
                    'score': float(item.get('score', 0)),
                    'algorithm': 'aws_similar_items_batch'
                })
            
            logger.debug(f"Got {len(similar_items)} cached similar items for {item_id}")
            return similar_items
            
        except Exception as e:
            logger.error(f"Error reading cached similar items: {e}")
            return []
    
    def record_event(
        self,
        user_id: str,
        item_id: str,
        event_type: str = 'purchase',
        event_value: float = 1.0
    ) -> bool:
        """
        Record a real-time event for model updates
        
        Args:
            user_id: Customer ID
            item_id: Product ID
            event_type: Type of event (purchase, view, etc.)
            event_value: Value of the event
            
        Returns:
            Success status
        """
        tracking_id = os.environ.get('PERSONALIZE_TRACKING_ID', '')
        
        if not tracking_id:
            logger.warning("Event tracking not configured")
            return False
        
        try:
            personalize_events = boto3.client(
                'personalize-events',
                region_name=self.region
            )
            
            import time
            
            personalize_events.put_events(
                trackingId=tracking_id,
                userId=str(user_id),
                sessionId=f"session-{user_id}-{int(time.time())}",
                eventList=[{
                    'eventType': event_type,
                    'eventValue': event_value,
                    'itemId': str(item_id),
                    'sentAt': int(time.time())
                }]
            )
            
            logger.info(f"Recorded event: {event_type} for user {user_id}, item {item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording event: {e}")
            return False
    
    def get_personalized_ranking(
        self,
        user_id: str,
        item_ids: List[str]
    ) -> List[Dict]:
        """
        Re-rank a list of items for a specific user
        
        Args:
            user_id: Customer ID
            item_ids: List of product IDs to rank
            
        Returns:
            Ranked list of items
        """
        ranking_campaign_arn = os.environ.get('PERSONALIZE_RANKING_CAMPAIGN_ARN', '')
        
        if not ranking_campaign_arn:
            logger.warning("Ranking campaign not configured")
            return [{'product_id': id, 'rank': i+1} for i, id in enumerate(item_ids)]
        
        try:
            response = self.personalize_runtime.get_personalized_ranking(
                campaignArn=ranking_campaign_arn,
                userId=str(user_id),
                inputList=[str(id) for id in item_ids]
            )
            
            ranked_items = []
            for i, item in enumerate(response.get('personalizedRanking', [])):
                ranked_items.append({
                    'product_id': item['itemId'],
                    'rank': i + 1,
                    'score': float(item.get('score', 0)),
                    'algorithm': 'aws_personalize_ranking'
                })
            
            return ranked_items
            
        except Exception as e:
            logger.error(f"Error getting personalized ranking: {e}")
            return [{'product_id': id, 'rank': i+1} for i, id in enumerate(item_ids)]


# Singleton instance
_personalize_service = None

def get_personalize_service() -> AWSPersonalizeService:
    """Get or create Personalize service instance"""
    global _personalize_service
    if _personalize_service is None:
        _personalize_service = AWSPersonalizeService()
    return _personalize_service
