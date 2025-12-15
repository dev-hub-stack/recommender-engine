"""
Data Ingestion Service
Orchestrates fetching data from APIs and storing in database
"""
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from .api_client import POSAPIClient, OEAPIClient
from .data_transformer import DataTransformer
from ..database.models import POSOrder, OEOrder, SyncLog
from ..database.connection import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionService:
    """Service for ingesting order data from APIs to database"""
    
    def __init__(self, 
                 pos_api_url: str,
                 oe_api_url: str,
                 auth_token: str,
                 timeout: int = 120):
        """
        Initialize Ingestion Service
        
        Args:
            pos_api_url: POS API endpoint URL
            oe_api_url: OE API endpoint URL
            auth_token: Authentication token for APIs
            timeout: API request timeout in seconds
        """
        self.pos_client = POSAPIClient(pos_api_url, auth_token, timeout=timeout)
        self.oe_client = OEAPIClient(oe_api_url, auth_token, timeout=timeout)
        self.transformer = DataTransformer()
    
    def ingest_pos_orders(self,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         max_pages: Optional[int] = None) -> Dict:
        """
        Ingest POS orders from API to database
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            max_pages: Maximum pages to fetch
            
        Returns:
            Dictionary with statistics
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING POS ORDERS INGESTION")
        logger.info("="*60)
        
        sync_start = datetime.now()
        stats = {
            'fetched': 0,
            'inserted': 0,
            'updated': 0,
            'failed': 0,
            'skipped': 0
        }
        
        try:
            # Fetch orders from API
            orders = self.pos_client.fetch_all_orders(
                start_date=start_date,
                end_date=end_date,
                max_pages=max_pages
            )
            
            stats['fetched'] = len(orders)
            
            if not orders:
                logger.info("No POS orders to ingest")
                self._log_sync('pos', sync_start, datetime.now(), stats, 'success')
                return stats
            
            # Process and store orders
            with get_db() as db:
                for order_data in orders:
                    try:
                        # Transform data
                        transformed = self.transformer.transform_pos_order(order_data)
                        
                        # Validate
                        if not self.transformer.validate_order_data(transformed, 'pos'):
                            stats['skipped'] += 1
                            continue
                        
                        # Upsert (insert or update)
                        result = self._upsert_pos_order(db, transformed)
                        
                        if result == 'inserted':
                            stats['inserted'] += 1
                        elif result == 'updated':
                            stats['updated'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process POS order {order_data.get('id')}: {str(e)}")
                        stats['failed'] += 1
                
                db.commit()
            
            # Log sync
            sync_end = datetime.now()
            self._log_sync('pos', sync_start, sync_end, stats, 'success')
            
            logger.info("\n" + "="*60)
            logger.info("POS ORDERS INGESTION COMPLETE")
            logger.info(f"  Fetched: {stats['fetched']}")
            logger.info(f"  Inserted: {stats['inserted']}")
            logger.info(f"  Updated: {stats['updated']}")
            logger.info(f"  Failed: {stats['failed']}")
            logger.info(f"  Skipped: {stats['skipped']}")
            logger.info("="*60)
            
            return stats
            
        except Exception as e:
            logger.error(f"POS ingestion failed: {str(e)}")
            sync_end = datetime.now()
            self._log_sync('pos', sync_start, sync_end, stats, 'failed', str(e))
            raise
    
    def ingest_oe_orders(self,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        max_pages: Optional[int] = None) -> Dict:
        """
        Ingest OE orders from API to database
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            max_pages: Maximum pages to fetch
            
        Returns:
            Dictionary with statistics
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING OE ORDERS INGESTION")
        logger.info("="*60)
        
        sync_start = datetime.now()
        stats = {
            'fetched': 0,
            'inserted': 0,
            'updated': 0,
            'failed': 0,
            'skipped': 0
        }
        
        try:
            # Fetch orders from API
            orders = self.oe_client.fetch_all_orders(
                start_date=start_date,
                end_date=end_date,
                max_pages=max_pages
            )
            
            stats['fetched'] = len(orders)
            
            if not orders:
                logger.info("No OE orders to ingest")
                self._log_sync('oe', sync_start, datetime.now(), stats, 'success')
                return stats
            
            # Process and store orders
            with get_db() as db:
                for order_data in orders:
                    try:
                        # Transform data
                        transformed = self.transformer.transform_oe_order(order_data)
                        
                        # Validate
                        if not self.transformer.validate_order_data(transformed, 'oe'):
                            stats['skipped'] += 1
                            continue
                        
                        # Upsert (insert or update)
                        result = self._upsert_oe_order(db, transformed)
                        
                        if result == 'inserted':
                            stats['inserted'] += 1
                        elif result == 'updated':
                            stats['updated'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process OE order {order_data.get('id')}: {str(e)}")
                        stats['failed'] += 1
                
                db.commit()
            
            # Log sync
            sync_end = datetime.now()
            self._log_sync('oe', sync_start, sync_end, stats, 'success')
            
            logger.info("\n" + "="*60)
            logger.info("OE ORDERS INGESTION COMPLETE")
            logger.info(f"  Fetched: {stats['fetched']}")
            logger.info(f"  Inserted: {stats['inserted']}")
            logger.info(f"  Updated: {stats['updated']}")
            logger.info(f"  Failed: {stats['failed']}")
            logger.info(f"  Skipped: {stats['skipped']}")
            logger.info("="*60)
            
            return stats
            
        except Exception as e:
            logger.error(f"OE ingestion failed: {str(e)}")
            sync_end = datetime.now()
            self._log_sync('oe', sync_start, sync_end, stats, 'failed', str(e))
            raise
    
    def ingest_all_orders(self,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         max_pages: Optional[int] = None) -> Dict:
        """
        Ingest both POS and OE orders
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            max_pages: Maximum pages to fetch per source
            
        Returns:
            Dictionary with combined statistics
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING COMPLETE DATA INGESTION (POS + OE)")
        logger.info("="*70)
        
        # Ingest POS orders
        pos_stats = self.ingest_pos_orders(start_date, end_date, max_pages)
        
        # Ingest OE orders
        oe_stats = self.ingest_oe_orders(start_date, end_date, max_pages)
        
        # Combined stats
        combined_stats = {
            'pos': pos_stats,
            'oe': oe_stats,
            'total_fetched': pos_stats['fetched'] + oe_stats['fetched'],
            'total_inserted': pos_stats['inserted'] + oe_stats['inserted'],
            'total_updated': pos_stats['updated'] + oe_stats['updated'],
            'total_failed': pos_stats['failed'] + oe_stats['failed'],
            'total_skipped': pos_stats['skipped'] + oe_stats['skipped']
        }
        
        logger.info("\n" + "="*70)
        logger.info("COMPLETE INGESTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total Fetched: {combined_stats['total_fetched']}")
        logger.info(f"Total Inserted: {combined_stats['total_inserted']}")
        logger.info(f"Total Updated: {combined_stats['total_updated']}")
        logger.info(f"Total Failed: {combined_stats['total_failed']}")
        logger.info(f"Total Skipped: {combined_stats['total_skipped']}")
        logger.info("="*70)
        
        return combined_stats
    
    def _upsert_pos_order(self, db: Session, data: Dict) -> str:
        """
        Insert or update POS order
        
        Returns:
            'inserted' or 'updated'
        """
        # Check if exists
        existing = db.query(POSOrder).filter(POSOrder.id == data['id']).first()
        
        if existing:
            # Update
            for key, value in data.items():
                setattr(existing, key, value)
            return 'updated'
        else:
            # Insert
            order = POSOrder(**data)
            db.add(order)
            return 'inserted'
    
    def _upsert_oe_order(self, db: Session, data: Dict) -> str:
        """
        Insert or update OE order
        
        Returns:
            'inserted' or 'updated'
        """
        # Check if exists
        existing = db.query(OEOrder).filter(OEOrder.id == data['id']).first()
        
        if existing:
            # Update
            for key, value in data.items():
                setattr(existing, key, value)
            return 'updated'
        else:
            # Insert
            order = OEOrder(**data)
            db.add(order)
            return 'inserted'
    
    def _log_sync(self, 
                  sync_type: str,
                  start_time: datetime,
                  end_time: datetime,
                  stats: Dict,
                  status: str,
                  error_message: Optional[str] = None):
        """Log sync operation to database"""
        try:
            with get_db() as db:
                duration = (end_time - start_time).total_seconds()
                
                sync_log = SyncLog(
                    sync_type=sync_type,
                    sync_start_date=start_time,
                    sync_end_date=end_time,
                    records_fetched=stats.get('fetched', 0),
                    records_inserted=stats.get('inserted', 0),
                    records_updated=stats.get('updated', 0),
                    records_failed=stats.get('failed', 0),
                    status=status,
                    error_message=error_message,
                    duration_seconds=duration,
                    started_at=start_time,
                    completed_at=end_time
                )
                
                db.add(sync_log)
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to log sync: {str(e)}")
    
    def test_connections(self) -> Dict[str, bool]:
        """
        Test API connections
        
        Returns:
            Dictionary with connection status for each API
        """
        logger.info("Testing API connections...")
        
        results = {
            'pos': self.pos_client.test_connection(),
            'oe': self.oe_client.test_connection()
        }
        
        if all(results.values()):
            logger.info("✅ All API connections successful")
        else:
            logger.warning("⚠️  Some API connections failed")
        
        return results
    
    def close(self):
        """Close API clients"""
        self.pos_client.close()
        self.oe_client.close()
