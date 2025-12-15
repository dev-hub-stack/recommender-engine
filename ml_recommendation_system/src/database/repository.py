"""
Data Access Layer (Repository Pattern)
"""
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import logging

from .models import POSOrder, OEOrder, SyncLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderRepository:
    """Repository for order data access"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== POS Orders ====================
    
    def bulk_insert_pos_orders(self, orders_df: pd.DataFrame) -> int:
        """
        Bulk insert POS orders (upsert - insert or update)
        
        Args:
            orders_df: DataFrame with POS orders
            
        Returns:
            Number of records inserted/updated
        """
        logger.info(f"Bulk inserting {len(orders_df)} POS orders...")
        
        records = orders_df.to_dict('records')
        inserted = 0
        
        for record in records:
            try:
                # Check if exists
                existing = self.db.query(POSOrder).filter(POSOrder.id == record['id']).first()
                
                if existing:
                    # Update
                    for key, value in record.items():
                        setattr(existing, key, value)
                else:
                    # Insert
                    order = POSOrder(**record)
                    self.db.add(order)
                
                inserted += 1
                
                # Commit in batches of 1000
                if inserted % 1000 == 0:
                    self.db.commit()
                    logger.info(f"  Committed {inserted} records...")
                    
            except Exception as e:
                logger.error(f"Error inserting POS order {record.get('id')}: {e}")
                continue
        
        self.db.commit()
        logger.info(f"✅ Inserted/updated {inserted} POS orders")
        return inserted
    
    def get_pos_orders(self, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get POS orders as DataFrame
        
        Args:
            start_date: Filter orders from this date
            end_date: Filter orders until this date
            
        Returns:
            DataFrame with POS orders
        """
        query = self.db.query(POSOrder)
        
        if start_date:
            query = query.filter(POSOrder.order_date >= start_date)
        if end_date:
            query = query.filter(POSOrder.order_date <= end_date)
        
        query = query.order_by(POSOrder.order_date)
        
        # Convert to DataFrame
        orders = query.all()
        if not orders:
            return pd.DataFrame()
        
        data = [{
            'id': o.id,
            'customer_phone': o.customer_phone,
            'customer_email': o.customer_email,
            'customer_name': o.customer_name,
            'customer_address': o.customer_address,
            'customer_city': o.customer_city,
            'customer_state': o.customer_state,
            'customer_country': o.customer_country,
            'order_date': o.order_date,
            'has_items': o.has_items,
            'total_price': o.total_price,
            'discount': o.discount,
            'payment_mode': o.payment_mode,
            'order_status': o.order_status,
            'brand_name': o.brand_name
        } for o in orders]
        
        return pd.DataFrame(data)
    
    def get_pos_order_count(self) -> int:
        """Get total count of POS orders"""
        return self.db.query(func.count(POSOrder.id)).scalar()
    
    def get_pos_date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get min and max order dates for POS orders"""
        result = self.db.query(
            func.min(POSOrder.order_date),
            func.max(POSOrder.order_date)
        ).first()
        return result if result else (None, None)
    
    # ==================== OE Orders ====================
    
    def bulk_insert_oe_orders(self, orders_df: pd.DataFrame) -> int:
        """
        Bulk insert OE orders (upsert - insert or update)
        
        Args:
            orders_df: DataFrame with OE orders
            
        Returns:
            Number of records inserted/updated
        """
        logger.info(f"Bulk inserting {len(orders_df)} OE orders...")
        
        records = orders_df.to_dict('records')
        inserted = 0
        
        for record in records:
            try:
                # Check if exists
                existing = self.db.query(OEOrder).filter(OEOrder.id == record['id']).first()
                
                if existing:
                    # Update
                    for key, value in record.items():
                        setattr(existing, key, value)
                else:
                    # Insert
                    order = OEOrder(**record)
                    self.db.add(order)
                
                inserted += 1
                
                # Commit in batches of 1000
                if inserted % 1000 == 0:
                    self.db.commit()
                    logger.info(f"  Committed {inserted} records...")
                    
            except Exception as e:
                logger.error(f"Error inserting OE order {record.get('id')}: {e}")
                continue
        
        self.db.commit()
        logger.info(f"✅ Inserted/updated {inserted} OE orders")
        return inserted
    
    def get_oe_orders(self,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get OE orders as DataFrame
        
        Args:
            start_date: Filter orders from this date
            end_date: Filter orders until this date
            
        Returns:
            DataFrame with OE orders
        """
        query = self.db.query(OEOrder)
        
        if start_date:
            query = query.filter(OEOrder.order_date >= start_date)
        if end_date:
            query = query.filter(OEOrder.order_date <= end_date)
        
        query = query.order_by(OEOrder.order_date)
        
        # Convert to DataFrame
        orders = query.all()
        if not orders:
            return pd.DataFrame()
        
        data = [{
            'id': o.id,
            'customer_phone': o.customer_phone,
            'customer_email': o.customer_email,
            'customer_name': o.customer_name,
            'customer_address': o.customer_address,
            'customer_city': o.customer_city,
            'customer_state': o.customer_state,
            'customer_country': o.customer_country,
            'order_date': o.order_date,
            'has_items': o.has_items,
            'total_price': o.total_price,
            'discount': o.discount,
            'payment_mode': o.payment_mode,
            'order_status': o.order_status,
            'brand_name': o.brand_name
        } for o in orders]
        
        return pd.DataFrame(data)
    
    def get_oe_order_count(self) -> int:
        """Get total count of OE orders"""
        return self.db.query(func.count(OEOrder.id)).scalar()
    
    def get_oe_date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get min and max order dates for OE orders"""
        result = self.db.query(
            func.min(OEOrder.order_date),
            func.max(OEOrder.order_date)
        ).first()
        return result if result else (None, None)
    
    # ==================== Sync Logs ====================
    
    def create_sync_log(self, sync_type: str, start_date: datetime, end_date: datetime) -> SyncLog:
        """Create a new sync log entry"""
        sync_log = SyncLog(
            sync_type=sync_type,
            sync_start_date=start_date,
            sync_end_date=end_date,
            status='running',
            started_at=datetime.now()
        )
        self.db.add(sync_log)
        self.db.commit()
        self.db.refresh(sync_log)
        return sync_log
    
    def update_sync_log(self, sync_log: SyncLog, **kwargs):
        """Update sync log with results"""
        for key, value in kwargs.items():
            setattr(sync_log, key, value)
        
        sync_log.completed_at = datetime.now()
        if sync_log.started_at:
            sync_log.duration_seconds = (sync_log.completed_at - sync_log.started_at).total_seconds()
        
        self.db.commit()
    
    def get_last_sync_date(self, sync_type: str) -> Optional[datetime]:
        """Get the last successful sync end date"""
        last_sync = self.db.query(SyncLog).filter(
            and_(
                SyncLog.sync_type == sync_type,
                SyncLog.status == 'success'
            )
        ).order_by(SyncLog.sync_end_date.desc()).first()
        
        return last_sync.sync_end_date if last_sync else None
    
    def get_sync_history(self, limit: int = 10) -> List[SyncLog]:
        """Get recent sync history"""
        return self.db.query(SyncLog).order_by(
            SyncLog.started_at.desc()
        ).limit(limit).all()
    
    # ==================== Combined Operations ====================
    
    def get_all_orders_for_training(self, 
                                    start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get all orders (POS + OE) combined for training
        
        Args:
            start_date: Filter orders from this date
            
        Returns:
            Combined DataFrame with all orders
        """
        logger.info("Fetching all orders for training...")
        
        # Get POS orders
        pos_orders = self.get_pos_orders(start_date=start_date)
        if not pos_orders.empty:
            pos_orders['source'] = 'POS'
        
        # Get OE orders
        oe_orders = self.get_oe_orders(start_date=start_date)
        if not oe_orders.empty:
            oe_orders['source'] = 'OE'
        
        # Combine
        if pos_orders.empty and oe_orders.empty:
            return pd.DataFrame()
        elif pos_orders.empty:
            all_orders = oe_orders
        elif oe_orders.empty:
            all_orders = pos_orders
        else:
            all_orders = pd.concat([pos_orders, oe_orders], ignore_index=True)
        
        logger.info(f"✅ Fetched {len(all_orders)} total orders for training")
        return all_orders
    
    def get_database_stats(self) -> dict:
        """Get database statistics"""
        pos_count = self.get_pos_order_count()
        oe_count = self.get_oe_order_count()
        pos_date_range = self.get_pos_date_range()
        oe_date_range = self.get_oe_date_range()
        
        return {
            'pos_orders': pos_count,
            'oe_orders': oe_count,
            'total_orders': pos_count + oe_count,
            'pos_date_range': {
                'min': pos_date_range[0].isoformat() if pos_date_range[0] else None,
                'max': pos_date_range[1].isoformat() if pos_date_range[1] else None
            },
            'oe_date_range': {
                'min': oe_date_range[0].isoformat() if oe_date_range[0] else None,
                'max': oe_date_range[1].isoformat() if oe_date_range[1] else None
            }
        }
