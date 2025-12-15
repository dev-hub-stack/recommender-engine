"""
SQLAlchemy ORM Models
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Index, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class POSOrder(Base):
    """POS (Point of Sale) Orders"""
    
    __tablename__ = 'pos_orders'
    
    # Primary Key
    id = Column(String(50), primary_key=True, index=True)
    
    # Customer Information
    customer_phone = Column(String(50), index=True)
    customer_email = Column(String(255), index=True)
    customer_name = Column(String(255))
    customer_address = Column(Text)
    customer_city = Column(String(100))
    customer_state = Column(String(100))
    customer_country = Column(String(100))
    
    # Order Information
    order_date = Column(DateTime, index=True, nullable=False)
    order_source = Column(String(50))
    order_status = Column(String(50))
    order_status_id = Column(Integer)
    
    # Product Information (JSON string)
    has_items = Column(Text, nullable=False)
    
    # Dealer Information
    dealer_id = Column(String(50))
    dealer_name = Column(String(255))
    dealership_id = Column(String(50))
    
    # Financial Information
    total_price = Column(Float)
    discount = Column(Float)
    dealer_discount = Column(Float)
    payment_mode = Column(String(50))
    
    # Other
    brand_name = Column(String(100))
    courier_id = Column(String(50))
    is_split = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_pos_customer_phone', 'customer_phone'),
        Index('idx_pos_customer_email', 'customer_email'),
        Index('idx_pos_order_date', 'order_date'),
        Index('idx_pos_customer_date', 'customer_phone', 'order_date'),
    )
    
    def __repr__(self):
        return f"<POSOrder(id={self.id}, customer={self.customer_phone}, date={self.order_date})>"


class OEOrder(Base):
    """OE (Order Entry) Orders"""
    
    __tablename__ = 'oe_orders'
    
    # Primary Key
    id = Column(String(50), primary_key=True, index=True)
    
    # Customer Information
    customer_phone = Column(String(50), index=True)
    customer_email = Column(String(255), index=True)
    customer_name = Column(String(255))
    customer_address = Column(Text)
    customer_city = Column(String(100))
    customer_state = Column(String(100))
    customer_country = Column(String(100))
    
    # Order Information
    order_date = Column(DateTime, index=True, nullable=False)
    order_name = Column(String(100))
    order_status = Column(String(50))
    order_status_id = Column(Integer)
    order_comments = Column(Text)
    
    # Product Information (JSON string)
    has_items = Column(Text, nullable=False)
    
    # Financial Information
    total_price = Column(Float)
    discount = Column(Float)
    payment_mode = Column(String(50))
    
    # Other
    brand_name = Column(String(100))
    courier_id = Column(String(50))
    is_split = Column(Boolean, default=False)
    assigned_tags = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_oe_customer_phone', 'customer_phone'),
        Index('idx_oe_customer_email', 'customer_email'),
        Index('idx_oe_order_date', 'order_date'),
        Index('idx_oe_customer_date', 'customer_phone', 'order_date'),
    )
    
    def __repr__(self):
        return f"<OEOrder(id={self.id}, customer={self.customer_phone}, date={self.order_date})>"


class SyncLog(Base):
    """Track data synchronization history"""
    
    __tablename__ = 'sync_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Sync Information
    sync_type = Column(String(20), nullable=False)  # 'pos' or 'oe'
    sync_start_date = Column(DateTime, nullable=False)
    sync_end_date = Column(DateTime, nullable=False)
    
    # Statistics
    records_fetched = Column(Integer, default=0)
    records_inserted = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    
    # Status
    status = Column(String(20), nullable=False)  # 'success', 'failed', 'partial'
    error_message = Column(Text)
    
    # Timing
    duration_seconds = Column(Float)
    started_at = Column(DateTime, default=func.now(), nullable=False)
    completed_at = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_sync_type', 'sync_type'),
        Index('idx_sync_status', 'status'),
        Index('idx_sync_started_at', 'started_at'),
    )
    
    def __repr__(self):
        return f"<SyncLog(id={self.id}, type={self.sync_type}, status={self.status})>"
