"""
Customer models for segmentation and analysis
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CustomerSegment:
    """Customer segment model"""
    segment_id: str
    name: str
    description: Optional[str] = None
    income_min: Optional[float] = None
    income_max: Optional[float] = None
    characteristics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = {}

@dataclass
class CustomerProfile:
    """Customer profile for recommendations"""
    customer_id: str
    segment: Optional[CustomerSegment] = None
    purchase_history: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.purchase_history is None:
            self.purchase_history = []
        if self.preferences is None:
            self.preferences = {}
