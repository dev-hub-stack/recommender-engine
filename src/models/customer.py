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
    name: str = ""
    segment_name: str = ""  # Alias for backwards compatibility
    description: Optional[str] = None
    income_min: Optional[float] = None
    income_max: Optional[float] = None
    characteristics: Optional[Dict[str, Any]] = None
    target_income_range: Optional[tuple] = None  # (min, max) tuple
    geographic_focus: Optional[List[str]] = None  # List of cities/regions
    
    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = {}
        if self.geographic_focus is None:
            self.geographic_focus = []
        # Handle segment_name alias
        if self.segment_name and not self.name:
            self.name = self.segment_name
        elif self.name and not self.segment_name:
            self.segment_name = self.name
        # Handle target_income_range
        if self.target_income_range:
            self.income_min = self.target_income_range[0]
            self.income_max = self.target_income_range[1]

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
