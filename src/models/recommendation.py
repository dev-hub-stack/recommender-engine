"""
Basic recommendation models for the recommendation engine
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Recommendation:
    """Basic recommendation model"""
    product_id: str
    score: float
    reason: Optional[str] = None
    purchase_count: Optional[int] = None
    co_purchase_count: Optional[int] = None
    category: Optional[str] = None
    product_name: Optional[str] = None

@dataclass
class RecommendationRequest:
    """Recommendation request model"""
    customer_id: str
    algorithm: Optional[str] = "collaborative"
    limit: Optional[int] = 10
    time_filter: Optional[str] = "all"

@dataclass
class RecommendationResponse:
    """Recommendation response model"""
    customer_id: str
    recommendations: List[Recommendation]
    algorithm: str
    cached: bool = False
    timestamp: Optional[str] = None

@dataclass
class RecommendationContext:
    """Context for generating recommendations"""
    customer_id: str
    session_id: Optional[str] = None
    platform: Optional[str] = None
    
@dataclass
class RecommendationMetadata:
    """Metadata for recommendations"""
    algorithm_version: str
    model_version: str
    performance_metrics: Optional[Dict[str, float]] = None

class RecommendationAlgorithm:
    """Base class for recommendation algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        
    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """Override in subclasses"""
        raise NotImplementedError
