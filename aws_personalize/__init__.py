"""
AWS Personalize Integration for MasterGroup Recommendations

Modules:
- personalize_service: Get recommendations from AWS Personalize
- personalize_sync: Real-time event sync to Personalize
- batch_sync: Daily/weekly batch export and model retrain
"""

from .personalize_service import AWSPersonalizeService, get_personalize_service
from .personalize_sync import PersonalizeSyncService, get_personalize_sync, sync_orders_to_personalize

__all__ = [
    'AWSPersonalizeService', 
    'get_personalize_service',
    'PersonalizeSyncService',
    'get_personalize_sync',
    'sync_orders_to_personalize'
]
