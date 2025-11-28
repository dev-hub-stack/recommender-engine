"""
APScheduler Service for Automatic Sync & Model Training
Handles:
- Periodic synchronization every 15 minutes
- Daily model training at 3 AM (Auto-Pilot Learning)
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import logging
from datetime import datetime
import redis

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.master_group_api import SYNC_CONFIG
from services.sync_service import get_sync_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchedulerService:
    """Service for scheduling automatic syncs and model training"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.sync_service = get_sync_service()
        self.is_running = False
        self.training_enabled = True  # Enable auto-pilot learning by default
    
    def sync_job(self):
        """Job that runs periodically to sync orders"""
        try:
            logger.info("🔄 Starting scheduled sync job...")
            result = self.sync_service.sync_incremental()
            
            if result['status'] == 'success':
                logger.info(f"✅ Scheduled sync completed: {result['orders_inserted']} orders synced")
            else:
                logger.error(f"❌ Scheduled sync failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"❌ Sync job error: {e}")
    
    def train_models_job(self):
        """
        Job that runs daily to retrain ALL ML models (Auto-Pilot Learning)
        
        Trains all 4 unified models:
        1. Collaborative Filtering (user-based & item-based)
        2. Content-Based Filtering (product features)
        3. Matrix Factorization (SVD)
        4. Popularity-Based (trending products)
        
        Models are saved to PostgreSQL for persistence across Heroku restarts.
        """
        try:
            logger.info("🤖 Starting automatic model training (Auto-Pilot)...")
            logger.info("=" * 60)
            logger.info("Training ALL 4 unified ML models...")
            logger.info("=" * 60)
            start_time = datetime.now()
            
            # Import the ML service
            try:
                from src.algorithms.ml_recommendation_service import get_ml_service
            except ImportError:
                from algorithms.ml_recommendation_service import get_ml_service
            
            ml_service = get_ml_service()
            
            # Train all models with 30 days of data (or full dataset)
            training_results = ml_service.train_all_models(
                time_filter='30days',  # Use recent data for daily training
                force_retrain=True     # Always retrain on schedule
            )
            
            # Also clear Redis cache
            try:
                import redis as redis_module
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
                redis_db = int(os.getenv("REDIS_DB", "0"))
                
                redis_client = redis_module.Redis(host=redis_host, port=redis_port, db=redis_db)
                
                # Clear all ML cache keys
                for pattern in ["ml:*", "recommendations:*"]:
                    keys = list(redis_client.scan_iter(pattern))
                    if keys:
                        redis_client.delete(*keys)
                        logger.info(f"🗑️  Cleared {len(keys)} cache entries for pattern: {pattern}")
                
            except Exception as e:
                logger.warning(f"⚠️  Could not clear Redis cache: {e}")
            
            # Calculate training time
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            # Log summary
            successful = training_results.get('successful_models', 0)
            total = training_results.get('total_models', 4)
            
            logger.info("=" * 60)
            logger.info(f"✅ AUTO-PILOT TRAINING COMPLETE: {successful}/{total} models in {training_duration:.2f}s")
            logger.info("Models trained:")
            for model_name, model_info in training_results.get('models', {}).items():
                status = "✅" if model_info.get('status') == 'success' else "❌"
                logger.info(f"  {status} {model_name}")
            logger.info("=" * 60)
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Model training job error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        if not SYNC_CONFIG['enable_auto_sync']:
            logger.info("Auto-sync is disabled in configuration")
            return
        
        interval_minutes = SYNC_CONFIG['interval_minutes']
        
        # Add the sync job (daily at 2:00 AM - before model training at 3 AM)
        self.scheduler.add_job(
            func=self.sync_job,
            trigger=CronTrigger(hour=2, minute=0),  # 2:00 AM daily
            id='master_group_sync',
            name='Master Group API Sync (Daily)',
            replace_existing=True,
            max_instances=1  # Prevent overlapping syncs
        )
        
        # Add the model training job (daily at 3 AM) - AUTO-PILOT LEARNING
        if self.training_enabled:
            self.scheduler.add_job(
                func=self.train_models_job,
                trigger=CronTrigger(hour=3, minute=0),  # 3:00 AM every day
                id='daily_model_training',
                name='Daily Model Training (Auto-Pilot)',
                replace_existing=True,
                max_instances=1  # Prevent overlapping training
            )
            logger.info("🤖 Auto-Pilot Learning enabled - Training daily at 3:00 AM")
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info(f"✅ Scheduler started - Daily sync at 2:00 AM, Training at 3:00 AM")
        logger.info(f"📅 Next sync scheduled for: {self.get_next_sync_time()}")
        
        if self.training_enabled:
            logger.info(f"🤖 Next training scheduled for: {self.get_next_training_time()}")
    
    def stop(self):
        """Stop the scheduler"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("🛑 Scheduler stopped")
    
    def trigger_manual_sync(self):
        """Manually trigger a sync"""
        logger.info("🔄 Manual sync triggered")
        return self.sync_service.sync_incremental()
    
    def trigger_manual_training(self):
        """Manually trigger model training"""
        logger.info("🤖 Manual training triggered")
        return self.train_models_job()
    
    def get_next_sync_time(self):
        """Get the next scheduled sync time"""
        if not self.is_running:
            return None
        
        job = self.scheduler.get_job('master_group_sync')
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
        return None
    
    def get_next_training_time(self):
        """Get the next scheduled training time"""
        if not self.is_running:
            return None
        
        job = self.scheduler.get_job('daily_model_training')
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
        return None
    
    def get_status(self):
        """Get scheduler status"""
        return {
            'scheduler_running': self.is_running,
            'auto_sync_enabled': SYNC_CONFIG['enable_auto_sync'],
            'sync_interval_minutes': SYNC_CONFIG['interval_minutes'],
            'next_sync_time': self.get_next_sync_time(),
            'auto_pilot_enabled': self.training_enabled,
            'next_training_time': self.get_next_training_time(),
            'sync_service_status': self.sync_service.get_sync_status()
        }


# Global scheduler instance
_scheduler = None

def get_scheduler() -> SchedulerService:
    """Get or create scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = SchedulerService()
    return _scheduler
