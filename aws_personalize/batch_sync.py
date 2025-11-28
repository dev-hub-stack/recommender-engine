"""
AWS Personalize Batch Sync & Retrain
Run daily/weekly to export new data and retrain the model
"""

import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
import csv
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
AWS_PROFILE = os.environ.get('AWS_PROFILE', 'mastergroup')
S3_BUCKET = 'mastergroup-personalize-data'
DATASET_GROUP_ARN = 'arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations'
ROLE_ARN = 'arn:aws:iam::657020414783:role/PersonalizeRole'
SOLUTION_ARN = 'arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-user-personalization'

DB_CONFIG = {
    'host': os.environ.get('PG_HOST', 'localhost'),
    'port': os.environ.get('PG_PORT', '5432'),
    'dbname': os.environ.get('PG_DB', 'mastergroup_recommendations'),
    'user': os.environ.get('PG_USER', 'postgres'),
    'password': os.environ.get('PG_PASSWORD', 'postgres')
}


class PersonalizeBatchSync:
    """
    Batch sync service for periodic data export and model retraining
    """
    
    def __init__(self):
        self.session = boto3.Session(
            profile_name=AWS_PROFILE,
            region_name=AWS_REGION
        )
        self.personalize = self.session.client('personalize')
        self.s3 = self.session.client('s3')
        self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
    def get_db_connection(self):
        return psycopg2.connect(**DB_CONFIG)
    
    def export_incremental_interactions(self, since_days: int = 7) -> str:
        """
        Export only new interactions since last sync
        
        Args:
            since_days: Export interactions from last N days
            
        Returns:
            S3 path of uploaded file
        """
        logger.info(f"Exporting interactions from last {since_days} days...")
        
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                o.unified_customer_id as user_id,
                oi.product_id as item_id,
                EXTRACT(EPOCH FROM o.order_date)::bigint as timestamp,
                'purchase' as event_type,
                oi.quantity as event_value
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE o.unified_customer_id IS NOT NULL 
            AND oi.product_id IS NOT NULL
            AND o.order_date IS NOT NULL
            AND o.order_date >= NOW() - INTERVAL '%s days'
            ORDER BY o.order_date
        """, (since_days,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Write to temp file
        temp_file = f'/tmp/interactions_incremental_{self.timestamp}.csv'
        with open(temp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['USER_ID', 'ITEM_ID', 'TIMESTAMP', 'EVENT_TYPE', 'EVENT_VALUE'])
            for row in rows:
                writer.writerow([
                    row['user_id'],
                    row['item_id'],
                    row['timestamp'],
                    row['event_type'],
                    row['event_value'] or 1
                ])
        
        # Upload to S3
        s3_key = f'incremental/interactions_{self.timestamp}.csv'
        self.s3.upload_file(temp_file, S3_BUCKET, s3_key)
        
        logger.info(f"Exported {len(rows)} interactions to s3://{S3_BUCKET}/{s3_key}")
        os.remove(temp_file)
        
        return f's3://{S3_BUCKET}/{s3_key}'
    
    def export_full_data(self) -> dict:
        """
        Export full dataset (interactions, items, users)
        Use for initial setup or monthly full refresh
        """
        logger.info("Exporting full dataset...")
        
        results = {
            'interactions': self._export_interactions(),
            'items': self._export_items(),
            'users': self._export_users()
        }
        
        return results
    
    def _export_interactions(self) -> str:
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                o.unified_customer_id as user_id,
                oi.product_id as item_id,
                EXTRACT(EPOCH FROM o.order_date)::bigint as timestamp,
                'purchase' as event_type,
                oi.quantity as event_value
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE o.unified_customer_id IS NOT NULL 
            AND oi.product_id IS NOT NULL
            AND o.order_date IS NOT NULL
            ORDER BY o.order_date
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        temp_file = f'/tmp/interactions_{self.timestamp}.csv'
        with open(temp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['USER_ID', 'ITEM_ID', 'TIMESTAMP', 'EVENT_TYPE', 'EVENT_VALUE'])
            for row in rows:
                writer.writerow([
                    row['user_id'], row['item_id'], row['timestamp'],
                    row['event_type'], row['event_value'] or 1
                ])
        
        s3_key = f'data/interactions.csv'
        self.s3.upload_file(temp_file, S3_BUCKET, s3_key)
        os.remove(temp_file)
        
        logger.info(f"Exported {len(rows)} interactions")
        return f's3://{S3_BUCKET}/{s3_key}'
    
    def _export_items(self) -> str:
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT DISTINCT
                oi.product_id as item_id,
                MAX(oi.product_name) as item_name,
                'General' as category,
                AVG(oi.unit_price) as price,
                COUNT(DISTINCT oi.order_id) as purchase_count
            FROM order_items oi
            WHERE oi.product_id IS NOT NULL
            GROUP BY oi.product_id
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        temp_file = f'/tmp/items_{self.timestamp}.csv'
        with open(temp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ITEM_ID', 'ITEM_NAME', 'CATEGORY', 'PRICE', 'PURCHASE_COUNT'])
            for row in rows:
                writer.writerow([
                    row['item_id'],
                    (row['item_name'] or 'Unknown')[:256],
                    row['category'],
                    round(float(row['price'] or 0), 2),
                    row['purchase_count']
                ])
        
        s3_key = f'data/items.csv'
        self.s3.upload_file(temp_file, S3_BUCKET, s3_key)
        os.remove(temp_file)
        
        logger.info(f"Exported {len(rows)} items")
        return f's3://{S3_BUCKET}/{s3_key}'
    
    def _export_users(self) -> str:
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                o.unified_customer_id as user_id,
                MAX(o.customer_city) as city,
                MAX(o.province) as province,
                COUNT(DISTINCT o.id) as order_count,
                COALESCE(SUM(oi.total_price), 0) as total_spend
            FROM orders o
            LEFT JOIN order_items oi ON o.id = oi.order_id
            WHERE o.unified_customer_id IS NOT NULL
            AND TRIM(o.unified_customer_id) != ''
            AND LENGTH(o.unified_customer_id) > 2
            GROUP BY o.unified_customer_id
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        temp_file = f'/tmp/users_{self.timestamp}.csv'
        with open(temp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['USER_ID', 'CITY', 'PROVINCE', 'ORDER_COUNT', 'TOTAL_SPEND'])
            for row in rows:
                writer.writerow([
                    str(row['user_id'])[:256],
                    (row['city'] or 'Unknown')[:256],
                    (row['province'] or 'Unknown')[:256],
                    row['order_count'],
                    round(float(row['total_spend'] or 0), 2)
                ])
        
        s3_key = f'data/users.csv'
        self.s3.upload_file(temp_file, S3_BUCKET, s3_key)
        os.remove(temp_file)
        
        logger.info(f"Exported {len(rows)} users")
        return f's3://{S3_BUCKET}/{s3_key}'
    
    def import_incremental_data(self, s3_path: str) -> str:
        """
        Import incremental interactions data
        """
        job_name = f"incremental-import-{self.timestamp}"
        
        response = self.personalize.create_dataset_import_job(
            jobName=job_name,
            datasetArn=f"{DATASET_GROUP_ARN.replace('dataset-group', 'dataset')}/INTERACTIONS",
            dataSource={'dataLocation': s3_path},
            roleArn=ROLE_ARN,
            importMode='INCREMENTAL'  # Append to existing data
        )
        
        logger.info(f"Started incremental import: {response['datasetImportJobArn']}")
        return response['datasetImportJobArn']
    
    def wait_for_import(self, import_job_arn: str, timeout_minutes: int = 30) -> bool:
        """Wait for import job to complete"""
        logger.info("Waiting for import to complete...")
        start_time = time.time()
        
        while True:
            response = self.personalize.describe_dataset_import_job(
                datasetImportJobArn=import_job_arn
            )
            status = response['datasetImportJob']['status']
            
            if status == 'ACTIVE':
                logger.info("Import completed successfully")
                return True
            elif status == 'CREATE FAILED':
                logger.error(f"Import failed: {response['datasetImportJob'].get('failureReason')}")
                return False
            
            if (time.time() - start_time) > timeout_minutes * 60:
                logger.error("Import timeout")
                return False
            
            logger.info(f"Import status: {status}")
            time.sleep(30)
    
    def create_new_solution_version(self) -> Optional[str]:
        """
        Create a new solution version (retrain the model)
        """
        try:
            response = self.personalize.create_solution_version(
                solutionArn=SOLUTION_ARN,
                trainingMode='UPDATE'  # Incremental training (faster)
            )
            
            solution_version_arn = response['solutionVersionArn']
            logger.info(f"Started training: {solution_version_arn}")
            return solution_version_arn
            
        except Exception as e:
            logger.error(f"Failed to create solution version: {e}")
            return None
    
    def wait_for_training(self, solution_version_arn: str, timeout_minutes: int = 90) -> bool:
        """Wait for solution training to complete"""
        logger.info("Waiting for training to complete...")
        start_time = time.time()
        
        while True:
            response = self.personalize.describe_solution_version(
                solutionVersionArn=solution_version_arn
            )
            status = response['solutionVersion']['status']
            
            if status == 'ACTIVE':
                logger.info("Training completed successfully")
                return True
            elif status == 'CREATE FAILED':
                logger.error(f"Training failed: {response['solutionVersion'].get('failureReason')}")
                return False
            
            if (time.time() - start_time) > timeout_minutes * 60:
                logger.error("Training timeout")
                return False
            
            logger.info(f"Training status: {status}")
            time.sleep(60)
    
    def update_campaign(self, solution_version_arn: str, campaign_arn: str) -> bool:
        """
        Update campaign to use new solution version
        """
        try:
            self.personalize.update_campaign(
                campaignArn=campaign_arn,
                solutionVersionArn=solution_version_arn
            )
            logger.info(f"Campaign updated to use: {solution_version_arn}")
            return True
        except Exception as e:
            logger.error(f"Failed to update campaign: {e}")
            return False


def run_daily_sync():
    """
    Run daily sync job:
    1. Export incremental interactions (last 7 days)
    2. Import to Personalize
    3. Retrain model (incremental)
    4. Update campaign
    """
    print("\n" + "="*60)
    print("DAILY PERSONALIZE SYNC")
    print("="*60 + "\n")
    
    sync = PersonalizeBatchSync()
    
    # Step 1: Export
    print("Step 1: Exporting incremental data...")
    s3_path = sync.export_incremental_interactions(since_days=7)
    
    # Step 2: Import
    print("\nStep 2: Importing to Personalize...")
    import_arn = sync.import_incremental_data(s3_path)
    
    if sync.wait_for_import(import_arn):
        # Step 3: Retrain
        print("\nStep 3: Retraining model...")
        version_arn = sync.create_new_solution_version()
        
        if version_arn and sync.wait_for_training(version_arn):
            # Step 4: Update campaign
            print("\nStep 4: Updating campaign...")
            campaign_arn = 'arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign'
            sync.update_campaign(version_arn, campaign_arn)
    
    print("\n" + "="*60)
    print("SYNC COMPLETE")
    print("="*60)


def run_weekly_full_sync():
    """
    Run weekly full sync:
    1. Export all data
    2. Full reimport
    3. Full retrain
    """
    print("\n" + "="*60)
    print("WEEKLY FULL PERSONALIZE SYNC")
    print("="*60 + "\n")
    
    sync = PersonalizeBatchSync()
    
    # Export full data
    print("Step 1: Exporting full dataset...")
    paths = sync.export_full_data()
    
    print(f"\nExported to:")
    for key, path in paths.items():
        print(f"  - {key}: {path}")
    
    print("\nFull sync requires manual import via AWS Console or setup script.")
    print("Run: python aws_personalize/setup_personalize.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        run_weekly_full_sync()
    else:
        run_daily_sync()
