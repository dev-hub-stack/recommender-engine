"""
Load AWS Personalize Batch Inference Results into PostgreSQL

This script downloads batch inference results from S3 and loads them
into the offline recommendation tables. Optimized for memory usage.
"""

import boto3
import json
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Database config
DB_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': int(os.getenv('PG_PORT', 5432)),
    'database': os.getenv('PG_DATABASE', 'mastergroup'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', '')
}

# AWS config
S3_BUCKET = os.getenv('S3_BUCKET', 'mastergroup-personalize-data')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Initialize AWS clients
try:
    s3 = boto3.client('s3', region_name=AWS_REGION)
except Exception as e:
    logger.error(f"Failed to initialize AWS S3 client: {e}")
    sys.exit(1)

def list_result_files(s3_prefix):
    """List all result JSON files in S3 prefix"""
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)
        files = []
        
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.json.out'):
                    files.append(obj['Key'])
        
        return files
    except Exception as e:
        logger.error(f"Error listing S3 files: {e}")
        return []

def process_file_stream(s3_key, chunk_size=1000):
    """Generator to process file line by line to save memory"""
    local_file = f"/tmp/{s3_key.replace('/', '_')}"
    try:
        logger.info(f"Downloading {s3_key} to {local_file}")
        s3.download_file(S3_BUCKET, s3_key, local_file)
        
        current_chunk = []
        with open(local_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        current_chunk.append(json.loads(line))
                        if len(current_chunk) >= chunk_size:
                            yield current_chunk
                            current_chunk = []
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {s3_key}")
        
        if current_chunk:
            yield current_chunk
            
    except Exception as e:
        logger.error(f"Error processing file {s3_key}: {e}")
    finally:
        if os.path.exists(local_file):
            try:
                os.remove(local_file)
            except:
                pass

def load_user_recommendations(s3_prefix='batch/output/users/'):
    """Load user personalization results into offline_user_recommendations"""
    logger.info(f"📥 Loading User Recommendations from {s3_prefix}...")
    
    files = list_result_files(s3_prefix)
    if not files:
        logger.warning("  ⚠️  No result files found!")
        return 0
    
    conn = None
    total_users = 0
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        for file_key in files:
            logger.info(f"  Processing {file_key}...")
            
            # Ultra-small chunk size to prevent OOM
            for chunk in process_file_stream(file_key, chunk_size=50):
                records = []
                for result in chunk:
                    if 'input' in result and 'output' in result:
                        user_id = result['input'].get('userId')
                        recommendations = result['output'].get('recommendedItems', [])
                        
                        if user_id and recommendations:
                            # Format recommendations as JSONB
                            formatted_recs = []
                            for item in recommendations:
                                if isinstance(item, str):
                                    formatted_recs.append({'product_id': item, 'score': 0})
                                else:
                                    formatted_recs.append({
                                        'product_id': item.get('itemId'),
                                        'score': item.get('score', 0)
                                    })
                            
                            recs_json = json.dumps(formatted_recs)
                            records.append((user_id, recs_json, 'aws-user-personalization'))
                
                if records:
                    execute_values(
                        cursor,
                        """
                        INSERT INTO offline_user_recommendations (user_id, recommendations, recipe_name)
                        VALUES %s
                        ON CONFLICT (user_id) 
                        DO UPDATE SET 
                            recommendations = EXCLUDED.recommendations,
                            recipe_name = EXCLUDED.recipe_name,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        records
                    )
                    conn.commit()
                    total_users += len(records)
                    logger.info(f"    Saved {len(records)} records...")
        
        logger.info(f"  ✅ Loaded {total_users} user recommendations total")
        return total_users
        
    except Exception as e:
        logger.error(f"Database error in load_user_recommendations: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def load_similar_items(s3_prefix='batch/output/items/'):
    """Load similar items results into offline_similar_items"""
    logger.info(f"📥 Loading Similar Items from {s3_prefix}...")
    
    files = list_result_files(s3_prefix)
    if not files:
        logger.warning("  ⚠️  No result files found!")
        return 0
    
    conn = None
    total_items = 0
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        for file_key in files:
            logger.info(f"  Processing {file_key}...")
            
            # Ultra-small chunk size to prevent OOM
            for chunk in process_file_stream(file_key, chunk_size=50):
                records = []
                for result in chunk:
                    if 'input' in result and 'output' in result:
                        product_id = result['input'].get('itemId')
                        similar_items = result['output'].get('recommendedItems', [])
                        
                        if product_id and similar_items:
                            formatted_recs = []
                            for item in similar_items:
                                if isinstance(item, str):
                                    formatted_recs.append({'product_id': item, 'score': 0})
                                else:
                                    formatted_recs.append({
                                        'product_id': item.get('itemId'),
                                        'score': item.get('score', 0)
                                    })
                            
                            similar_json = json.dumps(formatted_recs)
                            records.append((product_id, similar_json, 'aws-similar-items'))
                
                if records:
                    execute_values(
                        cursor,
                        """
                        INSERT INTO offline_similar_items (product_id, similar_products, recipe_name)
                        VALUES %s
                        ON CONFLICT (product_id) 
                        DO UPDATE SET 
                            similar_products = EXCLUDED.similar_products,
                            recipe_name = EXCLUDED.recipe_name,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        records
                    )
                    conn.commit()
                    total_items += len(records)
                    logger.info(f"    Saved {len(records)} records...")
        
        logger.info(f"  ✅ Loaded {total_items} product similarities total")
        return total_items
        
    except Exception as e:
        logger.error(f"Database error in load_similar_items: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_table_stats():
    """Get statistics on offline tables"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        stats = {}
        
        # Check tables existence first
        tables = ['offline_user_recommendations', 'offline_similar_items']
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*), MAX(updated_at) FROM {table}")
                count, last_update = cursor.fetchone()
                stats[table] = {'count': count, 'last_update': str(last_update)}
            except Exception as e:
                logger.warning(f"Could not get stats for {table}: {e}")
                conn.rollback()
        
        cursor.close()
        conn.close()
        
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {}

def main():
    logger.info("="*60)
    logger.info("AWS PERSONALIZE BATCH RESULTS LOADER")
    logger.info("="*60)
    logger.info(f"S3 Bucket: {S3_BUCKET}")
    logger.info(f"Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    logger.info("="*60)
    
    try:
        # Load result types
        # Use correct prefixes based on your S3 folder structure
        load_user_recommendations('batch-inference-output/user-personalization/')
        load_similar_items('batch-inference-output/similar-items/')
        # load_item_affinity() # Uncomment if you have this solution running
        
        # Show final stats
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL DATABASE STATISTICS")
        logger.info("="*60)
        
        stats = get_table_stats()
        for table, data in stats.items():
            logger.info(f"\n{table}:")
            logger.info(f"  Records: {data.get('count', 0):,}")
            logger.info(f"  Last Update: {data.get('last_update', 'N/A')}")
        
        logger.info("\n✅ Batch results loading process completed")
        
    except Exception as e:
        logger.error(f"\n❌ Fatal Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
