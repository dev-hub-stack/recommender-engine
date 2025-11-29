"""
Load AWS Personalize Batch Inference Results into PostgreSQL

This script downloads batch inference results from S3 and loads them
into the offline recommendation tables.
"""

import boto3
import json
import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
from datetime import datetime

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
s3 = boto3.client('s3', region_name=AWS_REGION)

def list_result_files(s3_prefix):
    """List all result JSON files in S3 prefix"""
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)
    files = []
    
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.json.out'):
                files.append(obj['Key'])
    
    return files

def download_and_parse_results(s3_key):
    """Download and parse a batch result file from S3"""
    local_file = f"/tmp/{s3_key.replace('/', '_')}"
    s3.download_file(S3_BUCKET, s3_key, local_file)
    
    results = []
    with open(local_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    return results

def load_user_recommendations(s3_prefix='batch/output/users/'):
    """Load user personalization results into offline_user_recommendations"""
    print(f"\n📥 Loading User Recommendations from {s3_prefix}...")
    
    files = list_result_files(s3_prefix)
    if not files:
        print("  ⚠️  No result files found!")
        return 0
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    total_users = 0
    for file_key in files:
        print(f"  Processing {file_key}...")
        results = download_and_parse_results(file_key)
        
        # Prepare batch insert
        records = []
        for result in results:
            if 'input' in result and 'output' in result:
                user_id = result['input'].get('userId')
                recommendations = result['output'].get('recommendedItems', [])
                
                # Format recommendations as JSONB
                recs_json = json.dumps([
                    {
                        'product_id': item['itemId'],
                        'score': item.get('score', 0)
                    }
                    for item in recommendations
                ])
                
                records.append((user_id, recs_json, 'aws-user-personalization'))
        
        if records:
            # Upsert into database
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
            total_users += len(records)
            conn.commit()
    
    cursor.close()
    conn.close()
    
    print(f"  ✅ Loaded {total_users} user recommendations")
    return total_users

def load_similar_items(s3_prefix='batch/output/items/'):
    """Load similar items results into offline_similar_items"""
    print(f"\n📥 Loading Similar Items from {s3_prefix}...")
    
    files = list_result_files(s3_prefix)
    if not files:
        print("  ⚠️  No result files found!")
        return 0
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    total_items = 0
    for file_key in files:
        print(f"  Processing {file_key}...")
        results = download_and_parse_results(file_key)
        
        records = []
        for result in results:
            if 'input' in result and 'output' in result:
                product_id = result['input'].get('itemId')
                similar_items = result['output'].get('recommendedItems', [])
                
                similar_json = json.dumps([
                    {
                        'product_id': item['itemId'],
                        'score': item.get('score', 0)
                    }
                    for item in similar_items
                ])
                
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
            total_items += len(records)
            conn.commit()
    
    cursor.close()
    conn.close()
    
    print(f"  ✅ Loaded {total_items} product similarities")
    return total_items

def load_item_affinity(s3_prefix='batch/output/affinity/'):
    """Load item affinity results into offline_item_affinity"""
    print(f"\n📥 Loading Item Affinity from {s3_prefix}...")
    
    files = list_result_files(s3_prefix)
    if not files:
        print("  ⚠️  No result files found!")
        return 0
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    total_affinity = 0
    for file_key in files:
        print(f"  Processing {file_key}...")
        results = download_and_parse_results(file_key)
        
        records = []
        for result in results:
            if 'input' in result and 'output' in result:
                user_id = result['input'].get('userId')
                affinities = result['output'].get('recommendedItems', [])
                
                affinity_json = json.dumps([
                    {
                        'product_id': item['itemId'],
                        'score': item.get('score', 0)
                    }
                    for item in affinities
                ])
                
                records.append((user_id, affinity_json, 'aws-item-affinity'))
        
        if records:
            execute_values(
                cursor,
                """
                INSERT INTO offline_item_affinity (user_id, item_affinities, recipe_name)
                VALUES %s
                ON CONFLICT (user_id) 
                DO UPDATE SET 
                    item_affinities = EXCLUDED.item_affinities,
                    recipe_name = EXCLUDED.recipe_name,
                    updated_at = CURRENT_TIMESTAMP
                """,
                records
            )
            total_affinity += len(records)
            conn.commit()
    
    cursor.close()
    conn.close()
    
    print(f"  ✅ Loaded {total_affinity} user affinities")
    return total_affinity

def get_table_stats():
    """Get statistics on offline tables"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    stats = {}
    
    # User recommendations
    cursor.execute("SELECT COUNT(*), MAX(updated_at) FROM offline_user_recommendations")
    count, last_update = cursor.fetchone()
    stats['user_recommendations'] = {'count': count, 'last_update': last_update}
    
    # Similar items
    cursor.execute("SELECT COUNT(*), MAX(updated_at) FROM offline_similar_items")
    count, last_update = cursor.fetchone()
    stats['similar_items'] = {'count': count, 'last_update': last_update}
    
    # Item affinity
    cursor.execute("SELECT COUNT(*), MAX(updated_at) FROM offline_item_affinity")
    count, last_update = cursor.fetchone()
    stats['item_affinity'] = {'count': count, 'last_update': last_update}
    
    cursor.close()
    conn.close()
    
    return stats

def main():
    print("="*60)
    print("AWS PERSONALIZE BATCH RESULTS LOADER")
    print("="*60)
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    print("="*60)
    
    try:
        # Load all result types
        load_user_recommendations()
        load_similar_items()
        load_item_affinity()
        
        # Show final stats
        print("\n" + "="*60)
        print("📊 FINAL DATABASE STATISTICS")
        print("="*60)
        
        stats = get_table_stats()
        for table, data in stats.items():
            print(f"\n{table}:")
            print(f"  Records: {data['count']:,}")
            print(f"  Last Update: {data['last_update']}")
        
        print("\n✅ Batch results loaded successfully!")
        print("\n🔄 Next: Update backend API to use offline tables")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
