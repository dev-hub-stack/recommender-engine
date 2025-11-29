"""
Generate Batch Inference Input Files from PostgreSQL

This script queries your PostgreSQL database and creates JSON input files
for AWS Personalize batch inference jobs.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
import boto3
from dotenv import load_dotenv

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

# Initialize S3 client
s3 = boto3.client('s3', region_name=AWS_REGION)

def get_all_users():
    """Get all unique user IDs from database"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT unified_customer_id 
        FROM orders 
        WHERE unified_customer_id IS NOT NULL
        ORDER BY unified_customer_id
    """)
    
    users = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    return users

def get_all_products():
    """Get all unique product IDs from database"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT product_id 
        FROM order_items 
        WHERE product_id IS NOT NULL
        ORDER BY product_id
    """)
    
    products = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    return products

def create_user_input_file():
    """
    Create input JSON file for User Personalization recipe
    Format: One JSON object per line: {"userId": "123"}
    """
    print("Generating user input file...")
    users = get_all_users()
    
    if not users:
        print("  ⚠️ No users found in database!")
        return None
    
    # Create local file
    local_file = '/tmp/batch_users_input.json'
    with open(local_file, 'w') as f:
        for user_id in users:
            json.dump({"userId": str(user_id)}, f)
            f.write('\n')
    
    print(f"  ✅ Generated input for {len(users)} users")
    
    # Upload to S3
    s3_key = 'batch/input/users.json'
    s3.upload_file(local_file, S3_BUCKET, s3_key)
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"  ✅ Uploaded to {s3_path}")
    
    return s3_path

def create_items_input_file():
    """
    Create input JSON file for Similar Items recipe
    Format: One JSON object per line: {"itemId": "ABC123"}
    """
    print("Generating items input file...")
    products = get_all_products()
    
    if not products:
        print("  ⚠️ No products found in database!")
        return None
    
    # Create local file
    local_file = '/tmp/batch_items_input.json'
    with open(local_file, 'w') as f:
        for product_id in products:
            json.dump({"itemId": str(product_id)}, f)
            f.write('\n')
    
    print(f"  ✅ Generated input for {len(products)} products")
    
    # Upload to S3
    s3_key = 'batch/input/items.json'
    s3.upload_file(local_file, S3_BUCKET, s3_key)
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"  ✅ Uploaded to {s3_path}")
    
    return s3_path

def create_item_affinity_input_file():
    """
    Create input JSON file for Item Affinity recipe
    Same as user personalization
    """
    print("Generating item affinity input file (same as users)...")
    users = get_all_users()
    
    if not users:
        print("  ⚠️ No users found in database!")
        return None
    
    local_file = '/tmp/batch_affinity_input.json'
    with open(local_file, 'w') as f:
        for user_id in users:
            json.dump({"userId": str(user_id)}, f)
            f.write('\n')
    
    print(f"  ✅ Generated input for {len(users)} users")
    
    s3_key = 'batch/input/affinity.json'
    s3.upload_file(local_file, S3_BUCKET, s3_key)
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"  ✅ Uploaded to {s3_path}")
    
    return s3_path

def main():
    print("="*60)
    print("AWS PERSONALIZE BATCH INPUT FILE GENERATOR")
    print("="*60)
    print(f"Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print("="*60)
    
    try:
        # Generate all input files
        user_path = create_user_input_file()
        items_path = create_items_input_file()
        affinity_path = create_item_affinity_input_file()
        
        print("\n" + "="*60)
        print("✅ BATCH INPUT FILES READY!")
        print("="*60)
        print(f"Users:     {user_path}")
        print(f"Items:     {items_path}")
        print(f"Affinity:  {affinity_path}")
        print("\nYou can now run train_hybrid_model.py to start batch inference.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
