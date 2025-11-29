#!/usr/bin/env python3
"""
Run AWS Personalize Batch Inference Jobs
Creates batch inference jobs to generate recommendations for all users/items

Usage:
    python run_batch_inference.py                    # Run all recipes
    python run_batch_inference.py --recipe user      # User personalization only
    python run_batch_inference.py --recipe similar   # Similar items only
    python run_batch_inference.py --recipe affinity  # Item affinity only
"""

import boto3
import json
import os
import sys
import argparse
import psycopg2
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment
REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'mastergroup-personalize-data')
S3_INPUT_PATH = f's3://{S3_BUCKET}/batch/input/'
S3_OUTPUT_PATH = f's3://{S3_BUCKET}/batch/output/'

# IAM Role for Batch Inference
ROLE_ARN = os.getenv('PERSONALIZE_ROLE_ARN', 'arn:aws:iam::657020414783:role/PersonalizeRole')

# Solution Version ARNs (already trained)
SOLUTION_VERSIONS = {
    'user-personalization': os.getenv('USER_PERSONALIZATION_VERSION', 
        'arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-user-personalization/d6537a4d'),
    'similar-items': os.getenv('SIMILAR_ITEMS_VERSION',
        'arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-similar-items/8a42398e'),
    'item-affinity': os.getenv('ITEM_AFFINITY_VERSION',
        'arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-item-affinity/latest'),
}

# PostgreSQL Configuration
PG_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'database': os.getenv('PG_DATABASE', 'mastergroup_recommendations'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', ''),
    'port': os.getenv('PG_PORT', '5432'),
    'sslmode': 'require'
}

# Initialize clients
personalize = boto3.client(
    'personalize',
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
s3 = boto3.client(
    's3',
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


def get_user_ids():
    """Get all user IDs from database"""
    print("📊 Fetching user IDs from database...")
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT unified_customer_id FROM orders WHERE unified_customer_id IS NOT NULL")
        users = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        print(f"   Found {len(users)} users")
        return users
    except Exception as e:
        print(f"❌ Database error: {e}")
        return []


def get_item_ids():
    """Get all product IDs from database"""
    print("📊 Fetching product IDs from database...")
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT product_id FROM order_items WHERE product_id IS NOT NULL")
        items = [str(row[0]) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        print(f"   Found {len(items)} products")
        return items
    except Exception as e:
        print(f"❌ Database error: {e}")
        return []


def create_batch_input(recipe_type):
    """Create batch input file in S3"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    if recipe_type in ['user-personalization', 'item-affinity']:
        ids = get_user_ids()
        id_key = 'userId'
        filename = f'users-{timestamp}.json'
    else:  # similar-items
        ids = get_item_ids()
        id_key = 'itemId'
        filename = f'items-{timestamp}.json'
    
    if not ids:
        print("❌ No IDs found")
        return None
    
    # Create JSONL content
    lines = [json.dumps({id_key: str(id)}) for id in ids]
    content = '\n'.join(lines)
    
    key = f'batch/input/{filename}'
    
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=content.encode('utf-8')
        )
        s3_path = f's3://{S3_BUCKET}/{key}'
        print(f"✅ Input file created: {s3_path} ({len(ids)} records)")
        return s3_path
    except Exception as e:
        print(f"❌ S3 error: {e}")
        return None


def run_batch_job(recipe_name, solution_version_arn, input_path):
    """Create and run a batch inference job"""
    print(f"\n🚀 Starting batch job for {recipe_name}...")
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    job_name = f"batch-{recipe_name}-{timestamp}"
    output_path = f"s3://{S3_BUCKET}/batch/output/{recipe_name}/"
    
    try:
        response = personalize.create_batch_inference_job(
            jobName=job_name,
            solutionVersionArn=solution_version_arn,
            jobInput={'s3DataSource': {'path': input_path}},
            jobOutput={'s3DataDestination': {'path': output_path}},
            roleArn=ROLE_ARN,
            numResults=25  # Top 25 recommendations per user/item
        )
        
        job_arn = response['batchInferenceJobArn']
        print(f"✅ Job created: {job_name}")
        print(f"   ARN: {job_arn}")
        print(f"   Output: {output_path}")
        return job_arn
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def check_job_status(job_arn):
    """Check batch job status"""
    try:
        response = personalize.describe_batch_inference_job(batchInferenceJobArn=job_arn)
        return response['batchInferenceJob']['status']
    except:
        return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description='Run AWS Personalize Batch Inference')
    parser.add_argument('--recipe', choices=['user', 'similar', 'affinity', 'all'], 
                        default='all', help='Which recipe to run')
    args = parser.parse_args()
    
    print("=" * 60)
    print("AWS PERSONALIZE BATCH INFERENCE")
    print("=" * 60)
    print(f"Bucket: {S3_BUCKET}")
    print(f"Region: {REGION}")
    
    jobs = {}
    
    # User Personalization
    if args.recipe in ['user', 'all']:
        print("\n" + "-" * 40)
        print("USER PERSONALIZATION")
        print("-" * 40)
        input_path = create_batch_input('user-personalization')
        if input_path:
            job_arn = run_batch_job('user-personalization', 
                                    SOLUTION_VERSIONS['user-personalization'], 
                                    input_path)
            if job_arn:
                jobs['user-personalization'] = job_arn
    
    # Similar Items
    if args.recipe in ['similar', 'all']:
        print("\n" + "-" * 40)
        print("SIMILAR ITEMS")
        print("-" * 40)
        input_path = create_batch_input('similar-items')
        if input_path:
            job_arn = run_batch_job('similar-items',
                                    SOLUTION_VERSIONS['similar-items'],
                                    input_path)
            if job_arn:
                jobs['similar-items'] = job_arn
    
    # Item Affinity
    if args.recipe in ['affinity', 'all']:
        print("\n" + "-" * 40)
        print("ITEM AFFINITY")
        print("-" * 40)
        input_path = create_batch_input('item-affinity')
        if input_path:
            job_arn = run_batch_job('item-affinity',
                                    SOLUTION_VERSIONS['item-affinity'],
                                    input_path)
            if job_arn:
                jobs['item-affinity'] = job_arn
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not jobs:
        print("❌ No jobs were created")
        return
    
    print(f"✅ Created {len(jobs)} batch job(s):")
    for name, arn in jobs.items():
        print(f"   • {name}: {check_job_status(arn)}")
    
    print("\n⏳ Jobs typically take 30-60 minutes to complete")
    print("\n📍 Next Steps:")
    print("   1. Monitor jobs in AWS Console > Personalize > Batch inference jobs")
    print("   2. Once ACTIVE, run: python load_batch_results.py")
    print("   3. Data will then appear in the dashboard")


if __name__ == "__main__":
    main()
