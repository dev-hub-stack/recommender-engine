#!/usr/bin/env python3
"""
Run AWS Personalize Batch Inference Jobs
Creates batch inference jobs to generate recommendations for all users
"""

import boto3
import json
from datetime import datetime
import time

# Configuration
REGION = 'us-east-1'
DATASET_GROUP_ARN = 'arn:aws:personalize:us-east-1:657020414783:dataset-group/mastergroup-recommendations'

# S3 Configuration
S3_BUCKET = 'mastergroup-personalize-data'  # Using existing bucket
S3_INPUT_PATH = f's3://{S3_BUCKET}/batch-inference-input/'
S3_OUTPUT_PATH = f's3://{S3_BUCKET}/batch-inference-output/'

# IAM Role for Batch Inference
ROLE_ARN = 'arn:aws:iam::657020414783:role/PersonalizeS3Role'

# Solution ARNs (from your active solutions)
SOLUTIONS = {
    'user-personalization': 'arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-user-personalization',
    'item-affinity': 'arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-item-affinity',
    'similar-items': 'arn:aws:personalize:us-east-1:657020414783:solution/mastergroup-similar-items'
}

personalize = boto3.client('personalize', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)


def create_batch_input_file():
    """Create batch input JSON file with all user IDs"""
    print("\n📝 Creating batch input file...")
    
    # For user-personalization, we need user IDs
    # For similar-items, we need item IDs
    
    # Example input format for user-personalization:
    # {"userId": "USER_ID"}
    
    # You'll need to generate this from your database
    # For now, creating a sample
    
    input_data = []
    # TODO: Query your database for all user IDs
    # For example:
    # SELECT DISTINCT unified_customer_id FROM orders LIMIT 100
    
    sample_users = [
        {"userId": "03005100928_Syed Hussain"},
        {"userId": "03005557017_Muhammad Shahrez Abbasi"},
        # Add more user IDs...
    ]
    
    # Write to S3
    input_key = f'batch-inference-input/users-{datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
    
    for user in sample_users:
        input_data.append(json.dumps(user))
    
    input_content = '\n'.join(input_data)
    
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=input_key,
            Body=input_content.encode('utf-8')
        )
        print(f"✅ Input file created: s3://{S3_BUCKET}/{input_key}")
        return f's3://{S3_BUCKET}/{input_key}'
    except Exception as e:
        print(f"❌ Error creating input file: {e}")
        return None


def create_batch_inference_job(solution_name, solution_arn, input_path):
    """Create a batch inference job"""
    print(f"\n🚀 Creating batch inference job for {solution_name}...")
    
    # Get latest solution version
    try:
        response = personalize.list_solution_versions(
            solutionArn=solution_arn,
            maxResults=1
        )
        
        if not response['solutionVersions']:
            print(f"❌ No solution versions found for {solution_name}")
            return None
        
        solution_version_arn = response['solutionVersions'][0]['solutionVersionArn']
        print(f"   Using solution version: {solution_version_arn}")
        
        # Create batch inference job
        job_name = f'mastergroup-{solution_name}-batch-{int(time.time())}'
        output_path = f'{S3_OUTPUT_PATH}{solution_name}/'
        
        response = personalize.create_batch_inference_job(
            jobName=job_name,
            solutionVersionArn=solution_version_arn,
            jobInput={
                's3DataSource': {
                    'path': input_path
                }
            },
            jobOutput={
                's3DataDestination': {
                    'path': output_path
                }
            },
            roleArn=ROLE_ARN
        )
        
        batch_job_arn = response['batchInferenceJobArn']
        print(f"✅ Batch inference job created: {job_name}")
        print(f"   ARN: {batch_job_arn}")
        print(f"   Output: {output_path}")
        
        return batch_job_arn
        
    except Exception as e:
        print(f"❌ Error creating batch job: {e}")
        return None


def check_job_status(job_arn):
    """Check status of a batch inference job"""
    try:
        response = personalize.describe_batch_inference_job(
            batchInferenceJobArn=job_arn
        )
        status = response['batchInferenceJob']['status']
        return status
    except Exception as e:
        print(f"Error checking status: {e}")
        return "UNKNOWN"


def main():
    print("=" * 60)
    print("AWS Personalize Batch Inference Job Creator")
    print("=" * 60)
    
    # Step 1: Create input file
    input_path = create_batch_input_file()
    if not input_path:
        print("\n❌ Failed to create input file. Exiting.")
        return
    
    print(f"\n✅ Input file ready: {input_path}")
    
    # Step 2: Create batch inference jobs
    print("\n" + "=" * 60)
    print("Creating Batch Inference Jobs")
    print("=" * 60)
    
    jobs = {}
    
    # Create job for user-personalization
    job_arn = create_batch_inference_job(
        'user-personalization',
        SOLUTIONS['user-personalization'],
        input_path
    )
    if job_arn:
        jobs['user-personalization'] = job_arn
    
    # Similar items would need item IDs, not user IDs
    # Skip for now or create separate input file
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\n✅ Created {len(jobs)} batch inference job(s)")
    
    for name, arn in jobs.items():
        print(f"\n{name}:")
        print(f"  ARN: {arn}")
        print(f"  Status: {check_job_status(arn)}")
    
    print("\n⏳ Jobs will take 2-4 hours to complete")
    print("\n📍 Next Steps:")
    print("1. Wait for jobs to complete (monitor in AWS Console)")
    print("2. Run load_batch_results.py to load results into PostgreSQL")
    print("3. Recommendations will then appear in the dashboard")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
