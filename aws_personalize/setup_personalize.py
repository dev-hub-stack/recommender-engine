"""
AWS Personalize Setup Script
Creates dataset group, schemas, datasets, and imports data
"""

import boto3
import json
import time
import os
from datetime import datetime

# Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
S3_BUCKET = os.environ.get('PERSONALIZE_S3_BUCKET', 'mastergroup-personalize-data')
DATASET_GROUP_NAME = 'mastergroup-recommendations'
ROLE_ARN = os.environ.get('PERSONALIZE_ROLE_ARN', 'arn:aws:iam::657020414783:role/PersonalizeRole')

# Initialize clients
personalize = boto3.client('personalize', region_name=AWS_REGION)
s3 = boto3.client('s3', region_name=AWS_REGION)

# Schemas for AWS Personalize
INTERACTIONS_SCHEMA = {
    "type": "record",
    "name": "Interactions",
    "namespace": "com.amazonaws.personalize.schema",
    "fields": [
        {"name": "USER_ID", "type": "string"},
        {"name": "ITEM_ID", "type": "string"},
        {"name": "TIMESTAMP", "type": "long"},
        {"name": "EVENT_TYPE", "type": "string"},
        {"name": "EVENT_VALUE", "type": "float"}
    ],
    "version": "1.0"
}

ITEMS_SCHEMA = {
    "type": "record",
    "name": "Items",
    "namespace": "com.amazonaws.personalize.schema",
    "fields": [
        {"name": "ITEM_ID", "type": "string"},
        {"name": "ITEM_NAME", "type": "string"},
        {"name": "CATEGORY", "type": "string", "categorical": True},
        {"name": "PRICE", "type": "float"},
        {"name": "PURCHASE_COUNT", "type": "int"}
    ],
    "version": "1.0"
}

USERS_SCHEMA = {
    "type": "record",
    "name": "Users",
    "namespace": "com.amazonaws.personalize.schema",
    "fields": [
        {"name": "USER_ID", "type": "string"},
        {"name": "CITY", "type": "string", "categorical": True},
        {"name": "PROVINCE", "type": "string", "categorical": True},
        {"name": "ORDER_COUNT", "type": "int"},
        {"name": "TOTAL_SPEND", "type": "float"}
    ],
    "version": "1.0"
}


def upload_to_s3(local_file, s3_key):
    """Upload file to S3"""
    print(f"  Uploading {local_file} to s3://{S3_BUCKET}/{s3_key}")
    s3.upload_file(local_file, S3_BUCKET, s3_key)
    return f"s3://{S3_BUCKET}/{s3_key}"


def create_schema(name, schema_dict):
    """Create a Personalize schema"""
    try:
        response = personalize.create_schema(
            name=name,
            schema=json.dumps(schema_dict)
        )
        print(f"  ✅ Created schema: {name}")
        return response['schemaArn']
    except personalize.exceptions.ResourceAlreadyExistsException:
        # Get existing schema
        response = personalize.list_schemas()
        for schema in response['schemas']:
            if schema['name'] == name:
                print(f"  ℹ️  Schema already exists: {name}")
                return schema['schemaArn']
        raise


def create_dataset_group():
    """Create dataset group"""
    try:
        response = personalize.create_dataset_group(name=DATASET_GROUP_NAME)
        dataset_group_arn = response['datasetGroupArn']
        print(f"  ✅ Created dataset group: {DATASET_GROUP_NAME}")
        
        # Wait for dataset group to be active
        print("  ⏳ Waiting for dataset group to be active...")
        while True:
            status = personalize.describe_dataset_group(
                datasetGroupArn=dataset_group_arn
            )['datasetGroup']['status']
            if status == 'ACTIVE':
                break
            elif status == 'CREATE FAILED':
                raise Exception("Dataset group creation failed")
            time.sleep(10)
        
        return dataset_group_arn
    except personalize.exceptions.ResourceAlreadyExistsException:
        response = personalize.list_dataset_groups()
        for dg in response['datasetGroups']:
            if dg['name'] == DATASET_GROUP_NAME:
                print(f"  ℹ️  Dataset group already exists: {DATASET_GROUP_NAME}")
                return dg['datasetGroupArn']
        raise


def create_dataset(dataset_group_arn, dataset_type, schema_arn):
    """Create a dataset"""
    name = f"{DATASET_GROUP_NAME}-{dataset_type.lower()}"
    try:
        response = personalize.create_dataset(
            name=name,
            schemaArn=schema_arn,
            datasetGroupArn=dataset_group_arn,
            datasetType=dataset_type
        )
        print(f"  ✅ Created dataset: {name}")
        return response['datasetArn']
    except personalize.exceptions.ResourceAlreadyExistsException:
        response = personalize.list_datasets(datasetGroupArn=dataset_group_arn)
        for ds in response['datasets']:
            if ds['name'] == name:
                print(f"  ℹ️  Dataset already exists: {name}")
                return ds['datasetArn']
        raise


def import_data(dataset_arn, s3_path, job_name):
    """Import data from S3 to dataset"""
    try:
        response = personalize.create_dataset_import_job(
            jobName=job_name,
            datasetArn=dataset_arn,
            dataSource={'dataLocation': s3_path},
            roleArn=ROLE_ARN
        )
        import_job_arn = response['datasetImportJobArn']
        print(f"  ✅ Started import job: {job_name}")
        return import_job_arn
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        raise


def wait_for_import(import_job_arn, timeout_minutes=30):
    """Wait for import job to complete"""
    print(f"  ⏳ Waiting for import to complete (timeout: {timeout_minutes}min)...")
    start_time = time.time()
    
    while True:
        response = personalize.describe_dataset_import_job(
            datasetImportJobArn=import_job_arn
        )
        status = response['datasetImportJob']['status']
        
        if status == 'ACTIVE':
            print("  ✅ Import completed!")
            return True
        elif status == 'CREATE FAILED':
            error = response['datasetImportJob'].get('failureReason', 'Unknown')
            print(f"  ❌ Import failed: {error}")
            return False
        
        if (time.time() - start_time) > timeout_minutes * 60:
            print("  ⚠️  Import timeout - check AWS console")
            return False
        
        print(f"    Status: {status}...")
        time.sleep(30)


def create_solution():
    """Create a solution (train the model)"""
    solution_name = f"{DATASET_GROUP_NAME}-user-personalization"
    
    # Get dataset group ARN
    response = personalize.list_dataset_groups()
    dataset_group_arn = None
    for dg in response['datasetGroups']:
        if dg['name'] == DATASET_GROUP_NAME:
            dataset_group_arn = dg['datasetGroupArn']
            break
    
    if not dataset_group_arn:
        raise Exception("Dataset group not found")
    
    try:
        response = personalize.create_solution(
            name=solution_name,
            datasetGroupArn=dataset_group_arn,
            recipeArn='arn:aws:personalize:::recipe/aws-user-personalization'
        )
        solution_arn = response['solutionArn']
        print(f"  ✅ Created solution: {solution_name}")
        
        # Create solution version (train)
        response = personalize.create_solution_version(solutionArn=solution_arn)
        solution_version_arn = response['solutionVersionArn']
        print(f"  ✅ Started training solution version")
        print(f"  ⏳ Training takes 30-60 minutes. Check AWS console for status.")
        
        return solution_arn, solution_version_arn
    except personalize.exceptions.ResourceAlreadyExistsException:
        print(f"  ℹ️  Solution already exists: {solution_name}")
        return None, None


def main():
    print("\n" + "="*60)
    print("AWS PERSONALIZE SETUP")
    print("="*60 + "\n")
    
    if not ROLE_ARN:
        print("❌ ERROR: PERSONALIZE_ROLE_ARN environment variable not set")
        print("\nFirst, create an IAM role with these permissions:")
        print("  1. AmazonPersonalizeFullAccess")
        print("  2. AmazonS3ReadOnlyAccess (for your bucket)")
        print("\nThen set: export PERSONALIZE_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/PersonalizeRole")
        return
    
    # Step 1: Upload data to S3
    print("STEP 1: Upload data to S3")
    print("-" * 40)
    interactions_s3 = upload_to_s3('aws_personalize/data/interactions.csv', 'data/interactions.csv')
    items_s3 = upload_to_s3('aws_personalize/data/items.csv', 'data/items.csv')
    users_s3 = upload_to_s3('aws_personalize/data/users.csv', 'data/users.csv')
    
    # Step 2: Create schemas
    print("\nSTEP 2: Create schemas")
    print("-" * 40)
    interactions_schema_arn = create_schema('mastergroup-interactions-schema', INTERACTIONS_SCHEMA)
    items_schema_arn = create_schema('mastergroup-items-schema', ITEMS_SCHEMA)
    users_schema_arn = create_schema('mastergroup-users-schema', USERS_SCHEMA)
    
    # Step 3: Create dataset group
    print("\nSTEP 3: Create dataset group")
    print("-" * 40)
    dataset_group_arn = create_dataset_group()
    
    # Step 4: Create datasets
    print("\nSTEP 4: Create datasets")
    print("-" * 40)
    interactions_dataset_arn = create_dataset(dataset_group_arn, 'Interactions', interactions_schema_arn)
    items_dataset_arn = create_dataset(dataset_group_arn, 'Items', items_schema_arn)
    users_dataset_arn = create_dataset(dataset_group_arn, 'Users', users_schema_arn)
    
    # Step 5: Import data
    print("\nSTEP 5: Import data")
    print("-" * 40)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    interactions_import = import_data(
        interactions_dataset_arn, 
        interactions_s3, 
        f"interactions-import-{timestamp}"
    )
    
    # Wait for interactions import (required before creating solution)
    if wait_for_import(interactions_import):
        # Import items and users in parallel
        items_import = import_data(items_dataset_arn, items_s3, f"items-import-{timestamp}")
        users_import = import_data(users_dataset_arn, users_s3, f"users-import-{timestamp}")
        
        wait_for_import(items_import)
        wait_for_import(users_import)
    
    # Step 6: Create solution
    print("\nSTEP 6: Create solution (ML model)")
    print("-" * 40)
    create_solution()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNEXT STEPS:")
    print("1. Wait for solution training to complete (30-60 min)")
    print("2. Create a campaign in AWS console")
    print("3. Update backend to use AWS Personalize API")
    print("")


if __name__ == "__main__":
    main()
