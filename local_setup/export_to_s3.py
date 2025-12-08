#!/usr/bin/env python3
"""
Export database tables to S3 as JSON files.
This is an alternative to pg_dump when you need portable data.
Reads configuration from .env file.

Usage:
    python export_to_s3.py [--bucket BUCKET] [--prefix PREFIX] [--tables TABLE1,TABLE2]

Examples:
    python export_to_s3.py                                    # Export all to default bucket
    python export_to_s3.py --bucket my-bucket --prefix backups/  # Custom bucket/prefix
    python export_to_s3.py --tables orders,order_items        # Specific tables only
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import boto3
import json
import gzip
import argparse
from datetime import datetime
from io import BytesIO
import sys
import os
from pathlib import Path

def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())
        print(f"✅ Loaded environment from {env_path}")
    else:
        print(f"⚠️ No .env file found at {env_path}")

# Load .env file
load_env()

# Production Database - from .env
PROD_CONFIG = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "database": os.environ.get("PG_DB", "mastergroup_recommendations"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", ""),
    "sslmode": "require"
}

# AWS Configuration - from .env
AWS_CONFIG = {
    "access_key": os.environ.get("AWS_ACCESS_KEY_ID"),
    "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    "region": os.environ.get("AWS_REGION", "us-east-1")
}

# Default S3 bucket - from .env
DEFAULT_BUCKET = os.environ.get("PERSONALIZE_S3_BUCKET", "mastergroup-personalize-data")
DEFAULT_PREFIX = "database-exports/"

def json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default"""
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def get_tables(cursor):
    """Get list of all tables"""
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    return [row['table_name'] for row in cursor.fetchall()]

def export_table_to_s3(cursor, s3_client, bucket, prefix, table_name, batch_size=10000):
    """Export a single table to S3 as gzipped JSON"""
    print(f"\n📦 Exporting {table_name}...")
    
    # Get row count
    cursor.execute(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
    total_rows = cursor.fetchone()['cnt']
    print(f"  📊 {total_rows:,} rows")
    
    if total_rows == 0:
        print(f"  ⏭️ Skipping empty table")
        return 0
    
    # Export in batches
    offset = 0
    part = 0
    total_exported = 0
    
    while offset < total_rows:
        # Fetch batch
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {batch_size} OFFSET {offset}')
        rows = cursor.fetchall()
        
        if not rows:
            break
        
        # Convert to JSON
        json_data = json.dumps(rows, default=json_serializer, ensure_ascii=False)
        
        # Compress
        compressed = gzip.compress(json_data.encode('utf-8'))
        
        # Upload to S3
        timestamp = datetime.now().strftime('%Y%m%d')
        key = f"{prefix}{timestamp}/{table_name}_part{part:04d}.json.gz"
        
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=compressed,
                ContentType='application/gzip',
                Metadata={
                    'table': table_name,
                    'rows': str(len(rows)),
                    'part': str(part),
                    'exported_at': datetime.now().isoformat()
                }
            )
            total_exported += len(rows)
            print(f"  ✅ Uploaded {key} ({len(rows):,} rows, {len(compressed)/1024:.1f} KB)")
        except Exception as e:
            print(f"  ❌ Failed to upload: {e}")
            return total_exported
        
        offset += batch_size
        part += 1
    
    return total_exported

def create_manifest(s3_client, bucket, prefix, tables_exported, timestamp):
    """Create a manifest file listing all exported files"""
    manifest = {
        "export_timestamp": timestamp,
        "database": PROD_CONFIG["database"],
        "tables": tables_exported,
        "total_tables": len(tables_exported),
        "total_rows": sum(t["rows"] for t in tables_exported)
    }
    
    key = f"{prefix}{timestamp}/manifest.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(manifest, indent=2),
        ContentType='application/json'
    )
    print(f"\n📋 Manifest created: s3://{bucket}/{key}")

def main():
    parser = argparse.ArgumentParser(description='Export database to S3')
    parser.add_argument('--bucket', type=str, default=DEFAULT_BUCKET, help='S3 bucket name')
    parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX, help='S3 key prefix')
    parser.add_argument('--tables', type=str, help='Comma-separated list of tables')
    parser.add_argument('--batch-size', type=int, default=10000, help='Rows per file')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print("📤 DATABASE EXPORT TO S3")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🪣 Bucket: {args.bucket}")
    print(f"📁 Prefix: {args.prefix}{timestamp}/")
    print("=" * 60)
    
    # Connect to database
    try:
        conn = psycopg2.connect(**PROD_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)
    
    # Connect to S3 - use default AWS CLI credentials (not .env)
    # Clear .env AWS credentials to use local AWS CLI profile
    if 'AWS_ACCESS_KEY_ID' in os.environ:
        del os.environ['AWS_ACCESS_KEY_ID']
    if 'AWS_SECRET_ACCESS_KEY' in os.environ:
        del os.environ['AWS_SECRET_ACCESS_KEY']
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        # Test connection
        s3_client.head_bucket(Bucket=args.bucket)
        print(f"✅ Connected to S3 bucket: {args.bucket}")
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        print("   Tip: Make sure you have AWS CLI configured with 'aws configure'")
        sys.exit(1)
    
    # Get tables
    if args.tables:
        tables = [t.strip() for t in args.tables.split(',')]
    else:
        tables = get_tables(cursor)
    
    print(f"\n📋 Tables to export: {len(tables)}")
    
    # Export each table
    tables_exported = []
    for table in tables:
        rows = export_table_to_s3(
            cursor, s3_client, args.bucket, 
            args.prefix, table, args.batch_size
        )
        tables_exported.append({"name": table, "rows": rows})
    
    # Create manifest
    create_manifest(s3_client, args.bucket, args.prefix, tables_exported, timestamp)
    
    # Close connections
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ EXPORT COMPLETE!")
    print(f"📊 Total tables: {len(tables_exported)}")
    print(f"📊 Total rows: {sum(t['rows'] for t in tables_exported):,}")
    print(f"📁 Location: s3://{args.bucket}/{args.prefix}{timestamp}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
