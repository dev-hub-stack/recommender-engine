#!/usr/bin/env python3
"""
Restore database from S3 backup to local PostgreSQL.
Downloads JSON files from S3 and imports them into your local database.

Usage:
    python restore_from_s3.py [--bucket BUCKET] [--prefix PREFIX] [--date DATE]

Examples:
    python restore_from_s3.py                                    # Restore latest backup
    python restore_from_s3.py --date 20251209                   # Restore specific date
    python restore_from_s3.py --bucket my-bucket --prefix backups/  # Custom bucket/prefix
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
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

# Load .env file
load_env()

# Local Database - for restoration (native PostgreSQL, not Docker)
LOCAL_CONFIG = {
    "host": os.environ.get("LOCAL_PG_HOST", "localhost"),
    "port": int(os.environ.get("LOCAL_PG_PORT", 5432)),  # Default PostgreSQL port
    "database": os.environ.get("LOCAL_PG_DB", "mastergroup_recommendations"),
    "user": os.environ.get("LOCAL_PG_USER", "postgres"),
    "password": os.environ.get("LOCAL_PG_PASSWORD", "")  # Empty password for local
}

# Default S3 bucket
DEFAULT_BUCKET = "mastergroup-db-backups-303498144074"
DEFAULT_PREFIX = "database-exports/"

def get_latest_backup_date(s3_client, bucket, prefix):
    """Find the latest backup date in S3"""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    
    dates = []
    for common_prefix in response.get('CommonPrefixes', []):
        folder = common_prefix['Prefix'].replace(prefix, '').rstrip('/')
        # Accept folders like 20251209 or 20251209_004821
        if folder and folder[:8].isdigit():
            dates.append(folder)
    
    if not dates:
        return None
    
    # Return the latest date (sorted alphabetically works for YYYYMMDD format)
    return sorted(dates)[-1]

def get_table_files(s3_client, bucket, prefix, date):
    """Get all table files for a specific backup date"""
    full_prefix = f"{prefix}{date}/"
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=full_prefix)
    
    files = {}
    for obj in response.get('Contents', []):
        key = obj['Key']
        filename = key.split('/')[-1]
        
        if filename.endswith('.json.gz') and '_part' in filename:
            table_name = filename.rsplit('_part', 1)[0]
            if table_name not in files:
                files[table_name] = []
            files[table_name].append(key)
    
    # Sort files by part number
    for table in files:
        files[table].sort()
    
    return files

def create_table_from_data(cursor, conn, table_name, sample_row):
    """Create table based on sample data structure"""
    columns = []
    for key, value in sample_row.items():
        if isinstance(value, bool):
            col_type = "BOOLEAN"
        elif isinstance(value, int):
            col_type = "BIGINT"
        elif isinstance(value, float):
            col_type = "NUMERIC"
        elif isinstance(value, list):
            col_type = "JSONB"
        elif isinstance(value, dict):
            col_type = "JSONB"
        else:
            col_type = "TEXT"
        columns.append(f'"{key}" {col_type}')
    
    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  ' + ',\n  '.join(columns) + '\n)'
    
    try:
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
        cursor.execute(create_sql)
        conn.commit()
        return True
    except Exception as e:
        print(f"  ⚠️ Error creating table: {e}")
        conn.rollback()
        return False

def restore_table(s3_client, cursor, conn, bucket, table_name, file_keys):
    """Restore a single table from S3 files"""
    print(f"\n📦 Restoring {table_name}...")
    print(f"  📁 {len(file_keys)} file(s) to process")
    
    total_rows = 0
    table_created = False
    
    for i, key in enumerate(file_keys):
        # Download and decompress
        response = s3_client.get_object(Bucket=bucket, Key=key)
        compressed_data = response['Body'].read()
        json_data = gzip.decompress(compressed_data).decode('utf-8')
        rows = json.loads(json_data)
        
        if not rows:
            continue
        
        # Create table on first file
        if not table_created:
            if not create_table_from_data(cursor, conn, table_name, rows[0]):
                return 0
            table_created = True
        
        # Get column names
        columns = list(rows[0].keys())
        
        # Prepare values - handle JSONB columns
        values = []
        for row in rows:
            row_values = []
            for col in columns:
                val = row.get(col)
                if isinstance(val, (list, dict)):
                    row_values.append(json.dumps(val))
                else:
                    row_values.append(val)
            values.append(tuple(row_values))
        
        # Insert data
        try:
            col_names = ", ".join([f'"{c}"' for c in columns])
            insert_sql = f'INSERT INTO "{table_name}" ({col_names}) VALUES %s'
            execute_values(cursor, insert_sql, values, page_size=1000)
            conn.commit()
            total_rows += len(rows)
            print(f"  ✅ Part {i+1}/{len(file_keys)}: {len(rows):,} rows")
        except Exception as e:
            print(f"  ❌ Error inserting data: {e}")
            conn.rollback()
            break
    
    print(f"  📊 Total: {total_rows:,} rows restored")
    return total_rows

def main():
    parser = argparse.ArgumentParser(description='Restore database from S3')
    parser.add_argument('--bucket', type=str, default=DEFAULT_BUCKET, help='S3 bucket name')
    parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX, help='S3 key prefix')
    parser.add_argument('--date', type=str, help='Backup date (e.g., 20251209)')
    parser.add_argument('--tables', type=str, help='Comma-separated list of tables to restore')
    parser.add_argument('--local-host', type=str, default='localhost', help='Local database host')
    parser.add_argument('--local-port', type=int, default=5433, help='Local database port')
    args = parser.parse_args()
    
    # Update local config
    LOCAL_CONFIG['host'] = args.local_host
    LOCAL_CONFIG['port'] = args.local_port
    
    print("=" * 60)
    print("📥 DATABASE RESTORE FROM S3")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🪣 Bucket: {args.bucket}")
    print(f"💻 Local DB: {LOCAL_CONFIG['host']}:{LOCAL_CONFIG['port']}")
    print("=" * 60)
    
    # Connect to S3 - use default AWS CLI credentials
    if 'AWS_ACCESS_KEY_ID' in os.environ:
        del os.environ['AWS_ACCESS_KEY_ID']
    if 'AWS_SECRET_ACCESS_KEY' in os.environ:
        del os.environ['AWS_SECRET_ACCESS_KEY']
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.head_bucket(Bucket=args.bucket)
        print(f"✅ Connected to S3 bucket: {args.bucket}")
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        sys.exit(1)
    
    # Find backup date
    backup_date = args.date
    if not backup_date:
        backup_date = get_latest_backup_date(s3_client, args.bucket, args.prefix)
        if not backup_date:
            print("❌ No backups found in S3")
            sys.exit(1)
        print(f"📅 Using latest backup: {backup_date}")
    else:
        print(f"📅 Using specified backup: {backup_date}")
    
    # Get table files
    table_files = get_table_files(s3_client, args.bucket, args.prefix, backup_date)
    
    if not table_files:
        print(f"❌ No backup files found for date {backup_date}")
        sys.exit(1)
    
    # Filter tables if specified
    if args.tables:
        requested_tables = [t.strip() for t in args.tables.split(',')]
        table_files = {k: v for k, v in table_files.items() if k in requested_tables}
    
    print(f"\n📋 Tables to restore: {len(table_files)}")
    for table in table_files:
        print(f"   • {table} ({len(table_files[table])} files)")
    
    # Connect to local database
    try:
        conn = psycopg2.connect(**LOCAL_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        print(f"\n✅ Connected to local database")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   Tip: Make sure Docker is running with 'docker-compose up -d'")
        sys.exit(1)
    
    # Restore each table
    total_restored = 0
    for table_name, file_keys in table_files.items():
        rows = restore_table(s3_client, cursor, conn, args.bucket, table_name, file_keys)
        total_restored += rows
    
    # Close connections
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ RESTORE COMPLETE!")
    print(f"📊 Total tables: {len(table_files)}")
    print(f"📊 Total rows: {total_restored:,}")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n🚀 Your local database is now ready!")
    print("   Test with: psql -h localhost -p 5433 -U postgres -d mastergroup_recommendations")

if __name__ == "__main__":
    main()
