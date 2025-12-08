#!/usr/bin/env python3
"""
Sync data from Production Lightsail PostgreSQL to Local PostgreSQL
This script copies all tables from production to your local database.
Reads configuration from .env file.

Usage:
    python sync_from_production.py [--tables TABLE1,TABLE2] [--limit N]

Examples:
    python sync_from_production.py                    # Sync all tables
    python sync_from_production.py --limit 10000     # Limit rows per table
    python sync_from_production.py --tables orders,order_items  # Specific tables
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import argparse
import json
from datetime import datetime
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

# Production Database (Lightsail) - from .env
PROD_CONFIG = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "database": os.environ.get("PG_DB", "mastergroup_recommendations"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", ""),
    "sslmode": "require"
}

# Local Database (Docker)
LOCAL_CONFIG = {
    "host": os.environ.get("LOCAL_PG_HOST", "localhost"),
    "port": int(os.environ.get("LOCAL_PG_PORT", 5433)),  # Docker mapped port
    "database": os.environ.get("LOCAL_PG_DB", "mastergroup_recommendations"),
    "user": os.environ.get("LOCAL_PG_USER", "postgres"),
    "password": os.environ.get("LOCAL_PG_PASSWORD", "MasterGroup2024Local!")
}

def get_connection(config, name="database"):
    """Create database connection"""
    try:
        conn = psycopg2.connect(**config)
        print(f"✅ Connected to {name}")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to {name}: {e}")
        sys.exit(1)

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

def get_table_schema(cursor, table_name):
    """Get CREATE TABLE statement"""
    cursor.execute(f"""
        SELECT column_name, data_type, is_nullable, column_default,
               character_maximum_length, numeric_precision
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
    """, (table_name,))
    return cursor.fetchall()

def create_table_if_not_exists(prod_cursor, local_cursor, local_conn, table_name):
    """Create table in local database if it doesn't exist"""
    # Get columns from production
    columns = get_table_schema(prod_cursor, table_name)
    
    if not columns:
        print(f"  ⚠️ No columns found for {table_name}")
        return False
    
    # Build CREATE TABLE statement
    column_defs = []
    for col in columns:
        col_name = col['column_name']
        data_type = col['data_type']
        
        # Handle special types
        if data_type == 'character varying':
            max_len = col['character_maximum_length']
            data_type = f"VARCHAR({max_len})" if max_len else "TEXT"
        elif data_type == 'numeric':
            data_type = "NUMERIC"
        elif data_type == 'ARRAY':
            data_type = "TEXT[]"
        elif data_type == 'jsonb':
            data_type = "JSONB"
        elif data_type == 'json':
            data_type = "JSON"
        
        nullable = "" if col['is_nullable'] == 'YES' else " NOT NULL"
        default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
        
        column_defs.append(f'"{col_name}" {data_type}{nullable}{default}')
    
    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  ' + ',\n  '.join(column_defs) + '\n)'
    
    try:
        local_cursor.execute(create_sql)
        local_conn.commit()
        return True
    except Exception as e:
        print(f"  ⚠️ Error creating table {table_name}: {e}")
        local_conn.rollback()
        return False

def sync_table(prod_cursor, local_cursor, local_conn, table_name, limit=None):
    """Sync a single table from production to local"""
    print(f"\n📦 Syncing {table_name}...")
    
    # Create table if not exists
    if not create_table_if_not_exists(prod_cursor, local_cursor, local_conn, table_name):
        return 0
    
    # Get row count
    prod_cursor.execute(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
    total_rows = prod_cursor.fetchone()['cnt']
    print(f"  📊 Production has {total_rows:,} rows")
    
    if total_rows == 0:
        print(f"  ⏭️ Skipping empty table")
        return 0
    
    # Clear local table
    local_cursor.execute(f'TRUNCATE TABLE "{table_name}" CASCADE')
    local_conn.commit()
    
    # Fetch data in batches
    batch_size = 5000
    offset = 0
    total_synced = 0
    max_rows = limit if limit else total_rows
    
    while offset < min(total_rows, max_rows):
        # Fetch batch
        query = f'SELECT * FROM "{table_name}" LIMIT {batch_size} OFFSET {offset}'
        prod_cursor.execute(query)
        rows = prod_cursor.fetchall()
        
        if not rows:
            break
        
        # Get column names
        columns = [desc[0] for desc in prod_cursor.description]
        
        # Insert into local
        try:
            insert_sql = f'INSERT INTO "{table_name}" ({", ".join([f\'"{c}\'" for c in columns])}) VALUES %s'
            values = [tuple(row[col] for col in columns) for row in rows]
            execute_values(local_cursor, insert_sql, values, page_size=1000)
            local_conn.commit()
            
            total_synced += len(rows)
            print(f"  ✅ Synced {total_synced:,}/{min(total_rows, max_rows):,} rows", end='\r')
        except Exception as e:
            print(f"\n  ❌ Error inserting data: {e}")
            local_conn.rollback()
            break
        
        offset += batch_size
    
    print(f"  ✅ Synced {total_synced:,} rows                    ")
    return total_synced

def main():
    parser = argparse.ArgumentParser(description='Sync production database to local')
    parser.add_argument('--tables', type=str, help='Comma-separated list of tables to sync')
    parser.add_argument('--limit', type=int, help='Limit rows per table')
    parser.add_argument('--local-host', type=str, default='localhost', help='Local database host')
    parser.add_argument('--local-port', type=int, default=5433, help='Local database port')
    args = parser.parse_args()
    
    # Update local config if provided
    LOCAL_CONFIG['host'] = args.local_host
    LOCAL_CONFIG['port'] = args.local_port
    
    print("=" * 60)
    print("🔄 PRODUCTION TO LOCAL DATABASE SYNC")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Production: {PROD_CONFIG['host']}")
    print(f"💻 Local: {LOCAL_CONFIG['host']}:{LOCAL_CONFIG['port']}")
    print("=" * 60)
    
    # Connect to databases
    prod_conn = get_connection(PROD_CONFIG, "Production (Lightsail)")
    local_conn = get_connection(LOCAL_CONFIG, "Local (Docker)")
    
    prod_cursor = prod_conn.cursor(cursor_factory=RealDictCursor)
    local_cursor = local_conn.cursor(cursor_factory=RealDictCursor)
    
    # Get tables to sync
    if args.tables:
        tables = [t.strip() for t in args.tables.split(',')]
    else:
        tables = get_tables(prod_cursor)
    
    print(f"\n📋 Tables to sync: {len(tables)}")
    for t in tables:
        print(f"   • {t}")
    
    # Sync each table
    total_synced = 0
    for table in tables:
        synced = sync_table(prod_cursor, local_cursor, local_conn, table, args.limit)
        total_synced += synced
    
    # Close connections
    prod_cursor.close()
    local_cursor.close()
    prod_conn.close()
    local_conn.close()
    
    print("\n" + "=" * 60)
    print("✅ SYNC COMPLETE!")
    print(f"📊 Total rows synced: {total_synced:,}")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
