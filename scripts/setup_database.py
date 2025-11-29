#!/usr/bin/env python3
"""
Database Setup Script for MasterGroup Recommendation System

This script handles complete database setup:
1. For FRESH databases: Runs initial schema SQL then migrations
2. For EXISTING databases: Runs pending migrations only
3. Populates order_items from orders JSON
4. Rebuilds all recommendation tables
5. Verifies setup is complete

Usage:
    python scripts/setup_database.py [--fresh]

Options:
    --fresh    Force fresh database setup (CAUTION: will recreate tables)

Environment Variables Required:
    DATABASE_URL or individual PG_* variables
"""
import os
import sys
import subprocess
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_database_url():
    """Get database URL from environment."""
    url = os.getenv('DATABASE_URL')
    if url:
        return url
    
    # Build from individual components
    host = os.getenv('PG_HOST', 'localhost')
    port = os.getenv('PG_PORT', '5432')
    database = os.getenv('PG_DATABASE', 'mastergroup_recommendations')
    user = os.getenv('PG_USER', 'postgres')
    password = os.getenv('PG_PASSWORD', '')
    sslmode = os.getenv('PG_SSLMODE', 'prefer')
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode={sslmode}"


def run_migrations():
    """Run Alembic migrations."""
    print("\n" + "=" * 60)
    print("STEP 1: Running Database Migrations")
    print("=" * 60)
    
    os.environ['DATABASE_URL'] = get_database_url()
    result = subprocess.run(
        ['alembic', 'upgrade', 'head'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print(f"❌ Migration failed: {result.stderr}")
        return False
    
    print("✅ Migrations completed successfully")
    print(result.stdout)
    return True


def connect_db():
    """Create database connection."""
    url = get_database_url()
    # Parse URL for psycopg2
    if url.startswith('postgresql://'):
        url = url.replace('postgresql://', '')
    
    # Extract components
    if '@' in url:
        auth, rest = url.split('@', 1)
        if ':' in auth:
            user, password = auth.split(':', 1)
        else:
            user, password = auth, ''
        
        if '/' in rest:
            host_port, db_opts = rest.split('/', 1)
        else:
            host_port, db_opts = rest, ''
        
        if ':' in host_port:
            host, port = host_port.split(':', 1)
        else:
            host, port = host_port, '5432'
        
        if '?' in db_opts:
            database, opts = db_opts.split('?', 1)
            sslmode = 'require' if 'sslmode=require' in opts else 'prefer'
        else:
            database, sslmode = db_opts, 'prefer'
    else:
        host = 'localhost'
        port = '5432'
        database = 'mastergroup_recommendations'
        user = 'postgres'
        password = ''
        sslmode = 'prefer'
    
    return psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        sslmode=sslmode
    )


def populate_order_items(conn):
    """Populate order_items from orders.items_json."""
    print("\n" + "=" * 60)
    print("STEP 2: Populating Order Items")
    print("=" * 60)
    
    cursor = conn.cursor()
    
    # Check if already populated
    cursor.execute("SELECT COUNT(*) FROM order_items")
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"ℹ️  Order items already populated ({count:,} rows)")
        return True
    
    # Check if orders exist
    cursor.execute("SELECT COUNT(*) FROM orders WHERE items_json IS NOT NULL")
    orders_count = cursor.fetchone()[0]
    
    if orders_count == 0:
        print("ℹ️  No orders with items_json found. Skipping...")
        return True
    
    print(f"📊 Processing {orders_count:,} orders...")
    
    cursor.execute("SELECT populate_order_items_from_orders()")
    result = cursor.fetchone()[0]
    conn.commit()
    
    print(f"✅ Inserted {result:,} order items")
    cursor.close()
    return True


def rebuild_recommendation_tables(conn):
    """Rebuild all recommendation tables."""
    print("\n" + "=" * 60)
    print("STEP 3: Rebuilding Recommendation Tables")
    print("=" * 60)
    
    cursor = conn.cursor()
    
    # Check if order_items has data
    cursor.execute("SELECT COUNT(*) FROM order_items")
    if cursor.fetchone()[0] == 0:
        print("ℹ️  No order items found. Skipping table rebuilds...")
        return True
    
    tables = [
        ('customer_purchases', 'rebuild_customer_purchases'),
        ('product_pairs', 'rebuild_product_pairs'),
        ('product_statistics', 'rebuild_product_statistics'),
        ('customer_statistics', 'rebuild_customer_statistics'),
    ]
    
    for table_name, function_name in tables:
        print(f"  📊 Rebuilding {table_name}...")
        try:
            cursor.execute(f"SELECT {function_name}()")
            result = cursor.fetchone()[0]
            conn.commit()
            print(f"     ✅ {result:,} rows")
        except Exception as e:
            print(f"     ⚠️  Error: {e}")
            conn.rollback()
    
    cursor.close()
    return True


def verify_setup(conn):
    """Verify database setup is complete."""
    print("\n" + "=" * 60)
    print("STEP 4: Verifying Setup")
    print("=" * 60)
    
    cursor = conn.cursor()
    
    tables = [
        'orders', 'order_items', 'users', 'customer_purchases',
        'product_pairs', 'product_statistics', 'customer_statistics',
        'recommendation_cache', 'sync_metadata',
        'offline_user_recommendations', 'offline_similar_items',
        'offline_item_affinity', 'city_province_mapping'
    ]
    
    print("\n📋 Table Status:")
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            status = "✅" if count > 0 or table in ['recommendation_cache', 'sync_metadata', 'offline_user_recommendations', 'offline_similar_items', 'offline_item_affinity'] else "⚪"
            print(f"   {status} {table}: {count:,} rows")
        except Exception as e:
            print(f"   ❌ {table}: {e}")
    
    # Check admin user
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'admin@mastergroup.com'")
    admin_exists = cursor.fetchone()[0] > 0
    print(f"\n👤 Admin User: {'✅ Created' if admin_exists else '❌ Missing'}")
    
    cursor.close()
    return True


def check_tables_exist(conn):
    """Check if core tables already exist."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'orders'
        """)
        return cursor.fetchone()[0] > 0
    except:
        return False
    finally:
        cursor.close()


def run_initial_schema(conn):
    """Run initial schema SQL for fresh database."""
    print("\n" + "=" * 60)
    print("STEP 0: Creating Initial Schema (Fresh Database)")
    print("=" * 60)
    
    sql_file = Path(__file__).parent.parent / 'sql' / 'legacy' / 'deploy_database.sql'
    
    if not sql_file.exists():
        print(f"⚠️  Schema file not found: {sql_file}")
        print("   Skipping initial schema - migrations will handle it")
        return True
    
    print(f"📄 Running {sql_file.name}...")
    
    try:
        with open(sql_file, 'r') as f:
            sql = f.read()
        
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        cursor.close()
        print("✅ Initial schema created")
        return True
    except Exception as e:
        print(f"⚠️  Schema creation error: {e}")
        conn.rollback()
        return True  # Continue anyway - migrations may still work


def seed_admin_user(conn):
    """Seed the admin user if not exists."""
    print("\n" + "=" * 60)
    print("STEP 5: Seeding Admin User")
    print("=" * 60)
    
    cursor = conn.cursor()
    
    # Check if admin exists
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'admin@mastergroup.com'")
    if cursor.fetchone()[0] > 0:
        print("ℹ️  Admin user already exists")
        cursor.close()
        return True
    
    # Create admin user (password: MG@2024#Secure!Pass)
    try:
        cursor.execute("""
            INSERT INTO users (email, password_hash, full_name, is_active) 
            VALUES (
                'admin@mastergroup.com',
                '$2b$12$8ALlBQw1UrHePD2QyYRy0uGz/mMEOsay4HzCwvPjMt8nOmGlQ/8MO',
                'Admin User',
                true
            )
            ON CONFLICT (email) DO NOTHING
        """)
        conn.commit()
        print("✅ Admin user created")
    except Exception as e:
        print(f"⚠️  Could not create admin user: {e}")
    
    cursor.close()
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='MasterGroup Database Setup')
    parser.add_argument('--fresh', action='store_true', 
                       help='Force fresh database setup')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MasterGroup Recommendation System - Database Setup")
    print("=" * 60)
    
    # Connect to database first to check state
    try:
        conn = connect_db()
    except Exception as e:
        print(f"\n❌ Cannot connect to database: {e}")
        sys.exit(1)
    
    # Check if this is a fresh database
    tables_exist = check_tables_exist(conn)
    
    if not tables_exist or args.fresh:
        print("\n📦 Fresh database detected - running full setup...")
        run_initial_schema(conn)
    else:
        print("\n📦 Existing database detected - running migrations only...")
    
    conn.close()
    
    # Step 1: Run migrations
    if not run_migrations():
        print("\n❌ Setup failed at migrations step")
        sys.exit(1)
    
    # Reconnect for remaining steps
    try:
        conn = connect_db()
    except Exception as e:
        print(f"\n❌ Cannot connect to database: {e}")
        sys.exit(1)
    
    try:
        # Step 2: Populate order items
        populate_order_items(conn)
        
        # Step 3: Rebuild recommendation tables
        rebuild_recommendation_tables(conn)
        
        # Step 4: Verify setup
        verify_setup(conn)
        
        # Step 5: Seed admin user
        seed_admin_user(conn)
        
    finally:
        conn.close()
    
    print("\n" + "=" * 60)
    print("✅ DATABASE SETUP COMPLETE!")
    print("=" * 60)
    print("""
Next Steps:
1. Configure .env file with API credentials
2. Start the API server: uvicorn src.main:app --host 0.0.0.0 --port 8001
3. Login to dashboard with: admin@mastergroup.com / MG@2024#Secure!Pass
4. Change the admin password after first login!
""")


if __name__ == '__main__':
    main()
