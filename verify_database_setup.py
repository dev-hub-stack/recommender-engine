#!/usr/bin/env python3
"""
Database Setup Verification Script for Recommendation Engine Service
This script verifies that the database is properly configured and all required tables exist.
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_database_url():
    """Get database URL from environment variables"""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        # Construct from individual components if DATABASE_URL not set
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        database = os.getenv('POSTGRES_DB', 'mastergroup_recommendations')
        user = os.getenv('POSTGRES_USER', 'postgres')
        password = os.getenv('POSTGRES_PASSWORD', 'postgres')
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    return db_url

def check_database_connection():
    """Test database connection"""
    try:
        db_url = get_database_url()
        logger.info(f"Testing connection to database...")
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"✓ Database connection successful: {version[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False

def check_required_tables():
    """Check if all required tables exist"""
    required_tables = [
        'order_items',
        'customer_purchases', 
        'product_pairs',
        'product_statistics',
        'customer_statistics',
        'recommendation_cache'
    ]
    
    try:
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Check each table
        missing_tables = []
        for table in required_tables:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            
            exists = cursor.fetchone()[0]
            if exists:
                logger.info(f"✓ Table '{table}' exists")
            else:
                logger.warning(f"✗ Table '{table}' missing")
                missing_tables.append(table)
        
        cursor.close()
        conn.close()
        
        return len(missing_tables) == 0, missing_tables
        
    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return False, []

def check_required_functions():
    """Check if all required functions exist"""
    required_functions = [
        'populate_order_items_from_orders',
        'rebuild_customer_purchases',
        'rebuild_product_pairs', 
        'rebuild_product_statistics',
        'rebuild_customer_statistics'
    ]
    
    try:
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        missing_functions = []
        for function in required_functions:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.routines 
                    WHERE routine_schema = 'public' 
                    AND routine_name = %s
                    AND routine_type = 'FUNCTION'
                );
            """, (function,))
            
            exists = cursor.fetchone()[0]
            if exists:
                logger.info(f"✓ Function '{function}' exists")
            else:
                logger.warning(f"✗ Function '{function}' missing")
                missing_functions.append(function)
        
        cursor.close()
        conn.close()
        
        return len(missing_functions) == 0, missing_functions
        
    except Exception as e:
        logger.error(f"Error checking functions: {e}")
        return False, []

def check_sample_data():
    """Check if there's sample data in key tables"""
    try:
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Check orders table (should exist from main system)
        cursor.execute("SELECT COUNT(*) FROM orders;")
        order_count = cursor.fetchone()[0]
        logger.info(f"✓ Orders table has {order_count} records")
        
        # Check recommendation tables
        tables_to_check = ['order_items', 'customer_purchases', 'product_statistics']
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = cursor.fetchone()[0]
                logger.info(f"✓ Table '{table}' has {count} records")
            except Exception as e:
                logger.warning(f"Could not check table '{table}': {e}")
        
        cursor.close()
        conn.close()
        
        return order_count > 0
        
    except Exception as e:
        logger.error(f"Error checking sample data: {e}")
        return False

def run_setup_script():
    """Run the setup script if tables are missing"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'setup_recommendation_tables.sql')
        if not os.path.exists(script_path):
            logger.error(f"Setup script not found at: {script_path}")
            return False
        
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        logger.info("Running setup script...")
        with open(script_path, 'r') as f:
            setup_sql = f.read()
        
        cursor.execute(setup_sql)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info("✓ Setup script executed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running setup script: {e}")
        return False

def main():
    """Main verification function"""
    logger.info("=== Recommendation Engine Database Verification ===")
    
    # Check environment variables
    if not os.getenv('DATABASE_URL') and not os.getenv('POSTGRES_HOST'):
        logger.error("No database configuration found. Please set DATABASE_URL or POSTGRES_* environment variables.")
        sys.exit(1)
    
    # Test database connection
    if not check_database_connection():
        logger.error("Cannot connect to database. Please check your configuration.")
        sys.exit(1)
    
    # Check required tables
    tables_ok, missing_tables = check_required_tables()
    
    # Check required functions  
    functions_ok, missing_functions = check_required_functions()
    
    # If missing components, offer to run setup
    if not tables_ok or not functions_ok:
        logger.warning("Missing database components detected.")
        if missing_tables:
            logger.warning(f"Missing tables: {', '.join(missing_tables)}")
        if missing_functions:
            logger.warning(f"Missing functions: {', '.join(missing_functions)}")
        
        response = input("Would you like to run the setup script? (y/N): ")
        if response.lower() == 'y':
            if run_setup_script():
                logger.info("Setup completed. Re-checking...")
                tables_ok, _ = check_required_tables()
                functions_ok, _ = check_required_functions()
            else:
                logger.error("Setup failed.")
                sys.exit(1)
    
    # Check sample data
    has_data = check_sample_data()
    
    # Final status
    if tables_ok and functions_ok:
        logger.info("✓ Database setup is complete and ready for recommendation engine!")
        if not has_data:
            logger.info("💡 To populate data, run the following functions:")
            logger.info("   SELECT populate_order_items_from_orders();")
            logger.info("   SELECT rebuild_customer_purchases();") 
            logger.info("   SELECT rebuild_product_pairs();")
            logger.info("   SELECT rebuild_product_statistics();")
            logger.info("   SELECT rebuild_customer_statistics();")
        sys.exit(0)
    else:
        logger.error("Database setup is incomplete. Please resolve the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
