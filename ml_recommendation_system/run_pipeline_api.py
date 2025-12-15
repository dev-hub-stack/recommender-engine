#!/usr/bin/env python3
"""
Run Data Pipeline - API Mode (Year-by-Year)
Fetches ALL orders from Masterverse APIs and stores in database
Date Range: 2000-01-01 to Yesterday 23:59:59
Strategy: Fetch year-by-year to avoid API timeouts
"""
import sys
import os
from datetime import datetime, timedelta

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the ml-recommendation-system directory to path
sys.path.insert(0, script_dir)

# Now import from src package
from src.ingestion.ingestion_service import IngestionService
from src.database.connection import get_db
from sqlalchemy import text
from dotenv import load_dotenv
import logging

# Load .env from ml-recommendation-system directory
load_dotenv(os.path.join(script_dir, '.env'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_year_by_year(service, source_type='pos'):
    """
    Fetch orders year by year to avoid API timeouts
    
    Args:
        service: IngestionService instance
        source_type: 'pos' or 'oe'
    
    Returns:
        Combined statistics
    """
    # Calculate years from 2000 to current year
    current_year = datetime.now().year
    yesterday = datetime.now() - timedelta(days=1)
    yesterday = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
    
    total_stats = {
        'fetched': 0,
        'inserted': 0,
        'updated': 0,
        'failed': 0,
        'skipped': 0
    }
    
    print(f"\n{'='*70}")
    print(f"FETCHING {source_type.upper()} ORDERS YEAR-BY-YEAR (2000-{current_year})")
    print(f"{'='*70}")
    
    for year in range(2000, current_year + 1):
        # Define year boundaries
        year_start = datetime(year, 1, 1, 0, 0, 0)
        
        if year == current_year:
            # For current year, use yesterday
            year_end = yesterday
        else:
            # For past years, use Dec 31
            year_end = datetime(year, 12, 31, 23, 59, 59)
        
        print(f"\n[Year {year}] Fetching {source_type.upper()} orders...")
        print(f"  Date range: {year_start.strftime('%Y-%m-%d')} to {year_end.strftime('%Y-%m-%d')}")
        
        try:
            if source_type == 'pos':
                year_stats = service.ingest_pos_orders(
                    start_date=year_start,
                    end_date=year_end
                )
            else:  # oe
                year_stats = service.ingest_oe_orders(
                    start_date=year_start,
                    end_date=year_end
                )
            
            # Accumulate stats
            total_stats['fetched'] += year_stats['fetched']
            total_stats['inserted'] += year_stats['inserted']
            total_stats['updated'] += year_stats['updated']
            total_stats['failed'] += year_stats['failed']
            total_stats['skipped'] += year_stats['skipped']
            
            print(f"  ✅ Year {year}: Fetched {year_stats['fetched']}, Inserted {year_stats['inserted']}")
            
        except Exception as e:
            print(f"  ❌ Year {year} failed: {str(e)}")
            print(f"  Continuing with next year...")
            continue
    
    print(f"\n{'='*70}")
    print(f"{source_type.upper()} YEAR-BY-YEAR FETCH COMPLETE")
    print(f"  Total Fetched: {total_stats['fetched']:,}")
    print(f"  Total Inserted: {total_stats['inserted']:,}")
    print(f"  Total Updated: {total_stats['updated']:,}")
    print(f"  Total Failed: {total_stats['failed']:,}")
    print(f"{'='*70}")
    
    return total_stats


def main():
    """Run the data pipeline - API mode with year-by-year fetching"""
    
    print("\n" + "="*70)
    print("ML RECOMMENDATION SYSTEM - DATA PIPELINE (YEAR-BY-YEAR)")
    print("="*70)
    
    # Get API configuration from environment
    pos_api_url = os.getenv('POS_API_URL')
    oe_api_url = os.getenv('OE_API_URL')
    auth_token = os.getenv('API_AUTH_TOKEN')
    api_timeout = int(os.getenv('API_TIMEOUT', '120'))
    
    if not all([pos_api_url, oe_api_url, auth_token]):
        print("\n❌ Error: Missing API configuration in .env file")
        print("Required variables:")
        print("  - POS_API_URL")
        print("  - OE_API_URL")
        print("  - API_AUTH_TOKEN")
        sys.exit(1)
    
    print(f"\nAPI Configuration:")
    print(f"  POS API: {pos_api_url}")
    print(f"  OE API: {oe_api_url}")
    print(f"  Timeout: {api_timeout}s")
    
    print(f"\nStrategy: Year-by-year fetching (2000 to {datetime.now().year})")
    print(f"  This avoids API timeouts with large date ranges")
    
    print(f"\n⚠️  WARNING: This will DELETE all existing records in pos_orders and oe_orders tables")
    print(f"⚠️  and fetch fresh data from APIs year-by-year")
    print(f"\nPress Ctrl+C within 5 seconds to cancel...")
    
    try:
        import time
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(0)
    
    # Initialize ingestion service
    service = IngestionService(
        pos_api_url=pos_api_url,
        oe_api_url=oe_api_url,
        auth_token=auth_token,
        timeout=api_timeout
    )
    
    try:
        # Clear existing data
        print("\n[STEP 1] Clearing existing data from database...")
        with get_db() as db:
            # Get counts before deletion
            pos_count = db.execute(text("SELECT COUNT(*) FROM pos_orders")).scalar()
            oe_count = db.execute(text("SELECT COUNT(*) FROM oe_orders")).scalar()
            
            print(f"  Current POS orders: {pos_count:,}")
            print(f"  Current OE orders: {oe_count:,}")
            
            # Delete all records
            db.execute(text("DELETE FROM pos_orders"))
            db.execute(text("DELETE FROM oe_orders"))
            db.commit()
            print("  ✅ Existing data cleared")
        
        # Fetch POS orders year-by-year
        print("\n[STEP 2] Fetching POS orders year-by-year...")
        pos_stats = fetch_year_by_year(service, source_type='pos')
        
        # Fetch OE orders year-by-year
        print("\n[STEP 3] Fetching OE orders year-by-year...")
        oe_stats = fetch_year_by_year(service, source_type='oe')
        
        # Create combined stats
        stats = {
            'pos': pos_stats,
            'oe': oe_stats
        }
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*70)
        print(f"\nPOS Orders:")
        print(f"  Fetched: {stats['pos']['fetched']:,}")
        print(f"  Inserted: {stats['pos']['inserted']:,}")
        print(f"  Updated: {stats['pos']['updated']:,}")
        print(f"  Failed: {stats['pos']['failed']:,}")
        
        print(f"\nOE Orders:")
        print(f"  Fetched: {stats['oe']['fetched']:,}")
        print(f"  Inserted: {stats['oe']['inserted']:,}")
        print(f"  Updated: {stats['oe']['updated']:,}")
        print(f"  Failed: {stats['oe']['failed']:,}")
        
        total_orders = stats['pos']['inserted'] + stats['oe']['inserted']
        print(f"\n📊 Total Orders in Database: {total_orders:,}")
        
        print(f"\n✅ Fresh data loaded into database")
        print(f"\nNext steps:")
        print(f"  1. Data cleaning (you'll handle this)")
        print(f"  2. Model training: python run_training.py")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        service.close()


if __name__ == '__main__':
    main()
