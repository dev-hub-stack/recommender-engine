#!/usr/bin/env python3
"""
Data Ingestion CLI
Run data ingestion from external APIs to database
"""
import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ingestion.ingestion_service import IngestionService
from src.database.connection import test_connection, init_db

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Ingest order data from external APIs to database'
    )
    
    parser.add_argument(
        '--source',
        choices=['pos', 'oe', 'all'],
        default='all',
        help='Data source to ingest (default: all)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for filtering (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for filtering (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        help='Number of days to fetch (from today backwards)'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        help='Maximum pages to fetch per source (for testing)'
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test API connections only'
    )
    
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database tables'
    )
    
    return parser.parse_args()


def main():
    """Main execution"""
    args = parse_args()
    
    # Get configuration from environment
    pos_api_url = os.getenv('POS_API_URL')
    oe_api_url = os.getenv('OE_API_URL')
    auth_token = os.getenv('API_AUTH_TOKEN')
    
    # Validate configuration
    if not all([pos_api_url, oe_api_url, auth_token]):
        logger.error("Missing required environment variables:")
        logger.error("  - POS_API_URL")
        logger.error("  - OE_API_URL")
        logger.error("  - API_AUTH_TOKEN")
        logger.error("\nPlease set these in your .env file")
        sys.exit(1)
    
    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database...")
        init_db()
        logger.info("✅ Database initialized")
        return
    
    # Test database connection
    logger.info("Testing database connection...")
    if not test_connection():
        logger.error("❌ Database connection failed")
        logger.error("Please check your DATABASE_URL in .env file")
        sys.exit(1)
    
    # Get timeout from environment
    api_timeout = int(os.getenv('API_TIMEOUT', '120'))
    
    # Initialize ingestion service
    service = IngestionService(
        pos_api_url=pos_api_url,
        oe_api_url=oe_api_url,
        auth_token=auth_token,
        timeout=api_timeout
    )
    
    # Test connections if requested
    if args.test_connection:
        logger.info("\nTesting API connections...")
        results = service.test_connections()
        
        logger.info("\nConnection Results:")
        logger.info(f"  POS API: {'✅ Success' if results['pos'] else '❌ Failed'}")
        logger.info(f"  OE API: {'✅ Success' if results['oe'] else '❌ Failed'}")
        
        service.close()
        return
    
    # Parse date filters
    start_date = None
    end_date = None
    
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        logger.info(f"Fetching last {args.days} days of data")
    else:
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Run ingestion
    try:
        if args.source == 'pos':
            stats = service.ingest_pos_orders(
                start_date=start_date,
                end_date=end_date,
                max_pages=args.max_pages
            )
        elif args.source == 'oe':
            stats = service.ingest_oe_orders(
                start_date=start_date,
                end_date=end_date,
                max_pages=args.max_pages
            )
        else:  # all
            stats = service.ingest_all_orders(
                start_date=start_date,
                end_date=end_date,
                max_pages=args.max_pages
            )
        
        logger.info("\n✅ Ingestion completed successfully")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Ingestion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        service.close()


if __name__ == '__main__':
    main()
