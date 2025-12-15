"""
Simple Database Loader
Reads orders directly from existing database (no API calls needed)
"""
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDBLoader:
    """Load orders directly from existing database"""
    
    def __init__(self, database_url: str = None):
        """
        Initialize loader
        
        Args:
            database_url: PostgreSQL connection string
                         If None, reads from DATABASE_URL env variable
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        
        if not self.database_url:
            raise ValueError("DATABASE_URL not provided")
        
        logger.info("Initialized SimpleDBLoader")
    
    def get_connection(self):
        """Get database connection"""
        import urllib.parse as urlparse
        
        url = urlparse.urlparse(self.database_url)
        
        return psycopg2.connect(
            host=url.hostname,
            port=url.port or 5432,
            database=url.path[1:],  # Remove leading /
            user=url.username,
            password=url.password
        )
    
    def load_orders_from_db(self, 
                           start_date: str = None,
                           end_date: str = None,
                           limit: int = None) -> pd.DataFrame:
        """
        Load orders directly from database (pos_orders + oe_orders)
        
        Args:
            start_date: Filter from this date (YYYY-MM-DD)
            end_date: Filter until this date (YYYY-MM-DD)
            limit: Maximum number of orders to fetch
            
        Returns:
            DataFrame with orders
        """
        logger.info("="*60)
        logger.info("LOADING ORDERS FROM DATABASE")
        logger.info("⚠️  POS ORDERS DISABLED - ONLY LOADING OE ORDERS")
        logger.info("="*60)
        
        # Build query for POS orders (DISABLED)
        # pos_query = """
        # SELECT 
        #     id,
        #     customer_phone,
        #     customer_email,
        #     customer_name,
        #     customer_city,
        #     customer_state,
        #     customer_country,
        #     order_date,
        #     has_items,
        #     'POS' as source
        # FROM pos_orders
        # WHERE 1=1
        # """
        
        # Build query for OE orders (ONLY SOURCE)
        oe_query = """
        SELECT 
            id,
            customer_phone,
            customer_email,
            customer_name,
            customer_city,
            customer_state,
            customer_country,
            order_date,
            has_items,
            'OE' as source
        FROM oe_orders
        WHERE 1=1
        """
        
        params = []
        
        # Add date filters
        date_filter = ""
        if start_date:
            date_filter += " AND order_date >= %s"
            params.append(start_date)
            logger.info(f"Start Date: {start_date}")
        
        if end_date:
            date_filter += " AND order_date <= %s"
            params.append(end_date)
            logger.info(f"End Date: {end_date}")
        
        # Add filters to OE query only
        oe_query += date_filter
        
        # Use only OE query (POS disabled)
        combined_query = f"""
        {oe_query}
        ORDER BY order_date DESC
        """
        
        if limit:
            combined_query += f" LIMIT {limit}"
            logger.info(f"Limit: {limit}")
        
        # Execute query
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            logger.info("Executing query...")
            # Only OE query now (no UNION), so pass params once
            cursor.execute(combined_query, params)
            
            results = cursor.fetchall()
            logger.info(f"✅ Fetched {len(results)} orders from database")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Convert has_items to string if needed (for compatibility)
            if not df.empty and 'has_items' in df.columns:
                df['has_items'] = df['has_items'].astype(str)
            
            cursor.close()
            conn.close()
            
            logger.info("="*60)
            logger.info(f"Total Orders: {len(df)}")
            if not df.empty:
                logger.info(f"Date Range: {df['order_date'].min()} to {df['order_date'].max()}")
                logger.info(f"Sources: {df['source'].value_counts().to_dict()}")
                logger.info(f"Unique Cities: {df['customer_city'].nunique()}")
            logger.info("="*60)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load orders: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            logger.info("Testing database connection...")
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            logger.info("✅ Database connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {str(e)}")
            return False
    
    def get_database_stats(self) -> dict:
        """Get statistics about orders in database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get POS stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    MIN(order_date) as earliest_order,
                    MAX(order_date) as latest_order,
                    COUNT(DISTINCT customer_city) as unique_cities
                FROM pos_orders
            """)
            pos_stats = cursor.fetchone()
            
            # Get OE stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    MIN(order_date) as earliest_order,
                    MAX(order_date) as latest_order,
                    COUNT(DISTINCT customer_city) as unique_cities
                FROM oe_orders
            """)
            oe_stats = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                'pos_orders': pos_stats['total_orders'],
                'oe_orders': oe_stats['total_orders'],
                'total_orders': pos_stats['total_orders'] + oe_stats['total_orders'],
                'pos_date_range': f"{pos_stats['earliest_order']} to {pos_stats['latest_order']}" if pos_stats['earliest_order'] else None,
                'oe_date_range': f"{oe_stats['earliest_order']} to {oe_stats['latest_order']}" if oe_stats['earliest_order'] else None,
                'pos_cities': pos_stats['unique_cities'],
                'oe_cities': oe_stats['unique_cities']
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}


# Example usage
if __name__ == '__main__':
    loader = SimpleDBLoader()
    
    # Test connection
    if loader.test_connection():
        # Get stats
        stats = loader.get_database_stats()
        print("\nDatabase Statistics:")
        print(f"  POS Orders: {stats.get('pos_orders', 0):,}")
        print(f"  OE Orders: {stats.get('oe_orders', 0):,}")
        print(f"  Total Orders: {stats.get('total_orders', 0):,}")
        print(f"  POS Date Range: {stats.get('pos_date_range')}")
        print(f"  OE Date Range: {stats.get('oe_date_range')}")
        print(f"  POS Cities: {stats.get('pos_cities', 0)}")
        print(f"  OE Cities: {stats.get('oe_cities', 0)}")
        
        # Load orders
        print("\nLoading orders...")
        orders = loader.load_orders_from_db(limit=10)
        print(f"\nLoaded {len(orders)} orders")
        if not orders.empty:
            print(orders[['id', 'customer_city', 'order_date', 'source']].head())
