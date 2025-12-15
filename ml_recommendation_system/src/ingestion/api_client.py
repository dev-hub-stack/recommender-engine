"""
API Client for fetching orders from external APIs
"""
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClient:
    """Client for fetching order data from external APIs"""
    
    def __init__(self, base_url: str, auth_token: str, timeout: int = 30):
        """
        Initialize API Client
        
        Args:
            base_url: Base URL for the API
            auth_token: Authentication token (Bearer token)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': auth_token,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def fetch_orders(self, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     page: int = 1,
                     per_page: int = 100) -> Dict:
        """
        Fetch orders from API with pagination
        
        Args:
            start_date: Start date for filtering orders
            end_date: End date for filtering orders
            page: Page number (1-indexed)
            per_page: Number of records per page
            
        Returns:
            Dictionary with 'data', 'total', 'page', 'per_page'
        """
        # Build query parameters
        params = {
            'page': page,
            'per_page': per_page
        }
        
        if start_date:
            params['start_date'] = start_date.strftime('%Y-%m-%d')
        
        if end_date:
            params['end_date'] = end_date.strftime('%Y-%m-%d')
        
        try:
            logger.info(f"Fetching orders: page={page}, per_page={per_page}")
            
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"✅ Fetched {len(data.get('data', []))} orders")
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ Request timeout after {self.timeout}s")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            raise
    
    def fetch_all_orders(self,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        per_page: int = 100,
                        max_pages: Optional[int] = None) -> List[Dict]:
        """
        Fetch all orders with automatic pagination
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            per_page: Records per page
            max_pages: Maximum pages to fetch (None = all)
            
        Returns:
            List of all order dictionaries
        """
        all_orders = []
        page = 1
        
        logger.info("="*60)
        logger.info("FETCHING ALL ORDERS FROM API")
        logger.info("="*60)
        
        if start_date:
            logger.info(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
        if end_date:
            logger.info(f"End Date: {end_date.strftime('%Y-%m-%d')}")
        
        while True:
            # Check max pages limit
            if max_pages and page > max_pages:
                logger.info(f"Reached max pages limit: {max_pages}")
                break
            
            try:
                # Fetch page
                result = self.fetch_orders(
                    start_date=start_date,
                    end_date=end_date,
                    page=page,
                    per_page=per_page
                )
                
                # Extract orders
                orders = result.get('data', [])
                
                if not orders:
                    logger.info("No more orders to fetch")
                    break
                
                all_orders.extend(orders)
                
                # Check if there are more pages
                total = result.get('total', 0)
                current_count = len(all_orders)
                
                logger.info(f"Progress: {current_count}/{total} orders fetched")
                
                # Break if we've fetched all
                if current_count >= total:
                    break
                
                page += 1
                
                # Rate limiting - be nice to the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {str(e)}")
                # Continue to next page or break based on error type
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                    # No more pages
                    break
                else:
                    # Other errors - re-raise
                    raise
        
        logger.info("="*60)
        logger.info(f"✅ Total orders fetched: {len(all_orders)}")
        logger.info("="*60)
        
        return all_orders
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Testing connection to: {self.base_url}")
            
            # Try to fetch first page with minimal data
            result = self.fetch_orders(page=1, per_page=1)
            
            logger.info("✅ API connection successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ API connection failed: {str(e)}")
            return False
    
    def close(self):
        """Close the session"""
        self.session.close()


class POSAPIClient(APIClient):
    """Client specifically for POS orders API"""
    
    def __init__(self, base_url: str, auth_token: str, timeout: int = 120):
        super().__init__(base_url, auth_token, timeout=timeout)
        logger.info("Initialized POS API Client")


class OEAPIClient(APIClient):
    """Client specifically for OE orders API"""
    
    def __init__(self, base_url: str, auth_token: str, timeout: int = 120):
        super().__init__(base_url, auth_token, timeout=timeout)
        logger.info("Initialized OE API Client")
