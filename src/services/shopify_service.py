"""
Shopify Integration Service
============================

Connects MasterGroup recommendation system with Shopify store.

Features:
- Fetch products from Shopify
- Map Shopify product IDs to recommendation system
- Sync Shopify orders for training data
- Webhook handlers for real-time updates
"""

import os
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Shopify Configuration (from environment or direct)
SHOPIFY_STORE = os.getenv('SHOPIFY_STORE', 'masterverse-project.myshopify.com')
SHOPIFY_API_KEY = os.getenv('SHOPIFY_API_KEY', '27d94ff1eaf5dc8b2765f61a9461f727')
SHOPIFY_API_SECRET = os.getenv('SHOPIFY_API_SECRET', '')  # Set in .env
SHOPIFY_ACCESS_TOKEN = os.getenv('SHOPIFY_ACCESS_TOKEN', '')  # Admin API access token
SHOPIFY_API_VERSION = os.getenv('SHOPIFY_API_VERSION', '2024-01')


class ShopifyService:
    """Service for Shopify store integration"""
    
    def __init__(self, store: str = None, access_token: str = None):
        self.store = store or SHOPIFY_STORE
        self.access_token = access_token or SHOPIFY_ACCESS_TOKEN
        self.base_url = f"https://{self.store}/admin/api/{SHOPIFY_API_VERSION}"
        
        self.headers = {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json"
        }
    
    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make authenticated request to Shopify API"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, params=data, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Shopify API error: {e}")
            raise
    
    # =========================================================================
    # PRODUCTS
    # =========================================================================
    
    def get_products(self, limit: int = 250, since_id: int = None) -> List[dict]:
        """Fetch products from Shopify"""
        params = {"limit": limit}
        if since_id:
            params["since_id"] = since_id
        
        result = self._make_request("GET", "products.json", params)
        return result.get("products", [])
    
    def get_all_products(self) -> List[dict]:
        """Fetch all products (handles pagination)"""
        all_products = []
        since_id = None
        
        while True:
            products = self.get_products(limit=250, since_id=since_id)
            if not products:
                break
            
            all_products.extend(products)
            since_id = products[-1]["id"]
            
            if len(products) < 250:
                break
        
        logger.info(f"Fetched {len(all_products)} products from Shopify")
        return all_products
    
    def get_product(self, product_id: int) -> dict:
        """Get single product by ID"""
        result = self._make_request("GET", f"products/{product_id}.json")
        return result.get("product", {})
    
    # =========================================================================
    # CUSTOMERS
    # =========================================================================
    
    def get_customers(self, limit: int = 250, since_id: int = None) -> List[dict]:
        """Fetch customers from Shopify"""
        params = {"limit": limit}
        if since_id:
            params["since_id"] = since_id
        
        result = self._make_request("GET", "customers.json", params)
        return result.get("customers", [])
    
    def search_customer(self, query: str) -> List[dict]:
        """Search for customer by email or phone"""
        params = {"query": query}
        result = self._make_request("GET", "customers/search.json", params)
        return result.get("customers", [])
    
    def get_customer(self, customer_id: int) -> dict:
        """Get single customer by ID"""
        result = self._make_request("GET", f"customers/{customer_id}.json")
        return result.get("customer", {})
    
    # =========================================================================
    # ORDERS
    # =========================================================================
    
    def get_orders(self, limit: int = 250, status: str = "any", since_id: int = None, 
                   created_at_min: str = None) -> List[dict]:
        """Fetch orders from Shopify"""
        params = {"limit": limit, "status": status}
        if since_id:
            params["since_id"] = since_id
        if created_at_min:
            params["created_at_min"] = created_at_min
        
        result = self._make_request("GET", "orders.json", params)
        return result.get("orders", [])
    
    def get_all_orders(self, days: int = 365) -> List[dict]:
        """Fetch all orders from last N days"""
        all_orders = []
        since_id = None
        created_at_min = (datetime.now() - timedelta(days=days)).isoformat()
        
        while True:
            orders = self.get_orders(limit=250, since_id=since_id, created_at_min=created_at_min)
            if not orders:
                break
            
            all_orders.extend(orders)
            since_id = orders[-1]["id"]
            
            if len(orders) < 250:
                break
        
        logger.info(f"Fetched {len(all_orders)} orders from Shopify")
        return all_orders
    
    # =========================================================================
    # PRODUCT MAPPING
    # =========================================================================
    
    def build_product_mapping(self) -> Dict[str, dict]:
        """
        Build mapping between Shopify product IDs and our system.
        
        Returns dict: {shopify_product_id: {title, handle, variants, price}}
        """
        products = self.get_all_products()
        
        mapping = {}
        for product in products:
            product_id = str(product["id"])
            
            # Get first variant price
            price = 0
            if product.get("variants"):
                price = float(product["variants"][0].get("price", 0))
            
            mapping[product_id] = {
                "id": product_id,
                "title": product.get("title", ""),
                "handle": product.get("handle", ""),
                "vendor": product.get("vendor", ""),
                "product_type": product.get("product_type", ""),
                "price": price,
                "variants": [
                    {
                        "id": str(v["id"]),
                        "title": v.get("title", ""),
                        "price": float(v.get("price", 0)),
                        "sku": v.get("sku", "")
                    }
                    for v in product.get("variants", [])
                ]
            }
        
        return mapping
    
    # =========================================================================
    # SYNC TO LOCAL DATABASE
    # =========================================================================
    
    def sync_orders_to_db(self, pg_conn, days: int = 365) -> Dict:
        """
        Sync Shopify orders to local PostgreSQL database.
        
        Creates entries in orders and order_items tables to be used
        for recommendation training.
        """
        from psycopg2.extras import execute_values
        
        orders = self.get_all_orders(days=days)
        
        if not orders:
            return {"orders_synced": 0, "items_synced": 0}
        
        cursor = pg_conn.cursor()
        
        orders_synced = 0
        items_synced = 0
        
        for order in orders:
            try:
                order_id = f"shopify_{order['id']}"
                
                # Get customer info
                customer = order.get("customer", {})
                customer_id = customer.get("id", "")
                customer_email = customer.get("email", "")
                customer_phone = customer.get("phone", "")
                customer_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
                
                # Generate unified customer ID
                if customer_phone:
                    unified_id = f"{customer_phone}_{customer_name.split()[0].lower() if customer_name else 'customer'}"
                elif customer_email:
                    unified_id = customer_email.split("@")[0]
                else:
                    unified_id = f"shopify_{customer_id}"
                
                # Get address info
                shipping = order.get("shipping_address", {})
                city = shipping.get("city", "")
                province = shipping.get("province", "")
                
                # Order date
                order_date = order.get("created_at", "")[:19]  # ISO format
                total = float(order.get("total_price", 0))
                
                # Insert order
                cursor.execute("""
                    INSERT INTO orders (id, unified_customer_id, customer_name, customer_email, 
                                       customer_phone, customer_city, province, order_date, 
                                       total_price, source_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        unified_customer_id = EXCLUDED.unified_customer_id,
                        total_price = EXCLUDED.total_price
                """, (order_id, unified_id, customer_name, customer_email, 
                      customer_phone, city, province, order_date, total, 'SHOPIFY'))
                
                orders_synced += 1
                
                # Insert order items
                for item in order.get("line_items", []):
                    product_id = str(item.get("product_id", ""))
                    product_name = item.get("title", "")
                    quantity = item.get("quantity", 1)
                    price = float(item.get("price", 0))
                    
                    if product_id:
                        cursor.execute("""
                            INSERT INTO order_items (order_id, product_id, product_name, 
                                                    quantity, unit_price)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (order_id, product_id) DO UPDATE SET
                                quantity = EXCLUDED.quantity
                        """, (order_id, product_id, product_name, quantity, price))
                        
                        items_synced += 1
                
            except Exception as e:
                logger.error(f"Error syncing order {order.get('id')}: {e}")
                continue
        
        pg_conn.commit()
        cursor.close()
        
        logger.info(f"Synced {orders_synced} orders, {items_synced} items from Shopify")
        
        return {
            "orders_synced": orders_synced,
            "items_synced": items_synced,
            "source": "shopify"
        }


# Singleton instance
_shopify_service = None

def get_shopify_service() -> ShopifyService:
    """Get or create Shopify service instance"""
    global _shopify_service
    if _shopify_service is None:
        _shopify_service = ShopifyService()
    return _shopify_service


# CLI for testing
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("SHOPIFY SERVICE TEST")
    print("=" * 60)
    
    # Check if access token is set
    if not SHOPIFY_ACCESS_TOKEN:
        print("\n⚠️  SHOPIFY_ACCESS_TOKEN not set!")
        print("Set it in .env file or as environment variable.")
        print("\nTo get the access token:")
        print("1. Go to Shopify Admin > Settings > Apps and sales channels")
        print("2. Click on your app (pods-recommendation)")
        print("3. Click 'API credentials'")
        print("4. Under 'Admin API access token', click 'Install app' if not done")
        print("5. Copy the access token")
        sys.exit(1)
    
    service = ShopifyService()
    
    print(f"\nStore: {service.store}")
    print(f"API Version: {SHOPIFY_API_VERSION}")
    
    # Test: Fetch products
    print("\n📦 Fetching products...")
    try:
        products = service.get_products(limit=5)
        print(f"   Found {len(products)} products (showing first 5)")
        for p in products[:5]:
            print(f"   - {p['id']}: {p['title']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test: Fetch orders
    print("\n🛒 Fetching recent orders...")
    try:
        orders = service.get_orders(limit=5)
        print(f"   Found {len(orders)} orders (showing first 5)")
        for o in orders[:5]:
            customer = o.get('customer', {})
            name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip() or "Guest"
            print(f"   - {o['id']}: {name} - ${o.get('total_price', 0)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
