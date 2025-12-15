"""
Shopify GraphQL API Client
Fetches product variant details by SKU efficiently using bulk queries
"""
import os
import requests
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShopifyClient:
    """Client for Shopify GraphQL Admin API"""
    
    def __init__(self):
        """Initialize Shopify client with credentials from environment"""
        self.access_token = os.getenv('SHOPIFY_ACCESS_TOKEN')
        self.store_name = os.getenv('SHOPIFY_STORE_NAME')
        self.api_version = os.getenv('SHOPIFY_API_VERSION', '2025-07')
        
        if not self.access_token or not self.store_name:
            raise ValueError("Missing Shopify credentials in environment variables")
        
        self.base_url = f"https://{self.store_name}.myshopify.com/admin/api/{self.api_version}/graphql.json"
        logger.info(f"Shopify client initialized for store: {self.store_name}")
    
    def get_variants_by_skus(self, skus: List[str]) -> Dict[str, dict]:
        """
        Fetch product variants from Shopify by SKUs (bulk query - ONE API call)
        
        Args:
            skus: List of product SKUs to fetch
            
        Returns:
            Dictionary mapping SKU to variant data
            
        Example:
            {
                "beauty-rest-Single-78*42-8": {
                    "id": 50639911518514,
                    "title": "Single - 78*42 / 8",
                    "sku": "beauty-rest-Single-78*42-8",
                    ...
                }
            }
        """
        if not skus:
            return {}
        
        logger.info(f"Fetching {len(skus)} variants from Shopify in ONE request")
        
        # Build query string: "sku:SKU1 OR sku:SKU2 OR sku:SKU3"
        query_string = " OR ".join([f"sku:{sku}" for sku in skus])
        
        # GraphQL query - get all fields we need
        graphql_query = """
        query getVariantsBySKU($query: String!) {
          productVariants(first: 50, query: $query) {
            edges {
              node {
                id
                legacyResourceId
                title
                sku
                price
                compareAtPrice
                inventoryQuantity
                availableForSale
                taxable
                barcode
                image {
                  url
                  altText
                }
                product {
                  id
                  title
                  featuredImage {
                    url
                    altText
                  }
                }
                selectedOptions {
                  name
                  value
                }
              }
            }
          }
        }
        """
        
        # Make request
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "X-Shopify-Access-Token": self.access_token,
                    "Content-Type": "application/json"
                },
                json={
                    "query": graphql_query,
                    "variables": {"query": query_string}
                },
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Shopify API error: {response.status_code} - {response.text}")
                return {}
            
            data = response.json()
            
            # Check for GraphQL errors
            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return {}
            
            # Parse response and map by SKU
            variants = {}
            for edge in data.get("data", {}).get("productVariants", {}).get("edges", []):
                node = edge["node"]
                sku = node["sku"]
                
                # Format variant data
                formatted_variant = self._format_variant(node)
                variants[sku] = formatted_variant
            
            logger.info(f"✅ Successfully fetched {len(variants)} variants from Shopify")
            return variants
            
        except requests.exceptions.Timeout:
            logger.error("Shopify API request timed out")
            return {}
        except Exception as e:
            logger.error(f"Error fetching variants from Shopify: {e}")
            return {}
    
    def _format_variant(self, node: dict) -> dict:
        """
        Format Shopify GraphQL response to required format
        
        Args:
            node: GraphQL node data
            
        Returns:
            Formatted variant object matching Shopify frontend format
        """
        # Extract options
        selected_options = node.get("selectedOptions", [])
        option1 = selected_options[0]["value"] if len(selected_options) > 0 else None
        option2 = selected_options[1]["value"] if len(selected_options) > 1 else None
        option3 = selected_options[2]["value"] if len(selected_options) > 2 else None
        options = [opt["value"] for opt in selected_options]
        
        # Get product title
        product_title = node.get("product", {}).get("title", "")
        variant_title = node.get("title", "")
        
        # Build full name
        name = f"{product_title} - {variant_title}" if product_title else variant_title
        
        # Get featured image (variant image or product image)
        featured_image = None
        if node.get("image"):
            featured_image = {
                "url": node["image"]["url"],
                "alt": node["image"].get("altText", "")
            }
        elif node.get("product", {}).get("featuredImage"):
            featured_image = {
                "url": node["product"]["featuredImage"]["url"],
                "alt": node["product"]["featuredImage"].get("altText", "")
            }
        
        # Convert price from dollars to cents (Shopify returns "17100.00", we need 1710000)
        price = int(float(node.get("price", "0")) * 100)
        compare_at_price = int(float(node.get("compareAtPrice", "0")) * 100) if node.get("compareAtPrice") else None
        
        return {
            "id": int(node.get("legacyResourceId", 0)),
            "title": variant_title,
            "option1": option1,
            "option2": option2,
            "option3": option3,
            "sku": node.get("sku", ""),
            "requires_shipping": True,  # Default - most products require shipping
            "taxable": node.get("taxable", False),
            "featured_image": featured_image,
            "available": node.get("availableForSale", False),
            "name": name,
            "public_title": variant_title,
            "options": options,
            "price": price,
            "weight": 500,  # Default weight - Shopify doesn't expose this in GraphQL easily
            "compare_at_price": compare_at_price,
            "inventory_management": "shopify",
            "barcode": node.get("barcode"),
            "requires_selling_plan": False,
            "selling_plan_allocations": [],
            "quantity_rule": {
                "min": 1,
                "max": None,
                "increment": 1
            }
        }


# Singleton instance
_shopify_client = None

def get_shopify_client() -> ShopifyClient:
    """Get singleton Shopify client instance"""
    global _shopify_client
    if _shopify_client is None:
        _shopify_client = ShopifyClient()
    return _shopify_client
