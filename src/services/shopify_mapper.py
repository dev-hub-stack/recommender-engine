"""
Shopify Integration Service
Handles product and customer mapping between Shopify and MasterGroup
"""

import os
import re
import logging
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class ShopifyMapper:
    """Service for mapping Shopify IDs to MasterGroup IDs"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
    
    def get_mastergroup_product_id(self, shopify_product_id: int) -> Optional[str]:
        """
        Translate Shopify product ID to MasterGroup product ID.
        
        Args:
            shopify_product_id: Shopify's product ID (e.g., 10045012017458)
            
        Returns:
            MasterGroup product ID (e.g., "1328") or None if not mapped
        """
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT mastergroup_product_id, match_confidence 
                FROM shopify_product_mapping 
                WHERE shopify_product_id = %s AND is_active = TRUE
            """, (shopify_product_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result and result['mastergroup_product_id']:
                return result['mastergroup_product_id']
            return None
        except Exception as e:
            logger.error(f"Error getting product mapping: {e}")
            return None
    
    def translate_product_id(self, product_id: str) -> str:
        """
        Smart translation of product ID - handles both Shopify and MasterGroup IDs.
        
        If it looks like a Shopify ID (>10 digits), translate it.
        Otherwise, return as-is (assumed to be MasterGroup ID).
        """
        if not product_id:
            return product_id
            
        # Check if it's a Shopify ID (long numeric string)
        if product_id.isdigit() and len(product_id) > 10:
            mg_id = self.get_mastergroup_product_id(int(product_id))
            if mg_id:
                return mg_id
        
        # Return original (either already MasterGroup ID or unmapped)
        return product_id
    
    def translate_product_ids(self, product_ids: List[str]) -> List[str]:
        """Translate a list of product IDs (Shopify or MasterGroup)."""
        return [self.translate_product_id(pid) for pid in product_ids if pid]
    
    def get_product_mapping(self, shopify_product_id: int) -> Optional[Dict]:
        """Get full mapping details for a Shopify product."""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM shopify_product_mapping 
                WHERE shopify_product_id = %s
            """, (shopify_product_id,))
            result = cursor.fetchone()
            cursor.close()
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error getting product mapping: {e}")
            return None
    
    def get_all_mappings(self) -> List[Dict]:
        """Get all product mappings."""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT shopify_product_id, shopify_title, mastergroup_product_id, 
                       mastergroup_product_name, match_confidence, match_method
                FROM shopify_product_mapping 
                WHERE is_active = TRUE
                ORDER BY match_confidence DESC
            """)
            results = cursor.fetchall()
            cursor.close()
            return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Error getting all mappings: {e}")
            return []


class ShopifyCustomerMapper:
    """Service for mapping Shopify customers to MasterGroup customers"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
    
    @staticmethod
    def normalize_pakistan_phone(phone: str) -> Optional[str]:
        """
        Normalize Pakistani phone number to standard format.
        
        Examples:
            +923001234567 -> 03001234567
            923001234567 -> 03001234567
            03001234567 -> 03001234567
            3001234567 -> 03001234567
        """
        if not phone:
            return None
        
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)
        
        # Handle various formats
        if digits.startswith('92') and len(digits) == 12:
            # +92XXXXXXXXXX or 92XXXXXXXXXX
            return '0' + digits[2:]
        elif digits.startswith('0') and len(digits) == 11:
            # 03XXXXXXXXX (already correct)
            return digits
        elif len(digits) == 10 and digits.startswith('3'):
            # 3XXXXXXXXX (missing leading 0)
            return '0' + digits
        elif len(digits) >= 10:
            # Take last 10 digits and add 0
            return '0' + digits[-10:]
        
        return phone  # Return original if can't normalize
    
    def find_mastergroup_customer(self, email: str = None, phone: str = None) -> Optional[str]:
        """
        Find MasterGroup customer ID from Shopify customer data.
        
        Matching priority:
        1. Phone number (most reliable)
        2. Email address
        
        Returns:
            MasterGroup user_id (unified_customer_id) or None
        """
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Try phone first (most reliable for Pakistani customers)
            if phone:
                normalized_phone = self.normalize_pakistan_phone(phone)
                if normalized_phone:
                    # Check existing mapping
                    cursor.execute("""
                        SELECT mastergroup_user_id FROM shopify_customer_mapping 
                        WHERE shopify_phone = %s
                    """, (normalized_phone,))
                    mapping = cursor.fetchone()
                    if mapping and mapping['mastergroup_user_id']:
                        cursor.close()
                        return mapping['mastergroup_user_id']
                    
                    # Search in orders by phone (last 10 digits)
                    phone_pattern = f"%{normalized_phone[-10:]}"
                    cursor.execute("""
                        SELECT DISTINCT unified_customer_id 
                        FROM orders 
                        WHERE customer_phone LIKE %s 
                        LIMIT 1
                    """, (phone_pattern,))
                    customer = cursor.fetchone()
                    if customer and customer['unified_customer_id']:
                        cursor.close()
                        return customer['unified_customer_id']
            
            # Try email
            if email:
                email_lower = email.lower().strip()
                
                # Check existing mapping
                cursor.execute("""
                    SELECT mastergroup_user_id FROM shopify_customer_mapping 
                    WHERE LOWER(shopify_email) = %s
                """, (email_lower,))
                mapping = cursor.fetchone()
                if mapping and mapping['mastergroup_user_id']:
                    cursor.close()
                    return mapping['mastergroup_user_id']
                
                # Note: MasterGroup orders don't typically have email
                # but check anyway
                cursor.execute("""
                    SELECT DISTINCT unified_customer_id 
                    FROM orders 
                    WHERE LOWER(customer_email) = %s 
                    LIMIT 1
                """, (email_lower,))
                customer = cursor.fetchone()
                if customer and customer['unified_customer_id']:
                    cursor.close()
                    return customer['unified_customer_id']
            
            cursor.close()
            return None
            
        except Exception as e:
            logger.error(f"Error finding customer: {e}")
            return None
    
    def save_customer_mapping(self, shopify_customer_id: int, email: str = None, 
                              phone: str = None, mastergroup_user_id: str = None,
                              match_method: str = None):
        """Save a customer mapping for future lookups."""
        try:
            cursor = self.conn.cursor()
            normalized_phone = self.normalize_pakistan_phone(phone) if phone else None
            
            cursor.execute("""
                INSERT INTO shopify_customer_mapping 
                (shopify_customer_id, shopify_email, shopify_phone, mastergroup_user_id, match_method)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (shopify_customer_id) DO UPDATE SET
                    shopify_email = COALESCE(EXCLUDED.shopify_email, shopify_customer_mapping.shopify_email),
                    shopify_phone = COALESCE(EXCLUDED.shopify_phone, shopify_customer_mapping.shopify_phone),
                    mastergroup_user_id = COALESCE(EXCLUDED.mastergroup_user_id, shopify_customer_mapping.mastergroup_user_id),
                    match_method = COALESCE(EXCLUDED.match_method, shopify_customer_mapping.match_method),
                    updated_at = NOW()
            """, (shopify_customer_id, email, normalized_phone, mastergroup_user_id, match_method))
            
            self.conn.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Error saving customer mapping: {e}")
            return False


def get_shopify_mapper(conn) -> ShopifyMapper:
    """Factory function to create ShopifyMapper instance."""
    return ShopifyMapper(conn)


def get_customer_mapper(conn) -> ShopifyCustomerMapper:
    """Factory function to create ShopifyCustomerMapper instance."""
    return ShopifyCustomerMapper(conn)
