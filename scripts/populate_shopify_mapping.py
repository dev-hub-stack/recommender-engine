#!/usr/bin/env python3
"""
Shopify Product Mapping Population Script

This script populates the shopify_product_mapping table by:
1. Fetching all products from the Shopify store
2. Matching them with MasterGroup products by title/keywords
3. Inserting the mappings into the database

Usage:
    python scripts/populate_shopify_mapping.py [--refresh]

Options:
    --refresh    Clear existing mappings and repopulate

Environment Variables Required:
    SHOPIFY_STORE          Shopify store domain
    SHOPIFY_ACCESS_TOKEN   Shopify Admin API access token
    PG_* variables         Database connection
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import execute_values
import requests
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Get database connection from environment variables."""
    host = os.getenv('PG_HOST', 'localhost')
    port = os.getenv('PG_PORT', '5432')
    database = os.getenv('PG_DB', os.getenv('PG_DATABASE', 'mastergroup_recommendations'))
    user = os.getenv('PG_USER', 'postgres')
    password = os.getenv('PG_PASSWORD', '')
    sslmode = os.getenv('PG_SSLMODE', 'prefer')
    
    # Use 'require' for cloud databases
    if host not in ['localhost', '127.0.0.1']:
        sslmode = 'require'
    
    return psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        sslmode=sslmode
    )


def fetch_shopify_products():
    """Fetch all products from Shopify store."""
    store = os.getenv('SHOPIFY_STORE', 'masterverse-project.myshopify.com')
    token = os.getenv('SHOPIFY_ACCESS_TOKEN')
    
    if not token:
        print("❌ SHOPIFY_ACCESS_TOKEN not set in environment")
        return []
    
    headers = {
        'X-Shopify-Access-Token': token,
        'Content-Type': 'application/json'
    }
    
    products = []
    url = f'https://{store}/admin/api/2024-01/products.json?limit=250'
    
    while url:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                products.extend(data.get('products', []))
                
                # Check for next page
                link_header = response.headers.get('Link', '')
                if 'rel="next"' in link_header:
                    # Extract next URL from Link header
                    for part in link_header.split(','):
                        if 'rel="next"' in part:
                            url = part.split('<')[1].split('>')[0]
                            break
                else:
                    url = None
            else:
                print(f"❌ Shopify API error: {response.status_code}")
                print(response.text[:500])
                break
        except Exception as e:
            print(f"❌ Error fetching Shopify products: {e}")
            break
    
    return products


def get_mastergroup_products(conn):
    """Get all MasterGroup products from order_items."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT product_id, LOWER(product_name), product_name 
        FROM order_items 
        WHERE product_name IS NOT NULL AND product_name != ''
    """)
    
    products = {}
    original_names = {}
    for row in cursor.fetchall():
        if row[1]:  # if lowercase name exists
            products[row[1]] = row[0]  # lowercase -> id
            original_names[row[1]] = row[2]  # lowercase -> original name
    
    cursor.close()
    return products, original_names


def match_products(shopify_products, mg_products, mg_names):
    """Match Shopify products to MasterGroup products."""
    mappings = []
    
    for sp in shopify_products:
        title = (sp.get('title') or '').lower().strip()
        variants = sp.get('variants') or [{}]
        sku = (variants[0].get('sku') or '')
        handle = sp.get('handle') or ''
        shopify_id = sp['id']
        
        # Get product image URL
        images = sp.get('images') or sp.get('image') or []
        if isinstance(images, dict):
            image_url = images.get('src', '')
        elif isinstance(images, list) and len(images) > 0:
            image_url = images[0].get('src', '')
        else:
            image_url = ''
        
        matched_mg_id = None
        matched_mg_name = None
        match_type = None
        confidence = 0.0
        
        for mg_name, mg_id in mg_products.items():
            if not mg_name:
                continue
            
            # Exact title match
            if title and title == mg_name:
                matched_mg_id = mg_id
                matched_mg_name = mg_names.get(mg_name, mg_name)
                match_type = 'exact'
                confidence = 1.0
                break
            
            # Title contains
            if title and len(title) > 3 and (title in mg_name or mg_name in title):
                matched_mg_id = mg_id
                matched_mg_name = mg_names.get(mg_name, mg_name)
                match_type = 'title'
                confidence = 0.9
                break
            
            # Keyword match (at least 2 matching words > 2 chars)
            title_words = set(w for w in title.split() if len(w) > 2)
            mg_words = set(w for w in mg_name.split() if len(w) > 2)
            common = title_words & mg_words
            if len(common) >= 2:
                matched_mg_id = mg_id
                matched_mg_name = mg_names.get(mg_name, mg_name)
                match_type = f'keywords({len(common)})'
                confidence = 0.7 + (len(common) * 0.05)
                break
        
        mappings.append({
            'shopify_product_id': shopify_id,
            'shopify_title': sp.get('title', '')[:255],
            'shopify_sku': sku[:100] if sku else None,
            'shopify_handle': handle[:255] if handle else None,
            'shopify_image_url': image_url if image_url else None,
            'mastergroup_product_id': matched_mg_id,
            'mastergroup_product_name': matched_mg_name[:255] if matched_mg_name else None,
            'match_confidence': confidence if matched_mg_id else 0.0,
            'match_method': match_type if matched_mg_id else 'unmatched'
        })
    
    return mappings


def insert_mappings(conn, mappings, refresh=False):
    """Insert mappings into the database."""
    cursor = conn.cursor()
    
    if refresh:
        print("  Clearing existing mappings...")
        cursor.execute("DELETE FROM shopify_product_mapping")
        conn.commit()
    
    # Insert new mappings
    insert_sql = """
        INSERT INTO shopify_product_mapping 
        (shopify_product_id, shopify_title, shopify_sku, shopify_handle, shopify_image_url,
         mastergroup_product_id, mastergroup_product_name, match_confidence, match_method)
        VALUES %s
        ON CONFLICT (shopify_product_id) DO UPDATE SET
            shopify_title = EXCLUDED.shopify_title,
            shopify_sku = EXCLUDED.shopify_sku,
            shopify_image_url = EXCLUDED.shopify_image_url,
            mastergroup_product_id = EXCLUDED.mastergroup_product_id,
            mastergroup_product_name = EXCLUDED.mastergroup_product_name,
            match_confidence = EXCLUDED.match_confidence,
            match_method = EXCLUDED.match_method,
            updated_at = CURRENT_TIMESTAMP
    """
    
    values = [
        (m['shopify_product_id'], m['shopify_title'], m['shopify_sku'], 
         m['shopify_handle'], m['shopify_image_url'], m['mastergroup_product_id'], 
         m['mastergroup_product_name'], m['match_confidence'], m['match_method'])
        for m in mappings
    ]
    
    execute_values(cursor, insert_sql, values)
    conn.commit()
    cursor.close()


def print_summary(conn):
    """Print mapping summary."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(mastergroup_product_id) as matched,
            COUNT(*) - COUNT(mastergroup_product_id) as unmatched
        FROM shopify_product_mapping
    """)
    counts = cursor.fetchone()
    
    print("\n" + "=" * 50)
    print("  SHOPIFY PRODUCT MAPPING SUMMARY")
    print("=" * 50)
    print(f"  Total Products:   {counts[0]}")
    print(f"  Matched:          {counts[1]} ({counts[1]*100//counts[0] if counts[0] else 0}%)")
    print(f"  Unmatched:        {counts[2]}")
    print("=" * 50)
    
    # Show unmatched products
    if counts[2] > 0:
        print("\n⚠️  Unmatched Products:")
        cursor.execute("""
            SELECT shopify_title FROM shopify_product_mapping 
            WHERE mastergroup_product_id IS NULL
            ORDER BY shopify_title
        """)
        for row in cursor.fetchall():
            print(f"    - {row[0]}")
    
    cursor.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Populate Shopify Product Mappings')
    parser.add_argument('--refresh', action='store_true', 
                       help='Clear existing mappings and repopulate')
    args = parser.parse_args()
    
    print("=" * 50)
    print("  SHOPIFY PRODUCT MAPPING POPULATION")
    print("=" * 50)
    
    # Check for Shopify credentials
    if not os.getenv('SHOPIFY_ACCESS_TOKEN'):
        print("\n❌ SHOPIFY_ACCESS_TOKEN not set")
        print("   Please set environment variables:")
        print("   - SHOPIFY_STORE")
        print("   - SHOPIFY_ACCESS_TOKEN")
        sys.exit(1)
    
    # Connect to database
    print("\n1. Connecting to database...")
    try:
        conn = get_db_connection()
        print("   ✅ Connected")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        sys.exit(1)
    
    # Fetch Shopify products
    print("\n2. Fetching Shopify products...")
    shopify_products = fetch_shopify_products()
    print(f"   ✅ Found {len(shopify_products)} products")
    
    if not shopify_products:
        print("   ⚠️  No products found. Check Shopify credentials.")
        conn.close()
        sys.exit(1)
    
    # Get MasterGroup products
    print("\n3. Loading MasterGroup products...")
    mg_products, mg_names = get_mastergroup_products(conn)
    print(f"   ✅ Found {len(mg_products)} products")
    
    # Match products
    print("\n4. Matching products...")
    mappings = match_products(shopify_products, mg_products, mg_names)
    matched = sum(1 for m in mappings if m['mastergroup_product_id'])
    print(f"   ✅ Matched {matched}/{len(mappings)} products")
    
    # Insert mappings
    print("\n5. Saving mappings to database...")
    insert_mappings(conn, mappings, refresh=args.refresh)
    print("   ✅ Saved")
    
    # Print summary
    print_summary(conn)
    
    conn.close()
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
