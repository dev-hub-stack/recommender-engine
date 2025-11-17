#!/usr/bin/env python3
"""
Content-Based Filtering Algorithm
Recommends products similar to what customer has purchased before
Based on product attributes: category, brand, price range
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentBasedFiltering:
    """
    Content-Based Recommendation using product similarity
    """
    
    def __init__(self, pg_conn):
        self.pg_conn = pg_conn
        self.product_features = {}
        self.product_vectors = None
        self.product_ids = []
        self.vectorizer = TfidfVectorizer()
        
    def build_product_features(self):
        """
        Extract product features from database
        Build feature vectors for similarity computation
        """
        logger.info("Building product feature vectors...")
        
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        
        # Get product information from order items
        cursor.execute("""
            SELECT DISTINCT
                oi.product_id,
                oi.product_name,
                COUNT(DISTINCT oi.order_id) as popularity,
                AVG(oi.unit_price) as avg_price,
                MIN(oi.unit_price) as min_price,
                MAX(oi.unit_price) as max_price
            FROM order_items oi
            WHERE oi.product_id IS NOT NULL
            AND oi.product_name IS NOT NULL
            GROUP BY oi.product_id, oi.product_name
        """)
        
        products = cursor.fetchall()
        
        # Build feature strings for each product
        feature_texts = []
        self.product_ids = []
        
        for product in products:
            product_id = product['product_id']
            product_name = product['product_name'] or ""
            avg_price = product['avg_price'] or 0
            
            # Create price category
            if avg_price < 1000:
                price_cat = "budget"
            elif avg_price < 5000:
                price_cat = "mid_range"
            elif avg_price < 20000:
                price_cat = "premium"
            else:
                price_cat = "luxury"
            
            # Extract category from product name (simple heuristic)
            product_lower = product_name.lower()
            categories = []
            
            # Common product categories
            if any(word in product_lower for word in ['mattress', 'bed', 'sleeping']):
                categories.append('mattress')
            if any(word in product_lower for word in ['pillow', 'cushion']):
                categories.append('pillow')
            if any(word in product_lower for word in ['sheet', 'cover', 'protector']):
                categories.append('bedding')
            if any(word in product_lower for word in ['foam', 'memory', 'latex']):
                categories.append('foam')
            if any(word in product_lower for word in ['king', 'queen', 'single', 'double']):
                size_words = ['king', 'queen', 'single', 'double']
                for size in size_words:
                    if size in product_lower:
                        categories.append(f'size_{size}')
            
            # Create feature string
            feature_text = f"{product_name} {' '.join(categories)} {price_cat}"
            
            self.product_features[product_id] = {
                'name': product_name,
                'categories': categories,
                'price_category': price_cat,
                'avg_price': avg_price,
                'popularity': product['popularity']
            }
            
            feature_texts.append(feature_text)
            self.product_ids.append(product_id)
        
        # Build TF-IDF vectors
        if feature_texts:
            self.product_vectors = self.vectorizer.fit_transform(feature_texts)
            logger.info(f"Built feature vectors for {len(self.product_ids)} products")
        
        cursor.close()
    
    def get_customer_profile(self, customer_id):
        """
        Get customer's purchase history and preferences
        """
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT DISTINCT oi.product_id
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE o.unified_customer_id = %s
            AND oi.product_id IS NOT NULL
        """, (customer_id,))
        
        purchased_products = [row['product_id'] for row in cursor.fetchall()]
        cursor.close()
        
        return purchased_products
    
    def get_recommendations(self, customer_id, limit=10, exclude_purchased=True):
        """
        Get content-based recommendations for a customer
        
        Args:
            customer_id: Customer identifier
            limit: Number of recommendations to return
            exclude_purchased: Whether to exclude already purchased products
            
        Returns:
            List of recommended products with similarity scores
        """
        # Build features if not already done
        if self.product_vectors is None:
            self.build_product_features()
        
        if self.product_vectors is None:
            logger.warning("No product features available")
            return []
        
        # Get customer's purchase history
        purchased_products = self.get_customer_profile(customer_id)
        
        if not purchased_products:
            # New customer - return popular products
            logger.info(f"No purchase history for {customer_id}, returning popular products")
            return self._get_popular_products(limit)
        
        # Find indices of purchased products
        purchased_indices = []
        for product_id in purchased_products:
            if product_id in self.product_ids:
                purchased_indices.append(self.product_ids.index(product_id))
        
        if not purchased_indices:
            return self._get_popular_products(limit)
        
        # Calculate average vector of purchased products (customer profile)
        customer_vector = np.asarray(self.product_vectors[purchased_indices].mean(axis=0))
        
        # Calculate similarity with all products
        similarities = cosine_similarity(customer_vector, self.product_vectors.toarray()).flatten()
        
        # Get top similar products
        similar_indices = similarities.argsort()[::-1]
        
        recommendations = []
        for idx in similar_indices:
            product_id = self.product_ids[idx]
            
            # Skip if already purchased
            if exclude_purchased and product_id in purchased_products:
                continue
            
            similarity_score = float(similarities[idx])
            
            # Skip very low similarity scores
            if similarity_score < 0.1:
                continue
            
            product_info = self.product_features.get(product_id, {})
            
            recommendations.append({
                'product_id': product_id,
                'score': similarity_score,
                'reason': f"Similar to products you purchased ({', '.join(product_info.get('categories', ['item'])[:2])})",
                'product_name': product_info.get('name', 'Unknown'),
                'categories': product_info.get('categories', []),
                'price': product_info.get('avg_price', 0)
            })
            
            if len(recommendations) >= limit:
                break
        
        logger.info(f"Generated {len(recommendations)} content-based recommendations for {customer_id}")
        return recommendations
    
    def _get_popular_products(self, limit=10):
        """Fallback to popular products for cold start"""
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                oi.product_id,
                oi.product_name,
                COUNT(*) as purchase_count,
                COUNT(DISTINCT o.unified_customer_id) as customer_count
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE oi.product_id IS NOT NULL
            GROUP BY oi.product_id, oi.product_name
            ORDER BY purchase_count DESC
            LIMIT %s
        """, (limit,))
        
        recommendations = []
        for row in cursor.fetchall():
            recommendations.append({
                'product_id': row['product_id'],
                'score': float(row['purchase_count']) / 1000.0,  # Normalize
                'reason': f"Popular product ({row['purchase_count']} purchases)",
                'product_name': row['product_name'],
                'categories': [],
                'price': 0
            })
        
        cursor.close()
        return recommendations


def test_content_based_filtering():
    """Test the content-based filtering"""
    import psycopg2
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='mastergroup_recommendations',
        user='postgres',
        password='postgres'
    )
    
    cbf = ContentBasedFiltering(conn)
    
    # Test with a customer
    test_customer = "03224266121_Sheikh  Annas"
    recommendations = cbf.get_recommendations(test_customer, limit=10)
    
    print(f"\n✅ Content-Based Recommendations for {test_customer}:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. Product {rec['product_id']} - Score: {rec['score']:.3f}")
        print(f"     {rec['reason']}")
        print(f"     Name: {rec['product_name']}")
    
    conn.close()


if __name__ == "__main__":
    test_content_based_filtering()
