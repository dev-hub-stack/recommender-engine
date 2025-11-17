#!/usr/bin/env python3
"""
Matrix Factorization using SVD (Singular Value Decomposition)
Advanced collaborative filtering that finds latent factors
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatrixFactorizationSVD:
    """
    Matrix Factorization using SVD for recommendations
    """
    
    def __init__(self, pg_conn, n_factors=50):
        self.pg_conn = pg_conn
        self.n_factors = n_factors  # Number of latent factors
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.global_mean = 0
        self.is_trained = False
        
    def build_interaction_matrix(self):
        """Build user-item interaction matrix from database"""
        logger.info("Building user-item interaction matrix...")
        
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        
        # Get all interactions (user-product pairs)
        cursor.execute("""
            SELECT 
                o.unified_customer_id as user_id,
                oi.product_id as item_id,
                COUNT(*) as interaction_count
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            WHERE o.unified_customer_id IS NOT NULL
            AND oi.product_id IS NOT NULL
            GROUP BY o.unified_customer_id, oi.product_id
        """)
        
        interactions = cursor.fetchall()
        cursor.close()
        
        if not interactions:
            logger.warning("No interactions found")
            return None, None, None
        
        # Create mappings
        users = sorted(set(row['user_id'] for row in interactions))
        items = sorted(set(row['item_id'] for row in interactions))
        
        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_users = len(users)
        n_items = len(items)
        
        # Build sparse matrix
        row_indices = []
        col_indices = []
        ratings = []
        
        for interaction in interactions:
            user_idx = self.user_to_idx[interaction['user_id']]
            item_idx = self.item_to_idx[interaction['item_id']]
            rating = min(interaction['interaction_count'], 10)  # Cap at 10
            
            row_indices.append(user_idx)
            col_indices.append(item_idx)
            ratings.append(rating)
        
        interaction_matrix = csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(n_users, n_items),
            dtype=np.float32
        )
        
        # Calculate global mean
        self.global_mean = np.mean(ratings)
        
        logger.info(f"Built interaction matrix: {n_users} users × {n_items} items")
        logger.info(f"Sparsity: {1 - (len(ratings) / (n_users * n_items)):.4f}")
        logger.info(f"Global mean rating: {self.global_mean:.2f}")
        
        return interaction_matrix, n_users, n_items
    
    def train(self):
        """Train the SVD model"""
        logger.info("Training SVD model...")
        
        # Build interaction matrix
        interaction_matrix, n_users, n_items = self.build_interaction_matrix()
        
        if interaction_matrix is None:
            logger.error("Cannot train without interaction data")
            return False
        
        # Use fewer factors if matrix is small
        n_factors = min(self.n_factors, min(n_users, n_items) - 1)
        
        try:
            # Perform SVD
            logger.info(f"Computing SVD with {n_factors} factors...")
            U, sigma, Vt = svds(interaction_matrix.astype(np.float64), k=n_factors)
            
            # Store factors
            self.user_factors = U
            self.item_factors = Vt.T
            self.sigma = sigma
            
            self.is_trained = True
            
            logger.info(f"✅ SVD training complete!")
            logger.info(f"User factors shape: {self.user_factors.shape}")
            logger.info(f"Item factors shape: {self.item_factors.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"SVD training failed: {e}")
            return False
    
    def predict_rating(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        if not self.is_trained:
            return self.global_mean
        
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        # Predicted rating = user_factors · sigma · item_factors
        prediction = np.dot(
            self.user_factors[user_idx] * self.sigma,
            self.item_factors[item_idx]
        )
        
        return float(prediction)
    
    def get_recommendations(self, customer_id, limit=10, exclude_purchased=True):
        """
        Get SVD-based recommendations for a customer
        
        Args:
            customer_id: Customer identifier
            limit: Number of recommendations
            exclude_purchased: Exclude already purchased items
            
        Returns:
            List of recommendations with predicted ratings
        """
        if not self.is_trained:
            logger.warning("Model not trained, training now...")
            if not self.train():
                return self._get_popular_fallback(limit)
        
        if customer_id not in self.user_to_idx:
            logger.info(f"User {customer_id} not in training data, using fallback")
            return self._get_popular_fallback(limit)
        
        user_idx = self.user_to_idx[customer_id]
        
        # Get purchased items
        purchased_items = set()
        if exclude_purchased:
            cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT DISTINCT oi.product_id
                FROM order_items oi
                JOIN orders o ON oi.order_id = o.id
                WHERE o.unified_customer_id = %s
            """, (customer_id,))
            purchased_items = {row['product_id'] for row in cursor.fetchall()}
            cursor.close()
        
        # Calculate predicted ratings for all items
        user_vector = self.user_factors[user_idx] * self.sigma
        predicted_ratings = np.dot(user_vector, self.item_factors.T)
        
        # Normalize to 0-10 scale
        if predicted_ratings.max() > predicted_ratings.min():
            predicted_ratings = (predicted_ratings - predicted_ratings.min()) / (predicted_ratings.max() - predicted_ratings.min()) * 10
        
        # Get top items
        top_indices = np.argsort(predicted_ratings)[::-1]
        
        recommendations = []
        for item_idx in top_indices:
            item_id = self.idx_to_item[item_idx]
            
            # Skip purchased items
            if exclude_purchased and item_id in purchased_items:
                continue
            
            predicted_rating = float(predicted_ratings[item_idx])
            
            # Skip very low predictions (less than 3/10)
            if predicted_rating < 3.0:
                continue
            
            # Get product info
            cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT product_name, unit_price
                FROM order_items
                WHERE product_id = %s
                LIMIT 1
            """, (item_id,))
            product = cursor.fetchone()
            cursor.close()
            
            recommendations.append({
                'product_id': item_id,
                'score': predicted_rating,
                'reason': f"Predicted rating: {predicted_rating:.2f}/10 (Matrix Factorization)",
                'product_name': product['product_name'] if product else 'Unknown',
                'price': float(product['unit_price']) if product and product['unit_price'] else 0
            })
            
            if len(recommendations) >= limit:
                break
        
        logger.info(f"Generated {len(recommendations)} SVD recommendations for {customer_id}")
        return recommendations
    
    def _get_popular_fallback(self, limit=10):
        """Fallback to popular products"""
        cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                oi.product_id,
                oi.product_name,
                COUNT(*) as purchase_count
            FROM order_items oi
            WHERE oi.product_id IS NOT NULL
            GROUP BY oi.product_id, oi.product_name
            ORDER BY purchase_count DESC
            LIMIT %s
        """, (limit,))
        
        recommendations = []
        for row in cursor.fetchall():
            recommendations.append({
                'product_id': row['product_id'],
                'score': float(row['purchase_count']) / 1000.0,
                'reason': f"Popular product ({row['purchase_count']} purchases)",
                'product_name': row['product_name'],
                'price': 0
            })
        
        cursor.close()
        return recommendations


def test_matrix_factorization():
    """Test the matrix factorization"""
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='mastergroup_recommendations',
        user='postgres',
        password='postgres'
    )
    
    mf = MatrixFactorizationSVD(conn, n_factors=30)
    
    # Train
    print("\n🔧 Training Matrix Factorization model...")
    mf.train()
    
    # Test
    test_customer = "03224266121_Sheikh  Annas"
    recommendations = mf.get_recommendations(test_customer, limit=10)
    
    print(f"\n✅ Matrix Factorization Recommendations for {test_customer}:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. Product {rec['product_id']} - Score: {rec['score']:.3f}")
        print(f"     {rec['reason']}")
        print(f"     Name: {rec['product_name']}")
    
    conn.close()


if __name__ == "__main__":
    test_matrix_factorization()
