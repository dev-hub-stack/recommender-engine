"""
Local Recommender Training Service
Trains recommendation models locally using the same data as AWS Personalize.

Models trained:
1. SVD (Singular Value Decomposition) - User personalization
2. KNN Item-Item Similarity - Similar items
3. Popularity-Based - Fallback/cold start

Uses scikit-surprise library for collaborative filtering.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import psycopg2
from psycopg2.extras import RealDictCursor
import structlog

# Surprise library for collaborative filtering
try:
    from surprise import Dataset, Reader, SVD, KNNBasic, KNNWithMeans
    from surprise.model_selection import cross_validate
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

# Scipy for sparse matrices and similarity
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from .local_model_storage import save_model, load_model, list_models

logger = structlog.get_logger()


def get_pg_connection():
    """Get PostgreSQL connection"""
    host = os.getenv('PG_HOST', 'localhost')
    # Use 'prefer' for localhost (no SSL), 'require' for remote
    sslmode = os.getenv('PG_SSLMODE', 'prefer' if host == 'localhost' else 'require')
    return psycopg2.connect(
        host=host,
        port=int(os.getenv('PG_PORT', '5432')),
        database=os.getenv('PG_DB', 'mastergroup_recommendations'),
        user=os.getenv('PG_USER', 'postgres'),
        password=os.getenv('PG_PASSWORD', ''),
        sslmode=sslmode
    )


class LocalRecommender:
    """
    Local recommendation system using collaborative filtering.
    Mirrors AWS Personalize capabilities without cloud dependency.
    """
    
    def __init__(self):
        self.svd_model = None
        self.item_similarity_matrix = None
        self.popularity_scores = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.item_id_to_name = {}
        self.user_item_matrix = None
        
        # Try to load existing models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            # Load SVD model
            svd_data = load_model("svd_recommender")
            if svd_data:
                self.svd_model = svd_data.get("model", {}).get("svd")
                self.user_encoder = svd_data.get("model", {}).get("user_encoder", self.user_encoder)
                self.item_encoder = svd_data.get("model", {}).get("item_encoder", self.item_encoder)
                logger.info("✅ Loaded SVD model from disk")
            
            # Load item similarity
            sim_data = load_model("item_similarity")
            if sim_data:
                self.item_similarity_matrix = sim_data.get("model", {}).get("similarity_matrix")
                self.item_id_to_name = sim_data.get("model", {}).get("item_id_to_name", {})
                logger.info("✅ Loaded item similarity model from disk")
            
            # Load popularity
            pop_data = load_model("popularity_scores")
            if pop_data:
                self.popularity_scores = pop_data.get("model", {}).get("scores")
                logger.info("✅ Loaded popularity scores from disk")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    def fetch_training_data(self, days: int = 90) -> pd.DataFrame:
        """
        Fetch interaction data from PostgreSQL.
        Same format as AWS Personalize interactions dataset.
        
        Returns:
            DataFrame with columns: user_id, item_id, timestamp, event_type, item_name
        """
        logger.info(f"Fetching training data for last {days} days...")
        
        conn = get_pg_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Fetch order interactions
        cursor.execute("""
            SELECT 
                o.unified_customer_id as user_id,
                oi.product_id as item_id,
                oi.product_name as item_name,
                o.order_date as timestamp,
                'purchase' as event_type,
                oi.quantity,
                oi.unit_price as price
            FROM orders o
            JOIN order_items oi ON o.id::text = oi.order_id
            WHERE o.order_date >= %s
            AND o.unified_customer_id IS NOT NULL
            AND oi.product_id IS NOT NULL
            ORDER BY o.order_date
        """, (start_date,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            logger.warning("No training data found!")
            return df
        
        # Clean data
        df = df.dropna(subset=['user_id', 'item_id'])
        df['user_id'] = df['user_id'].astype(str)
        df['item_id'] = df['item_id'].astype(str)
        
        logger.info(
            f"Fetched {len(df)} interactions",
            users=df['user_id'].nunique(),
            items=df['item_id'].nunique()
        )
        
        return df
    
    def train_all_models(self, days: int = 90) -> Dict[str, Any]:
        """
        Train all recommendation models.
        
        Args:
            days: Number of days of historical data to use
        
        Returns:
            Dict with training results for each model
        """
        logger.info("🚀 Starting local model training...")
        start_time = datetime.now()
        
        # Fetch data
        df = self.fetch_training_data(days)
        
        if df.empty:
            return {"success": False, "error": "No training data available"}
        
        results = {
            "training_started": start_time.isoformat(),
            "data_points": len(df),
            "unique_users": df['user_id'].nunique(),
            "unique_items": df['item_id'].nunique(),
            "models": {}
        }
        
        # Store item names mapping
        self.item_id_to_name = df.groupby('item_id')['item_name'].first().to_dict()
        
        # 1. Train popularity model (always works)
        pop_result = self._train_popularity_model(df)
        results["models"]["popularity"] = pop_result
        
        # 2. Train item similarity model
        sim_result = self._train_item_similarity(df)
        results["models"]["item_similarity"] = sim_result
        
        # 3. Train SVD model (requires surprise library)
        if SURPRISE_AVAILABLE:
            svd_result = self._train_svd_model(df)
            results["models"]["svd"] = svd_result
        else:
            results["models"]["svd"] = {"success": False, "error": "surprise library not installed"}
        
        # Training summary
        elapsed = (datetime.now() - start_time).total_seconds()
        results["training_completed"] = datetime.now().isoformat()
        results["training_duration_seconds"] = round(elapsed, 2)
        results["success"] = True
        
        logger.info(f"✅ Training completed in {elapsed:.1f}s")
        
        return results
    
    def _train_popularity_model(self, df: pd.DataFrame) -> Dict:
        """Train popularity-based model (most purchased items)"""
        logger.info("Training popularity model...")
        
        try:
            # Calculate popularity scores
            # Weighted by recency and purchase count
            df['days_ago'] = (datetime.now() - pd.to_datetime(df['timestamp'])).dt.days
            df['recency_weight'] = np.exp(-df['days_ago'] / 30)  # Exponential decay
            
            # Score = sum of (quantity * recency_weight)
            popularity = df.groupby('item_id').agg({
                'quantity': 'sum',
                'recency_weight': 'sum',
                'price': 'mean',
                'item_name': 'first'
            }).reset_index()
            
            popularity['score'] = popularity['quantity'] * popularity['recency_weight']
            popularity = popularity.sort_values('score', ascending=False)
            
            self.popularity_scores = popularity.set_index('item_id')['score'].to_dict()
            
            # Save to disk
            save_result = save_model(
                model_name="popularity_scores",
                model_object={
                    "scores": self.popularity_scores,
                    "item_details": popularity.to_dict('records')
                },
                metadata={
                    "algorithm": "weighted_popularity",
                    "total_items": len(popularity),
                    "top_item": popularity.iloc[0]['item_name'] if len(popularity) > 0 else None
                }
            )
            
            return {
                "success": True,
                "total_items": len(popularity),
                "top_5": popularity.head()['item_name'].tolist()
            }
            
        except Exception as e:
            logger.error(f"Popularity training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _train_item_similarity(self, df: pd.DataFrame) -> Dict:
        """Train item-item similarity model using co-purchase patterns"""
        logger.info("Training item similarity model...")
        
        try:
            # Encode users and items
            self.user_encoder.fit(df['user_id'].unique())
            self.item_encoder.fit(df['item_id'].unique())
            
            user_indices = self.user_encoder.transform(df['user_id'])
            item_indices = self.item_encoder.transform(df['item_id'])
            
            # Create user-item matrix (sparse)
            n_users = len(self.user_encoder.classes_)
            n_items = len(self.item_encoder.classes_)
            
            # Use purchase count as value
            values = df['quantity'].values.astype(float)
            
            self.user_item_matrix = csr_matrix(
                (values, (user_indices, item_indices)),
                shape=(n_users, n_items)
            )
            
            # Calculate item-item cosine similarity
            # Transpose to get item vectors (each column = item)
            item_vectors = self.user_item_matrix.T.toarray()
            
            # Normalize to unit vectors
            norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized = item_vectors / norms
            
            # Cosine similarity matrix
            self.item_similarity_matrix = cosine_similarity(normalized)
            
            # Save to disk
            save_result = save_model(
                model_name="item_similarity",
                model_object={
                    "similarity_matrix": self.item_similarity_matrix,
                    "item_encoder": self.item_encoder,
                    "user_encoder": self.user_encoder,
                    "item_id_to_name": self.item_id_to_name
                },
                metadata={
                    "algorithm": "cosine_similarity",
                    "n_items": n_items,
                    "n_users": n_users,
                    "matrix_shape": list(self.item_similarity_matrix.shape)
                }
            )
            
            return {
                "success": True,
                "n_items": n_items,
                "n_users": n_users,
                "matrix_shape": list(self.item_similarity_matrix.shape)
            }
            
        except Exception as e:
            logger.error(f"Item similarity training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _train_svd_model(self, df: pd.DataFrame) -> Dict:
        """Train SVD model for user personalization (like AWS User Personalization)"""
        logger.info("Training SVD model...")
        
        if not SURPRISE_AVAILABLE:
            return {"success": False, "error": "surprise library not available"}
        
        try:
            # Prepare data for Surprise
            # Create implicit ratings from purchase counts
            user_item_counts = df.groupby(['user_id', 'item_id']).agg({
                'quantity': 'sum'
            }).reset_index()
            
            # Normalize to 1-5 rating scale
            max_count = user_item_counts['quantity'].max()
            user_item_counts['rating'] = 1 + 4 * (user_item_counts['quantity'] / max_count)
            
            # Create Surprise dataset
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                user_item_counts[['user_id', 'item_id', 'rating']], 
                reader
            )
            
            # Build full trainset
            trainset = data.build_full_trainset()
            
            # Train SVD model
            self.svd_model = SVD(
                n_factors=50,
                n_epochs=20,
                lr_all=0.005,
                reg_all=0.02,
                verbose=False
            )
            self.svd_model.fit(trainset)
            
            # Cross-validate for metrics
            cv_results = cross_validate(
                SVD(n_factors=50, n_epochs=20),
                data,
                measures=['RMSE', 'MAE'],
                cv=3,
                verbose=False
            )
            
            avg_rmse = cv_results['test_rmse'].mean()
            avg_mae = cv_results['test_mae'].mean()
            
            # Save to disk
            save_result = save_model(
                model_name="svd_recommender",
                model_object={
                    "svd": self.svd_model,
                    "user_encoder": self.user_encoder,
                    "item_encoder": self.item_encoder,
                    "item_id_to_name": self.item_id_to_name
                },
                metadata={
                    "algorithm": "SVD",
                    "n_factors": 50,
                    "n_epochs": 20,
                    "rmse": round(avg_rmse, 4),
                    "mae": round(avg_mae, 4),
                    "n_users": trainset.n_users,
                    "n_items": trainset.n_items,
                    "n_ratings": trainset.n_ratings
                }
            )
            
            return {
                "success": True,
                "rmse": round(avg_rmse, 4),
                "mae": round(avg_mae, 4),
                "n_users": trainset.n_users,
                "n_items": trainset.n_items
            }
            
        except Exception as e:
            logger.error(f"SVD training failed: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== INFERENCE METHODS ====================
    
    def get_user_recommendations(
        self,
        user_id: str,
        limit: int = 10,
        exclude_purchased: bool = True
    ) -> List[Dict]:
        """
        Get personalized recommendations for a user.
        Uses SVD model if available, falls back to popularity.
        
        Args:
            user_id: Customer ID
            limit: Number of recommendations
            exclude_purchased: Whether to exclude items already purchased
        
        Returns:
            List of recommended items with scores
        """
        recommendations = []
        
        # Try SVD first
        if self.svd_model is not None and SURPRISE_AVAILABLE:
            recommendations = self._svd_recommendations(user_id, limit, exclude_purchased)
        
        # Fall back to popularity
        if not recommendations:
            recommendations = self._popularity_recommendations(limit)
        
        return recommendations[:limit]
    
    def _svd_recommendations(
        self,
        user_id: str,
        limit: int,
        exclude_purchased: bool
    ) -> List[Dict]:
        """Get SVD-based recommendations"""
        try:
            # Get all items
            all_items = self.item_encoder.classes_
            
            # Get items already purchased by user (to exclude)
            purchased_items = set()
            if exclude_purchased:
                conn = get_pg_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT oi.product_id
                    FROM order_items oi
                    JOIN orders o ON oi.order_id = o.id
                    WHERE o.unified_customer_id = %s
                """, (user_id,))
                purchased_items = {row[0] for row in cursor.fetchall()}
                cursor.close()
                conn.close()
            
            # Predict scores for all items
            predictions = []
            for item_id in all_items:
                if item_id in purchased_items:
                    continue
                
                pred = self.svd_model.predict(user_id, item_id)
                predictions.append({
                    "item_id": item_id,
                    "item_name": self.item_id_to_name.get(item_id, "Unknown"),
                    "score": round(pred.est, 4),
                    "algorithm": "SVD"
                })
            
            # Sort by score descending
            predictions.sort(key=lambda x: x['score'], reverse=True)
            
            return predictions[:limit]
            
        except Exception as e:
            logger.warning(f"SVD recommendation failed: {e}")
            return []
    
    def _popularity_recommendations(self, limit: int) -> List[Dict]:
        """Get popularity-based recommendations"""
        if not self.popularity_scores:
            return []
        
        # Sort by popularity score
        sorted_items = sorted(
            self.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                "item_id": item_id,
                "item_name": self.item_id_to_name.get(item_id, "Unknown"),
                "score": round(score, 4),
                "algorithm": "Popularity"
            }
            for item_id, score in sorted_items
        ]
    
    def get_similar_items(
        self,
        item_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get items similar to a given item.
        Uses item-item similarity matrix.
        
        Args:
            item_id: Product ID
            limit: Number of similar items
        
        Returns:
            List of similar items with similarity scores
        """
        if self.item_similarity_matrix is None:
            return self._popularity_recommendations(limit)
        
        try:
            # Get item index
            if item_id not in self.item_encoder.classes_:
                logger.warning(f"Item {item_id} not in training data")
                return self._popularity_recommendations(limit)
            
            item_idx = self.item_encoder.transform([item_id])[0]
            
            # Get similarity scores for this item
            similarities = self.item_similarity_matrix[item_idx]
            
            # Get top similar items (excluding itself)
            top_indices = np.argsort(similarities)[::-1][1:limit+1]
            
            results = []
            for idx in top_indices:
                similar_item_id = self.item_encoder.inverse_transform([idx])[0]
                results.append({
                    "item_id": similar_item_id,
                    "item_name": self.item_id_to_name.get(similar_item_id, "Unknown"),
                    "similarity_score": round(float(similarities[idx]), 4),
                    "algorithm": "ItemSimilarity"
                })
            
            return results
            
        except Exception as e:
            logger.warning(f"Similar items failed: {e}")
            return self._popularity_recommendations(limit)
    
    def get_frequently_bought_together(
        self,
        item_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Get items frequently bought together with a given item.
        Based on co-purchase patterns.
        """
        try:
            conn = get_pg_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Find orders containing this item
            cursor.execute("""
                WITH item_orders AS (
                    SELECT DISTINCT oi.order_id
                    FROM order_items oi
                    WHERE oi.product_id = %s
                )
                SELECT 
                    oi.product_id as item_id,
                    oi.product_name as item_name,
                    COUNT(*) as co_purchase_count,
                    SUM(oi.quantity) as total_quantity
                FROM order_items oi
                JOIN item_orders io ON oi.order_id = io.order_id
                WHERE oi.product_id != %s
                GROUP BY oi.product_id, oi.product_name
                ORDER BY co_purchase_count DESC, total_quantity DESC
                LIMIT %s
            """, (item_id, item_id, limit))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [
                {
                    "item_id": row['item_id'],
                    "item_name": row['item_name'],
                    "co_purchase_count": row['co_purchase_count'],
                    "algorithm": "FrequentlyBoughtTogether"
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Frequently bought together failed: {e}")
            return []
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        models = list_models()
        
        return {
            "svd_loaded": self.svd_model is not None,
            "item_similarity_loaded": self.item_similarity_matrix is not None,
            "popularity_loaded": self.popularity_scores is not None,
            "models_on_disk": [m['model_name'] for m in models],
            "models_details": models
        }


# Global recommender instance
_recommender_instance = None


def get_recommender() -> LocalRecommender:
    """Get or create the global recommender instance"""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = LocalRecommender()
    return _recommender_instance
