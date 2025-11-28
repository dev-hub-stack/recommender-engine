"""
ML-Based Recommendation Service
Integrates all existing algorithms and provides production-ready ML recommendations
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import joblib
import structlog
from pathlib import Path

# Add algorithms and config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'algorithms'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'config'))

try:
    from algorithms.collaborative_filtering import CollaborativeFilteringEngine, UserItemMatrix
    from algorithms.content_based_filtering import ContentBasedFiltering
    from algorithms.matrix_factorization import MatrixFactorizationSVD
    from algorithms.popularity_based import PopularityBasedEngine
except ModuleNotFoundError:
    from src.algorithms.collaborative_filtering import CollaborativeFilteringEngine, UserItemMatrix
    from src.algorithms.content_based_filtering import ContentBasedFiltering
    from src.algorithms.matrix_factorization import MatrixFactorizationSVD
    from src.algorithms.popularity_based import PopularityBasedEngine

# Load centralized config
try:
    from config.master_group_api import PG_CONFIG
except ImportError:
    try:
        from src.config.master_group_api import PG_CONFIG
    except ImportError:
        # Fallback to env vars if config not found
        PG_CONFIG = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': int(os.getenv('PG_PORT', '5432')),
            'database': os.getenv('PG_DB', 'mastergroup_recommendations'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'postgres'),
        }

logger = structlog.get_logger()

# Database configuration from centralized config
PG_HOST = PG_CONFIG.get('host', 'localhost')
PG_PORT = PG_CONFIG.get('port', 5432)
PG_DB = PG_CONFIG.get('database', 'mastergroup_recommendations')
PG_USER = PG_CONFIG.get('user', 'postgres')
PG_PASSWORD = PG_CONFIG.get('password', '')

# Model storage
MODEL_DIR = os.getenv('MODEL_DIR', '/tmp/ml_models')
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)


class MLRecommendationService:
    """
    Unified ML Recommendation Service
    Trains and serves recommendations from all algorithms
    """
    
    def __init__(self):
        self.collaborative_engine = None
        self.content_based_engine = None
        self.matrix_factorization_engine = None
        self.popularity_engine = None
        
        self.is_trained = False
        self.training_timestamp = None
        self.model_metadata = {}
        self.precomputed_cache = {}  # Cache for pre-computed recommendations
        
        logger.info("ML Recommendation Service initialized")
    
    def get_db_connection(self):
        """Create database connection"""
        return psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        )
    
    def load_interaction_data(self, time_filter: str = 'all', limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load user-item interaction data from database
        
        Args:
            time_filter: Time filter (all, 1year, 6months, 90days, 30days, 7days)
            limit: Optional limit on number of interactions
            
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        logger.info(f"Loading interaction data (time_filter={time_filter})")
        
        # Calculate date range
        time_ranges = {
            '7days': 7,
            '30days': 30,
            '90days': 90,
            '6months': 180,
            '1year': 365,
            'all': None
        }
        
        days = time_ranges.get(time_filter)
        
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build query
        query = """
            SELECT 
                o.unified_customer_id as user_id,
                oi.product_id as item_id,
                oi.product_name,
                CAST(oi.quantity as FLOAT) as rating,  -- Use quantity as implicit rating
                o.order_date as timestamp,
                oi.total_price as revenue
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            WHERE o.unified_customer_id IS NOT NULL
            AND oi.product_id IS NOT NULL
        """
        
        params = []
        if days:
            query += " AND o.order_date >= NOW() - INTERVAL '%s days'"
            params.append(days)
        
        query += " ORDER BY o.order_date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        df = pd.DataFrame(rows)
        
        if len(df) == 0:
            logger.warning(f"No interactions found for time_filter={time_filter}")
            return pd.DataFrame(columns=['user_id', 'item_id', 'product_name', 'rating', 'timestamp', 'revenue'])
        
        logger.info(f"Loaded {len(df)} interactions from {df['user_id'].nunique()} users and {df['item_id'].nunique()} items")
        
        return df
    
    def load_product_metadata(self) -> pd.DataFrame:
        """Load product metadata for content-based filtering"""
        logger.info("Loading product metadata")
        
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                oi.product_id,
                MAX(oi.product_name) as product_name,
                AVG(oi.unit_price) as avg_price,
                SUM(oi.quantity) as total_quantity_sold,
                SUM(oi.total_price) as total_revenue,
                COUNT(DISTINCT oi.order_id) as total_orders,
                COUNT(DISTINCT o.unified_customer_id) as unique_customers
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE oi.product_id IS NOT NULL
            GROUP BY oi.product_id
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        df = pd.DataFrame(rows)
        logger.info(f"Loaded metadata for {len(df)} products")
        
        return df
    
    def train_all_models(self, time_filter: str = 'all', force_retrain: bool = False) -> Dict:
        """
        Train all ML algorithms
        
        Args:
            time_filter: Time filter for training data
            force_retrain: Force retraining even if models exist
            
        Returns:
            Training metrics dictionary
        """
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING ML MODEL TRAINING")
        logger.info("=" * 80)
        
        # Check if models already trained
        if self.is_trained and not force_retrain:
            logger.info("Models already trained. Use force_retrain=True to retrain.")
            return self.model_metadata
        
        results = {
            'training_started': start_time.isoformat(),
            'time_filter': time_filter,
            'models': {}
        }
        
        # Load data
        logger.info("\n📊 LOADING DATA...")
        interactions_df = self.load_interaction_data(time_filter=time_filter)
        product_metadata_df = self.load_product_metadata()
        
        if len(interactions_df) == 0:
            logger.error("No interaction data available for training")
            return {'error': 'No training data available'}
        
        results['data_stats'] = {
            'n_interactions': len(interactions_df),
            'n_users': interactions_df['user_id'].nunique(),
            'n_items': interactions_df['item_id'].nunique(),
            'n_products_with_metadata': len(product_metadata_df)
        }
        
        # ========================================
        # 1. TRAIN COLLABORATIVE FILTERING
        # ========================================
        try:
            logger.info("\n🤖 TRAINING COLLABORATIVE FILTERING...")
            cf_start = datetime.now()
            
            self.collaborative_engine = CollaborativeFilteringEngine(
                user_cf_weight=0.6,
                item_cf_weight=0.4,
                min_interactions=3  # Lower threshold for production
            )
            
            cf_metrics = self.collaborative_engine.train(interactions_df)
            cf_time = (datetime.now() - cf_start).total_seconds()
            
            results['models']['collaborative_filtering'] = {
                'status': 'success',
                'training_time_seconds': cf_time,
                'metrics': cf_metrics,
                'accuracy': cf_metrics.get('rmse_accuracy', 0),
                'coverage': cf_metrics.get('coverage', 0)
            }
            
            # Save model
            cf_path = os.path.join(MODEL_DIR, f'collaborative_filtering_{time_filter}.pkl')
            self.collaborative_engine.save_model(cf_path)
            
            logger.info(f"✅ Collaborative Filtering trained in {cf_time:.2f}s - Accuracy: {cf_metrics.get('rmse_accuracy', 0):.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Collaborative Filtering training failed: {e}", exc_info=True)
            results['models']['collaborative_filtering'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # ========================================
        # 2. TRAIN CONTENT-BASED FILTERING
        # ========================================
        try:
            logger.info("\n🤖 TRAINING CONTENT-BASED FILTERING...")
            cb_start = datetime.now()
            
            # Create product features from metadata
            product_features = []
            for _, row in product_metadata_df.iterrows():
                product_features.append({
                    'product_id': str(row['product_id']),
                    'product_name': row['product_name'] or '',
                    'price': float(row['avg_price'] or 0),
                    'popularity': int(row['total_orders'] or 0)
                })
            
            # Create database connection for Content-Based
            conn = self.get_db_connection()
            self.content_based_engine = ContentBasedFiltering(pg_conn=conn)
            
            # Build product features (it queries the database internally)
            self.content_based_engine.build_product_features()
            # Keep connection open for recommendations
            
            cb_time = (datetime.now() - cb_start).total_seconds()
            
            results['models']['content_based'] = {
                'status': 'success',
                'training_time_seconds': cb_time,
                'n_products': len(product_features),
                'feature_dimensions': 0  # Not using feature matrix in this implementation
            }
            
            # Note: ContentBasedFiltering doesn't have save_model, will rebuild on each load
            
            logger.info(f"✅ Content-Based Filtering trained in {cb_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Content-Based Filtering training failed: {e}", exc_info=True)
            results['models']['content_based'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # ========================================
        # 3. TRAIN MATRIX FACTORIZATION (SVD)
        # ========================================
        try:
            logger.info("\n🤖 TRAINING MATRIX FACTORIZATION (SVD)...")
            mf_start = datetime.now()
            
            # Create database connection for MF (it needs direct DB access)
            conn = self.get_db_connection()
            
            self.matrix_factorization_engine = MatrixFactorizationSVD(
                pg_conn=conn,
                n_factors=30
            )
            
            # Train model (uses build_interaction_matrix and train internally)
            self.matrix_factorization_engine.build_interaction_matrix()
            self.matrix_factorization_engine.train()
            
            mf_time = (datetime.now() - mf_start).total_seconds()
            
            results['models']['matrix_factorization'] = {
                'status': 'success',
                'training_time_seconds': mf_time,
                'n_factors': 30,
                'n_epochs': 20
            }
            
            # Note: MatrixFactorizationSVD doesn't have save_model, will rebuild on each load
            # Keep connection open for recommendations
            
            logger.info(f"✅ Matrix Factorization trained in {mf_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Matrix Factorization training failed: {e}", exc_info=True)
            results['models']['matrix_factorization'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # ========================================
        # 4. TRAIN POPULARITY-BASED (No Training Needed)
        # ========================================
        try:
            logger.info("\n🤖 INITIALIZING POPULARITY-BASED...")
            pop_start = datetime.now()
            
            # PopularityBasedEngine doesn't take conn parameter
            self.popularity_engine = PopularityBasedEngine(
                min_sales_threshold=5,
                popularity_weight=0.4,
                trend_weight=0.3,
                segment_weight=0.3
            )
            
            # Note: Popularity engine needs training data to work properly
            # For now, we'll initialize it without training
            
            pop_time = (datetime.now() - pop_start).total_seconds()
            
            results['models']['popularity_based'] = {
                'status': 'success',
                'initialization_time_seconds': pop_time,
                'note': 'Computed on-demand from database'
            }
            
            logger.info(f"✅ Popularity-Based initialized in {pop_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Popularity-Based initialization failed: {e}", exc_info=True)
            results['models']['popularity_based'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # ========================================
        # FINALIZE TRAINING
        # ========================================
        total_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        self.training_timestamp = datetime.now()
        self.model_metadata = results
        
        results['training_completed'] = datetime.now().isoformat()
        results['total_training_time_seconds'] = total_time
        
        # Count successful models
        successful_models = sum(1 for m in results['models'].values() if m.get('status') == 'success')
        results['successful_models'] = successful_models
        results['total_models'] = len(results['models'])
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ TRAINING COMPLETE: {successful_models}/{len(results['models'])} models trained successfully in {total_time:.2f}s")
        logger.info("=" * 80)
        
        # Save models to PostgreSQL for persistence (important for Heroku)
        try:
            self.save_models_to_db(time_filter)
            logger.info("✅ Models saved to PostgreSQL for persistence")
        except Exception as e:
            logger.warning(f"⚠️ Could not save models to DB: {e}")
        
        return results
    
    def get_hybrid_recommendations(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        algorithm_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Get hybrid recommendations combining all algorithms
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            algorithm_weights: Optional custom weights for each algorithm
            
        Returns:
            List of recommendation dictionaries
        """
        if not self.is_trained:
            logger.warning("Models not trained. Training now with default settings...")
            self.train_all_models()
        
        # Default weights
        if algorithm_weights is None:
            algorithm_weights = {
                'collaborative': 0.40,
                'content_based': 0.20,
                'matrix_factorization': 0.30,
                'popularity': 0.10
            }
        
        logger.info(f"Generating hybrid recommendations for user {user_id}")
        
        combined_scores = {}
        algorithms_used = []
        
        # Get recommendations from each algorithm
        try:
            if self.collaborative_engine and self.collaborative_engine.is_trained:
                cf_recs = self.collaborative_engine.user_cf.get_recommendations(
                    user_id, 
                    n_recommendations * 3
                )
                for item_id, score in cf_recs:
                    combined_scores[item_id] = combined_scores.get(item_id, 0) + (
                        score * algorithm_weights['collaborative']
                    )
                if cf_recs:
                    algorithms_used.append('collaborative')
        except Exception as e:
            logger.warning(f"Collaborative filtering failed: {e}")
        
        # Matrix Factorization
        try:
            if self.matrix_factorization_engine and self.matrix_factorization_engine.is_trained:
                mf_recs = self.matrix_factorization_engine.get_recommendations(
                    user_id,
                    limit=n_recommendations * 3
                )
                for rec in mf_recs:
                    item_id = rec['product_id']
                    score = rec['score']
                    combined_scores[item_id] = combined_scores.get(item_id, 0) + (
                        score * algorithm_weights['matrix_factorization']
                    )
                if mf_recs:
                    algorithms_used.append('matrix_factorization')
        except Exception as e:
            logger.warning(f"Matrix factorization failed: {e}")
        
        # Content-Based Filtering
        try:
            if self.content_based_engine and self.content_based_engine.product_vectors is not None:
                cb_recs = self.content_based_engine.get_recommendations(
                    user_id,
                    limit=n_recommendations * 3
                )
                for rec in cb_recs:
                    item_id = rec['product_id']
                    score = rec['score']
                    combined_scores[item_id] = combined_scores.get(item_id, 0) + (
                        score * algorithm_weights['content_based']
                    )
                if cb_recs:
                    algorithms_used.append('content_based')
        except Exception as e:
            logger.warning(f"Content-based filtering failed: {e}")
        
        # Popularity fallback if no recommendations from other algorithms
        if not combined_scores:
            logger.info("No personalized recommendations, falling back to popularity")
            return self.get_popular_recommendations(n_recommendations)
        
        # Sort and return top recommendations
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_recs = sorted_recs[:n_recommendations]
        
        # Enrich with product details
        recommendations = []
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        for item_id, score in top_recs:
            cursor.execute("""
                SELECT 
                    product_id,
                    MAX(product_name) as product_name,
                    AVG(unit_price) as price,
                    SUM(total_price) as revenue
                FROM order_items
                WHERE product_id = %s
                GROUP BY product_id
            """, (item_id,))
            
            product_data = cursor.fetchone()
            
            if product_data:
                recommendations.append({
                    'product_id': str(product_data['product_id']),
                    'product_name': product_data['product_name'],
                    'score': float(score),
                    'price': float(product_data['price'] or 0),
                    'revenue': float(product_data['revenue'] or 0),
                    'algorithm': 'hybrid',
                    'algorithms_used': algorithms_used,
                    'confidence': min(score / max(combined_scores.values()), 1.0) if combined_scores else 0
                })
        
        cursor.close()
        conn.close()
        
        logger.info(f"Generated {len(recommendations)} hybrid recommendations for user {user_id} using {algorithms_used}")
        
        return recommendations
    
    def get_popular_recommendations(self, n_recommendations: int = 10, time_filter: str = '30days') -> List[Dict]:
        """
        Get popular product recommendations (fallback for cold start)
        
        Args:
            n_recommendations: Number of recommendations to return
            time_filter: Time filter for popularity calculation
            
        Returns:
            List of recommendation dictionaries
        """
        logger.info(f"Getting popular recommendations (n={n_recommendations})")
        
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Calculate date range
        time_ranges = {'7days': 7, '30days': 30, '90days': 90, 'all': None}
        days = time_ranges.get(time_filter)
        
        date_filter = ""
        params = []
        if days:
            date_filter = "AND o.order_date >= NOW() - INTERVAL '%s days'"
            params.append(days)
        
        cursor.execute(f"""
            SELECT 
                oi.product_id,
                MAX(oi.product_name) as product_name,
                AVG(oi.unit_price) as price,
                SUM(oi.total_price) as revenue,
                COUNT(*) as purchase_count,
                COUNT(DISTINCT o.unified_customer_id) as unique_customers
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE oi.product_id IS NOT NULL
            {date_filter}
            GROUP BY oi.product_id
            ORDER BY purchase_count DESC
            LIMIT %s
        """, params + [n_recommendations])
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        recommendations = []
        max_count = rows[0]['purchase_count'] if rows else 1
        
        for row in rows:
            recommendations.append({
                'product_id': str(row['product_id']),
                'product_name': row['product_name'],
                'score': float(row['purchase_count']) / max_count,
                'price': float(row['price'] or 0),
                'revenue': float(row['revenue'] or 0),
                'algorithm': 'popularity',
                'purchase_count': row['purchase_count'],
                'unique_customers': row['unique_customers'],
                'confidence': 0.8  # High confidence for popular items
            })
        
        logger.info(f"Generated {len(recommendations)} popular recommendations")
        return recommendations
    
    def get_collaborative_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using only collaborative filtering"""
        if not self.collaborative_engine or not self.collaborative_engine.is_trained:
            logger.warning("Collaborative engine not trained, returning popular items")
            return self.get_popular_recommendations(n_recommendations)
        
        try:
            cf_recs = self.collaborative_engine.user_cf.get_recommendations(user_id, n_recommendations)
            return self._enrich_recommendations(cf_recs, 'collaborative_filtering')
        except Exception as e:
            logger.error(f"Collaborative recommendations failed: {e}")
            return self.get_popular_recommendations(n_recommendations)
    
    def get_content_based_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using only content-based filtering"""
        if not self.content_based_engine:
            logger.warning("Content-based engine not initialized, returning popular items")
            return self.get_popular_recommendations(n_recommendations)
        
        try:
            cb_recs = self.content_based_engine.get_recommendations(user_id, limit=n_recommendations)
            return cb_recs if cb_recs else self.get_popular_recommendations(n_recommendations)
        except Exception as e:
            logger.error(f"Content-based recommendations failed: {e}")
            return self.get_popular_recommendations(n_recommendations)
    
    def get_matrix_factorization_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using only matrix factorization"""
        if not self.matrix_factorization_engine or not self.matrix_factorization_engine.is_trained:
            logger.warning("Matrix factorization engine not trained, returning popular items")
            return self.get_popular_recommendations(n_recommendations)
        
        try:
            mf_recs = self.matrix_factorization_engine.get_recommendations(user_id, limit=n_recommendations)
            return mf_recs if mf_recs else self.get_popular_recommendations(n_recommendations)
        except Exception as e:
            logger.error(f"Matrix factorization recommendations failed: {e}")
            return self.get_popular_recommendations(n_recommendations)
    
    def _enrich_recommendations(self, recs: List[Tuple[str, float]], algorithm: str) -> List[Dict]:
        """Enrich raw recommendations with product details"""
        if not recs:
            return []
        
        conn = self.get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        recommendations = []
        max_score = recs[0][1] if recs else 1
        
        for item_id, score in recs:
            cursor.execute("""
                SELECT 
                    product_id,
                    MAX(product_name) as product_name,
                    AVG(unit_price) as price,
                    SUM(total_price) as revenue
                FROM order_items
                WHERE product_id = %s
                GROUP BY product_id
            """, (item_id,))
            
            product_data = cursor.fetchone()
            
            if product_data:
                recommendations.append({
                    'product_id': str(product_data['product_id']),
                    'product_name': product_data['product_name'],
                    'score': float(score),
                    'price': float(product_data['price'] or 0),
                    'revenue': float(product_data['revenue'] or 0),
                    'algorithm': algorithm,
                    'confidence': min(score / max_score, 1.0) if max_score > 0 else 0
                })
        
        cursor.close()
        conn.close()
        
        return recommendations
    
    def get_model_status(self) -> Dict:
        """Get current status of all ML models"""
        return {
            'is_trained': self.is_trained,
            'training_timestamp': self.training_timestamp.isoformat() if self.training_timestamp else None,
            'model_metadata': self.model_metadata,
            'algorithms': {
                'collaborative_filtering': self.collaborative_engine is not None and self.collaborative_engine.is_trained,
                'content_based': self.content_based_engine is not None and self.content_based_engine.product_vectors is not None,
                'matrix_factorization': self.matrix_factorization_engine is not None and self.matrix_factorization_engine.is_trained,
                'popularity_based': self.popularity_engine is not None
            }
        }
    
    def load_trained_models(self, time_filter: str = 'all'):
        """Load previously trained models - tries DB first, then disk"""
        logger.info(f"Loading trained models for time_filter={time_filter}")
        
        # Try loading from PostgreSQL first (persists across Heroku restarts)
        if self._load_models_from_db(time_filter):
            return
        
        # Fallback to disk (local development)
        try:
            # Load Collaborative Filtering
            cf_path = os.path.join(MODEL_DIR, f'collaborative_filtering_{time_filter}.pkl')
            if os.path.exists(cf_path):
                self.collaborative_engine = CollaborativeFilteringEngine()
                self.collaborative_engine.load_model(cf_path)
                logger.info("✅ Loaded Collaborative Filtering model from disk")
            
            # Load Content-Based
            cb_path = os.path.join(MODEL_DIR, f'content_based_{time_filter}.pkl')
            if os.path.exists(cb_path):
                conn = self.get_db_connection()
                self.content_based_engine = ContentBasedFiltering(pg_conn=conn)
                self.content_based_engine.load_model(cb_path)
                conn.close()
                logger.info("✅ Loaded Content-Based model from disk")
            
            # Load Matrix Factorization
            mf_path = os.path.join(MODEL_DIR, f'matrix_factorization_{time_filter}.pkl')
            if os.path.exists(mf_path):
                conn = self.get_db_connection()
                self.matrix_factorization_engine = MatrixFactorizationSVD(pg_conn=conn, n_factors=30)
                self.matrix_factorization_engine.load_model(mf_path)
                conn.close()
                logger.info("✅ Loaded Matrix Factorization model from disk")
            
            # Initialize Popularity-Based (no loading needed, doesn't use conn)
            self.popularity_engine = PopularityBasedEngine()
            logger.info("✅ Initialized Popularity-Based model")
            
            self.is_trained = True
            logger.info("All models loaded successfully from disk")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            raise
    
    def _load_models_from_db(self, time_filter: str) -> bool:
        """Try to load models from PostgreSQL database"""
        try:
            from services.model_storage import load_model_from_db
        except ImportError:
            try:
                from src.services.model_storage import load_model_from_db
            except ImportError:
                logger.warning("Model storage service not available")
                return False
        
        try:
            loaded_count = 0
            
            # Load Collaborative Filtering
            cf_model, cf_meta = load_model_from_db(f'collaborative_filtering_{time_filter}')
            if cf_model:
                self.collaborative_engine = cf_model
                loaded_count += 1
                logger.info("✅ Loaded Collaborative Filtering from DB")
            
            # Load Content-Based
            cb_model, cb_meta = load_model_from_db(f'content_based_{time_filter}')
            if cb_model:
                self.content_based_engine = cb_model
                loaded_count += 1
                logger.info("✅ Loaded Content-Based from DB")
            
            # Load Matrix Factorization
            mf_model, mf_meta = load_model_from_db(f'matrix_factorization_{time_filter}')
            if mf_model:
                self.matrix_factorization_engine = mf_model
                loaded_count += 1
                logger.info("✅ Loaded Matrix Factorization from DB")
            
            # Load metadata
            meta_obj, _ = load_model_from_db(f'model_metadata_{time_filter}')
            if meta_obj:
                self.model_metadata = meta_obj.get('metadata', {})
                self.training_timestamp = meta_obj.get('training_timestamp')
            
            # Initialize Popularity-Based
            self.popularity_engine = PopularityBasedEngine()
            
            if loaded_count > 0:
                self.is_trained = True
                logger.info(f"✅ Loaded {loaded_count} models from PostgreSQL DB")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Could not load models from DB: {e}")
            return False
    
    def save_models_to_db(self, time_filter: str = '30days'):
        """Save all trained models to PostgreSQL for persistence"""
        try:
            from services.model_storage import save_model_to_db
        except ImportError:
            try:
                from src.services.model_storage import save_model_to_db
            except ImportError:
                logger.warning("Model storage service not available, skipping DB save")
                return False
        
        try:
            saved_count = 0
            
            if self.collaborative_engine:
                if save_model_to_db(f'collaborative_filtering_{time_filter}', 
                                   self.collaborative_engine, time_filter):
                    saved_count += 1
            
            if self.content_based_engine:
                if save_model_to_db(f'content_based_{time_filter}', 
                                   self.content_based_engine, time_filter):
                    saved_count += 1
            
            if self.matrix_factorization_engine:
                if save_model_to_db(f'matrix_factorization_{time_filter}', 
                                   self.matrix_factorization_engine, time_filter):
                    saved_count += 1
            
            # Save metadata
            meta_obj = {
                'metadata': self.model_metadata,
                'training_timestamp': self.training_timestamp
            }
            save_model_to_db(f'model_metadata_{time_filter}', meta_obj, time_filter)
            
            logger.info(f"✅ Saved {saved_count} models to PostgreSQL DB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models to DB: {e}")
            return False
    
    # ==========================================================================
    # PRE-COMPUTED RECOMMENDATIONS (for faster frontend responses)
    # ==========================================================================
    
    def precompute_recommendations(self, time_filter: str = '30days') -> Dict:
        """
        Pre-compute and cache top products, product pairs, and popular items.
        This enables instant responses on the frontend.
        """
        logger.info(f"Pre-computing recommendations for time_filter={time_filter}")
        
        results = {
            'status': 'success',
            'time_filter': time_filter,
            'precomputed': {}
        }
        
        try:
            # 1. Pre-compute Top Products
            top_products = self.get_popular_recommendations(n_recommendations=50)
            self.precomputed_cache['top_products'] = {
                'data': top_products,
                'timestamp': datetime.now().isoformat(),
                'time_filter': time_filter
            }
            results['precomputed']['top_products'] = len(top_products)
            logger.info(f"✅ Pre-computed {len(top_products)} top products")
            
            # 2. Pre-compute Product Pairs from collaborative filtering
            product_pairs = self._compute_product_pairs(time_filter)
            self.precomputed_cache['product_pairs'] = {
                'data': product_pairs,
                'timestamp': datetime.now().isoformat(),
                'time_filter': time_filter
            }
            results['precomputed']['product_pairs'] = len(product_pairs)
            logger.info(f"✅ Pre-computed {len(product_pairs)} product pairs")
            
            # 3. Pre-compute Customer Segments
            customer_segments = self._compute_customer_segments(time_filter)
            self.precomputed_cache['customer_segments'] = {
                'data': customer_segments,
                'timestamp': datetime.now().isoformat(),
                'time_filter': time_filter
            }
            results['precomputed']['customer_segments'] = len(customer_segments)
            logger.info(f"✅ Pre-computed {len(customer_segments)} customer segments")
            
            # Save pre-computed data to disk
            cache_path = os.path.join(MODEL_DIR, f'precomputed_{time_filter}.pkl')
            with open(cache_path, 'wb') as f:
                pickle.dump(self.precomputed_cache, f)
            logger.info(f"✅ Saved pre-computed cache to {cache_path}")
            
            results['cache_path'] = cache_path
            
        except Exception as e:
            logger.error(f"Pre-computation failed: {e}", exc_info=True)
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def _compute_product_pairs(self, time_filter: str) -> List[Dict]:
        """Compute frequently bought together product pairs"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get product pairs from order data
            cursor.execute("""
                WITH order_products AS (
                    SELECT order_id, product_id, product_name
                    FROM order_items
                    WHERE order_id IN (
                        SELECT DISTINCT order_id FROM orders
                        WHERE order_date >= NOW() - INTERVAL '30 days'
                    )
                )
                SELECT 
                    a.product_id as product_a_id,
                    a.product_name as product_a_name,
                    b.product_id as product_b_id,
                    b.product_name as product_b_name,
                    COUNT(*) as co_purchase_count
                FROM order_products a
                JOIN order_products b ON a.order_id = b.order_id AND a.product_id < b.product_id
                GROUP BY a.product_id, a.product_name, b.product_id, b.product_name
                HAVING COUNT(*) >= 3
                ORDER BY co_purchase_count DESC
                LIMIT 100
            """)
            
            pairs = []
            for row in cursor.fetchall():
                pairs.append({
                    'product_a_id': row[0],
                    'product_a_name': row[1],
                    'product_b_id': row[2],
                    'product_b_name': row[3],
                    'co_purchase_count': row[4],
                    'confidence_score': min(row[4] / 10 * 100, 100)  # Normalize to 0-100
                })
            
            cursor.close()
            conn.close()
            return pairs
            
        except Exception as e:
            logger.error(f"Failed to compute product pairs: {e}")
            return []
    
    def _compute_customer_segments(self, time_filter: str) -> List[Dict]:
        """Compute customer similarity segments"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    customer_id,
                    customer_name,
                    COUNT(DISTINCT order_id) as order_count,
                    SUM(total_amount) as total_spent
                FROM orders
                WHERE order_date >= NOW() - INTERVAL '90 days'
                GROUP BY customer_id, customer_name
                HAVING COUNT(DISTINCT order_id) >= 2
                ORDER BY total_spent DESC
                LIMIT 100
            """)
            
            segments = []
            for row in cursor.fetchall():
                segments.append({
                    'customer_id': row[0],
                    'customer_name': row[1],
                    'order_count': row[2],
                    'total_spent': float(row[3]) if row[3] else 0,
                    'similar_customers_count': 5,  # Placeholder
                    'actual_recommendations': 10  # Placeholder
                })
            
            cursor.close()
            conn.close()
            return segments
            
        except Exception as e:
            logger.error(f"Failed to compute customer segments: {e}")
            return []
    
    def get_precomputed(self, cache_key: str) -> Optional[Dict]:
        """Get pre-computed data from cache"""
        if cache_key in self.precomputed_cache:
            return self.precomputed_cache[cache_key]
        
        # Try to load from disk
        for time_filter in ['30days', 'all', '7days']:
            cache_path = os.path.join(MODEL_DIR, f'precomputed_{time_filter}.pkl')
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        self.precomputed_cache = pickle.load(f)
                    if cache_key in self.precomputed_cache:
                        return self.precomputed_cache[cache_key]
                except Exception as e:
                    logger.error(f"Failed to load precomputed cache: {e}")
        
        return None
    
    # ==========================================================================
    # A/B TESTING SUPPORT
    # ==========================================================================
    
    def get_ab_test_recommendation(
        self, 
        user_id: str, 
        algorithm: str,  # 'collaborative', 'content_based', 'matrix_factorization', 'hybrid', 'popularity'
        n_recommendations: int = 10
    ) -> Dict:
        """
        Get recommendations using a specific algorithm for A/B testing.
        Allows frontend to compare different algorithms.
        """
        result = {
            'algorithm': algorithm,
            'user_id': user_id,
            'recommendations': [],
            'metadata': {}
        }
        
        try:
            if algorithm == 'collaborative':
                recs = self.get_collaborative_recommendations(user_id, n_recommendations)
                result['recommendations'] = recs
                result['metadata']['description'] = 'User-based collaborative filtering'
                
            elif algorithm == 'content_based':
                recs = self.get_content_based_recommendations(user_id, n_recommendations)
                result['recommendations'] = recs
                result['metadata']['description'] = 'Content-based filtering using product features'
                
            elif algorithm == 'matrix_factorization':
                recs = self.get_matrix_factorization_recommendations(user_id, n_recommendations)
                result['recommendations'] = recs
                result['metadata']['description'] = 'Matrix factorization (SVD) latent factors'
                
            elif algorithm == 'popularity':
                recs = self.get_popular_recommendations(n_recommendations)
                result['recommendations'] = recs
                result['metadata']['description'] = 'Popularity-based recommendations'
                
            elif algorithm == 'hybrid':
                recs = self.get_hybrid_recommendations(user_id, n_recommendations)
                result['recommendations'] = recs
                result['metadata']['description'] = 'Hybrid ensemble of all algorithms'
                
            else:
                result['error'] = f"Unknown algorithm: {algorithm}"
            
            result['count'] = len(result['recommendations'])
            
        except Exception as e:
            logger.error(f"A/B test recommendation failed for {algorithm}: {e}")
            result['error'] = str(e)
        
        return result
    
    def get_ab_test_config(self) -> Dict:
        """Get A/B test configuration for frontend"""
        return {
            'algorithms': [
                {
                    'id': 'hybrid',
                    'name': 'Hybrid Ensemble',
                    'description': 'Combines all algorithms with weighted scoring',
                    'weight': 40,  # 40% of traffic
                    'is_default': True
                },
                {
                    'id': 'collaborative',
                    'name': 'Collaborative Filtering',
                    'description': 'Recommendations based on similar users',
                    'weight': 25,  # 25% of traffic
                    'is_default': False
                },
                {
                    'id': 'content_based',
                    'name': 'Content-Based',
                    'description': 'Recommendations based on product similarity',
                    'weight': 15,  # 15% of traffic
                    'is_default': False
                },
                {
                    'id': 'matrix_factorization',
                    'name': 'Matrix Factorization',
                    'description': 'SVD-based latent factor model',
                    'weight': 10,  # 10% of traffic
                    'is_default': False
                },
                {
                    'id': 'popularity',
                    'name': 'Popularity-Based',
                    'description': 'Most popular products (baseline)',
                    'weight': 10,  # 10% of traffic
                    'is_default': False
                }
            ],
            'enabled': True,
            'tracking_enabled': True
        }


# Global service instance
_ml_service = None

def get_ml_service() -> MLRecommendationService:
    """Get or create global ML service instance"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLRecommendationService()
    return _ml_service

