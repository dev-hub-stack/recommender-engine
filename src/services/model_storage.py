"""
Model Storage Service
Stores trained ML models in PostgreSQL for persistence across Heroku dyno restarts
"""

import pickle
import base64
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import structlog
import os

logger = structlog.get_logger()

def get_pg_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv('PG_HOST', 'localhost'),
        port=int(os.getenv('PG_PORT', '5432')),
        database=os.getenv('PG_DB', 'mastergroup_recommendations'),
        user=os.getenv('PG_USER', 'postgres'),
        password=os.getenv('PG_PASSWORD', 'postgres')
    )


def ensure_model_table():
    """Create model storage table if not exists"""
    conn = get_pg_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_models (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) UNIQUE NOT NULL,
            model_data BYTEA NOT NULL,
            time_filter VARCHAR(50),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    logger.info("✅ Model storage table ready")


def save_model_to_db(model_name: str, model_object, time_filter: str = '30days', metadata: dict = None):
    """
    Save a trained model to PostgreSQL
    
    Args:
        model_name: Unique name for the model (e.g., 'collaborative_filtering')
        model_object: The trained model object to pickle and store
        time_filter: Time filter used for training
        metadata: Additional metadata (training time, accuracy, etc.)
    """
    try:
        ensure_model_table()
        
        # Serialize model
        model_bytes = pickle.dumps(model_object)
        
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Upsert model
        cursor.execute("""
            INSERT INTO ml_models (model_name, model_data, time_filter, metadata, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (model_name) 
            DO UPDATE SET 
                model_data = EXCLUDED.model_data,
                time_filter = EXCLUDED.time_filter,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """, (model_name, model_bytes, time_filter, psycopg2.extras.Json(metadata or {})))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        size_mb = len(model_bytes) / (1024 * 1024)
        logger.info(f"✅ Saved model '{model_name}' to DB ({size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save model '{model_name}': {e}")
        return False


def load_model_from_db(model_name: str):
    """
    Load a trained model from PostgreSQL
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model_object, metadata) or (None, None) if not found
    """
    try:
        ensure_model_table()
        
        conn = get_pg_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT model_data, metadata, time_filter, updated_at 
            FROM ml_models 
            WHERE model_name = %s
        """, (model_name,))
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row:
            model_object = pickle.loads(row['model_data'])
            metadata = row['metadata'] or {}
            metadata['time_filter'] = row['time_filter']
            metadata['last_updated'] = row['updated_at'].isoformat() if row['updated_at'] else None
            
            logger.info(f"✅ Loaded model '{model_name}' from DB")
            return model_object, metadata
        else:
            logger.warning(f"⚠️ Model '{model_name}' not found in DB")
            return None, None
            
    except Exception as e:
        logger.error(f"❌ Failed to load model '{model_name}': {e}")
        return None, None


def get_model_info():
    """Get info about all stored models"""
    try:
        ensure_model_table()
        
        conn = get_pg_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                model_name,
                time_filter,
                metadata,
                LENGTH(model_data) as size_bytes,
                created_at,
                updated_at
            FROM ml_models
            ORDER BY updated_at DESC
        """)
        
        models = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [{
            'name': m['model_name'],
            'time_filter': m['time_filter'],
            'size_mb': round(m['size_bytes'] / (1024 * 1024), 2),
            'metadata': m['metadata'],
            'created_at': m['created_at'].isoformat() if m['created_at'] else None,
            'updated_at': m['updated_at'].isoformat() if m['updated_at'] else None
        } for m in models]
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return []


def delete_model_from_db(model_name: str):
    """Delete a model from the database"""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ml_models WHERE model_name = %s", (model_name,))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"🗑️ Deleted model '{model_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        return False
