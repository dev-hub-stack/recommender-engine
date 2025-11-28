"""
S3 Model Storage Service
Stores large ML models in AWS S3 for persistence across Heroku/server restarts.

Usage:
    # Save model after training
    save_model_to_s3('collaborative_filtering_all', model_object)
    
    # Load model on startup
    model = load_model_from_s3('collaborative_filtering_all')

Environment Variables Required:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_S3_BUCKET (default: mastergroup-ml-models)
    AWS_REGION (default: us-east-1)
"""

import boto3
import pickle
import os
import io
import structlog
from datetime import datetime
from typing import Optional, Tuple, Dict, List

logger = structlog.get_logger()

# S3 Configuration
S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'mastergroup-ml-models')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')


def get_s3_client():
    """Get S3 client with credentials from environment"""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=AWS_REGION
    )


def ensure_bucket_exists():
    """Create S3 bucket if it doesn't exist"""
    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"✅ S3 bucket exists: {S3_BUCKET}")
    except Exception:
        try:
            s3 = get_s3_client()
            if AWS_REGION == 'us-east-1':
                s3.create_bucket(Bucket=S3_BUCKET)
            else:
                s3.create_bucket(
                    Bucket=S3_BUCKET,
                    CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                )
            logger.info(f"✅ Created S3 bucket: {S3_BUCKET}")
        except Exception as e:
            logger.error(f"❌ Could not create bucket: {e}")
            raise


def save_model_to_s3(
    model_name: str, 
    model_object, 
    time_filter: str = 'all',
    metadata: Optional[Dict] = None
) -> bool:
    """
    Save a trained model to S3.
    
    Args:
        model_name: Unique name for the model (e.g., 'collaborative_filtering')
        model_object: The trained model object to pickle and upload
        time_filter: Time filter used for training
        metadata: Additional metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        s3 = get_s3_client()
        
        # Serialize model to bytes
        logger.info(f"📦 Serializing model '{model_name}'...")
        model_bytes = pickle.dumps(model_object)
        size_gb = len(model_bytes) / (1024 * 1024 * 1024)
        logger.info(f"   Size: {size_gb:.2f} GB")
        
        # S3 key
        s3_key = f"models/{model_name}_{time_filter}.pkl"
        
        # Upload to S3
        logger.info(f"☁️  Uploading to S3: s3://{S3_BUCKET}/{s3_key}")
        
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=model_bytes,
            Metadata={
                'time_filter': time_filter,
                'created_at': datetime.now().isoformat(),
                'size_bytes': str(len(model_bytes))
            }
        )
        
        logger.info(f"✅ Model '{model_name}' uploaded to S3 ({size_gb:.2f} GB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save model to S3: {e}")
        return False


def load_model_from_s3(
    model_name: str, 
    time_filter: str = 'all'
) -> Tuple[Optional[object], Optional[Dict]]:
    """
    Load a trained model from S3.
    
    Args:
        model_name: Name of the model to load
        time_filter: Time filter used for training
        
    Returns:
        Tuple of (model_object, metadata) or (None, None) if not found
    """
    try:
        s3 = get_s3_client()
        s3_key = f"models/{model_name}_{time_filter}.pkl"
        
        logger.info(f"☁️  Downloading from S3: s3://{S3_BUCKET}/{s3_key}")
        
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        model_bytes = response['Body'].read()
        
        size_gb = len(model_bytes) / (1024 * 1024 * 1024)
        logger.info(f"   Downloaded: {size_gb:.2f} GB")
        
        # Deserialize
        logger.info(f"📦 Deserializing model...")
        model_object = pickle.loads(model_bytes)
        
        # Get metadata
        metadata = response.get('Metadata', {})
        metadata['size_bytes'] = len(model_bytes)
        
        logger.info(f"✅ Model '{model_name}' loaded from S3")
        return model_object, metadata
        
    except s3.exceptions.NoSuchKey:
        logger.warning(f"⚠️ Model '{model_name}' not found in S3")
        return None, None
    except Exception as e:
        logger.error(f"❌ Failed to load model from S3: {e}")
        return None, None


def list_models_in_s3() -> List[Dict]:
    """List all models stored in S3"""
    try:
        s3 = get_s3_client()
        
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='models/'
        )
        
        models = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            size_gb = obj['Size'] / (1024 * 1024 * 1024)
            
            models.append({
                'key': key,
                'name': key.replace('models/', '').replace('.pkl', ''),
                'size_gb': round(size_gb, 2),
                'last_modified': obj['LastModified'].isoformat()
            })
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to list S3 models: {e}")
        return []


def delete_model_from_s3(model_name: str, time_filter: str = 'all') -> bool:
    """Delete a model from S3"""
    try:
        s3 = get_s3_client()
        s3_key = f"models/{model_name}_{time_filter}.pkl"
        
        s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        logger.info(f"🗑️ Deleted model from S3: {s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete from S3: {e}")
        return False


def get_model_info_from_s3(model_name: str, time_filter: str = 'all') -> Optional[Dict]:
    """Get model info without downloading the full model"""
    try:
        s3 = get_s3_client()
        s3_key = f"models/{model_name}_{time_filter}.pkl"
        
        response = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        
        return {
            'name': model_name,
            'time_filter': time_filter,
            'size_gb': round(response['ContentLength'] / (1024 * 1024 * 1024), 2),
            'last_modified': response['LastModified'].isoformat(),
            'metadata': response.get('Metadata', {})
        }
        
    except Exception as e:
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS FOR ML SERVICE
# =============================================================================

def save_all_models_to_s3(ml_service, time_filter: str = 'all') -> Dict:
    """
    Save all trained models from MLRecommendationService to S3.
    
    Args:
        ml_service: Instance of MLRecommendationService
        time_filter: Time filter used for training
        
    Returns:
        Dict with results for each model
    """
    results = {}
    
    if ml_service.collaborative_engine:
        results['collaborative_filtering'] = save_model_to_s3(
            'collaborative_filtering', 
            ml_service.collaborative_engine, 
            time_filter
        )
    
    if ml_service.content_based_engine:
        results['content_based'] = save_model_to_s3(
            'content_based', 
            ml_service.content_based_engine, 
            time_filter
        )
    
    if ml_service.matrix_factorization_engine:
        results['matrix_factorization'] = save_model_to_s3(
            'matrix_factorization', 
            ml_service.matrix_factorization_engine, 
            time_filter
        )
    
    # Save metadata
    metadata = {
        'training_timestamp': ml_service.training_timestamp,
        'model_metadata': ml_service.model_metadata
    }
    results['metadata'] = save_model_to_s3('model_metadata', metadata, time_filter)
    
    return results


def load_all_models_from_s3(ml_service, time_filter: str = 'all') -> bool:
    """
    Load all models from S3 into MLRecommendationService.
    
    Args:
        ml_service: Instance of MLRecommendationService
        time_filter: Time filter used for training
        
    Returns:
        True if at least one model loaded successfully
    """
    loaded_count = 0
    
    # Load Collaborative Filtering (main model - contains user/item similarity matrices)
    cf_model, _ = load_model_from_s3('collaborative_filtering', time_filter)
    if cf_model:
        ml_service.collaborative_engine = cf_model
        loaded_count += 1
        logger.info("✅ Loaded collaborative_filtering from S3")
    
    # Load Content-Based (optional - can rebuild if not present)
    cb_model, _ = load_model_from_s3('content_based', time_filter)
    if cb_model:
        ml_service.content_based_engine = cb_model
        loaded_count += 1
        logger.info("✅ Loaded content_based from S3")
    else:
        logger.info("ℹ️ content_based not in S3 - will use DB queries")
    
    # Load Matrix Factorization (optional - can rebuild if not present)
    mf_model, _ = load_model_from_s3('matrix_factorization', time_filter)
    if mf_model:
        ml_service.matrix_factorization_engine = mf_model
        loaded_count += 1
        logger.info("✅ Loaded matrix_factorization from S3")
    else:
        logger.info("ℹ️ matrix_factorization not in S3 - will use DB queries")
    
    # Load Metadata
    metadata, _ = load_model_from_s3('model_metadata', time_filter)
    if metadata:
        ml_service.training_timestamp = metadata.get('training_timestamp')
        ml_service.model_metadata = metadata.get('model_metadata', {})
    
    # Initialize popularity engine (doesn't need saved model)
    try:
        from src.algorithms.popularity_based import PopularityBasedEngine
        ml_service.popularity_engine = PopularityBasedEngine()
        logger.info("✅ Initialized popularity_based engine")
    except Exception as e:
        logger.warning(f"Could not initialize popularity engine: {e}")
    
    if loaded_count > 0:
        ml_service.is_trained = True
        logger.info(f"✅ Loaded {loaded_count} models from S3")
        return True
    
    return False
