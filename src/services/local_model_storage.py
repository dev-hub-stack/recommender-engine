"""
Local File-Based Model Storage Service
Stores trained ML models as joblib files for fast loading and persistence.

Benefits over database storage:
- Faster load times (no network overhead)
- Easier debugging (can inspect files directly)
- Simple versioning with timestamps
- Works offline
"""

import os
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import structlog

logger = structlog.get_logger()

# Default models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_METADATA_FILE = "metadata.json"


def ensure_models_dir():
    """Create models directory if it doesn't exist"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def get_model_path(model_name: str, version: Optional[str] = None) -> Path:
    """
    Get path to a model file.
    
    Args:
        model_name: Name of the model (e.g., 'svd_recommender', 'item_similarity')
        version: Optional version string (e.g., '20231215_143022'). If None, uses 'latest'
    
    Returns:
        Path to the model file
    """
    models_dir = ensure_models_dir()
    
    if version:
        return models_dir / f"{model_name}_{version}.joblib"
    else:
        # Return latest version
        return models_dir / f"{model_name}_latest.joblib"


def save_model(
    model_name: str,
    model_object: Any,
    metadata: Optional[Dict] = None,
    create_version: bool = True
) -> Dict[str, Any]:
    """
    Save a trained model to disk as a joblib file.
    
    Args:
        model_name: Unique name for the model
        model_object: The trained model object to save
        metadata: Additional metadata (training time, metrics, etc.)
        create_version: If True, also saves a timestamped version
    
    Returns:
        Dict with save info (path, version, size_mb)
    """
    try:
        models_dir = ensure_models_dir()
        
        # Generate version timestamp
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare metadata
        model_metadata = {
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_type": type(model_object).__name__,
            **(metadata or {})
        }
        
        # Save latest version
        latest_path = models_dir / f"{model_name}_latest.joblib"
        joblib.dump({
            "model": model_object,
            "metadata": model_metadata
        }, latest_path, compress=3)
        
        # Save versioned copy if requested
        versioned_path = None
        if create_version:
            versioned_path = models_dir / f"{model_name}_{version}.joblib"
            joblib.dump({
                "model": model_object,
                "metadata": model_metadata
            }, versioned_path, compress=3)
        
        # Calculate file size
        size_bytes = latest_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        # Update global metadata file
        _update_metadata_index(model_name, model_metadata)
        
        logger.info(
            f"✅ Saved model '{model_name}' (v{version})",
            size_mb=f"{size_mb:.2f}",
            path=str(latest_path)
        )
        
        return {
            "success": True,
            "model_name": model_name,
            "version": version,
            "path": str(latest_path),
            "versioned_path": str(versioned_path) if versioned_path else None,
            "size_mb": round(size_mb, 2),
            "metadata": model_metadata
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to save model '{model_name}': {e}")
        return {
            "success": False,
            "error": str(e)
        }


def load_model(
    model_name: str,
    version: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Load a trained model from disk.
    
    Args:
        model_name: Name of the model to load
        version: Optional version string. If None, loads latest
    
    Returns:
        Dict with 'model' and 'metadata', or None if not found
    """
    try:
        model_path = get_model_path(model_name, version)
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        data = joblib.load(model_path)
        
        logger.info(
            f"✅ Loaded model '{model_name}'",
            version=data.get("metadata", {}).get("version", "unknown")
        )
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Failed to load model '{model_name}': {e}")
        return None


def list_models() -> List[Dict[str, Any]]:
    """
    List all available models and their versions.
    
    Returns:
        List of dicts with model info
    """
    models_dir = ensure_models_dir()
    models = []
    
    for path in models_dir.glob("*_latest.joblib"):
        model_name = path.stem.replace("_latest", "")
        
        try:
            data = joblib.load(path)
            metadata = data.get("metadata", {})
            
            # Find all versions
            versions = []
            for v_path in models_dir.glob(f"{model_name}_*.joblib"):
                if "_latest" not in v_path.stem:
                    v = v_path.stem.replace(f"{model_name}_", "")
                    versions.append(v)
            
            models.append({
                "model_name": model_name,
                "latest_version": metadata.get("version"),
                "created_at": metadata.get("created_at"),
                "model_type": metadata.get("model_type"),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "versions": sorted(versions, reverse=True)[:5],  # Last 5 versions
                "metadata": metadata
            })
        except Exception as e:
            logger.warning(f"Could not read model {path}: {e}")
    
    return models


def delete_model(model_name: str, version: Optional[str] = None) -> bool:
    """
    Delete a model (specific version or all versions).
    
    Args:
        model_name: Name of the model
        version: If provided, deletes only that version. If None, deletes all.
    
    Returns:
        True if successful
    """
    models_dir = ensure_models_dir()
    deleted = False
    
    try:
        if version:
            # Delete specific version
            path = models_dir / f"{model_name}_{version}.joblib"
            if path.exists():
                path.unlink()
                deleted = True
        else:
            # Delete all versions
            for path in models_dir.glob(f"{model_name}_*.joblib"):
                path.unlink()
                deleted = True
        
        if deleted:
            logger.info(f"🗑️ Deleted model '{model_name}'" + (f" v{version}" if version else " (all versions)"))
        
        return deleted
        
    except Exception as e:
        logger.error(f"❌ Failed to delete model '{model_name}': {e}")
        return False


def cleanup_old_versions(model_name: str, keep_versions: int = 3) -> int:
    """
    Clean up old model versions, keeping only the most recent N versions.
    
    Args:
        model_name: Name of the model
        keep_versions: Number of versions to keep (default 3)
    
    Returns:
        Number of deleted versions
    """
    models_dir = ensure_models_dir()
    
    # Find all versioned files (excluding latest)
    versions = []
    for path in models_dir.glob(f"{model_name}_*.joblib"):
        if "_latest" not in path.stem:
            version = path.stem.replace(f"{model_name}_", "")
            versions.append((version, path))
    
    # Sort by version (timestamp) descending
    versions.sort(key=lambda x: x[0], reverse=True)
    
    # Delete old versions
    deleted_count = 0
    for version, path in versions[keep_versions:]:
        path.unlink()
        deleted_count += 1
    
    if deleted_count > 0:
        logger.info(f"🧹 Cleaned up {deleted_count} old versions of '{model_name}'")
    
    return deleted_count


def _update_metadata_index(model_name: str, metadata: Dict):
    """Update the global metadata index file"""
    models_dir = ensure_models_dir()
    metadata_path = models_dir / MODEL_METADATA_FILE
    
    try:
        # Load existing metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}
        
        # Update entry for this model
        all_metadata[model_name] = {
            "latest_version": metadata.get("version"),
            "updated_at": datetime.now().isoformat(),
            **metadata
        }
        
        # Save back
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2, default=str)
            
    except Exception as e:
        logger.warning(f"Could not update metadata index: {e}")


def get_model_info(model_name: str) -> Optional[Dict]:
    """Get metadata for a specific model without loading the full model"""
    models_dir = ensure_models_dir()
    metadata_path = models_dir / MODEL_METADATA_FILE
    
    try:
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
                return all_metadata.get(model_name)
    except Exception as e:
        logger.warning(f"Could not read metadata: {e}")
    
    return None
