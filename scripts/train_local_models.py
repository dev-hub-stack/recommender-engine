#!/usr/bin/env python3
"""
Local Model Training Script

Trains recommendation models locally and saves them as joblib files.
Can be run manually or scheduled via cron.

Usage:
    python scripts/train_local_models.py
    python scripts/train_local_models.py --days 90
    python scripts/train_local_models.py --days 30 --cleanup

Environment variables required:
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.services.local_recommender import LocalRecommender, get_recommender
from src.services.local_model_storage import list_models, cleanup_old_versions

import structlog
logger = structlog.get_logger()


def train_models(days: int = 90, cleanup: bool = False) -> dict:
    """
    Main training function.
    
    Args:
        days: Number of days of historical data to use
        cleanup: Whether to clean up old model versions
    
    Returns:
        Training results dict
    """
    print("=" * 60)
    print(f"  LOCAL MODEL TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Training data: Last {days} days")
    print(f"  Cleanup old versions: {cleanup}")
    print(f"  Database: {os.getenv('PG_HOST', 'localhost')}:{os.getenv('PG_PORT', '5432')}")
    print()
    
    # Create recommender and train
    recommender = LocalRecommender()
    
    print("Starting training...")
    results = recommender.train_all_models(days=days)
    
    # Print results
    print("\n" + "=" * 60)
    print("  TRAINING RESULTS")
    print("=" * 60)
    
    if results.get("success"):
        print(f"\n✅ Training completed successfully!")
        print(f"   Duration: {results.get('training_duration_seconds', 0):.1f} seconds")
        print(f"   Data points: {results.get('data_points', 0):,}")
        print(f"   Unique users: {results.get('unique_users', 0):,}")
        print(f"   Unique items: {results.get('unique_items', 0):,}")
        
        print("\n📊 Model Results:")
        for model_name, model_result in results.get("models", {}).items():
            status = "✅" if model_result.get("success") else "❌"
            print(f"\n   {status} {model_name}:")
            for key, value in model_result.items():
                if key != "success":
                    print(f"      {key}: {value}")
    else:
        print(f"\n❌ Training failed: {results.get('error', 'Unknown error')}")
    
    # Cleanup old versions if requested
    if cleanup:
        print("\n🧹 Cleaning up old model versions...")
        for model in ['svd_recommender', 'item_similarity', 'popularity_scores']:
            deleted = cleanup_old_versions(model, keep_versions=3)
            if deleted > 0:
                print(f"   Deleted {deleted} old versions of {model}")
    
    # Show all models on disk
    print("\n📁 Models on disk:")
    models = list_models()
    for model in models:
        print(f"   - {model['model_name']} (v{model['latest_version']}, {model['size_mb']} MB)")
    
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    
    return results


def test_recommendations(user_id: str = None, item_id: str = None):
    """Test the trained models with sample recommendations"""
    print("\n" + "=" * 60)
    print("  TESTING RECOMMENDATIONS")
    print("=" * 60)
    
    recommender = get_recommender()
    status = recommender.get_model_status()
    
    print(f"\nModel Status:")
    print(f"   SVD loaded: {status['svd_loaded']}")
    print(f"   Item similarity loaded: {status['item_similarity_loaded']}")
    print(f"   Popularity loaded: {status['popularity_loaded']}")
    
    # Test popularity recommendations
    print("\n📈 Top 5 Popular Items:")
    popular = recommender._popularity_recommendations(5)
    for i, item in enumerate(popular, 1):
        print(f"   {i}. {item['item_name']} (score: {item['score']})")
    
    # Test user recommendations if user_id provided
    if user_id:
        print(f"\n👤 Recommendations for user {user_id}:")
        recs = recommender.get_user_recommendations(user_id, limit=5)
        for i, item in enumerate(recs, 1):
            print(f"   {i}. {item['item_name']} (score: {item['score']}, algo: {item['algorithm']})")
    
    # Test similar items if item_id provided
    if item_id:
        print(f"\n🔗 Similar items to {item_id}:")
        similar = recommender.get_similar_items(item_id, limit=5)
        for i, item in enumerate(similar, 1):
            score_key = 'similarity_score' if 'similarity_score' in item else 'score'
            print(f"   {i}. {item['item_name']} ({score_key}: {item.get(score_key, 'N/A')})")


def main():
    parser = argparse.ArgumentParser(description='Train local recommendation models')
    parser.add_argument(
        '--days', 
        type=int, 
        default=90,
        help='Number of days of historical data to use (default: 90)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old model versions after training'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only test existing models, do not train'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        default=None,
        help='User ID to test recommendations for'
    )
    parser.add_argument(
        '--item-id',
        type=str,
        default=None,
        help='Item ID to test similar items for'
    )
    
    args = parser.parse_args()
    
    if args.test_only:
        test_recommendations(args.user_id, args.item_id)
    else:
        results = train_models(days=args.days, cleanup=args.cleanup)
        
        # Also run tests after training
        test_recommendations(args.user_id, args.item_id)
        
        return 0 if results.get("success") else 1


if __name__ == "__main__":
    exit(main())
