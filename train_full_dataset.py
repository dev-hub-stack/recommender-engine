#!/usr/bin/env python3
"""
Train ML Models on Full Dataset - Direct Script
Runs training without needing the web server.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

print("=" * 60)
print("   TRAINING ML MODELS ON FULL DATASET")
print("=" * 60)
print()

# Import ML service
try:
    from src.algorithms.ml_recommendation_service import get_ml_service
    print("✅ Imported ML service")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Get service instance
ml_service = get_ml_service()

# Train all models on full dataset
print()
print("🚀 Training all 4 ML models on FULL dataset...")
print("   This may take 2-5 minutes...")
print()

try:
    results = ml_service.train_all_models(
        time_filter='all',  # Use ALL data
        force_retrain=True
    )
    
    print()
    print("=" * 60)
    print("   TRAINING RESULTS")
    print("=" * 60)
    print()
    
    for model_name, model_info in results.get('models', {}).items():
        status = "✅" if model_info.get('status') == 'success' else "❌"
        time_sec = model_info.get('training_time_seconds', model_info.get('initialization_time_seconds', 0))
        print(f"   {status} {model_name}: {time_sec:.1f}s")
    
    print()
    print(f"   Total training time: {results.get('total_training_time_seconds', 0):.1f}s")
    print(f"   Successful models: {results.get('successful_models', 0)}/{results.get('total_models', 4)}")
    
    # Save to PostgreSQL
    print()
    print("💾 Saving models to PostgreSQL for Heroku persistence...")
    
    try:
        success = ml_service.save_models_to_db(time_filter='all')
        if success:
            print("✅ Models saved to PostgreSQL!")
        else:
            print("⚠️  Could not save to PostgreSQL (will save on first deploy)")
    except Exception as e:
        print(f"⚠️  Save to DB skipped: {e}")
    
    print()
    print("=" * 60)
    print("   NEXT STEPS - DEPLOY TO HEROKU")
    print("=" * 60)
    print("""
1. Commit changes:
   git add .
   git commit -m "Trained ML models on full dataset"

2. Deploy to Heroku:
   git push heroku main

3. After deploy, verify:
   curl https://YOUR-APP.herokuapp.com/api/v1/ml/status

Models are saved to PostgreSQL, so they persist across restarts!
""")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
