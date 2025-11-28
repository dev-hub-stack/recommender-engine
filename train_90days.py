#!/usr/bin/env python3
"""
Train ML Models on 90 Days of Data
Optimal balance of data freshness, model size, and training time.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

print("=" * 60)
print("   TRAINING ML MODELS ON 90 DAYS OF DATA")
print("=" * 60)
print()
print("Why 90 days?")
print("  • Recent data is more relevant for predictions")
print("  • Model size: ~300 MB (vs 41 GB for all data)")
print("  • Training time: ~15 min (vs 35 min)")
print("  • Can store in PostgreSQL for Heroku")
print()

from src.algorithms.ml_recommendation_service import get_ml_service

ml_service = get_ml_service()

print("🚀 Starting training...")
print()

try:
    results = ml_service.train_all_models(
        time_filter='90days',
        force_retrain=True
    )
    
    print()
    print("=" * 60)
    print("   TRAINING RESULTS")
    print("=" * 60)
    
    for model_name, info in results.get('models', {}).items():
        status = "✅" if info.get('status') == 'success' else "❌"
        time_sec = info.get('training_time_seconds', info.get('initialization_time_seconds', 0))
        print(f"   {status} {model_name}: {time_sec:.1f}s")
    
    print()
    print(f"   Total: {results.get('total_training_time_seconds', 0):.1f}s")
    print(f"   Models: {results.get('successful_models', 0)}/{results.get('total_models', 4)}")
    
    # Check model file sizes
    print()
    print("=" * 60)
    print("   MODEL FILE SIZES")
    print("=" * 60)
    
    model_dir = '/tmp/ml_models'
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if '90days' in f:
                size = os.path.getsize(os.path.join(model_dir, f))
                size_mb = size / (1024 * 1024)
                print(f"   {f}: {size_mb:.1f} MB")
    
    print()
    print("✅ Training complete! Models ready for Heroku deployment.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
