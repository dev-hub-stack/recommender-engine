#!/usr/bin/env python3
"""
Train ML Models on Full Dataset and Deploy to Heroku

Usage:
    # Train locally first
    python3 train_and_deploy.py --train
    
    # Then deploy to Heroku
    git push heroku main
    
    # After deploy, save models to Heroku PostgreSQL
    python3 train_and_deploy.py --save-to-heroku
"""

import argparse
import requests
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOCAL_URL = "http://localhost:8001"
HEROKU_URL = os.getenv("HEROKU_APP_URL", "https://your-app.herokuapp.com")


def train_locally(time_filter='all'):
    """Train models locally on full dataset"""
    print("=" * 60)
    print("   TRAINING ML MODELS ON FULL DATASET")
    print("=" * 60)
    print()
    
    try:
        print(f"🚀 Starting training with time_filter={time_filter}...")
        print("   This may take 2-5 minutes for full dataset...")
        print()
        
        response = requests.post(
            f"{LOCAL_URL}/api/v1/ml/train",
            params={"time_filter": time_filter, "force_retrain": "true"},
            timeout=600  # 10 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training completed!")
            print()
            print("Models trained:")
            for model, info in result.get('training_results', {}).get('models', {}).items():
                status = "✅" if info.get('status') == 'success' else "❌"
                time_sec = info.get('training_time_seconds', 0)
                print(f"   {status} {model}: {time_sec:.1f}s")
            print()
            print(f"Total time: {result.get('training_results', {}).get('total_training_time_seconds', 0):.1f}s")
            return True
        else:
            print(f"❌ Training failed: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to local server. Please start it first:")
        print("   PORT=8001 python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8001")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def save_models_locally():
    """Save trained models to PostgreSQL"""
    print()
    print("💾 Saving models to PostgreSQL...")
    
    try:
        response = requests.post(
            f"{LOCAL_URL}/api/v1/ml/models/save",
            params={"time_filter": "all"},
            timeout=120
        )
        
        if response.status_code == 200:
            print("✅ Models saved to PostgreSQL")
            return True
        else:
            print(f"❌ Failed to save: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_stored_models():
    """Check what models are stored"""
    print()
    print("📋 Checking stored models...")
    
    try:
        response = requests.get(f"{LOCAL_URL}/api/v1/ml/models/stored", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            models = result.get('stored_models', [])
            
            if models:
                print(f"✅ Found {len(models)} models in PostgreSQL:")
                for m in models:
                    print(f"   • {m['name']} ({m['size_mb']:.2f} MB) - {m['updated_at']}")
            else:
                print("⚠️  No models stored in PostgreSQL yet")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_endpoints():
    """Test all ML endpoints"""
    print()
    print("🧪 Testing ML endpoints...")
    
    endpoints = [
        ("Top Products", "/api/v1/ml/top-products?limit=3"),
        ("Product Pairs", "/api/v1/ml/product-pairs?limit=3"),
        ("Customer Similarity", "/api/v1/ml/customer-similarity?limit=3"),
        ("Collaborative Products", "/api/v1/ml/collaborative-products?limit=3"),
        ("RFM Segments", "/api/v1/ml/rfm-segments"),
    ]
    
    success_count = 0
    for name, endpoint in endpoints:
        try:
            response = requests.get(f"{LOCAL_URL}{endpoint}", timeout=60)
            if response.status_code == 200:
                print(f"   ✅ {name}")
                success_count += 1
            else:
                print(f"   ❌ {name}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    print()
    print(f"   {success_count}/{len(endpoints)} endpoints working")
    return success_count == len(endpoints)


def print_deploy_instructions():
    """Print Heroku deployment instructions"""
    print()
    print("=" * 60)
    print("   HEROKU DEPLOYMENT INSTRUCTIONS")
    print("=" * 60)
    print("""
1. COMMIT CHANGES:
   git add .
   git commit -m "ML models trained on full dataset"

2. DEPLOY TO HEROKU:
   git push heroku main

3. AFTER DEPLOY - Train models on Heroku (if needed):
   curl -X POST "https://YOUR-APP.herokuapp.com/api/v1/ml/train?time_filter=all&force_retrain=true"

4. VERIFY MODELS:
   curl "https://YOUR-APP.herokuapp.com/api/v1/ml/status"
   curl "https://YOUR-APP.herokuapp.com/api/v1/ml/models/stored"

5. TEST ENDPOINTS:
   curl "https://YOUR-APP.herokuapp.com/api/v1/ml/top-products?limit=5"

NOTE: Models are automatically saved to PostgreSQL after training,
      so they persist across Heroku dyno restarts!
""")


def main():
    parser = argparse.ArgumentParser(description='Train and Deploy ML Models')
    parser.add_argument('--train', action='store_true', help='Train models on full dataset')
    parser.add_argument('--time-filter', default='all', help='Time filter (all, 30days, 90days)')
    parser.add_argument('--save', action='store_true', help='Save models to PostgreSQL')
    parser.add_argument('--check', action='store_true', help='Check stored models')
    parser.add_argument('--test', action='store_true', help='Test ML endpoints')
    parser.add_argument('--deploy-info', action='store_true', help='Show deployment instructions')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    if args.all or args.train:
        if not train_locally(args.time_filter):
            sys.exit(1)
    
    if args.all or args.save:
        save_models_locally()
    
    if args.all or args.check:
        check_stored_models()
    
    if args.all or args.test:
        test_endpoints()
    
    if args.all or args.deploy_info:
        print_deploy_instructions()
    
    if not any([args.train, args.save, args.check, args.test, args.deploy_info, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
