"""
Run model training
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training import ModelTrainer
from config.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRAIN_TEST_SPLIT_METHOD,
    TRAIN_TEST_SPLIT_RATIO,
    TEST_DAYS,
    MIN_CUSTOMER_INTERACTIONS,
    MIN_PRODUCT_INTERACTIONS,
    SIMILARITY_METRIC,
    TOP_N_SIMILAR_ITEMS,
    TOP_N_SIMILAR_USERS,
    DEFAULT_N_RECOMMENDATIONS
)


def main():
    """Run model training"""
    
    print("\n" + "="*70)
    print("ML RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("="*70)
    
    # Configuration
    config = {
        'split_method': TRAIN_TEST_SPLIT_METHOD,
        'test_ratio': 1 - TRAIN_TEST_SPLIT_RATIO,
        'test_days': TEST_DAYS,
        'min_customer_interactions': MIN_CUSTOMER_INTERACTIONS,
        'min_product_interactions': MIN_PRODUCT_INTERACTIONS,
        'similarity_metric': SIMILARITY_METRIC,
        'top_n_items': TOP_N_SIMILAR_ITEMS,
        'top_n_users': TOP_N_SIMILAR_USERS,
        'n_recommendations': DEFAULT_N_RECOMMENDATIONS
    }
    
    print("\nTraining Configuration:")
    print(f"  Split method: {config['split_method']}")
    if config['split_method'] == 'time_based':
        print(f"  Test period: Last {config['test_days']} days")
    else:
        print(f"  Test ratio: {config['test_ratio']*100:.0f}%")
    print(f"  Min customer interactions: {config['min_customer_interactions']}")
    print(f"  Min product interactions: {config['min_product_interactions']}")
    print(f"  Similarity metric: {config['similarity_metric']}")
    print(f"  Top-N similar items: {config['top_n_items']}")
    print(f"  Top-N similar users: {config['top_n_users']}")
    
    # Paths
    data_path = os.path.join(PROCESSED_DATA_DIR, 'processed_orders_latest.csv')
    output_dir = os.path.join(MODELS_DIR, 'production')
    
    print(f"\nData source: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Data file not found: {data_path}")
        print("Please run the data pipeline first: python run_pipeline.py")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train model
    try:
        metrics = trainer.train(data_path, output_dir)
        
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"✅ Training completed successfully!")
        print(f"\nValidation Metrics:")
        print(f"  Precision@10: {metrics['precision@10']:.4f} ({metrics['precision@10']*100:.2f}%)")
        print(f"  Recall@10: {metrics['recall@10']:.4f} ({metrics['recall@10']*100:.2f}%)")
        print(f"  Hit Rate: {metrics['hit_rate']:.4f} ({metrics['hit_rate']*100:.2f}%)")
        print(f"  Coverage: {metrics['coverage']:.4f} ({metrics['coverage']*100:.2f}%)")
        print(f"\nModels saved to: {output_dir}")
        print(f"\nNext step: Deploy models for serving recommendations")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
