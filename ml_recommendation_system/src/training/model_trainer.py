"""
Model Trainer Module
Main orchestrator for training collaborative filtering models
"""
import pandas as pd
import pickle
import json
import os
import logging
from datetime import datetime
from .data_splitter import DataSplitter
from .matrix_builder import MatrixBuilder
from .similarity_computer import SimilarityComputer
from .model_validator import ModelValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Main trainer for collaborative filtering models"""
    
    def __init__(self, config: dict):
        """
        Initialize ModelTrainer
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.splitter = DataSplitter(
            method=config.get('split_method', 'time_based'),
            test_ratio=config.get('test_ratio', 0.1),
            test_days=config.get('test_days', 60)
        )
        self.matrix_builder = MatrixBuilder(
            min_location_interactions=config.get('min_location_interactions', 1),
            min_product_interactions=config.get('min_product_interactions', 2)
        )
        self.similarity_computer = SimilarityComputer(
            similarity_metric=config.get('similarity_metric', 'cosine'),
            top_n_items=config.get('top_n_items', 50),
            top_n_users=config.get('top_n_users', 100)
        )
        self.validator = ModelValidator(
            n_recommendations=config.get('n_recommendations', 10)
        )
        
        self.train_data = None
        self.test_data = None
        self.interaction_matrix = None
        self.item_similarity = None
        self.user_similarity = None
        self.validation_metrics = None
        self.training_start_time = None
        self.training_end_time = None
    
    def train(self, data_path: str, output_dir: str = 'data/models/production'):
        """
        Train collaborative filtering model
        
        Args:
            data_path: Path to processed data CSV
            output_dir: Directory to save trained models
        """
        self.training_start_time = datetime.now()
        
        logger.info("\n" + "="*70)
        logger.info("ML RECOMMENDATION SYSTEM - MODEL TRAINING")
        logger.info("="*70)
        logger.info(f"Start time: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Data source: {data_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Step 1: Load data
        logger.info("\n[STEP 1] Loading processed data...")
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data):,} interactions")
        
        # Step 2: Train/test split
        logger.info("\n[STEP 2] Splitting data into train/test sets...")
        self.train_data, self.test_data = self.splitter.split(data)
        
        # Step 3: Build interaction matrix
        logger.info("\n[STEP 3] Building interaction matrix...")
        self.interaction_matrix = self.matrix_builder.build_matrix(self.train_data)
        
        # Step 4: Compute similarity matrices
        logger.info("\n[STEP 4] Computing similarity matrices...")
        self.item_similarity, self.user_similarity = self.similarity_computer.compute_all(
            self.interaction_matrix
        )
        
        # Step 5: Validate model
        logger.info("\n[STEP 5] Validating model on test data...")
        self.validation_metrics = self.validator.validate(
            self.test_data,
            self.interaction_matrix,
            self.item_similarity,
            self.matrix_builder
        )
        
        # Step 6: Save models
        logger.info("\n[STEP 6] Saving model artifacts...")
        self._save_models(output_dir)
        
        self.training_end_time = datetime.now()
        training_duration = self.training_end_time - self.training_start_time
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"End time: {self.training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {training_duration}")
        logger.info(f"Models saved to: {output_dir}")
        logger.info("="*70)
        
        return self.validation_metrics
    
    def _save_models(self, output_dir: str):
        """
        Save all model artifacts
        
        Args:
            output_dir: Directory to save models
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save interaction matrix
        logger.info("  Saving interaction matrix...")
        with open(os.path.join(output_dir, 'interaction_matrix.pkl'), 'wb') as f:
            pickle.dump(self.interaction_matrix, f)
        
        # Save item similarity
        logger.info("  Saving item similarity matrix...")
        with open(os.path.join(output_dir, 'item_similarity.pkl'), 'wb') as f:
            pickle.dump(self.item_similarity, f)
        
        # Save user similarity
        logger.info("  Saving user similarity matrix...")
        with open(os.path.join(output_dir, 'user_similarity.pkl'), 'wb') as f:
            pickle.dump(self.user_similarity, f)
        
        # Save encoders
        logger.info("  Saving location encoder...")
        with open(os.path.join(output_dir, 'location_encoder.pkl'), 'wb') as f:
            pickle.dump(self.matrix_builder.location_encoder, f)
        
        logger.info("  Saving product encoder...")
        with open(os.path.join(output_dir, 'product_encoder.pkl'), 'wb') as f:
            pickle.dump(self.matrix_builder.product_encoder, f)
        
        # Save metadata
        logger.info("  Saving metadata...")
        training_duration = (datetime.now() - self.training_start_time).total_seconds()
        metadata = {
            'training_date': self.training_start_time.isoformat(),
            'training_duration_seconds': training_duration,
            'config': self.config,
            'data_stats': {
                'total_interactions': len(self.train_data) + len(self.test_data),
                'train_interactions': len(self.train_data),
                'test_interactions': len(self.test_data),
                'n_locations': self.matrix_builder.n_locations,
                'n_products': self.matrix_builder.n_products,
                'matrix_sparsity': 100 * (1 - self.interaction_matrix.nnz / (self.matrix_builder.n_locations * self.matrix_builder.n_products))
            },
            'model_stats': {
                'item_similarity_nnz': self.item_similarity.nnz,
                'user_similarity_nnz': self.user_similarity.nnz,
                'item_similarity_size_mb': self.item_similarity.data.nbytes / 1024 / 1024,
                'user_similarity_size_mb': self.user_similarity.data.nbytes / 1024 / 1024
            }
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save validation metrics
        logger.info("  Saving validation report...")
        with open(os.path.join(output_dir, 'validation_report.json'), 'w') as f:
            json.dump(self.validation_metrics, f, indent=2)
        
        # Calculate total size
        total_size = 0
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
        
        logger.info(f"\n  Total model size: {total_size / 1024 / 1024:.1f} MB")
        logger.info(f"  Files saved:")
        for filename in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, filename)
            if os.path.isfile(filepath):
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                logger.info(f"    - {filename} ({size_mb:.1f} MB)")
