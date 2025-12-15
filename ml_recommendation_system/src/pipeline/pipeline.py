"""
Main Data Pipeline
Orchestrates data loading, processing, and saving
"""
import pandas as pd
import os
import logging
from datetime import datetime
from .data_loader import DataLoader
from .data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Main pipeline for data preparation"""
    
    def __init__(self, 
                 source_type: str = 'csv',
                 time_window_days: int = 730,
                 output_dir: str = 'data/processed'):
        """
        Initialize DataPipeline
        
        Args:
            source_type: 'csv' or 'api'
            time_window_days: Number of days of historical data (default: 2 years)
            output_dir: Directory to save processed data
        """
        self.loader = DataLoader(source_type=source_type)
        self.processor = DataProcessor(time_window_days=time_window_days)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def run(self, **kwargs) -> pd.DataFrame:
        """
        Run the complete data pipeline
        
        Args:
            **kwargs: Arguments for data loader
                For CSV: pos_file, oe_file
                For API: api_config
                
        Returns:
            Processed DataFrame
        """
        logger.info("="*60)
        logger.info("STARTING DATA PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load data
        logger.info("\n[STEP 1] Loading data...")
        raw_orders = self.loader.load(**kwargs)
        
        # Step 2: Process data (includes customer ID cleaning)
        logger.info("\n[STEP 2] Processing data...")
        processed_orders = self.processor.process(raw_orders)
        
        # Step 3: Aggregate interactions
        logger.info("\n[STEP 3] Aggregating interactions...")
        aggregated_orders = self.processor.aggregate_interactions(processed_orders)
        
        # Step 4: Save processed data
        logger.info("\n[STEP 4] Saving processed data...")
        self._save_processed_data(aggregated_orders)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        
        return aggregated_orders
    
    def _save_processed_data(self, data: pd.DataFrame):
        """
        Save processed data to files
        
        Args:
            data: Processed DataFrame
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f'processed_orders_{timestamp}.csv')
        data.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # Save as Parquet (more efficient for large datasets)
        parquet_path = os.path.join(self.output_dir, f'processed_orders_{timestamp}.parquet')
        data.to_parquet(parquet_path, index=False)
        logger.info(f"Saved Parquet: {parquet_path}")
        
        # Save latest version (for easy access)
        latest_csv = os.path.join(self.output_dir, 'processed_orders_latest.csv')
        data.to_csv(latest_csv, index=False)
        logger.info(f"Saved latest CSV: {latest_csv}")
        
        latest_parquet = os.path.join(self.output_dir, 'processed_orders_latest.parquet')
        data.to_parquet(latest_parquet, index=False)
        logger.info(f"Saved latest Parquet: {latest_parquet}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'total_interactions': len(data),
            'unique_locations': data['location_id'].nunique(),
            'unique_products': data['product_id'].nunique(),
            'date_range_start': str(data['order_date'].min()),
            'date_range_end': str(data['order_date'].max()),
            'sources': data['source'].value_counts().to_dict()
        }
        
        import json
        metadata_path = os.path.join(self.output_dir, f'metadata_{timestamp}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
        
        # Print summary
        logger.info("\n" + "-"*60)
        logger.info("DATA SUMMARY")
        logger.info("-"*60)
        logger.info(f"Total interactions: {metadata['total_interactions']:,}")
        logger.info(f"Unique locations: {metadata['unique_locations']:,}")
        logger.info(f"Unique products: {metadata['unique_products']:,}")
        logger.info(f"Date range: {metadata['date_range_start']} to {metadata['date_range_end']}")
        logger.info(f"Sources: {metadata['sources']}")
        logger.info("-"*60)
