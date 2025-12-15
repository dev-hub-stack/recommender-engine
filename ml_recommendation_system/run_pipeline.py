"""
Run the data pipeline
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import DataPipeline
from config.config import (
    POS_ORDERS_CSV, 
    OE_ORDERS_CSV, 
    TIME_WINDOW_DAYS,
    SOURCE_TYPE,
    PROCESSED_DATA_DIR
)


def main():
    """Run the data pipeline"""
    
    print("\n" + "="*70)
    print("ML RECOMMENDATION SYSTEM - DATA PIPELINE")
    print("="*70)
    
    # Initialize pipeline
    pipeline = DataPipeline(
        source_type=SOURCE_TYPE,
        time_window_days=TIME_WINDOW_DAYS,
        output_dir=PROCESSED_DATA_DIR
    )
    
    # Run pipeline based on source type
    if SOURCE_TYPE == 'csv':
        print(f"\nLoading from CSV files:")
        print(f"  - POS Orders: {POS_ORDERS_CSV}")
        print(f"  - OE Orders: {OE_ORDERS_CSV}")
        
        processed_data = pipeline.run(
            pos_file=POS_ORDERS_CSV,
            oe_file=OE_ORDERS_CSV
        )
    elif SOURCE_TYPE == 'database':
        print(f"\nLoading from Database:")
        print(f"  - Tables: pos_orders, oe_orders")
        
        # Get database URL from environment
        import os
        from dotenv import load_dotenv
        load_dotenv()
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            print("\n❌ Error: DATABASE_URL not found in .env file")
            sys.exit(1)
        
        processed_data = pipeline.run(
            database_url=database_url
        )
    else:  # api
        print("\nLoading from API endpoints")
        from config.config import API_CONFIG
        processed_data = pipeline.run(api_config=API_CONFIG)
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}")
    print(f"Total records: {len(processed_data):,}")
    print(f"\nNext step: Run model training with this processed data")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
