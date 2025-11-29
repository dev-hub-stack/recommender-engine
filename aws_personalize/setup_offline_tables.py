"""
Setup Offline Recommendation Tables in PostgreSQL

This script creates the necessary tables for storing batch inference results
from AWS Personalize. This enables cost-saving by avoiding real-time campaigns.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection from environment
DB_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': int(os.getenv('PG_PORT', 5432)),
    'database': os.getenv('PG_DATABASE', 'mastergroup'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', '')
}

SQL_SCHEMA = """
-- Table for Offline User Recommendations (User Personalization Recipe)
CREATE TABLE IF NOT EXISTS offline_user_recommendations (
    user_id VARCHAR(255) PRIMARY KEY,
    recommendations JSONB NOT NULL, -- List of {product_id, score}
    recipe_name VARCHAR(100) DEFAULT 'aws-user-personalization',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for Similar Items (Similar Items Recipe)
CREATE TABLE IF NOT EXISTS offline_similar_items (
    product_id VARCHAR(255) PRIMARY KEY,
    similar_products JSONB NOT NULL, -- List of {product_id, score}
    recipe_name VARCHAR(100) DEFAULT 'aws-similar-items',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for Item Affinity (User → Categories/Items affinity)
CREATE TABLE IF NOT EXISTS offline_item_affinity (
    user_id VARCHAR(255) PRIMARY KEY,
    item_affinities JSONB NOT NULL, -- List of {product_id, score}
    recipe_name VARCHAR(100) DEFAULT 'aws-item-affinity',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for Personalized Ranking Cache
-- Stores pre-ranked item lists for users (optional, for frequently accessed lists)
CREATE TABLE IF NOT EXISTS offline_personalized_ranking (
    user_id VARCHAR(255),
    input_items TEXT, -- Hash or list of input item IDs
    ranked_products JSONB NOT NULL,
    recipe_name VARCHAR(100) DEFAULT 'aws-personalized-ranking',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, input_items)
);

-- Batch job metadata tracking
CREATE TABLE IF NOT EXISTS batch_job_metadata (
    job_id SERIAL PRIMARY KEY,
    job_name VARCHAR(255) UNIQUE NOT NULL,
    job_arn VARCHAR(512),
    recipe_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'PENDING',
    s3_input_path TEXT,
    s3_output_path TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Indexes for fast retrieval
CREATE INDEX IF NOT EXISTS idx_offline_user_recs_updated 
    ON offline_user_recommendations(updated_at DESC);
    
CREATE INDEX IF NOT EXISTS idx_offline_similar_items_updated 
    ON offline_similar_items(updated_at DESC);
    
CREATE INDEX IF NOT EXISTS idx_offline_item_affinity_updated 
    ON offline_item_affinity(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_batch_jobs_status 
    ON batch_job_metadata(status, started_at DESC);

-- Comments for documentation
COMMENT ON TABLE offline_user_recommendations IS 
    'Stores personalized product recommendations for each user from AWS Personalize batch inference';
    
COMMENT ON TABLE offline_similar_items IS 
    'Stores similar/related products for each product from AWS Personalize';
    
COMMENT ON TABLE offline_item_affinity IS 
    'Stores user affinity scores for products/categories';
    
COMMENT ON TABLE batch_job_metadata IS 
    'Tracks AWS Personalize batch inference jobs for auditing and monitoring';
"""

def setup_tables():
    """Create offline recommendation tables"""
    try:
        print("Connecting to PostgreSQL...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("Creating offline recommendation tables...")
        cursor.execute(SQL_SCHEMA)
        conn.commit()
        
        print("✅ Tables created successfully!")
        
        # Verify tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'offline_%' OR table_name = 'batch_job_metadata'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print("\nCreated tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("AWS PERSONALIZE OFFLINE TABLES SETUP")
    print("="*60)
    print(f"Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    print("="*60)
    
    success = setup_tables()
    
    if success:
        print("\n✅ Setup complete! Ready for batch inference workflow.")
    else:
        print("\n❌ Setup failed. Please check database connection and permissions.")
