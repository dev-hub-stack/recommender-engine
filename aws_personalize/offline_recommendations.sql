-- Table for Offline User Recommendations (User Personalization)
CREATE TABLE IF NOT EXISTS offline_user_recommendations (
    user_id VARCHAR(255) PRIMARY KEY,
    recommendations JSONB, -- List of {product_id, score}
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for Similar Items (Item-to-Item)
CREATE TABLE IF NOT EXISTS offline_similar_items (
    product_id VARCHAR(255) PRIMARY KEY,
    similar_products JSONB, -- List of {product_id, score}
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for Personalized Ranking (Optional - usually computed on demand, but can be pre-computed for popular lists)
CREATE TABLE IF NOT EXISTS offline_personalized_ranking (
    user_id VARCHAR(255),
    category VARCHAR(255),
    ranked_products JSONB,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, category)
);

-- Index for fast retrieval
CREATE INDEX IF NOT EXISTS idx_offline_user_recs_updated ON offline_user_recommendations(updated_at);
CREATE INDEX IF NOT EXISTS idx_offline_similar_items_updated ON offline_similar_items(updated_at);
