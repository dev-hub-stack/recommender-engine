-- Drop the old function
DROP FUNCTION IF EXISTS rebuild_product_pairs();

-- Create optimized version
CREATE OR REPLACE FUNCTION rebuild_product_pairs() RETURNS INTEGER AS $$
DECLARE
    rows_inserted INTEGER := 0;
BEGIN
    -- Clear existing data
    TRUNCATE product_pairs;

    -- Create temporary table with product counts for efficiency
    CREATE TEMP TABLE temp_product_counts AS
    SELECT product_id, COUNT(*) as total_count
    FROM order_items
    GROUP BY product_id;

    -- Create index for faster lookups
    CREATE INDEX idx_temp_product_counts ON temp_product_counts(product_id);

    -- Rebuild product pairs from co-purchases with optimized query
    INSERT INTO product_pairs (
        product_1,
        product_2,
        co_purchase_count,
        confidence,
        last_updated
    )
    SELECT
        LEAST(oi1.product_id, oi2.product_id) as product_1,
        GREATEST(oi1.product_id, oi2.product_id) as product_2,
        COUNT(*) as co_purchase_count,
        CAST(COUNT(*) AS DECIMAL) / NULLIF(pc.total_count, 0) as confidence,
        CURRENT_TIMESTAMP as last_updated
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
    JOIN temp_product_counts pc ON pc.product_id = oi1.product_id
    GROUP BY oi1.product_id, oi2.product_id, pc.total_count
    HAVING COUNT(*) > 1  -- Only pairs bought together at least twice
    ON CONFLICT (product_1, product_2) DO UPDATE SET
        co_purchase_count = EXCLUDED.co_purchase_count,
        confidence = EXCLUDED.confidence,
        last_updated = EXCLUDED.last_updated;

    -- Clean up temp table
    DROP TABLE temp_product_counts;

    GET DIAGNOSTICS rows_inserted = ROW_COUNT;
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;
