-- Check cities for NULL provinces
SELECT 
    COALESCE(city, 'No City') as city,
    COUNT(*) as count
FROM orders
WHERE province IS NULL
GROUP BY city
ORDER BY count DESC
LIMIT 20;
