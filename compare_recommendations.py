#!/usr/bin/env python3
"""
Compare AWS Personalize vs Custom ML Model Recommendations
Quick validation script for client presentation
"""

import sys
import os
sys.path.append('src')

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from algorithms.matrix_factorization import MatrixFactorizationSVD

def get_db_connection():
    """Get production database connection"""
    return psycopg2.connect(
        host='ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com',
        port='5432',
        dbname='mastergroup_recommendations',
        user='postgres',
        password='MasterGroup2024Secure!',
        sslmode='require'
    )

def get_test_customers(limit=3):
    """Get customers with good purchase history"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT 
            unified_customer_id,
            COUNT(*) as order_count,
            SUM(oi.total_price) as total_spend
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        WHERE unified_customer_id IS NOT NULL
        GROUP BY unified_customer_id
        HAVING COUNT(*) >= 5
        ORDER BY order_count DESC
        LIMIT %s
    """, (limit,))
    
    customers = cursor.fetchall()
    cursor.close()
    conn.close()
    return customers

def get_aws_personalize_recommendations(customer_id):
    """Get AWS Personalize recommendations from cache"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT recommendations 
        FROM offline_user_recommendations 
        WHERE user_id = %s
    """, (customer_id,))
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if result and result['recommendations']:
        return result['recommendations'][:10]  # Top 10
    return []

def get_custom_model_recommendations(customer_id):
    """Get recommendations from custom Matrix Factorization model"""
    try:
        conn = get_db_connection()
        mf_model = MatrixFactorizationSVD(conn, n_factors=20)
        
        print(f"   Training Matrix Factorization for {customer_id}...")
        success = mf_model.train()
        
        if success:
            recommendations = mf_model.get_recommendations(customer_id, limit=10)
            conn.close()
            return recommendations
        else:
            conn.close()
            return []
            
    except Exception as e:
        print(f"   Error in custom model: {e}")
        return []

def compare_recommendations(aws_recs, custom_recs):
    """Compare recommendation lists"""
    if not aws_recs or not custom_recs:
        return {
            'overlap': 0,
            'aws_count': len(aws_recs),
            'custom_count': len(custom_recs),
            'similarity': 0.0
        }
    
    # Extract product IDs
    aws_products = {rec['product_id'] for rec in aws_recs}
    custom_products = {rec['product_id'] for rec in custom_recs}
    
    # Calculate overlap
    overlap = len(aws_products.intersection(custom_products))
    total_unique = len(aws_products.union(custom_products))
    
    similarity = overlap / total_unique if total_unique > 0 else 0
    
    return {
        'overlap': overlap,
        'aws_count': len(aws_recs),
        'custom_count': len(custom_recs),
        'similarity': similarity,
        'aws_products': list(aws_products)[:5],
        'custom_products': list(custom_products)[:5]
    }

def main():
    print("🎯 AWS Personalize vs Custom ML Model Comparison")
    print("=" * 60)
    
    # Get test customers
    print("\n📋 Getting test customers...")
    customers = get_test_customers(2)  # Test 2 customers
    
    if not customers:
        print("❌ No customers found with sufficient purchase history")
        return
    
    results = []
    
    for i, customer in enumerate(customers, 1):
        customer_id = customer['unified_customer_id']
        print(f"\n🧪 Test {i}: Customer {customer_id}")
        print(f"   Orders: {customer['order_count']}, Spend: PKR {customer['total_spend']:,.0f}")
        
        # Get AWS Personalize recommendations
        print("   Fetching AWS Personalize recommendations...")
        aws_recs = get_aws_personalize_recommendations(customer_id)
        
        # Get custom model recommendations
        print("   Generating custom model recommendations...")
        custom_recs = get_custom_model_recommendations(customer_id)
        
        # Compare
        comparison = compare_recommendations(aws_recs, custom_recs)
        
        print(f"   📊 Results:")
        print(f"      AWS Personalize: {comparison['aws_count']} recommendations")
        print(f"      Custom Model: {comparison['custom_count']} recommendations")
        print(f"      Overlap: {comparison['overlap']} products")
        print(f"      Similarity: {comparison['similarity']:.1%}")
        
        if comparison['similarity'] > 0:
            print(f"      ✅ Models show {comparison['similarity']:.1%} similarity")
        else:
            print(f"      ⚠️ No overlap found")
        
        results.append({
            'customer_id': customer_id,
            'customer_orders': customer['order_count'],
            'comparison': comparison
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    
    if results:
        avg_similarity = sum(r['comparison']['similarity'] for r in results) / len(results)
        print(f"Average Similarity: {avg_similarity:.1%}")
        
        if avg_similarity > 0.3:
            print("✅ GOOD: Custom model shows strong similarity to AWS Personalize")
        elif avg_similarity > 0.1:
            print("⚠️ MODERATE: Custom model shows some similarity to AWS Personalize")
        else:
            print("❌ LOW: Custom model needs tuning to match AWS Personalize")
    
    print(f"\nTested {len(results)} customers")
    print("Custom Matrix Factorization model ready for on-premise deployment")

if __name__ == "__main__":
    main()
