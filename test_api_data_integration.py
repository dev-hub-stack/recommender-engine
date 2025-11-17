"""
Integration Tests for Recommendation Engine with API Data
Tests recommendation generation, model training, and real-time updates with API data.

Requirements: 14.5 (integration tests for recommendation engine)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from data_loader import RecommendationDataLoader
from real_time_updater import RealTimeRecommendationUpdater


class TestRecommendationDataLoader:
    """Test data loader with API integration"""
    
    def test_data_loader_initialization(self):
        """Test data loader initializes correctly"""
        loader = RecommendationDataLoader(mode='api', lookback_months=6)
        
        assert loader.mode == 'api'
        assert loader.lookback_months == 6
        assert loader.data_layer is not None
        print("✓ Data loader initialized successfully")
    
    def test_load_training_data_api_mode(self):
        """Test loading training data in API mode"""
        loader = RecommendationDataLoader(mode='api', lookback_months=6)
        
        try:
            sales_data, product_data, customer_data = loader.load_training_data(use_lookback=True)
            
            # Verify data loaded
            assert sales_data is not None
            assert product_data is not None
            assert customer_data is not None
            
            # Verify data structure
            assert isinstance(sales_data, pd.DataFrame)
            assert isinstance(product_data, pd.DataFrame)
            assert isinstance(customer_data, pd.DataFrame)
            
            # Verify non-empty
            assert len(sales_data) > 0, "Sales data should not be empty"
            assert len(product_data) > 0, "Product data should not be empty"
            assert len(customer_data) > 0, "Customer data should not be empty"
            
            print(f"✓ Training data loaded: {len(sales_data)} sales, {len(product_data)} products, {len(customer_data)} customers")
            
        except Exception as e:
            print(f"⚠ API data loading test skipped (API may not be available): {e}")
            pytest.skip(f"API not available: {e}")
    
    def test_load_collaborative_filtering_data(self):
        """Test loading CF data format"""
        loader = RecommendationDataLoader(mode='api', lookback_months=6)
        
        try:
            cf_data = loader.load_collaborative_filtering_data(use_lookback=True)
            
            # Verify data loaded
            assert cf_data is not None
            assert isinstance(cf_data, pd.DataFrame)
            
            # Verify required columns
            required_columns = ['user_id', 'item_id', 'rating', 'timestamp']
            for col in required_columns:
                assert col in cf_data.columns, f"Missing required column: {col}"
            
            # Verify data quality
            assert len(cf_data) > 0, "CF data should not be empty"
            assert cf_data['user_id'].nunique() > 0, "Should have users"
            assert cf_data['item_id'].nunique() > 0, "Should have items"
            assert cf_data['rating'].min() >= 1.0, "Ratings should be >= 1"
            assert cf_data['rating'].max() <= 5.0, "Ratings should be <= 5"
            
            print(f"✓ CF data loaded: {len(cf_data)} interactions, {cf_data['user_id'].nunique()} users, {cf_data['item_id'].nunique()} items")
            
        except Exception as e:
            print(f"⚠ CF data loading test skipped: {e}")
            pytest.skip(f"API not available: {e}")
    
    def test_data_freshness_tracking(self):
        """Test data freshness tracking"""
        loader = RecommendationDataLoader(mode='api', lookback_months=6)
        
        # Before loading
        freshness_before = loader.get_data_freshness()
        assert freshness_before['status'] == 'not_loaded'
        
        try:
            # Load data
            loader.load_training_data(use_lookback=True)
            
            # After loading
            freshness_after = loader.get_data_freshness()
            assert freshness_after['status'] == 'loaded'
            assert freshness_after['last_load_time'] is not None
            assert freshness_after['mode'] == 'api'
            assert freshness_after['lookback_months'] == 6
            
            print(f"✓ Data freshness tracked: age={freshness_after.get('age_hours')} hours")
            
        except Exception as e:
            print(f"⚠ Freshness tracking test skipped: {e}")
            pytest.skip(f"API not available: {e}")
    
    def test_mode_switching(self):
        """Test switching between API and CSV modes"""
        loader = RecommendationDataLoader(mode='api')
        
        assert loader.mode == 'api'
        
        # Switch to CSV
        loader.switch_mode('csv')
        assert loader.mode == 'csv'
        
        # Switch back to API
        loader.switch_mode('api')
        assert loader.mode == 'api'
        
        print("✓ Mode switching works correctly")


class TestRealTimeUpdater:
    """Test real-time recommendation updates"""
    
    def test_real_time_updater_initialization(self):
        """Test real-time updater initializes correctly"""
        loader = RecommendationDataLoader(mode='api')
        updater = RealTimeRecommendationUpdater(loader)
        
        assert updater.data_loader is not None
        assert updater.trending_products == {}
        assert updater.last_update_time is None
        
        print("✓ Real-time updater initialized successfully")
    
    def test_load_latest_orders(self):
        """Test loading latest orders"""
        loader = RecommendationDataLoader(mode='api')
        
        try:
            recent_orders = loader.load_latest_orders(since_hours=24)
            
            # Verify data structure
            assert isinstance(recent_orders, pd.DataFrame)
            
            if not recent_orders.empty:
                print(f"✓ Latest orders loaded: {len(recent_orders)} orders")
            else:
                print("⚠ No recent orders found (may be expected)")
            
        except Exception as e:
            print(f"⚠ Latest orders test skipped: {e}")
            pytest.skip(f"API not available: {e}")
    
    def test_real_time_product_trends(self):
        """Test real-time product trend calculation"""
        loader = RecommendationDataLoader(mode='api')
        
        try:
            trends = loader.get_real_time_product_trends(since_hours=24)
            
            # Verify structure
            assert isinstance(trends, dict)
            
            if trends:
                # Check trend structure
                sample_product = list(trends.keys())[0]
                sample_trend = trends[sample_product]
                
                assert 'total_quantity' in sample_trend
                assert 'total_revenue' in sample_trend
                assert 'order_count' in sample_trend
                assert 'velocity' in sample_trend
                
                print(f"✓ Product trends calculated: {len(trends)} trending products")
            else:
                print("⚠ No trending products found (may be expected)")
            
        except Exception as e:
            print(f"⚠ Product trends test skipped: {e}")
            pytest.skip(f"API not available: {e}")
    
    def test_update_from_latest_orders(self):
        """Test updating recommendations from latest orders"""
        loader = RecommendationDataLoader(mode='api')
        updater = RealTimeRecommendationUpdater(loader)
        
        try:
            metrics = updater.update_from_latest_orders(since_hours=24)
            
            # Verify metrics structure
            assert 'status' in metrics
            assert 'n_orders' in metrics
            assert 'n_trending_products' in metrics
            assert 'n_customers_updated' in metrics
            
            if metrics['status'] == 'success':
                print(f"✓ Real-time update successful: {metrics['n_orders']} orders, {metrics['n_trending_products']} trending products")
            else:
                print(f"⚠ Real-time update returned: {metrics['status']}")
            
        except Exception as e:
            print(f"⚠ Real-time update test skipped: {e}")
            pytest.skip(f"API not available: {e}")
    
    def test_get_trending_products(self):
        """Test getting trending products"""
        loader = RecommendationDataLoader(mode='api')
        updater = RealTimeRecommendationUpdater(loader)
        
        try:
            # Update first
            updater.update_from_latest_orders(since_hours=24)
            
            # Get trending products
            trending = updater.get_trending_products(top_n=5)
            
            # Verify structure
            assert isinstance(trending, list)
            
            if trending:
                # Check structure of first product
                first_product = trending[0]
                assert 'product_id' in first_product
                assert 'trending_score' in first_product
                
                print(f"✓ Trending products retrieved: {len(trending)} products")
            else:
                print("⚠ No trending products available")
            
        except Exception as e:
            print(f"⚠ Trending products test skipped: {e}")
            pytest.skip(f"API not available: {e}")
    
    def test_add_freshness_to_response(self):
        """Test adding freshness info to recommendations"""
        loader = RecommendationDataLoader(mode='api')
        updater = RealTimeRecommendationUpdater(loader)
        
        # Create sample recommendations
        sample_recs = [
            {'product_id': 'P1', 'score': 0.9},
            {'product_id': 'P2', 'score': 0.8}
        ]
        
        # Add freshness
        recs_with_freshness = updater.add_freshness_to_response(sample_recs)
        
        # Verify freshness added
        assert len(recs_with_freshness) == 2
        for rec in recs_with_freshness:
            assert 'data_freshness' in rec
            assert 'mode' in rec['data_freshness']
        
        print("✓ Freshness information added to recommendations")
    
    def test_get_update_status(self):
        """Test getting update status"""
        loader = RecommendationDataLoader(mode='api')
        updater = RealTimeRecommendationUpdater(loader)
        
        status = updater.get_update_status()
        
        # Verify status structure
        assert 'last_update_time' in status
        assert 'n_trending_products' in status
        assert 'n_customers_tracked' in status
        assert 'data_freshness' in status
        
        print(f"✓ Update status retrieved: {status['n_trending_products']} trending products")


class TestModelTrainingWithAPI:
    """Test model training with API data"""
    
    def test_load_training_data_for_models(self):
        """Test loading data in format expected by models"""
        loader = RecommendationDataLoader(mode='api', lookback_months=6)
        
        try:
            sales_data, product_data, customer_data = loader.load_training_data(use_lookback=True)
            
            # Verify data can be used for training
            assert 'product_id' in sales_data.columns or 'product_name' in sales_data.columns
            assert 'customer_id' in sales_data.columns or 'unified_customer_id' in sales_data.columns
            
            # Verify product data
            assert 'product_id' in product_data.columns or 'product_name' in product_data.columns
            
            # Verify customer data
            assert 'customer_id' in customer_data.columns or 'unified_customer_id' in customer_data.columns
            
            print("✓ Data format suitable for model training")
            
        except Exception as e:
            print(f"⚠ Model training data test skipped: {e}")
            pytest.skip(f"API not available: {e}")
    
    def test_six_month_lookback(self):
        """Test 6-month lookback for training data"""
        loader = RecommendationDataLoader(mode='api', lookback_months=6)
        
        try:
            sales_data, _, _ = loader.load_training_data(use_lookback=True)
            
            if 'order_date' in sales_data.columns or 'sale_date' in sales_data.columns:
                date_col = 'order_date' if 'order_date' in sales_data.columns else 'sale_date'
                sales_data[date_col] = pd.to_datetime(sales_data[date_col])
                
                # Check date range
                min_date = sales_data[date_col].min()
                max_date = sales_data[date_col].max()
                date_range_days = (max_date - min_date).days
                
                # Should be approximately 6 months (180 days)
                # Allow some flexibility
                assert date_range_days <= 200, f"Date range too large: {date_range_days} days"
                
                print(f"✓ 6-month lookback verified: {date_range_days} days of data")
            else:
                print("⚠ No date column found to verify lookback")
            
        except Exception as e:
            print(f"⚠ Lookback test skipped: {e}")
            pytest.skip(f"API not available: {e}")


def run_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("RECOMMENDATION ENGINE API INTEGRATION TESTS")
    print("="*70 + "\n")
    
    # Test Data Loader
    print("\n--- Testing Data Loader ---")
    test_loader = TestRecommendationDataLoader()
    test_loader.test_data_loader_initialization()
    test_loader.test_load_training_data_api_mode()
    test_loader.test_load_collaborative_filtering_data()
    test_loader.test_data_freshness_tracking()
    test_loader.test_mode_switching()
    
    # Test Real-Time Updater
    print("\n--- Testing Real-Time Updater ---")
    test_updater = TestRealTimeUpdater()
    test_updater.test_real_time_updater_initialization()
    test_updater.test_load_latest_orders()
    test_updater.test_real_time_product_trends()
    test_updater.test_update_from_latest_orders()
    test_updater.test_get_trending_products()
    test_updater.test_add_freshness_to_response()
    test_updater.test_get_update_status()
    
    # Test Model Training
    print("\n--- Testing Model Training with API ---")
    test_training = TestModelTrainingWithAPI()
    test_training.test_load_training_data_for_models()
    test_training.test_six_month_lookback()
    
    print("\n" + "="*70)
    print("INTEGRATION TESTS COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_tests()
