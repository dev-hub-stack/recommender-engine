"""
Unit tests for automated model training and monitoring pipeline
Tests MLflow integration, automated training, and performance monitoring
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from training.mlflow_integration import MLflowManager
from training.model_trainer import ModelTrainer, TrainingConfig, DataValidator
from training.monitoring_pipeline import (
    MonitoringPipeline, AlertConfig, PerformanceMetrics, 
    MetricsCollector, AlertManager
)
from models.recommendation import RecommendationAlgorithm


class TestMLflowManager:
    """Test cases for MLflow integration"""
    
    def setup_method(self):
        """Setup test MLflow manager"""
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        tracking_uri = f"file://{self.temp_dir}/mlflow"
        
        self.mlflow_manager = MLflowManager(
            tracking_uri=tracking_uri,
            experiment_name="test_experiment"
        )
    
    def test_setup_mlflow(self):
        """Test MLflow setup"""
        assert self.mlflow_manager.client is not None
        assert self.mlflow_manager.experiment_id is not None
        assert self.mlflow_manager.experiment_name == "test_experiment"
    
    @patch('mlflow.start_run')
    def test_start_run(self, mock_start_run):
        """Test starting MLflow run"""
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_start_run.return_value = mock_run
        
        run_id = self.mlflow_manager.start_run(
            "test_run",
            RecommendationAlgorithm.ENSEMBLE,
            tags={"test": "true"}
        )
        
        assert run_id == "test_run_123"
        mock_start_run.assert_called_once()
    
    @patch('mlflow.log_param')
    def test_log_parameters(self, mock_log_param):
        """Test logging parameters"""
        params = {
            "algorithm": "collaborative_filtering",
            "accuracy_threshold": 0.85,
            "data_size": 1000
        }
        
        self.mlflow_manager.log_parameters(params)
        
        assert mock_log_param.call_count == len(params)
    
    @patch('mlflow.log_metric')
    def test_log_metrics(self, mock_log_metric):
        """Test logging metrics"""
        metrics = {
            "accuracy": 0.87,
            "rmse": 0.45,
            "training_time": 120.5
        }
        
        self.mlflow_manager.log_metrics(metrics)
        
        assert mock_log_metric.call_count == len(metrics)
    
    @patch('mlflow.log_artifact')
    @patch('joblib.dump')
    def test_log_model(self, mock_joblib_dump, mock_log_artifact):
        """Test logging model"""
        mock_model = Mock()
        
        with patch('mlflow.active_run') as mock_active_run:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_123"
            mock_active_run.return_value = mock_run
            
            model_uri = self.mlflow_manager.log_model(
                mock_model, "test_model", RecommendationAlgorithm.ENSEMBLE
            )
            
            assert "runs:/test_run_123/model" in model_uri
            mock_log_artifact.assert_called()


class TestDataValidator:
    """Test cases for data validation"""
    
    def setup_method(self):
        """Setup test data validator"""
        self.validator = DataValidator(min_users=5, min_items=5, min_interactions=50)
    
    def test_validate_cf_data_valid(self):
        """Test validation of valid CF data"""
        cf_data = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(10)] * 10,
            'item_id': [f'item_{i%8}' for i in range(100)],
            'rating': np.random.uniform(1, 5, 100),
            'timestamp': [datetime.now()] * 100
        })
        
        is_valid, issues = self.validator.validate_cf_data(cf_data)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_cf_data_insufficient_interactions(self):
        """Test validation with insufficient interactions"""
        cf_data = pd.DataFrame({
            'user_id': ['user_1', 'user_2'],
            'item_id': ['item_1', 'item_2'],
            'rating': [4.0, 3.5],
            'timestamp': [datetime.now()] * 2
        })
        
        is_valid, issues = self.validator.validate_cf_data(cf_data)
        
        assert not is_valid
        assert any("Insufficient interactions" in issue for issue in issues)
    
    def test_validate_cf_data_missing_columns(self):
        """Test validation with missing columns"""
        cf_data = pd.DataFrame({
            'user_id': ['user_1', 'user_2'],
            'item_id': ['item_1', 'item_2']
            # Missing 'rating' and 'timestamp'
        })
        
        is_valid, issues = self.validator.validate_cf_data(cf_data)
        
        assert not is_valid
        assert any("Missing columns" in issue for issue in issues)
    
    def test_validate_cf_data_invalid_ratings(self):
        """Test validation with invalid rating range"""
        cf_data = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(10)] * 10,
            'item_id': [f'item_{i%8}' for i in range(100)],
            'rating': np.random.uniform(0, 6, 100),  # Invalid range
            'timestamp': [datetime.now()] * 100
        })
        
        is_valid, issues = self.validator.validate_cf_data(cf_data)
        
        assert not is_valid
        assert any("Rating out of range" in issue for issue in issues)
    
    def test_validate_sales_data_valid(self):
        """Test validation of valid sales data"""
        sales_data = pd.DataFrame({
            'product_id': [f'product_{i%10}' for i in range(100)],
            'customer_id': [f'customer_{i%20}' for i in range(100)],
            'sale_date': [datetime.now() - timedelta(days=i%30) for i in range(100)],
            'quantity': np.random.randint(1, 5, 100),
            'amount': np.random.uniform(100, 1000, 100)
        })
        
        is_valid, issues = self.validator.validate_sales_data(sales_data)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_sales_data_negative_values(self):
        """Test validation with negative values"""
        sales_data = pd.DataFrame({
            'product_id': ['product_1', 'product_2'],
            'customer_id': ['customer_1', 'customer_2'],
            'sale_date': [datetime.now()] * 2,
            'quantity': [-1, 2],  # Negative quantity
            'amount': [100, -50]  # Negative amount
        })
        
        is_valid, issues = self.validator.validate_sales_data(sales_data)
        
        assert not is_valid
        assert any("Negative quantities" in issue for issue in issues)
        assert any("Negative amounts" in issue for issue in issues)


class TestModelTrainer:
    """Test cases for model trainer"""
    
    def setup_method(self):
        """Setup test model trainer"""
        config = TrainingConfig(
            cf_min_interactions=3,
            cf_accuracy_threshold=0.80,
            popularity_min_sales=2,
            max_training_time_minutes=5
        )
        
        # Mock MLflow manager
        self.mock_mlflow = Mock(spec=MLflowManager)
        self.mock_mlflow.start_run.return_value = "test_run_123"
        
        self.trainer = ModelTrainer(config, self.mock_mlflow)
    
    def test_training_config_defaults(self):
        """Test training configuration defaults"""
        config = TrainingConfig()
        
        assert config.cf_min_interactions == 5
        assert config.cf_accuracy_threshold == 0.85
        assert config.popularity_min_sales == 3
        assert config.retrain_schedule == "daily"
        assert config.performance_threshold == 0.80
        assert config.ensemble_weights is not None
    
    def test_train_collaborative_filtering_success(self):
        """Test successful CF training"""
        # Create test data
        cf_data = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(10)] * 5,
            'item_id': [f'item_{i%8}' for i in range(50)],
            'rating': np.random.uniform(1, 5, 50),
            'timestamp': [datetime.now()] * 50
        })
        
        # Mock successful training
        self.trainer.cf_engine.train = Mock(return_value={'rmse_accuracy': 0.85})
        
        result = self.trainer.train_collaborative_filtering(cf_data)
        
        assert result.success
        assert result.algorithm == RecommendationAlgorithm.ENSEMBLE
        assert 'rmse_accuracy' in result.metrics
        assert result.training_time_seconds > 0
    
    def test_train_collaborative_filtering_failure(self):
        """Test CF training failure"""
        cf_data = pd.DataFrame({
            'user_id': ['user_1'],
            'item_id': ['item_1'],
            'rating': [4.0],
            'timestamp': [datetime.now()]
        })
        
        # Mock training failure
        self.trainer.cf_engine.train = Mock(side_effect=Exception("Training failed"))
        
        result = self.trainer.train_collaborative_filtering(cf_data)
        
        assert not result.success
        assert result.error_message == "Training failed"
    
    def test_train_popularity_based_success(self):
        """Test successful popularity-based training"""
        # Create test data
        sales_data = pd.DataFrame({
            'product_id': [f'product_{i%5}' for i in range(30)],
            'customer_id': [f'customer_{i%10}' for i in range(30)],
            'sale_date': [datetime.now()] * 30,
            'quantity': np.random.randint(1, 4, 30),
            'amount': np.random.uniform(100, 500, 30)
        })
        
        product_data = pd.DataFrame({
            'product_id': [f'product_{i}' for i in range(5)],
            'category_id': [f'category_{i%2}' for i in range(5)]
        })
        
        customer_data = pd.DataFrame({
            'customer_id': [f'customer_{i}' for i in range(10)],
            'city': ['Karachi'] * 10,
            'income_bracket': ['300k-500k PKR'] * 10
        })
        
        # Mock successful training
        self.trainer.popularity_engine.train = Mock(return_value={'n_products_analyzed': 5})
        
        result = self.trainer.train_popularity_based(sales_data, product_data, customer_data)
        
        assert result.success
        assert result.algorithm == RecommendationAlgorithm.POPULARITY_BASED
        assert 'n_products_analyzed' in result.metrics
    
    def test_get_training_status(self):
        """Test getting training status"""
        status = self.trainer.get_training_status()
        
        assert 'is_training' in status
        assert 'last_training_time' in status
        assert 'training_history_count' in status
        assert 'config' in status
        assert 'recent_results' in status
        
        assert not status['is_training']  # Should not be training initially
        assert status['training_history_count'] == 0


class TestMetricsCollector:
    """Test cases for metrics collector"""
    
    def setup_method(self):
        """Setup test metrics collector"""
        self.collector = MetricsCollector(window_hours=24)
    
    def test_record_request(self):
        """Test recording request metrics"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        self.collector.record_request(algorithm, 150.0, True)
        self.collector.record_request(algorithm, 200.0, False)
        
        assert algorithm in self.collector.request_logs
        assert len(self.collector.request_logs[algorithm]) == 2
        
        # Check log structure
        log_entry = self.collector.request_logs[algorithm][0]
        assert 'timestamp' in log_entry
        assert 'response_time_ms' in log_entry
        assert 'success' in log_entry
        assert 'user_feedback' in log_entry
    
    def test_record_user_feedback(self):
        """Test recording user feedback"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        # First record a request
        self.collector.record_request(algorithm, 150.0, True)
        
        # Then record feedback
        self.collector.record_user_feedback(
            algorithm, 'user_1', 'product_1', True, False
        )
        
        # Check that feedback was added to recent request
        log_entry = self.collector.request_logs[algorithm][0]
        feedback = log_entry['user_feedback']
        
        assert feedback['user_id'] == 'user_1'
        assert feedback['product_id'] == 'product_1'
        assert feedback['clicked'] is True
        assert feedback['purchased'] is False
    
    def test_calculate_current_metrics(self):
        """Test calculating current performance metrics"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        # Record multiple requests with feedback
        for i in range(20):
            success = i % 4 != 0  # 75% success rate
            response_time = 100 + (i * 10)  # Varying response times
            
            self.collector.record_request(algorithm, response_time, success)
            
            # Add feedback for some requests
            if i % 2 == 0:
                clicked = i % 3 == 0
                purchased = i % 6 == 0
                self.collector.record_user_feedback(
                    algorithm, f'user_{i}', f'product_{i}', clicked, purchased
                )
        
        metrics = self.collector.calculate_current_metrics(algorithm)
        
        assert metrics is not None
        assert metrics.algorithm == algorithm
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert metrics.response_time_ms > 0
        assert 0 <= metrics.error_rate <= 1
        assert metrics.sample_size == 20
    
    def test_calculate_current_metrics_no_data(self):
        """Test calculating metrics with no data"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        metrics = self.collector.calculate_current_metrics(algorithm)
        
        assert metrics is None


class TestAlertManager:
    """Test cases for alert manager"""
    
    def setup_method(self):
        """Setup test alert manager"""
        config = AlertConfig(
            accuracy_threshold=0.85,
            response_time_threshold_ms=200.0,
            error_rate_threshold=0.05,
            max_alerts_per_hour=3
        )
        
        self.alert_manager = AlertManager(config)
    
    def test_check_and_create_alerts_good_performance(self):
        """Test alert creation with good performance"""
        metrics = PerformanceMetrics(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            timestamp=datetime.utcnow(),
            accuracy=0.90,  # Above threshold
            precision=0.88,
            recall=0.85,
            f1_score=0.86,
            response_time_ms=150.0,  # Below threshold
            error_rate=0.02,  # Below threshold
            throughput_rps=10.0,
            sample_size=200  # Above minimum
        )
        
        alerts = self.alert_manager.check_and_create_alerts(metrics)
        
        assert len(alerts) == 0  # No alerts should be created
    
    def test_check_and_create_alerts_poor_performance(self):
        """Test alert creation with poor performance"""
        metrics = PerformanceMetrics(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            timestamp=datetime.utcnow(),
            accuracy=0.70,  # Below threshold
            precision=0.65,
            recall=0.60,
            f1_score=0.62,
            response_time_ms=300.0,  # Above threshold
            error_rate=0.10,  # Above threshold
            throughput_rps=5.0,
            sample_size=50  # Below minimum
        )
        
        alerts = self.alert_manager.check_and_create_alerts(metrics)
        
        assert len(alerts) > 0  # Alerts should be created
        
        # Check alert properties
        for alert in alerts:
            assert alert.algorithm == RecommendationAlgorithm.ENSEMBLE
            assert alert.severity in ['warning', 'critical']
            assert not alert.resolved
            assert alert.alert_id in self.alert_manager.active_alerts
    
    def test_alert_rate_limiting(self):
        """Test alert rate limiting"""
        metrics = PerformanceMetrics(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            timestamp=datetime.utcnow(),
            accuracy=0.70,  # Below threshold
            precision=0.65,
            recall=0.60,
            f1_score=0.62,
            response_time_ms=150.0,
            error_rate=0.02,
            throughput_rps=10.0,
            sample_size=200
        )
        
        # Create alerts up to the limit
        total_alerts = 0
        for i in range(5):  # Try to create more than the limit (3)
            alerts = self.alert_manager.check_and_create_alerts(metrics)
            total_alerts += len(alerts)
        
        # Should be limited to max_alerts_per_hour
        assert total_alerts <= self.alert_manager.config.max_alerts_per_hour
    
    def test_resolve_alert(self):
        """Test resolving alerts"""
        # Create an alert first
        metrics = PerformanceMetrics(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            timestamp=datetime.utcnow(),
            accuracy=0.70,
            precision=0.65,
            recall=0.60,
            f1_score=0.62,
            response_time_ms=150.0,
            error_rate=0.02,
            throughput_rps=10.0,
            sample_size=200
        )
        
        alerts = self.alert_manager.check_and_create_alerts(metrics)
        assert len(alerts) > 0
        
        alert_id = alerts[0].alert_id
        
        # Resolve the alert
        success = self.alert_manager.resolve_alert(alert_id)
        
        assert success
        assert alert_id not in self.alert_manager.active_alerts
        assert alerts[0].resolved
        assert alerts[0].resolved_at is not None
    
    def test_get_alert_summary(self):
        """Test getting alert summary"""
        summary = self.alert_manager.get_alert_summary()
        
        assert 'active_alerts' in summary
        assert 'alerts_last_24h' in summary
        assert 'critical_alerts_last_24h' in summary
        assert 'most_recent_alert' in summary
        
        assert summary['active_alerts'] == 0  # Initially no alerts


class TestMonitoringPipeline:
    """Test cases for monitoring pipeline"""
    
    def setup_method(self):
        """Setup test monitoring pipeline"""
        config = AlertConfig(
            accuracy_threshold=0.85,
            response_time_threshold_ms=200.0,
            monitoring_window_hours=24
        )
        
        # Mock MLflow manager
        self.mock_mlflow = Mock(spec=MLflowManager)
        
        self.pipeline = MonitoringPipeline(config, self.mock_mlflow)
    
    def test_record_request(self):
        """Test recording requests through pipeline"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        self.pipeline.record_request(algorithm, 150.0, True)
        
        # Check that metrics collector received the request
        assert algorithm in self.pipeline.metrics_collector.request_logs
        assert len(self.pipeline.metrics_collector.request_logs[algorithm]) == 1
    
    def test_record_user_feedback(self):
        """Test recording user feedback through pipeline"""
        algorithm = RecommendationAlgorithm.ENSEMBLE
        
        # Record request first
        self.pipeline.record_request(algorithm, 150.0, True)
        
        # Record feedback
        self.pipeline.record_user_feedback(
            algorithm, 'user_1', 'product_1', True, False, 4.0
        )
        
        # Check that feedback was recorded
        log_entry = self.pipeline.metrics_collector.request_logs[algorithm][0]
        assert 'user_id' in log_entry['user_feedback']
    
    def test_register_performance_callback(self):
        """Test registering performance callbacks"""
        callback_called = False
        
        def test_callback(metrics, alerts):
            nonlocal callback_called
            callback_called = True
        
        self.pipeline.register_performance_callback(test_callback)
        
        assert len(self.pipeline.performance_callbacks) == 1
        
        # Trigger callback
        metrics = Mock()
        alerts = []
        self.pipeline._trigger_performance_callbacks(metrics, alerts)
        
        assert callback_called
    
    def test_get_monitoring_dashboard_data(self):
        """Test getting dashboard data"""
        dashboard_data = self.pipeline.get_monitoring_dashboard_data()
        
        assert 'timestamp' in dashboard_data
        assert 'algorithms' in dashboard_data
        assert 'alerts' in dashboard_data
        assert 'system_status' in dashboard_data
        
        assert dashboard_data['system_status'] in ['healthy', 'warning', 'critical']
    
    def test_export_metrics_report(self):
        """Test exporting metrics report"""
        report = self.pipeline.export_metrics_report(hours=24)
        
        assert 'report_generated' in report
        assert 'time_window_hours' in report
        assert 'algorithms' in report
        assert 'alerts' in report
        
        assert report['time_window_hours'] == 24


class TestPerformanceThresholds:
    """Test performance threshold validation"""
    
    def test_accuracy_threshold_85_percent(self):
        """Test 85% accuracy threshold requirement"""
        config = AlertConfig(accuracy_threshold=0.85)
        
        # Test metrics that meet threshold
        good_metrics = PerformanceMetrics(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            timestamp=datetime.utcnow(),
            accuracy=0.87,  # Above 85%
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            response_time_ms=150.0,
            error_rate=0.02,
            throughput_rps=10.0,
            sample_size=200
        )
        
        meets_threshold, issues = good_metrics.meets_thresholds(config)
        assert meets_threshold
        assert len(issues) == 0
        
        # Test metrics that don't meet threshold
        poor_metrics = PerformanceMetrics(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            timestamp=datetime.utcnow(),
            accuracy=0.82,  # Below 85%
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            response_time_ms=150.0,
            error_rate=0.02,
            throughput_rps=10.0,
            sample_size=200
        )
        
        meets_threshold, issues = poor_metrics.meets_thresholds(config)
        assert not meets_threshold
        assert any("Accuracy" in issue for issue in issues)
    
    def test_response_time_threshold_200ms(self):
        """Test 200ms response time threshold requirement"""
        config = AlertConfig(response_time_threshold_ms=200.0)
        
        # Test metrics that meet threshold
        fast_metrics = PerformanceMetrics(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            timestamp=datetime.utcnow(),
            accuracy=0.90,
            precision=0.88,
            recall=0.85,
            f1_score=0.86,
            response_time_ms=150.0,  # Below 200ms
            error_rate=0.02,
            throughput_rps=10.0,
            sample_size=200
        )
        
        meets_threshold, issues = fast_metrics.meets_thresholds(config)
        assert meets_threshold
        
        # Test metrics that don't meet threshold
        slow_metrics = PerformanceMetrics(
            algorithm=RecommendationAlgorithm.ENSEMBLE,
            timestamp=datetime.utcnow(),
            accuracy=0.90,
            precision=0.88,
            recall=0.85,
            f1_score=0.86,
            response_time_ms=250.0,  # Above 200ms
            error_rate=0.02,
            throughput_rps=10.0,
            sample_size=200
        )
        
        meets_threshold, issues = slow_metrics.meets_thresholds(config)
        assert not meets_threshold
        assert any("Response time" in issue for issue in issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])