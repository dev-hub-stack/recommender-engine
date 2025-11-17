#!/usr/bin/env python3
"""
Production Setup Verification Script
Comprehensive verification for production deployment of the recommendation system
"""

import os
import sys
import json
import psycopg2
import redis
import requests
from datetime import datetime, timedelta
import subprocess
from typing import Dict, List, Any
import time

class ProductionVerifier:
    def __init__(self):
        self.results = {
            'verification_time': datetime.now().isoformat(),
            'environment': 'production',
            'checks': [],
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
    def log_check(self, category: str, name: str, status: bool, details: str = "", error: str = ""):
        """Log a verification check"""
        check = {
            'category': category,
            'name': name,
            'status': 'PASS' if status else 'FAIL',
            'details': details,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.results['checks'].append(check)
        
        if not status:
            self.results['errors'].append(f"{category} - {name}: {error}")
        
        # Print real-time feedback
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {category} - {name}")
        if details:
            print(f"   {details}")
        if error:
            print(f"   Error: {error}")
    
    def log_warning(self, message: str):
        """Log a warning"""
        self.results['warnings'].append(message)
        print(f"⚠️  Warning: {message}")
    
    def check_environment_variables(self):
        """Verify all required environment variables"""
        print("\n🔧 Environment Variables")
        print("-" * 40)
        
        required_vars = [
            ('DATABASE_URL', 'Database connection string'),
            ('MASTER_GROUP_API_BASE', 'Master Group API base URL'),
            ('MASTER_GROUP_AUTH_TOKEN', 'API authentication token')
        ]
        
        optional_vars = [
            ('REDIS_URL', 'Redis cache connection'),
            ('POSTGRES_HOST', 'PostgreSQL host'),
            ('POSTGRES_USER', 'PostgreSQL username'),
            ('POSTGRES_PASSWORD', 'PostgreSQL password'),
            ('POSTGRES_DB', 'PostgreSQL database name'),
            ('SYNC_INTERVAL_MINUTES', 'Sync interval (default: 15)'),
            ('ENVIRONMENT', 'Environment setting')
        ]
        
        # Check required variables
        for var_name, description in required_vars:
            value = os.getenv(var_name)
            if value:
                # Mask sensitive values
                display_value = value[:10] + "..." if len(value) > 10 else value
                if 'token' in var_name.lower() or 'password' in var_name.lower():
                    display_value = "*" * len(value)
                self.log_check("Environment", var_name, True, f"Set: {display_value}")
            else:
                self.log_check("Environment", var_name, False, "", f"Required variable {var_name} not set")
        
        # Check optional variables
        for var_name, description in optional_vars:
            value = os.getenv(var_name)
            if value:
                display_value = value[:20] + "..." if len(value) > 20 else value
                if 'password' in var_name.lower():
                    display_value = "*" * len(value)
                self.log_check("Environment", var_name, True, f"Set: {display_value}")
            else:
                self.log_warning(f"Optional variable {var_name} not set - using defaults")
    
    def check_database_connection(self):
        """Test database connectivity and setup"""
        print("\n🗄️  Database Verification")
        print("-" * 40)
        
        try:
            # Get database connection string
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                # Try to construct from parts
                host = os.getenv('POSTGRES_HOST', 'localhost')
                port = os.getenv('POSTGRES_PORT', '5432')
                database = os.getenv('POSTGRES_DB', 'mastergroup_recommendations')
                user = os.getenv('POSTGRES_USER', 'postgres')
                password = os.getenv('POSTGRES_PASSWORD', 'postgres')
                db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            
            # Test connection
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # Test basic connectivity
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            self.log_check("Database", "Connection", True, f"PostgreSQL: {version[:50]}...")
            
            # Check required tables
            required_tables = [
                'orders', 'order_items', 'customer_purchases', 
                'product_pairs', 'product_statistics', 'customer_statistics',
                'recommendation_cache'
            ]
            
            missing_tables = []
            for table in required_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                """, (table,))
                
                exists = cursor.fetchone()[0]
                if exists:
                    # Count records
                    cursor.execute(f"SELECT COUNT(*) FROM {table};")
                    count = cursor.fetchone()[0]
                    self.log_check("Database", f"Table {table}", True, f"{count:,} records")
                else:
                    missing_tables.append(table)
                    self.log_check("Database", f"Table {table}", False, "", "Table missing")
            
            # Check required functions
            required_functions = [
                'populate_order_items_from_orders',
                'rebuild_customer_purchases',
                'rebuild_product_pairs',
                'rebuild_product_statistics',
                'rebuild_customer_statistics'
            ]
            
            for function in required_functions:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.routines 
                        WHERE routine_schema = 'public' 
                        AND routine_name = %s
                        AND routine_type = 'FUNCTION'
                    );
                """, (function,))
                
                exists = cursor.fetchone()[0]
                self.log_check("Database", f"Function {function}", exists, 
                             "Available" if exists else "", "Function missing" if not exists else "")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            self.log_check("Database", "Connection", False, "", str(e))
    
    def check_redis_connection(self):
        """Test Redis connectivity"""
        print("\n🔴 Redis Cache Verification")
        print("-" * 40)
        
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            r = redis.from_url(redis_url)
            
            # Test connection
            r.ping()
            self.log_check("Redis", "Connection", True, "Connected successfully")
            
            # Test cache operations
            test_key = "test_production_verification"
            test_value = {"test": True, "timestamp": datetime.now().isoformat()}
            
            # Set test value
            r.setex(test_key, 60, json.dumps(test_value))
            
            # Get test value
            retrieved = json.loads(r.get(test_key))
            r.delete(test_key)
            
            self.log_check("Redis", "Cache Operations", True, "Set/Get/Delete operations working")
            
        except Exception as e:
            self.log_check("Redis", "Connection", False, "", str(e))
            self.log_warning("Redis cache not available - recommendations will work but without caching")
    
    def check_api_endpoints(self):
        """Verify Master Group API connectivity"""
        print("\n🌐 API Connectivity")
        print("-" * 40)
        
        try:
            base_url = os.getenv('MASTER_GROUP_API_BASE', 'https://mes.master.com.pk')
            auth_token = os.getenv('MASTER_GROUP_AUTH_TOKEN', '')
            
            if not auth_token:
                self.log_check("API", "Authentication", False, "", "No auth token provided")
                return
            
            headers = {
                'Authorization': auth_token,
                'Content-Type': 'application/json'
            }
            
            # Test POS orders endpoint
            try:
                pos_url = f"{base_url}/get_pos_orders"
                response = requests.get(pos_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    order_count = len(data) if isinstance(data, list) else len(data.get('data', []))
                    self.log_check("API", "POS Orders", True, f"Accessible - {order_count} orders available")
                else:
                    self.log_check("API", "POS Orders", False, "", f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_check("API", "POS Orders", False, "", str(e))
            
            # Test OE orders endpoint
            try:
                oe_url = f"{base_url}/get_oe_orders"
                response = requests.get(oe_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    order_count = len(data) if isinstance(data, list) else len(data.get('data', []))
                    self.log_check("API", "OE Orders", True, f"Accessible - {order_count} orders available")
                else:
                    self.log_check("API", "OE Orders", False, "", f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_check("API", "OE Orders", False, "", str(e))
                
        except Exception as e:
            self.log_check("API", "Configuration", False, "", str(e))
    
    def check_recommendation_service(self):
        """Test recommendation service endpoints"""
        print("\n🎯 Recommendation Service")
        print("-" * 40)
        
        service_url = os.getenv('RECOMMENDATION_SERVICE_URL', 'http://localhost:8001')
        
        # Key endpoints to test
        endpoints_to_test = [
            ('/health', 'Health Check'),
            ('/api/v1/sync/status', 'Sync Status'),
            ('/api/v1/sync/scheduler-status', 'Scheduler Status'),
            ('/api/v1/recommendations/popular', 'Popular Recommendations'),
            ('/api/v1/cache/stats', 'Cache Statistics')
        ]
        
        for endpoint, name in endpoints_to_test:
            try:
                response = requests.get(f"{service_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    self.log_check("Service", name, True, f"Responding correctly")
                else:
                    self.log_check("Service", name, False, "", f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_check("Service", name, False, "", str(e))
    
    def check_scheduler_status(self):
        """Verify scheduler is working"""
        print("\n⏰ Scheduler Verification")
        print("-" * 40)
        
        try:
            # Try to import scheduler
            sys.path.append('/Users/clustox_1/Documents/MasterGroup-RecommendationSystem/recommendation-engine-service')
            from services.scheduler import get_scheduler
            
            scheduler = get_scheduler()
            status = scheduler.get_status()
            
            # Check scheduler status
            self.log_check("Scheduler", "Service", status['scheduler_running'], 
                         f"Auto-sync: {status['auto_sync_enabled']}, Interval: {status['sync_interval_minutes']}min")
            
            # Check next sync time
            if status['next_sync_time']:
                next_sync = datetime.fromisoformat(status['next_sync_time'].replace('Z', '+00:00'))
                time_until = next_sync - datetime.now(next_sync.tzinfo)
                self.log_check("Scheduler", "Next Sync", True, 
                             f"Scheduled for {next_sync.strftime('%Y-%m-%d %H:%M:%S')} ({time_until})")
            else:
                self.log_check("Scheduler", "Next Sync", False, "", "No sync scheduled")
            
            # Check auto-pilot training
            if status['auto_pilot_enabled'] and status['next_training_time']:
                next_training = datetime.fromisoformat(status['next_training_time'].replace('Z', '+00:00'))
                self.log_check("Scheduler", "Auto-Pilot Training", True,
                             f"Next training: {next_training.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                self.log_check("Scheduler", "Auto-Pilot Training", False, "", "Auto-pilot not scheduled")
                
        except Exception as e:
            self.log_check("Scheduler", "Service", False, "", str(e))
    
    def check_dependencies(self):
        """Check Python dependencies"""
        print("\n📦 Dependencies")
        print("-" * 40)
        
        required_packages = [
            'fastapi', 'uvicorn', 'psycopg2', 'redis', 'pandas', 
            'numpy', 'scikit-learn', 'apscheduler', 'requests'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.log_check("Dependencies", package, True, "Installed")
            except ImportError:
                self.log_check("Dependencies", package, False, "", f"Package {package} not installed")
    
    def run_production_verification(self):
        """Run complete production verification"""
        print("🚀 Production Setup Verification")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        print()
        
        # Run all checks
        self.check_dependencies()
        self.check_environment_variables()
        self.check_database_connection()
        self.check_redis_connection()
        self.check_api_endpoints()
        self.check_recommendation_service()
        self.check_scheduler_status()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def generate_summary(self):
        """Generate verification summary"""
        total_checks = len(self.results['checks'])
        passed_checks = len([c for c in self.results['checks'] if c['status'] == 'PASS'])
        failed_checks = total_checks - passed_checks
        
        self.results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'success_rate': round((passed_checks / total_checks) * 100, 1) if total_checks > 0 else 0,
            'warnings_count': len(self.results['warnings']),
            'ready_for_production': failed_checks == 0
        }
        
        print("\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("-" * 30)
        print(f"Total Checks: {total_checks}")
        print(f"✅ Passed: {passed_checks}")
        print(f"❌ Failed: {failed_checks}")
        print(f"⚠️  Warnings: {len(self.results['warnings'])}")
        print(f"Success Rate: {self.results['summary']['success_rate']}%")
        
        if self.results['summary']['ready_for_production']:
            print("\n🎉 SYSTEM IS READY FOR PRODUCTION!")
        else:
            print("\n🚨 SYSTEM NOT READY - Please fix the following issues:")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        if self.results['warnings']:
            print("\n⚠️  Warnings to consider:")
            for warning in self.results['warnings']:
                print(f"   • {warning}")
    
    def save_results(self, filename: str = None):
        """Save verification results"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"production_verification_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📝 Results saved to: {filename}")

def main():
    verifier = ProductionVerifier()
    results = verifier.run_production_verification()
    verifier.save_results()
    
    # Exit with error code if not ready
    if not results['summary']['ready_for_production']:
        sys.exit(1)

if __name__ == "__main__":
    main()
