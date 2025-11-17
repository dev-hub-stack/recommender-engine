# Production Deployment Checklist

## Pre-Deployment Verification ✅

### Database Setup
- [ ] PostgreSQL server running and accessible
- [ ] Database `mastergroup_recommendations` created
- [ ] Database user has proper permissions (CREATE, INSERT, UPDATE, DELETE, SELECT)
- [ ] `setup_recommendation_tables.sql` script executed successfully
- [ ] All 6 tables created (orders, order_items, customer_purchases, product_pairs, product_statistics, customer_statistics, recommendation_cache)
- [ ] All 5 functions created (populate_order_items_from_orders, rebuild_customer_purchases, etc.)
- [ ] Sample data populated and verified

### Environment Configuration
- [ ] `.env` file configured with production values
- [ ] `DATABASE_URL` set correctly
- [ ] `MASTER_GROUP_API_BASE` pointing to production API
- [ ] `MASTER_GROUP_AUTH_TOKEN` configured with valid token
- [ ] `REDIS_URL` configured (optional but recommended)
- [ ] `ENVIRONMENT=production` set
- [ ] `DEBUG=false` set

### API Connectivity
- [ ] Master Group POS API accessible (`/get_pos_orders`)
- [ ] Master Group OE API accessible (`/get_oe_orders`)
- [ ] API authentication working
- [ ] Sample data retrieval successful
- [ ] Network connectivity from production server verified

### Service Dependencies
- [ ] Python 3.8+ installed
- [ ] All packages from `requirements.txt` installed
- [ ] Redis server running (for caching)
- [ ] APScheduler dependencies available

## Deployment Steps 🚀

### 1. Server Preparation
```bash
# Create application directory
sudo mkdir -p /opt/recommendation-engine
cd /opt/recommendation-engine

# Clone repository
git clone <repository-url> .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment file
cp .env.example .env

# Edit configuration (use production values)
nano .env
```

### 3. Database Setup
```bash
# Run verification script
python verify_database_setup.py

# If needed, manually run setup
psql -d mastergroup_recommendations -f setup_recommendation_tables.sql

# Populate initial data
psql -d mastergroup_recommendations -c "
  SELECT populate_order_items_from_orders();
  SELECT rebuild_customer_purchases();
  SELECT rebuild_product_pairs();
  SELECT rebuild_product_statistics();
  SELECT rebuild_customer_statistics();
"
```

### 4. Production Verification
```bash
# Run comprehensive verification
python verify_production_setup.py

# Test all endpoints
python test_all_endpoints.py
```

### 5. Service Deployment
```bash
# Option 1: Direct Python execution
python src/main.py

# Option 2: Using systemd service (recommended)
sudo cp recommendation-engine.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable recommendation-engine
sudo systemctl start recommendation-engine
```

## Systemd Service Configuration 🔧

Create `/etc/systemd/system/recommendation-engine.service`:

```ini
[Unit]
Description=Master Group Recommendation Engine
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/recommendation-engine
Environment=PATH=/opt/recommendation-engine/venv/bin
ExecStart=/opt/recommendation-engine/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Monitoring & Health Checks 📊

### Health Check Endpoints
- **Service Health**: `GET /health`
- **Sync Status**: `GET /api/v1/sync/status`
- **Scheduler Status**: `GET /api/v1/sync/scheduler-status`
- **Cache Statistics**: `GET /api/v1/cache/stats`
- **System Metrics**: `GET /metrics`

### Automated Monitoring
```bash
# Add to cron for regular health checks
# Every 5 minutes
*/5 * * * * curl -f http://localhost:8001/health || systemctl restart recommendation-engine

# Daily verification
0 6 * * * /opt/recommendation-engine/venv/bin/python /opt/recommendation-engine/verify_production_setup.py
```

## Auto-Sync & Scheduler Verification ⏰

### Scheduler Features
✅ **Auto-Sync**: Synchronizes orders every 15 minutes (configurable)
✅ **Auto-Pilot Learning**: Retrains ML models daily at 3:00 AM
✅ **Manual Triggers**: Manual sync and training via API endpoints

### Scheduler Endpoints
- **Status**: `GET /api/v1/sync/scheduler-status`
- **Manual Sync**: `POST /api/v1/sync/trigger`
- **Manual Training**: `POST /api/v1/training/trigger`
- **Sync History**: `GET /api/v1/sync/history`

### Configuration
```env
# Scheduler settings
SYNC_INTERVAL_MINUTES=15          # How often to sync (default: 15 min)
SYNC_POS_ORDERS=true              # Enable POS order sync
SYNC_OE_ORDERS=true               # Enable OE order sync
SYNC_BATCH_SIZE=1000              # Records per batch
```

## Production Database Script Verification 🗄️

### Script Features
The `setup_recommendation_tables.sql` script is **production-ready** and includes:

✅ **Safe Operations**: Uses `IF NOT EXISTS` for tables and `CREATE OR REPLACE` for functions
✅ **Data Integrity**: Foreign key constraints and proper indexes
✅ **Performance**: Optimized indexes for recommendation queries
✅ **Idempotent**: Can be run multiple times safely
✅ **Rollback Safe**: Won't break existing data

### What the Script Creates

#### Tables (6 total):
1. **order_items** - Product items extracted from orders
2. **customer_purchases** - Customer purchase history for collaborative filtering
3. **product_pairs** - Product co-purchase relationships for cross-selling
4. **product_statistics** - Product popularity metrics
5. **customer_statistics** - Customer segmentation data
6. **recommendation_cache** - Caching layer for performance

#### Functions (5 total):
1. **populate_order_items_from_orders()** - Extract items from orders.items_json
2. **rebuild_customer_purchases()** - Build customer purchase aggregations
3. **rebuild_product_pairs()** - Calculate product relationships
4. **rebuild_product_statistics()** - Generate product analytics
5. **rebuild_customer_statistics()** - Create customer profiles

#### Indexes (15 total):
- Performance indexes on all critical query paths
- Composite indexes for complex recommendation queries
- Proper ordering for DESC queries (top products, etc.)

### Production Safety
- **Non-destructive**: Script won't delete existing data
- **Incremental**: Can add new tables/functions without affecting existing ones
- **Backwards Compatible**: Safe to run on existing databases
- **Transaction Safe**: Uses proper PostgreSQL transactions

## Load Balancing & Scaling 📈

### Horizontal Scaling
- **Multiple Instances**: Run multiple service instances behind load balancer
- **Database Pooling**: Configure connection pooling for high load
- **Redis Clustering**: Use Redis cluster for distributed caching

### Vertical Scaling
- **CPU**: Recommendation algorithms are CPU-intensive
- **RAM**: In-memory caching and ML models need adequate RAM
- **Storage**: Database storage for order history and analytics

### Recommended Resources
- **Minimum**: 2 CPU cores, 4GB RAM, 50GB storage
- **Recommended**: 4 CPU cores, 8GB RAM, 100GB+ storage
- **High Load**: 8+ CPU cores, 16GB+ RAM, SSD storage

## Backup & Recovery 💾

### Database Backups
```bash
# Daily database backup
pg_dump mastergroup_recommendations > backup_$(date +%Y%m%d).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/opt/backups/recommendation-db"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump mastergroup_recommendations | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete  # Keep 7 days
```

### Recovery Process
```bash
# Restore from backup
psql -d mastergroup_recommendations < backup_20241117.sql

# Verify data integrity
python verify_database_setup.py
```

## Troubleshooting 🔧

### Common Issues

#### Service Won't Start
```bash
# Check logs
journalctl -u recommendation-engine -f

# Check configuration
python verify_production_setup.py

# Test database connection
python -c "import psycopg2; psycopg2.connect('postgresql://...')"
```

#### Sync Not Working
```bash
# Check scheduler status
curl http://localhost:8001/api/v1/sync/scheduler-status

# Manual sync test
curl -X POST http://localhost:8001/api/v1/sync/trigger

# Check API connectivity
curl -H "Authorization: TOKEN" https://mes.master.com.pk/get_pos_orders
```

#### Performance Issues
```bash
# Check cache stats
curl http://localhost:8001/api/v1/cache/stats

# Monitor database performance
# Add indexes, optimize queries as needed

# Check system resources
htop
```

## Security Considerations 🔒

### Environment Security
- [ ] Database credentials secured
- [ ] API tokens stored securely
- [ ] Network access restricted (firewall rules)
- [ ] SSL/TLS for API communications
- [ ] Regular security updates

### Application Security
- [ ] Input validation on all endpoints
- [ ] Rate limiting configured
- [ ] Authentication for administrative endpoints
- [ ] Audit logging enabled
- [ ] Error handling doesn't expose sensitive data

## Success Metrics 📊

### System Health
- [ ] All endpoints responding within 2 seconds
- [ ] Auto-sync running every 15 minutes
- [ ] Daily training completing successfully
- [ ] Cache hit rate > 80%
- [ ] Database queries optimized (< 1 second average)

### Business Metrics
- [ ] Recommendations generating for all customers
- [ ] Cross-selling suggestions available
- [ ] Popular products updated regularly
- [ ] Customer segmentation working
- [ ] Analytics dashboard functional

---

## Final Production Readiness Check ✅

Run these commands to verify everything is working:

```bash
# 1. Comprehensive system verification
python verify_production_setup.py

# 2. Test all endpoints
python test_all_endpoints.py

# 3. Verify scheduler is running
curl http://localhost:8001/api/v1/sync/scheduler-status

# 4. Check service health
curl http://localhost:8001/health

# 5. Test recommendations
curl "http://localhost:8001/api/v1/recommendations/popular?limit=10"
```

If all checks pass ✅, your system is **PRODUCTION READY** 🎉
