# 🚀 MasterGroup On-Premise Deployment Guide

## Quick Start (HP ProLiant / Ubuntu Server)

### Prerequisites
Your server should have:
- **OS**: Ubuntu 22.04 LTS (recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 100GB+ SSD
- **Network**: Internet access for initial sync

---

## Option 1: One-Click Full Deployment (Recommended)

```bash
# 1. Download the master script
curl -O https://raw.githubusercontent.com/dev-hub-stack/recommender-engine/dev/scripts/deploy_full_stack.sh

# 2. Make executable
chmod +x deploy_full_stack.sh

# 3. Run as root
sudo ./deploy_full_stack.sh
```

This will automatically:
- ✅ Install Python 3.11, Node.js 20, PostgreSQL 14, Redis, Nginx
- ✅ Clone both frontend and backend repositories
- ✅ Setup database with tables and migrations
- ✅ Configure all environment variables
- ✅ Start all services (API, Dashboard, Nginx)
- ✅ Setup cron jobs for daily data sync
- ✅ Begin initial 4-year data sync

---

## Option 2: Manual Clone + Quick Setup

```bash
# 1. Create directory
sudo mkdir -p /opt/mastergroup && cd /opt/mastergroup

# 2. Clone repositories
git clone https://github.com/dev-hub-stack/recommender-engine.git backend
git clone https://github.com/dev-hub-stack/recommender-dashboard.git frontend

# 3. Run quick setup
cd backend
chmod +x scripts/quick_setup.sh
sudo ./scripts/quick_setup.sh
```

---

## What Gets Installed

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11 | Backend API runtime |
| Node.js | 20 LTS | Frontend build/serve |
| PostgreSQL | 14 | Primary database |
| Redis | 6+ | Caching layer |
| Nginx | Latest | Reverse proxy |
| PM2 | Latest | Frontend process manager |

---

## Access Points (After Deployment)

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | `http://SERVER_IP` | Analytics frontend |
| API | `http://SERVER_IP:8001` | Backend API |
| API Docs | `http://SERVER_IP:8001/docs` | Swagger documentation |
| Health Check | `http://SERVER_IP:8001/health` | Service status |

---

## Default Credentials

### Dashboard Login
- **Email**: `admin@mastergroup.com`
- **Password**: `MG@2024#Secure!Pass`

### Database
- **Host**: localhost
- **Port**: 5432
- **Database**: mastergroup_recommendations
- **User**: mastergroup_user
- **Password**: (Generated during setup - check `/opt/mastergroup/CREDENTIALS.txt`)

---

## Service Management

### Check Status
```bash
# All services at once
/opt/mastergroup/check_status.sh

# Individual services
sudo systemctl status mastergroup-api
pm2 status
sudo systemctl status postgresql
sudo systemctl status redis-server
sudo systemctl status nginx
```

### Restart Services
```bash
# Restart everything
/opt/mastergroup/restart_services.sh

# Restart individually
sudo systemctl restart mastergroup-api     # Backend
pm2 restart mastergroup-dashboard          # Frontend
sudo systemctl reload nginx                # Nginx
```

### View Logs
```bash
# Backend API logs
tail -f /var/log/mastergroup/api.log
sudo journalctl -u mastergroup-api -f

# Frontend logs
pm2 logs mastergroup-dashboard

# Nginx logs
tail -f /var/log/nginx/mastergroup-access.log

# Data sync logs
tail -f /var/log/mastergroup/sync.log
```

---

## Data Sync Commands

### Initial Historical Sync (4 years)
```bash
cd /opt/mastergroup/backend
source venv/bin/activate
python scripts/local_ml_pipeline.py --sync-days 1500
```
*Note: Takes 30-60 minutes depending on network*

### Daily Sync (Automatic via Cron)
```bash
# Already configured - runs at 2:00 AM daily
# Manual trigger:
python scripts/local_ml_pipeline.py --sync-days 2
```

### Train Models Only (No Sync)
```bash
python scripts/local_ml_pipeline.py --train-only
```

### Pre-warm Cache
```bash
python scripts/prewarm_cache.py
```

---

## Update Code

```bash
# Pull latest changes and restart
/opt/mastergroup/update_code.sh

# Or manually:
cd /opt/mastergroup/backend
git pull origin dev
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart mastergroup-api

cd /opt/mastergroup/frontend
git pull origin main
npm install
npm run build
pm2 restart mastergroup-dashboard
```

---

## Backup & Restore

### Database Backup (Automatic Daily at 3 AM)
```bash
# Manual backup
pg_dump -U mastergroup_user mastergroup_recommendations | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Database Restore
```bash
gunzip -c backup_20260118.sql.gz | psql -U mastergroup_user mastergroup_recommendations
```

### Full Application Backup
```bash
tar -czf mastergroup_backup_$(date +%Y%m%d).tar.gz \
    /opt/mastergroup/backend/.env \
    /opt/mastergroup/backend/models \
    /var/log/mastergroup
```

---

## Troubleshooting

### API Not Starting
```bash
# Check logs
sudo journalctl -u mastergroup-api -n 50

# Common fixes:
sudo systemctl restart postgresql
sudo systemctl restart redis-server
sudo systemctl restart mastergroup-api
```

### Database Connection Issues
```bash
# Test connection
psql -h localhost -U mastergroup_user -d mastergroup_recommendations

# Check PostgreSQL is running
sudo systemctl status postgresql
```

### Redis Issues
```bash
# Test Redis
redis-cli ping

# Restart Redis
sudo systemctl restart redis-server
```

### Frontend Not Loading
```bash
# Check PM2
pm2 list
pm2 logs mastergroup-dashboard

# Rebuild frontend
cd /opt/mastergroup/frontend
npm run build
pm2 restart mastergroup-dashboard
```

### Nginx 502 Bad Gateway
```bash
# Check backend is running
curl http://localhost:8001/health

# Check Nginx config
sudo nginx -t
sudo systemctl reload nginx
```

---

## Firewall Configuration

```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow API port (if accessing directly)
sudo ufw allow 8001/tcp

# Enable firewall
sudo ufw enable
```

---

## Performance Tuning

### PostgreSQL (for large datasets)
Edit `/etc/postgresql/14/main/postgresql.conf`:
```ini
shared_buffers = 4GB        # 25% of RAM
effective_cache_size = 12GB # 75% of RAM
work_mem = 256MB
maintenance_work_mem = 1GB
max_connections = 200
```

### Redis (for caching)
Edit `/etc/redis/redis.conf`:
```ini
maxmemory 2gb
maxmemory-policy allkeys-lru
```

### Uvicorn Workers
Edit `/etc/systemd/system/mastergroup-api.service`:
```ini
# Increase workers based on CPU cores
--workers 8  # for 8-core server
```

---

## Support

- **Documentation**: `/opt/mastergroup/backend/docs/`
- **Logs**: `/var/log/mastergroup/`
- **Credentials**: `/opt/mastergroup/CREDENTIALS.txt`

---

## Quick Commands Reference

| Action | Command |
|--------|---------|
| Check all services | `/opt/mastergroup/check_status.sh` |
| Restart all | `/opt/mastergroup/restart_services.sh` |
| Update code | `/opt/mastergroup/update_code.sh` |
| View API logs | `tail -f /var/log/mastergroup/api.log` |
| View sync logs | `tail -f /var/log/mastergroup/sync.log` |
| Manual sync | `python scripts/local_ml_pipeline.py --sync-days 2` |
| Restart API | `sudo systemctl restart mastergroup-api` |
| Restart frontend | `pm2 restart mastergroup-dashboard` |
