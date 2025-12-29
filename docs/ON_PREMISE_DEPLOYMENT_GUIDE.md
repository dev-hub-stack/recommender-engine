# On-Premise Server Deployment Guide
**MasterGroup Recommendation Engine**  
*Complete guide for deploying to client's own infrastructure*

---

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Network & Security](#network--security)
3. [Prerequisites Installation](#prerequisites-installation)
4. [Application Deployment](#application-deployment)
5. [Database Setup](#database-setup)
6. [Service Configuration](#service-configuration)
7. [SSL/TLS Configuration](#ssltls-configuration)
8. [Firewall Configuration](#firewall-configuration)
9. [Monitoring & Logging](#monitoring--logging)
10. [Backup & Disaster Recovery](#backup--disaster-recovery)
11. [Maintenance & Updates](#maintenance--updates)
12. [Troubleshooting](#troubleshooting)

---

## 1. System Requirements

### Minimum Hardware Specifications
```
- **CPU:** 4 cores (Intel Xeon or AMD EPYC recommended)
- **RAM:** 16 GB minimum (32 GB recommended)
- **Storage:** 100 GB SSD (for OS + Application)
- **Database Storage:** 500 GB SSD minimum (1 TB recommended for growth)
- **Network:** 1 Gbps network interface
```

### Recommended Hardware Specifications
```
- **CPU:** 8+ cores
- **RAM:** 32-64 GB
- **Storage:** 250 GB NVMe SSD
- **Database Storage:** 2 TB SSD RAID 10
- **Network:** 10 Gbps network interface
- **Backup Storage:** 2 TB external/NAS
```

### Operating System
```
✅ Ubuntu Server 22.04 LTS (Recommended)
✅ Ubuntu Server 20.04 LTS
✅ CentOS 8 / Rocky Linux 8
✅ Red Hat Enterprise Linux 8+
❌ Windows Server (Not supported)
```

---

## 2. Network & Security

### Required Network Access

#### Outbound (From Your Server)
```
1. Master Group API: https://mes.master.com.pk (Port 443)
   - Purpose: Data synchronization
   - Required: YES
   
2. GitHub: https://github.com (Port 443)
   - Purpose: Code deployment/updates
   - Required: YES
   
3. Package Repositories:
   - http://archive.ubuntu.com (Port 80/443)
   - https://pypi.org (Port 443)
   - Purpose: System and Python package installation
   - Required: YES (during installation/updates)
```

#### Inbound (To Your Server)
```
1. API Access: Port 8001 (or 443 with reverse proxy)
   - Purpose: Backend API for frontend dashboard
   - Source: Frontend server / User workstations
   
2. SSH Access: Port 22
   - Purpose: Server administration
   - Source: IT admin workstations only
   - Security: Restrict to specific IPs, use key authentication

3. PostgreSQL: Port 5432 (OPTIONAL)
   - Purpose: Database access (if external DB server)
   - Source: Application server only
   - Security: Restrict to application server IP only
```

### Firewall Recommendations
```bash
# Allow SSH (from admin IPs only)
ufw allow from <admin_ip> to any port 22

# Allow API access (from frontend server)
ufw allow from <frontend_server_ip> to any port 8001

# Allow PostgreSQL (if using external DB)
ufw allow from <app_server_ip> to any port 5432

# Allow outbound HTTPS
ufw allow out 443/tcp

# Enable firewall
ufw enable
```

---

## 3. Prerequisites Installation

### Step 1: Update System
```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git curl wget vim htop net-tools build-essential
```

### Step 2: Install Python 3.10+
```bash
# Install Python
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Verify installation
python3 --version  # Should show Python 3.10+
```

### Step 3: Install PostgreSQL 14+
```bash
# Add PostgreSQL repository
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt update

# Install PostgreSQL
sudo apt install -y postgresql-14 postgresql-contrib-14

# Verify installation
sudo systemctl status postgresql
```

### Step 4: Install Redis
```bash
# Install Redis
sudo apt install -y redis-server

# Configure Redis to start on boot
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Verify installation
redis-cli ping  # Should return "PONG"
```

### Step 5: Install Nginx (Reverse Proxy)
```bash
# Install Nginx
sudo apt install -y nginx

# Enable and start Nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

---

## 4. Application Deployment

### Step 1: Create Application User
```bash
# Create dedicated user for application
sudo useradd -m -s /bin/bash mastergroup
sudo passwd mastergroup

# Create application directory
sudo mkdir -p /opt/mastergroup-ml
sudo chown mastergroup:mastergroup /opt/mastergroup-ml
```

### Step 2: Clone Repository
```bash
# Switch to mastergroup user
sudo su - mastergroup

# Clone repository
cd /opt
git clone https://github.com/dev-hub-stack/recommender-engine.git mastergroup-ml
cd mastergroup-ml

# Checkout production branch
git checkout main  # or 'dev' if deploying dev version
```

### Step 3: Setup Python Environment
```bash
# Still as mastergroup user
cd /opt/mastergroup-ml

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```bash
# Create .env file
cp .env.example .env

# Edit .env with actual credentials
nano .env
```

**Required `.env` Configuration:**
```ini
# ==========================================
# DATABASE CONFIGURATION
# ==========================================
PG_HOST=localhost                          # Use 'localhost' if DB on same server
PG_PORT=5432
PG_DB=mastergroup_recommendations
PG_USER=mastergroup_user
PG_PASSWORD=<STRONG_PASSWORD_HERE>        # Use strong password!
PG_SSLMODE=require                         # Use 'disable' for local DB

# ==========================================
# MASTER GROUP API
# ==========================================
MASTER_GROUP_API_BASE=https://mes.master.com.pk
MASTER_GROUP_AUTH_TOKEN=<YOUR_API_TOKEN>   # Obtain from Master Group

# ==========================================
# ML CONFIGURATION
# ==========================================
USE_LOCAL_ML=true                          # Always true for on-premise
MODEL_PATH=models/                         # Local model storage path

# ==========================================
# REDIS CONFIGURATION
# ==========================================
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=                            # Set if Redis has password
REDIS_DB=0

# ==========================================
# API CONFIGURATION
# ==========================================
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=4                              # Adjust based on CPU cores
DEBUG=false                                # MUST be false in production
LOG_LEVEL=INFO

# ==========================================
# SECURITY
# ==========================================
SECRET_KEY=<GENERATE_RANDOM_SECRET>        # Use: openssl rand -hex 32
JWT_SECRET=<GENERATE_RANDOM_SECRET>        # Use: openssl rand -hex 32
JWT_EXPIRY_HOURS=24

# ==========================================
# CORS (Frontend Access)
# ==========================================
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://analytics.mastergroup.com
```

**Generate Secrets:**
```bash
# Generate SECRET_KEY
openssl rand -hex 32

# Generate JWT_SECRET
openssl rand -hex 32
```

---

## 5. Database Setup

### Step 1: Create Database and User
```bash
# Switch to postgres user
sudo -u postgres psql

# Inside PostgreSQL shell:
CREATE DATABASE mastergroup_recommendations;
CREATE USER mastergroup_user WITH ENCRYPTED PASSWORD 'YOUR_STRONG_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE mastergroup_recommendations TO mastergroup_user;

# Exit PostgreSQL
\q
```

### Step 2: Configure PostgreSQL for Remote Access (if needed)
```bash
# Edit postgresql.conf
sudo nano /etc/postgresql/14/main/postgresql.conf

# Find and modify:
listen_addresses = '0.0.0.0'  # Allow connections from any IP
max_connections = 200         # Increase connection limit
shared_buffers = 4GB          # 25% of total RAM
effective_cache_size = 12GB   # 75% of total RAM
work_mem = 64MB
maintenance_work_mem = 1GB

# Edit pg_hba.conf for access control
sudo nano /etc/postgresql/14/main/pg_hba.conf

# Add this line (replace <app_server_ip> with actual IP)
host    mastergroup_recommendations    mastergroup_user    <app_server_ip>/32    md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### Step 3: Run Database Migrations
```bash
# As mastergroup user
cd /opt/mastergroup-ml
source venv/bin/activate

# Run setup script
python3 scripts/setup_database.py

# Verify database setup
python3 -c "
from dotenv import load_dotenv
load_dotenv()
import psycopg2, os
conn = psycopg2.connect(
    host=os.getenv('PG_HOST'),
    port=os.getenv('PG_PORT'),
    database=os.getenv('PG_DB'),
    user=os.getenv('PG_USER'),
    password=os.getenv('PG_PASSWORD')
)
print('✅ Database connection successful!')
conn.close()
"
```

### Step 4: Initial Data Sync (4 Years Historical Data)
```bash
# This will take 2-4 hours depending on data volume
# Run in screen/tmux session to avoid interruption
screen -S ml_pipeline

# Activate environment
cd /opt/mastergroup-ml
source venv/bin/activate

# Run initial sync (1500 days = ~4 years)
python3 scripts/local_ml_pipeline.py --sync-days 1500

# Detach from screen: Ctrl+A, then D
# Reattach: screen -r ml_pipeline
```

---

## 6. Service Configuration

### Step 1: Create Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/mastergroup-api.service
```

**Service File Content:**
```ini
[Unit]
Description=MasterGroup Recommendation API
After=network.target postgresql.service redis-server.service
Requires=postgresql.service redis-server.service

[Service]
Type=simple
User=mastergroup
Group=mastergroup
WorkingDirectory=/opt/mastergroup-ml
Environment="PATH=/opt/mastergroup-ml/venv/bin"
EnvironmentFile=/opt/mastergroup-ml/.env

# Start command
ExecStart=/opt/mastergroup-ml/venv/bin/uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 4 \
    --log-level info \
    --access-log \
    --log-config logging_config.json

# Restart policy
Restart=always
RestartSec=10

# Resource limits
LimitNOFILE=65536
TimeoutStartSec=300

# Logging
StandardOutput=append:/var/log/mastergroup/api.log
StandardError=append:/var/log/mastergroup/api-error.log

[Install]
WantedBy=multi-user.target
```

### Step 2: Create Log Directory
```bash
sudo mkdir -p /var/log/mastergroup
sudo chown mastergroup:mastergroup /var/log/mastergroup
```

### Step 3: Enable and Start Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable mastergroup-api

# Start service
sudo systemctl start mastergroup-api

# Check status
sudo systemctl status mastergroup-api

# View logs
sudo journalctl -u mastergroup-api -f
```

### Step 4: Configure Nginx Reverse Proxy
```bash
# Create Nginx config
sudo nano /etc/nginx/sites-available/mastergroup-api
```

**Nginx Configuration:**
```nginx
# Upstream backend
upstream mastergroup_backend {
    server 127.0.0.1:8001;
    keepalive 64;
}

# HTTP -> HTTPS Redirect
server {
    listen 80;
    server_name api.mastergroup.local;  # Change to your domain
    return 301 https://$server_name$request_uri;
}

# HTTPS Server
server {
    listen 443 ssl http2;
    server_name api.mastergroup.local;  # Change to your domain

    # SSL Certificates (if you have them)
    ssl_certificate /etc/ssl/certs/mastergroup-api.crt;
    ssl_certificate_key /etc/ssl/private/mastergroup-api.key;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Logging
    access_log /var/log/nginx/mastergroup-api-access.log;
    error_log /var/log/nginx/mastergroup-api-error.log;

    # Max upload size
    client_max_body_size 100M;

    # Proxy settings
    location / {
        proxy_pass http://mastergroup_backend;
        proxy_http_version 1.1;
        
        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        send_timeout 300s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://mastergroup_backend/health;
        access_log off;
    }
}
```

**Enable Nginx Site:**
```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/mastergroup-api /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

---

## 7. SSL/TLS Configuration

### Option 1: Self-Signed Certificate (Testing Only)
```bash
# Generate self-signed certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/mastergroup-api.key \
    -out /etc/ssl/certs/mastergroup-api.crt

# Set permissions
sudo chmod 600 /etc/ssl/private/mastergroup-api.key
```

### Option 2: Let's Encrypt (Production - Free)
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.mastergroup.com

# Auto-renewal is configured automatically
# Test renewal:
sudo certbot renew --dry-run
```

### Option 3: Corporate CA Certificate (Enterprise)
```bash
# Copy certificates provided by your IT department
sudo cp company-ca.crt /etc/ssl/certs/mastergroup-api.crt
sudo cp company-ca.key /etc/ssl/private/mastergroup-api.key

# Set permissions
sudo chmod 600 /etc/ssl/private/mastergroup-api.key
sudo chmod 644 /etc/ssl/certs/mastergroup-api.crt
```

---

## 8. Firewall Configuration

### Using UFW (Ubuntu)
```bash
# Reset firewall (if needed)
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (replace <admin_ip> with actual admin IP)
sudo ufw allow from <admin_ip> to any port 22

# Allow HTTP/HTTPS (from frontend server or users)
sudo ufw allow from <frontend_server_ip> to any port 80
sudo ufw allow from <frontend_server_ip> to any port 443

# Or allow from entire internal network
sudo ufw allow from 192.168.1.0/24 to any port 80
sudo ufw allow from 192.168.1.0/24 to any port 443

# Allow PostgreSQL (if using external DB)
sudo ufw allow from <app_server_ip> to any port 5432

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status numbered
```

### Using iptables (Advanced)
```bash
# Flush existing rules
sudo iptables -F

# Default policies
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT

# Allow loopback
sudo iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH from admin IP
sudo iptables -A INPUT -p tcp -s <admin_ip> --dport 22 -j ACCEPT

# Allow HTTP/HTTPS from frontend
sudo iptables -A INPUT -p tcp -s <frontend_ip> --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp -s <frontend_ip> --dport 443 -j ACCEPT

# Save rules
sudo apt install -y iptables-persistent
sudo netfilter-persistent save
```

---

## 9. Monitoring & Logging

### Setup Log Rotation
```bash
# Create logrotate config
sudo nano /etc/logrotate.d/mastergroup
```

**Logrotate Configuration:**
```
/var/log/mastergroup/*.log {
    daily
    rotate 30
    missingok
    notifempty
    compress
    delaycompress
    copytruncate
    create 0644 mastergroup mastergroup
}
```

### Setup System Monitoring
```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs sysstat

# Enable sysstat
sudo systemctl enable sysstat
sudo systemctl start sysstat
```

### Setup Application Monitoring Script
```bash
# Create monitoring script
sudo nano /opt/mastergroup-ml/scripts/health_check.sh
```

**Health Check Script:**
```bash
#!/bin/bash
# Health check script for MasterGroup API

API_URL="http://localhost:8001/health"
LOG_FILE="/var/log/mastergroup/health_check.log"
ALERT_EMAIL="admin@your-company.com"

# Check API health
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL")

if [ "$response" != "200" ]; then
    echo "$(date): API health check FAILED (HTTP $response)" >> "$LOG_FILE"
    
    # Send email alert
    echo "MasterGroup API is DOWN. HTTP Status: $response" | \
        mail -s "ALERT: MasterGroup API Health Check Failed" "$ALERT_EMAIL"
    
    # Try to restart service
    sudo systemctl restart mastergroup-api
    sleep 10
    
    # Check again
    response2=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL")
    if [ "$response2" == "200" ]; then
        echo "$(date): API restarted successfully" >> "$LOG_FILE"
    else
        echo "$(date): API restart FAILED" >> "$LOG_FILE"
    fi
else
    echo "$(date): API health check OK" >> "$LOG_FILE"
fi
```

**Make executable and schedule:**
```bash
chmod +x /opt/mastergroup-ml/scripts/health_check.sh

# Add to cron (check every 5 minutes)
crontab -e

# Add this line:
*/5 * * * * /opt/mastergroup-ml/scripts/health_check.sh
```

---

## 10. Backup & Disaster Recovery

### Database Backup Script
```bash
# Create backup script
sudo nano /opt/mastergroup-ml/scripts/backup_database.sh
```

**Backup Script:**
```bash
#!/bin/bash
# Database backup script

BACKUP_DIR="/backup/mastergroup"
DB_NAME="mastergroup_recommendations"
DB_USER="mastergroup_user"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Perform backup
PGPASSWORD="$PG_PASSWORD" pg_dump \
    -h localhost \
    -U "$DB_USER" \
    -F c \
    -b \
    -v \
    -f "$BACKUP_DIR/backup_$DATE.dump" \
    "$DB_NAME"

# Compress backup
gzip "$BACKUP_DIR/backup_$DATE.dump"

# Delete old backups (older than RETENTION_DAYS)
find "$BACKUP_DIR" -name "backup_*.dump.gz" -mtime +$RETENTION_DAYS -delete

echo "$(date): Database backup completed successfully"
```

**Schedule Daily Backups:**
```bash
chmod +x /opt/mastergroup-ml/scripts/backup_database.sh

# Add to cron (daily at 3 AM)
crontab -e

# Add:
0 3 * * * /opt/mastergroup-ml/scripts/backup_database.sh >> /var/log/mastergroup/backup.log 2>&1
```

### Database Restore Procedure
```bash
# Restore from backup
BACKUP_FILE="/backup/mastergroup/backup_20241228_030000.dump.gz"

# Uncompress
gunzip "$BACKUP_FILE"

# Restore (will prompt for password)
pg_restore \
    -h localhost \
    -U mastergroup_user \
    -d mastergroup_recommendations \
    -v \
    -c \
    "${BACKUP_FILE%.gz}"
```

### Application Backup
```bash
# Backup application files (models, logs, config)
tar -czf /backup/mastergroup/app_backup_$(date +%Y%m%d).tar.gz \
    /opt/mastergroup-ml/models \
    /opt/mastergroup-ml/.env \
    /var/log/mastergroup

# Backup to remote server (optional)
rsync -avz -e "ssh -p 22" \
    /backup/mastergroup/ \
    backup-server:/backups/mastergroup/
```

---

## 11. Maintenance & Updates

### Daily Maintenance Tasks (Automated)
```bash
# Edit mastergroup user crontab
crontab -e

# Add these tasks:

# 1. Daily data sync and model training (2 AM)
0 2 * * * cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python3 scripts/local_ml_pipeline.py --sync-days 2 >> /var/log/mastergroup/pipeline.log 2>&1

# 2. Province data cleaning (weekly, Sundays at 2:30 AM)
30 2 * * 0 cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python3 src/scheduled_tasks.py >> /var/log/mastergroup/province_cleanup.log 2>&1

# 3. Cache pre-warming (after training, 3 AM)
0 3 * * * cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python3 scripts/prewarm_cache.py >> /var/log/mastergroup/cache_prewarm.log 2>&1

# 4. Database backup (daily, 3:30 AM)
30 3 * * * /opt/mastergroup-ml/scripts/backup_database.sh >> /var/log/mastergroup/backup.log 2>&1

# 5. Log cleanup (weekly, Sundays at 4 AM)
0 4 * * 0 find /var/log/mastergroup -name "*.log" -mtime +30 -delete
```

### Manual Update Procedure
```bash
# 1. Stop API service
sudo systemctl stop mastergroup-api

# 2. Backup current version
cd /opt/mastergroup-ml
cp -r /opt/mastergroup-ml /opt/mastergroup-ml.backup.$(date +%Y%m%d)

# 3. Pull latest code
git fetch origin
git checkout main  # or specific version tag
git pull

# 4. Update dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# 5. Run database migrations (if any)
python3 scripts/setup_database.py

# 6. Clear cache
redis-cli FLUSHALL

# 7. Restart service
sudo systemctl start mastergroup-api

# 8. Verify health
curl http://localhost:8001/health
```

---

## 12. Troubleshooting

### Issue: API Not Starting
**Check logs:**
```bash
sudo journalctl -u mastergroup-api -n 100 --no-pager
sudo tail -f /var/log/mastergroup/api-error.log
```

**Common causes:**
1. Database connection failure → Check `.env` credentials
2. Port already in use → `sudo lsof -i :8001`
3. Missing dependencies → `pip install -r requirements.txt`

### Issue: Database Connection Timeout
**Check PostgreSQL:**
```bash
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"
```

**Check connection:**
```bash
psql -h localhost -U mastergroup_user -d mastergroup_recommendations
```

### Issue: High Memory Usage
**Check processes:**
```bash
htop
ps aux --sort=-%mem | head -10
```

**Restart services:**
```bash
sudo systemctl restart mastergroup-api
sudo systemctl restart redis-server
```

### Issue: Slow API Response
**Check cache:**
```bash
redis-cli INFO stats | grep -E 'keyspace_hits|keyspace_misses'
```

**Clear and rebuild cache:**
```bash
redis-cli FLUSHALL
python3 scripts/prewarm_cache.py
```

### Issue: Data Sync Failures
**Check Master Group API connectivity:**
```bash
curl -I https://mes.master.com.pk
```

**Check auth token:**
```bash
source venv/bin/activate
python3 -c "
from dotenv import load_dotenv
import os, requests
load_dotenv()
token = os.getenv('MASTER_GROUP_AUTH_TOKEN')
response = requests.get('https://mes.master.com.pk/api/v1/orders', headers={'Authorization': f'Bearer {token}'})
print('Status:', response.status_code)
"
```

---

## 📞 Support Contacts

**Technical Issues:**
- Email: dev-support@clustox.com
- Phone: [Your support number]

**Emergency Contacts:**
- 24/7 Hotline: [Emergency number]

**Documentation:**
- Deployment Guide: This document
- API Documentation: `/docs` endpoint
- Backend Setup: `BACKEND_SETUP_GUIDE_V2.md`

---

## ✅ Deployment Checklist

**Pre-Deployment:**
- [ ] Server hardware meets specifications
- [ ] Network connectivity verified
- [ ] Firewall rules configured
- [ ] SSL certificates obtained
- [ ] Master Group API token obtained

**Installation:**
- [ ] Operating system installed and updated
- [ ] Python 3.10+ installed
- [ ] PostgreSQL 14+ installed and configured
- [ ] Redis installed and running
- [ ] Nginx installed and configured
- [ ] Application code cloned
- [ ] Python dependencies installed
- [ ] `.env` file configured with correct credentials

**Database Setup:**
- [ ] Database created
- [ ] User created with permissions
- [ ] Migrations run successfully
- [ ] Initial data sync completed (4 years)
- [ ] Models trained successfully

**Service Configuration:**
- [ ] Systemd service created and enabled
- [ ] Service starts without errors
- [ ] Nginx reverse proxy configured
- [ ] SSL/TLS configured
- [ ] Health check endpoint accessible

**Automation:**
- [ ] Daily sync cron job scheduled
- [ ] Backup cron job scheduled
- [ ] Health check cron job scheduled
- [ ] Log rotation configured

**Testing:**
- [ ] API health check passes
- [ ] Frontend can connect to backend
- [ ] Login works
- [ ] Data loads in dashboard
- [ ] Recommendations generate successfully

**Documentation:**
- [ ] Admin credentials documented (securely)
- [ ] Firewall rules documented
- [ ] Backup procedures documented
- [ ] Update procedures documented

**Production Readiness:**
- [ ] Monitoring configured
- [ ] Logging configured
- [ ] Backup tested and verified
- [ ] Disaster recovery plan documented
- [ ] Support team trained

---

**Deployment Date:** _______________  
**Deployed By:** _______________  
**Version:** _______________  

---

*End of On-Premise Deployment Guide*
