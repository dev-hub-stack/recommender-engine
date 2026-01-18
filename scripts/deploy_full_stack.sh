#!/bin/bash
#===============================================================================
# MasterGroup Analytics Platform - On-Premise Full Deployment Script
# For HP ProLiant Servers running Ubuntu 22.04 LTS
#
# This script installs and configures:
#   - System prerequisites (git, curl, etc.)
#   - Python 3.11 with virtual environment
#   - Node.js 20 LTS with npm
#   - PostgreSQL 14 with database setup
#   - Redis for caching
#   - Nginx as reverse proxy
#   - Backend API service (FastAPI/Uvicorn)
#   - Frontend dashboard (React/Vite)
#   - Data sync and ML training pipeline
#   - Systemd services for auto-start
#   - Cron jobs for daily sync
#
# Usage:
#   chmod +x deploy_full_stack.sh
#   sudo ./deploy_full_stack.sh
#
# After deployment:
#   Backend API:  http://your-server-ip:8001
#   Frontend:     http://your-server-ip (port 80)
#   Dashboard:    http://your-server-ip/dashboard
#
#===============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - MODIFY THESE AS NEEDED
APP_USER="mastergroup"
APP_DIR="/opt/mastergroup"
BACKEND_DIR="${APP_DIR}/backend"
FRONTEND_DIR="${APP_DIR}/frontend"
LOG_DIR="/var/log/mastergroup"
BACKUP_DIR="/backup/mastergroup"

# Git repositories
BACKEND_REPO="https://github.com/dev-hub-stack/recommender-engine.git"
FRONTEND_REPO="https://github.com/dev-hub-stack/recommender-dashboard.git"
GIT_BRANCH="dev"

# Database configuration
DB_NAME="mastergroup_recommendations"
DB_USER="mastergroup_user"
DB_PASSWORD=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)

# API Configuration
API_PORT=8001
API_WORKERS=4

# Frontend Configuration
FRONTEND_PORT=3000

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

header() {
    echo ""
    echo -e "${BLUE}======================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}======================================================${NC}"
    echo ""
}

#===============================================================================
# STEP 0: Pre-flight checks
#===============================================================================
header "STEP 0: Pre-flight Checks"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "Please run this script with sudo or as root"
fi

# Check if Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    warn "This script is designed for Ubuntu. Proceed with caution on other distributions."
fi

log "✅ Pre-flight checks passed"

#===============================================================================
# STEP 1: System Update & Essential Tools
#===============================================================================
header "STEP 1: System Update & Essential Tools"

log "Updating system packages..."
apt update && apt upgrade -y

log "Installing essential tools..."
apt install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    net-tools \
    build-essential \
    software-properties-common \
    gnupg2 \
    lsb-release \
    ca-certificates \
    apt-transport-https \
    unzip \
    screen \
    tmux

log "✅ Essential tools installed"

#===============================================================================
# STEP 2: Install Python 3.11
#===============================================================================
header "STEP 2: Installing Python 3.11"

log "Adding deadsnakes PPA for Python 3.11..."
add-apt-repository -y ppa:deadsnakes/ppa
apt update

log "Installing Python 3.11..."
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Set Python 3.11 as default python3
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

log "Python version: $(python3 --version)"
log "✅ Python 3.11 installed"

#===============================================================================
# STEP 3: Install Node.js 20 LTS
#===============================================================================
header "STEP 3: Installing Node.js 20 LTS"

log "Adding NodeSource repository..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -

log "Installing Node.js..."
apt install -y nodejs

log "Node version: $(node --version)"
log "NPM version: $(npm --version)"

# Install PM2 for process management (frontend)
log "Installing PM2 globally..."
npm install -g pm2

log "✅ Node.js 20 installed"

#===============================================================================
# STEP 4: Install PostgreSQL 14
#===============================================================================
header "STEP 4: Installing PostgreSQL 14"

log "Adding PostgreSQL repository..."
echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | tee /etc/apt/sources.list.d/pgdg.list
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
apt update

log "Installing PostgreSQL 14..."
apt install -y postgresql-14 postgresql-contrib-14

# Start PostgreSQL
systemctl enable postgresql
systemctl start postgresql

log "PostgreSQL version: $(psql --version)"
log "✅ PostgreSQL 14 installed"

#===============================================================================
# STEP 5: Install Redis
#===============================================================================
header "STEP 5: Installing Redis"

log "Installing Redis..."
apt install -y redis-server

# Configure Redis
sed -i 's/^supervised no/supervised systemd/' /etc/redis/redis.conf

# Start Redis
systemctl enable redis-server
systemctl restart redis-server

# Verify Redis
if redis-cli ping | grep -q "PONG"; then
    log "✅ Redis installed and running"
else
    error "Redis installation failed"
fi

#===============================================================================
# STEP 6: Install Nginx
#===============================================================================
header "STEP 6: Installing Nginx"

log "Installing Nginx..."
apt install -y nginx

systemctl enable nginx
systemctl start nginx

log "✅ Nginx installed"

#===============================================================================
# STEP 7: Create Application User and Directories
#===============================================================================
header "STEP 7: Creating Application User and Directories"

log "Creating application user: ${APP_USER}"
if id "$APP_USER" &>/dev/null; then
    log "User ${APP_USER} already exists"
else
    useradd -m -s /bin/bash "$APP_USER"
fi

log "Creating directories..."
mkdir -p "$APP_DIR"
mkdir -p "$BACKEND_DIR"
mkdir -p "$FRONTEND_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$BACKUP_DIR"
mkdir -p "${APP_DIR}/models"

chown -R ${APP_USER}:${APP_USER} "$APP_DIR"
chown -R ${APP_USER}:${APP_USER} "$LOG_DIR"
chown -R ${APP_USER}:${APP_USER} "$BACKUP_DIR"

log "✅ Directories created"

#===============================================================================
# STEP 8: Setup PostgreSQL Database
#===============================================================================
header "STEP 8: Setting Up PostgreSQL Database"

log "Creating database and user..."

sudo -u postgres psql <<EOF
-- Create user if not exists
DO
\$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${DB_USER}') THEN
      CREATE USER ${DB_USER} WITH ENCRYPTED PASSWORD '${DB_PASSWORD}';
   END IF;
END
\$\$;

-- Create database if not exists
SELECT 'CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
ALTER USER ${DB_USER} CREATEDB;
EOF

# Configure PostgreSQL for connections
log "Configuring PostgreSQL..."

# Allow connections from localhost
cat >> /etc/postgresql/14/main/pg_hba.conf <<EOF

# MasterGroup Application
local   ${DB_NAME}    ${DB_USER}                      md5
host    ${DB_NAME}    ${DB_USER}    127.0.0.1/32      md5
host    ${DB_NAME}    ${DB_USER}    ::1/128           md5
EOF

# Performance tuning
cat >> /etc/postgresql/14/main/postgresql.conf <<EOF

# MasterGroup Optimizations
max_connections = 200
shared_buffers = 2GB
effective_cache_size = 6GB
maintenance_work_mem = 512MB
work_mem = 128MB
random_page_cost = 1.1
effective_io_concurrency = 200
EOF

systemctl restart postgresql

log "✅ PostgreSQL database configured"

#===============================================================================
# STEP 9: Clone Backend Repository
#===============================================================================
header "STEP 9: Cloning Backend Repository"

log "Cloning backend repository..."
cd "$APP_DIR"

if [ -d "$BACKEND_DIR/.git" ]; then
    log "Backend repo exists, pulling latest..."
    cd "$BACKEND_DIR"
    sudo -u "$APP_USER" git fetch origin
    sudo -u "$APP_USER" git checkout "$GIT_BRANCH"
    sudo -u "$APP_USER" git pull origin "$GIT_BRANCH"
else
    log "Cloning fresh..."
    rm -rf "$BACKEND_DIR"
    sudo -u "$APP_USER" git clone "$BACKEND_REPO" "$BACKEND_DIR"
    cd "$BACKEND_DIR"
    sudo -u "$APP_USER" git checkout "$GIT_BRANCH"
fi

log "✅ Backend repository cloned"

#===============================================================================
# STEP 10: Setup Backend Python Environment
#===============================================================================
header "STEP 10: Setting Up Backend Python Environment"

cd "$BACKEND_DIR"

log "Creating Python virtual environment..."
sudo -u "$APP_USER" python3 -m venv venv

log "Installing Python dependencies..."
sudo -u "$APP_USER" ./venv/bin/pip install --upgrade pip
sudo -u "$APP_USER" ./venv/bin/pip install -r requirements.txt

log "✅ Backend Python environment setup complete"

#===============================================================================
# STEP 11: Configure Backend Environment Variables
#===============================================================================
header "STEP 11: Configuring Backend Environment"

log "Creating .env file..."

cat > "${BACKEND_DIR}/.env" <<EOF
# =============================================
# MasterGroup Recommendation Engine - Production
# Generated: $(date)
# =============================================

# DATABASE CONFIGURATION
PG_HOST=localhost
PG_PORT=5432
PG_DB=${DB_NAME}
PG_USER=${DB_USER}
PG_PASSWORD=${DB_PASSWORD}
PG_SSLMODE=disable

# MASTER GROUP API (For Data Sync)
MASTER_GROUP_API_BASE=https://mes.master.com.pk
MASTER_GROUP_AUTH_TOKEN=H2rcLQPfzYoV55k9ZyT5aWkyyMKEyxHhX1r3ntrkrvrGeVL4dOsGv3EcQMY2

# ML CONFIGURATION
USE_LOCAL_ML=true
MODEL_PATH=${APP_DIR}/models/

# REDIS CONFIGURATION
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# API CONFIGURATION
API_HOST=0.0.0.0
API_PORT=${API_PORT}
API_WORKERS=${API_WORKERS}
DEBUG=false
LOG_LEVEL=INFO

# SECURITY
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
JWT_EXPIRY_HOURS=24

# CORS (Allow frontend access)
ALLOWED_ORIGINS=http://localhost,http://localhost:${FRONTEND_PORT},http://127.0.0.1
EOF

chown ${APP_USER}:${APP_USER} "${BACKEND_DIR}/.env"
chmod 600 "${BACKEND_DIR}/.env"

log "✅ Backend environment configured"

#===============================================================================
# STEP 12: Run Database Migrations
#===============================================================================
header "STEP 12: Running Database Migrations"

cd "$BACKEND_DIR"

log "Running database setup script..."
sudo -u "$APP_USER" ./venv/bin/python scripts/setup_database.py

log "✅ Database migrations complete"

#===============================================================================
# STEP 13: Clone Frontend Repository
#===============================================================================
header "STEP 13: Cloning Frontend Repository"

log "Cloning frontend repository..."
cd "$APP_DIR"

if [ -d "$FRONTEND_DIR/.git" ]; then
    log "Frontend repo exists, pulling latest..."
    cd "$FRONTEND_DIR"
    sudo -u "$APP_USER" git fetch origin
    sudo -u "$APP_USER" git checkout main
    sudo -u "$APP_USER" git pull origin main
else
    log "Cloning fresh..."
    rm -rf "$FRONTEND_DIR"
    sudo -u "$APP_USER" git clone "$FRONTEND_REPO" "$FRONTEND_DIR"
    cd "$FRONTEND_DIR"
    sudo -u "$APP_USER" git checkout main
fi

log "✅ Frontend repository cloned"

#===============================================================================
# STEP 14: Setup Frontend
#===============================================================================
header "STEP 14: Setting Up Frontend"

cd "$FRONTEND_DIR"

log "Installing npm dependencies..."
sudo -u "$APP_USER" npm install

log "Creating frontend .env file..."
cat > "${FRONTEND_DIR}/.env" <<EOF
# MasterGroup Analytics Dashboard - Production
VITE_API_BASE_URL=http://localhost:${API_PORT}/api/v1
EOF

chown ${APP_USER}:${APP_USER} "${FRONTEND_DIR}/.env"

log "Building frontend for production..."
sudo -u "$APP_USER" npm run build

log "✅ Frontend setup complete"

#===============================================================================
# STEP 15: Create Systemd Service for Backend API
#===============================================================================
header "STEP 15: Creating Backend Systemd Service"

log "Creating mastergroup-api.service..."

cat > /etc/systemd/system/mastergroup-api.service <<EOF
[Unit]
Description=MasterGroup Recommendation API
After=network.target postgresql.service redis-server.service
Requires=postgresql.service redis-server.service

[Service]
Type=simple
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${BACKEND_DIR}
Environment="PATH=${BACKEND_DIR}/venv/bin"
EnvironmentFile=${BACKEND_DIR}/.env

ExecStart=${BACKEND_DIR}/venv/bin/uvicorn src.main:app \\
    --host 0.0.0.0 \\
    --port ${API_PORT} \\
    --workers ${API_WORKERS} \\
    --log-level info

Restart=always
RestartSec=10
LimitNOFILE=65536

StandardOutput=append:${LOG_DIR}/api.log
StandardError=append:${LOG_DIR}/api-error.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable mastergroup-api

log "✅ Backend service created"

#===============================================================================
# STEP 16: Create PM2 Frontend Service
#===============================================================================
header "STEP 16: Setting Up Frontend with PM2"

cd "$FRONTEND_DIR"

# Create PM2 ecosystem file
cat > ecosystem.config.js <<EOF
module.exports = {
  apps: [{
    name: 'mastergroup-dashboard',
    script: 'node_modules/vite/bin/vite.js',
    args: 'preview --host 0.0.0.0 --port ${FRONTEND_PORT}',
    cwd: '${FRONTEND_DIR}',
    env: {
      NODE_ENV: 'production'
    },
    instances: 1,
    exec_mode: 'fork',
    watch: false,
    autorestart: true,
    max_memory_restart: '500M'
  }]
};
EOF

chown ${APP_USER}:${APP_USER} ecosystem.config.js

# Start with PM2
sudo -u "$APP_USER" pm2 start ecosystem.config.js
sudo -u "$APP_USER" pm2 save

# Setup PM2 startup
pm2 startup systemd -u ${APP_USER} --hp /home/${APP_USER}

log "✅ Frontend PM2 service created"

#===============================================================================
# STEP 17: Configure Nginx Reverse Proxy
#===============================================================================
header "STEP 17: Configuring Nginx"

log "Creating Nginx configuration..."

cat > /etc/nginx/sites-available/mastergroup <<EOF
# MasterGroup Analytics Platform

# Upstream for Backend API
upstream backend_api {
    server 127.0.0.1:${API_PORT};
    keepalive 64;
}

# Upstream for Frontend
upstream frontend_app {
    server 127.0.0.1:${FRONTEND_PORT};
}

server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    # Logging
    access_log /var/log/nginx/mastergroup-access.log;
    error_log /var/log/nginx/mastergroup-error.log;

    # Max upload size
    client_max_body_size 100M;

    # API Endpoints
    location /api/ {
        proxy_pass http://backend_api;
        proxy_http_version 1.1;
        
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # CORS Headers
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Authorization, Content-Type" always;
        
        if (\$request_method = OPTIONS) {
            return 204;
        }
    }

    # Health check endpoint
    location /health {
        proxy_pass http://backend_api/health;
        access_log off;
    }

    # Frontend Dashboard
    location / {
        proxy_pass http://frontend_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/mastergroup /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test and reload
nginx -t && systemctl reload nginx

log "✅ Nginx configured"

#===============================================================================
# STEP 18: Setup Cron Jobs for Data Sync
#===============================================================================
header "STEP 18: Setting Up Cron Jobs"

log "Creating cron jobs..."

# Create cron script
cat > ${BACKEND_DIR}/scripts/daily_sync.sh <<'EOF'
#!/bin/bash
# Daily data sync and training script
cd /opt/mastergroup/backend
source venv/bin/activate
python scripts/local_ml_pipeline.py --sync-days 2 >> /var/log/mastergroup/sync.log 2>&1
python scripts/prewarm_cache.py >> /var/log/mastergroup/cache.log 2>&1
EOF

chmod +x ${BACKEND_DIR}/scripts/daily_sync.sh
chown ${APP_USER}:${APP_USER} ${BACKEND_DIR}/scripts/daily_sync.sh

# Add cron jobs
(crontab -u ${APP_USER} -l 2>/dev/null || echo "") | cat - <<EOF | crontab -u ${APP_USER} -
# MasterGroup Recommendation Engine - Cron Jobs

# Daily sync and training at 2:00 AM
0 2 * * * ${BACKEND_DIR}/scripts/daily_sync.sh

# Hourly data sync (orders only) 
0 */4 * * * cd ${BACKEND_DIR} && ./venv/bin/python scripts/local_ml_pipeline.py --sync-days 1 >> ${LOG_DIR}/hourly_sync.log 2>&1

# Daily database backup at 3:00 AM
0 3 * * * pg_dump -U ${DB_USER} ${DB_NAME} | gzip > ${BACKUP_DIR}/db_backup_\$(date +\%Y\%m\%d).sql.gz

# Cache pre-warming after daily sync
30 2 * * * cd ${BACKEND_DIR} && ./venv/bin/python scripts/prewarm_cache.py >> ${LOG_DIR}/cache_prewarm.log 2>&1
EOF

log "✅ Cron jobs configured"

#===============================================================================
# STEP 19: Run Initial Data Sync
#===============================================================================
header "STEP 19: Running Initial Data Sync (This may take 30-60 minutes)"

cd "$BACKEND_DIR"

log "Starting historical data sync (4 years)..."
log "This will run in the background. Check progress with: tail -f ${LOG_DIR}/initial_sync.log"

# Run in background
sudo -u "$APP_USER" nohup ./venv/bin/python scripts/local_ml_pipeline.py --sync-days 1500 > ${LOG_DIR}/initial_sync.log 2>&1 &

log "✅ Initial sync started in background (PID: $!)"

#===============================================================================
# STEP 20: Start Services
#===============================================================================
header "STEP 20: Starting All Services"

log "Starting backend API..."
systemctl start mastergroup-api

log "Waiting for API to be ready..."
sleep 10

# Check if API is running
if curl -s http://localhost:${API_PORT}/health > /dev/null 2>&1; then
    log "✅ Backend API is running"
else
    warn "Backend API may still be starting up. Check: systemctl status mastergroup-api"
fi

log "✅ All services started"

#===============================================================================
# STEP 21: Create Management Scripts
#===============================================================================
header "STEP 21: Creating Management Scripts"

# Create status script
cat > ${APP_DIR}/check_status.sh <<'EOF'
#!/bin/bash
echo "=== MasterGroup Platform Status ==="
echo ""
echo "=== Backend API ==="
systemctl status mastergroup-api --no-pager | head -5
echo ""
echo "=== Frontend Dashboard ==="
pm2 list
echo ""
echo "=== PostgreSQL ==="
systemctl status postgresql --no-pager | head -3
echo ""
echo "=== Redis ==="
redis-cli ping
echo ""
echo "=== Nginx ==="
systemctl status nginx --no-pager | head -3
echo ""
echo "=== API Health Check ==="
curl -s http://localhost:8001/health | head -c 200
echo ""
EOF

chmod +x ${APP_DIR}/check_status.sh

# Create restart script
cat > ${APP_DIR}/restart_services.sh <<'EOF'
#!/bin/bash
echo "Restarting all MasterGroup services..."
sudo systemctl restart mastergroup-api
sudo -u mastergroup pm2 restart all
sudo systemctl reload nginx
echo "Done!"
EOF

chmod +x ${APP_DIR}/restart_services.sh

# Create update script
cat > ${APP_DIR}/update_code.sh <<'EOF'
#!/bin/bash
echo "Updating MasterGroup from Git..."

cd /opt/mastergroup/backend
git pull origin dev
source venv/bin/activate
pip install -r requirements.txt

cd /opt/mastergroup/frontend
git pull origin main
npm install
npm run build

sudo systemctl restart mastergroup-api
sudo -u mastergroup pm2 restart all

echo "Update complete!"
EOF

chmod +x ${APP_DIR}/update_code.sh

log "✅ Management scripts created"

#===============================================================================
# DEPLOYMENT COMPLETE
#===============================================================================
header "🎉 DEPLOYMENT COMPLETE!"

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║               MasterGroup Analytics Platform                        ║${NC}"
echo -e "${GREEN}║                   Deployment Successful!                            ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                                     ║${NC}"
echo -e "${GREEN}║  Access URLs:                                                       ║${NC}"
echo -e "${GREEN}║    Dashboard:  http://${SERVER_IP}                                  ║${NC}"
echo -e "${GREEN}║    API:        http://${SERVER_IP}:${API_PORT}                      ║${NC}"
echo -e "${GREEN}║    API Docs:   http://${SERVER_IP}:${API_PORT}/docs                 ║${NC}"
echo -e "${GREEN}║                                                                     ║${NC}"
echo -e "${GREEN}║  Database Credentials (SAVE THESE!):                                ║${NC}"
echo -e "${GREEN}║    Host:     localhost                                              ║${NC}"
echo -e "${GREEN}║    Database: ${DB_NAME}                                             ║${NC}"
echo -e "${GREEN}║    User:     ${DB_USER}                                             ║${NC}"
echo -e "${GREEN}║    Password: ${DB_PASSWORD}                                         ║${NC}"
echo -e "${GREEN}║                                                                     ║${NC}"
echo -e "${GREEN}║  Management Commands:                                               ║${NC}"
echo -e "${GREEN}║    Check status:   ${APP_DIR}/check_status.sh                       ║${NC}"
echo -e "${GREEN}║    Restart all:    ${APP_DIR}/restart_services.sh                   ║${NC}"
echo -e "${GREEN}║    Update code:    ${APP_DIR}/update_code.sh                        ║${NC}"
echo -e "${GREEN}║                                                                     ║${NC}"
echo -e "${GREEN}║  Logs:                                                              ║${NC}"
echo -e "${GREEN}║    API:            ${LOG_DIR}/api.log                               ║${NC}"
echo -e "${GREEN}║    Sync:           ${LOG_DIR}/sync.log                              ║${NC}"
echo -e "${GREEN}║    Initial Sync:   ${LOG_DIR}/initial_sync.log                      ║${NC}"
echo -e "${GREEN}║                                                                     ║${NC}"
echo -e "${GREEN}║  NOTE: Initial data sync is running in background.                 ║${NC}"
echo -e "${GREEN}║        Check progress: tail -f ${LOG_DIR}/initial_sync.log          ║${NC}"
echo -e "${GREEN}║                                                                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Save credentials to file
cat > ${APP_DIR}/CREDENTIALS.txt <<EOF
===============================================
MasterGroup Platform Credentials
Generated: $(date)
Server: $(hostname)
IP: ${SERVER_IP}
===============================================

DATABASE:
  Host: localhost
  Port: 5432
  Database: ${DB_NAME}
  User: ${DB_USER}
  Password: ${DB_PASSWORD}

ACCESS URLS:
  Dashboard: http://${SERVER_IP}
  API: http://${SERVER_IP}:${API_PORT}
  API Docs: http://${SERVER_IP}:${API_PORT}/docs

IMPORTANT PATHS:
  Backend: ${BACKEND_DIR}
  Frontend: ${FRONTEND_DIR}
  Logs: ${LOG_DIR}
  Backups: ${BACKUP_DIR}

SERVICE COMMANDS:
  sudo systemctl status mastergroup-api
  sudo -u mastergroup pm2 status
  ${APP_DIR}/check_status.sh
  ${APP_DIR}/restart_services.sh
  ${APP_DIR}/update_code.sh
===============================================
EOF

chmod 600 ${APP_DIR}/CREDENTIALS.txt
log "Credentials saved to: ${APP_DIR}/CREDENTIALS.txt"

echo ""
log "✅ Deployment script completed successfully!"
