#!/bin/bash
#===============================================================================
# MasterGroup - Quick Setup After Git Clone
#
# Run this AFTER you've manually cloned both repos:
#   git clone https://github.com/dev-hub-stack/recommender-engine.git backend
#   git clone https://github.com/dev-hub-stack/recommender-dashboard.git frontend
#
# Usage:
#   chmod +x quick_setup.sh
#   sudo ./quick_setup.sh
#===============================================================================

set -e

echo "======================================"
echo "  MasterGroup Quick Setup"
echo "======================================"

# Check both directories exist
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "ERROR: Please clone both repos first:"
    echo "  git clone https://github.com/dev-hub-stack/recommender-engine.git backend"
    echo "  git clone https://github.com/dev-hub-stack/recommender-dashboard.git frontend"
    exit 1
fi

# Step 1: Install system dependencies
echo ""
echo "[1/8] Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    postgresql-14 postgresql-contrib-14 \
    redis-server \
    nginx \
    curl wget git

# Step 2: Install Node.js 20
echo ""
echo "[2/8] Installing Node.js 20..."
if ! command -v node &> /dev/null || [[ $(node -v) != v20* ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt install -y nodejs
fi
sudo npm install -g pm2

# Step 3: Setup PostgreSQL
echo ""
echo "[3/8] Setting up PostgreSQL..."
DB_PASSWORD="MG_$(openssl rand -hex 8)"

sudo -u postgres psql -c "CREATE USER mastergroup_user WITH PASSWORD '${DB_PASSWORD}';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE mastergroup_recommendations OWNER mastergroup_user;" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL ON DATABASE mastergroup_recommendations TO mastergroup_user;"

# Step 4: Setup Backend
echo ""
echo "[4/8] Setting up Backend..."
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create .env
cat > .env <<EOF
PG_HOST=localhost
PG_PORT=5432
PG_DB=mastergroup_recommendations
PG_USER=mastergroup_user
PG_PASSWORD=${DB_PASSWORD}
PG_SSLMODE=disable

MASTER_GROUP_API_BASE=https://mes.master.com.pk
MASTER_GROUP_AUTH_TOKEN=H2rcLQPfzYoV55k9ZyT5aWkyyMKEyxHhX1r3ntrkrvrGeVL4dOsGv3EcQMY2

USE_LOCAL_ML=true
REDIS_HOST=localhost
REDIS_PORT=6379

SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
EOF

# Run migrations
python scripts/setup_database.py
cd ..

# Step 5: Setup Frontend
echo ""
echo "[5/8] Setting up Frontend..."
cd frontend
npm install

cat > .env <<EOF
VITE_API_BASE_URL=http://localhost:8001/api/v1
EOF

npm run build
cd ..

# Step 6: Create systemd service for backend
echo ""
echo "[6/8] Creating backend service..."
CURRENT_DIR=$(pwd)

sudo tee /etc/systemd/system/mastergroup-api.service > /dev/null <<EOF
[Unit]
Description=MasterGroup API
After=network.target postgresql.service redis-server.service

[Service]
Type=simple
User=$USER
WorkingDirectory=${CURRENT_DIR}/backend
Environment="PATH=${CURRENT_DIR}/backend/venv/bin"
ExecStart=${CURRENT_DIR}/backend/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8001 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mastergroup-api
sudo systemctl start mastergroup-api

# Step 7: Setup frontend with PM2
echo ""
echo "[7/8] Setting up frontend service..."
cd frontend
pm2 start "npm run build && npm run preview -- --host 0.0.0.0 --port 3000" --name mastergroup-dashboard
pm2 save
cd ..

# Step 8: Configure Nginx
echo ""
echo "[8/8] Configuring Nginx..."
sudo tee /etc/nginx/sites-available/mastergroup > /dev/null <<'EOF'
server {
    listen 80 default_server;
    server_name _;

    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/mastergroup /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# Done!
echo ""
echo "======================================"
echo "  Setup Complete!"
echo "======================================"
echo ""
echo "Database Password: ${DB_PASSWORD}"
echo ""
echo "Access:"
echo "  Dashboard: http://$(hostname -I | awk '{print $1}')"
echo "  API Docs:  http://$(hostname -I | awk '{print $1}'):8001/docs"
echo ""
echo "Commands:"
echo "  View API logs:      sudo journalctl -u mastergroup-api -f"
echo "  Restart API:        sudo systemctl restart mastergroup-api"
echo "  View frontend logs: pm2 logs mastergroup-dashboard"
echo ""
echo "Next: Run initial data sync (may take 30-60 minutes):"
echo "  cd backend && source venv/bin/activate"
echo "  python scripts/local_ml_pipeline.py --sync-days 1500"
echo ""
