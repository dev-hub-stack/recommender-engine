#!/bin/bash
# ============================================================
# MasterGroup API Server Setup Script
# Run this on the Lightsail instance via SSH
# ============================================================

set -e

echo "============================================================"
echo "Setting up MasterGroup API Server"
echo "============================================================"

# Update system
echo "Step 1: Updating system..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install dependencies
echo "Step 2: Installing dependencies..."
sudo apt-get install -y python3.11 python3.11-venv python3-pip git libpq-dev nginx curl

# Create app directory
echo "Step 3: Creating app directory..."
sudo mkdir -p /opt/mastergroup-api
sudo chown ubuntu:ubuntu /opt/mastergroup-api
cd /opt/mastergroup-api

# Clone repository (specific branch)
echo "Step 4: Cloning repository..."
git clone -b feature/aws-personalize-deployment https://github.com/dev-hub-stack/recommender-engine.git . || {
    echo "Git clone failed. You may need to copy files manually."
}

# Setup Python environment
echo "Step 5: Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create environment file
echo "Step 6: Creating environment file..."
cat > /opt/mastergroup-api/.env << 'EOF'
PG_HOST=ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com
PG_PORT=5432
PG_DB=mastergroup_recommendations
PG_USER=postgres
PG_PASSWORD=MasterGroup2024Secure!
REDIS_HOST=localhost
REDIS_PORT=6379
AWS_REGION=us-east-1
PERSONALIZE_CAMPAIGN_ARN=arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign
PERSONALIZE_TRACKING_ID=6b8748e4-4cbe-412e-8247-b6978d2814ac
EOF

# Create systemd service
echo "Step 7: Creating systemd service..."
sudo tee /etc/systemd/system/mastergroup-api.service > /dev/null << 'EOF'
[Unit]
Description=MasterGroup Recommendation API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/mastergroup-api
EnvironmentFile=/opt/mastergroup-api/.env
ExecStart=/opt/mastergroup-api/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Setup Nginx reverse proxy
echo "Step 8: Setting up Nginx..."
sudo tee /etc/nginx/sites-available/mastergroup-api > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/mastergroup-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Enable and start services
echo "Step 9: Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable mastergroup-api
sudo systemctl start mastergroup-api
sudo systemctl restart nginx

# Wait and check
sleep 5
echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "API Status:"
sudo systemctl status mastergroup-api --no-pager | head -10
echo ""
echo "Test the API:"
echo "  curl http://localhost:8001/health"
echo "  curl http://44.201.11.243/health"
echo ""
