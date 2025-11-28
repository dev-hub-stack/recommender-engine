#!/bin/bash
# Lightsail Instance Launch Script for MasterGroup API

# Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python 3.11 and dependencies
sudo apt-get install -y python3.11 python3.11-venv python3-pip git libpq-dev nginx

# Create app directory
sudo mkdir -p /opt/mastergroup-api
sudo chown ubuntu:ubuntu /opt/mastergroup-api
cd /opt/mastergroup-api

# Clone repository (or we'll copy files)
# git clone https://github.com/dev-hub-stack/recommender-engine.git .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Environment variables
cat > /opt/mastergroup-api/.env << 'EOF'
PG_HOST=ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com
PG_PORT=5432
PG_DB=mastergroup_recommendations
PG_USER=postgres
PG_PASSWORD=MasterGroup2024Secure!
AWS_REGION=us-east-1
PERSONALIZE_CAMPAIGN_ARN=arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign
PERSONALIZE_TRACKING_ID=6b8748e4-4cbe-412e-8247-b6978d2814ac
EOF

# Create systemd service
sudo cat > /etc/systemd/system/mastergroup-api.service << 'EOF'
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

# Enable service
sudo systemctl daemon-reload
sudo systemctl enable mastergroup-api

# Setup Nginx as reverse proxy
sudo cat > /etc/nginx/sites-available/mastergroup-api << 'EOF'
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8001/health;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/mastergroup-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

echo "Setup complete! Upload your code to /opt/mastergroup-api and run: sudo systemctl start mastergroup-api"
