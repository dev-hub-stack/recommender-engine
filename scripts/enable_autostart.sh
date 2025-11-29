#!/bin/bash
# ============================================================
# Enable Auto-Start for MasterGroup API
# This script creates a systemd service to auto-start the API on boot.
# Run this on the server: bash scripts/enable_autostart.sh
# ============================================================

set -e

# Detect current directory (must be run from project root or scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if main.py exists to verify root
if [ ! -f "$PROJECT_ROOT/src/main.py" ]; then
    echo "❌ Error: Could not find src/main.py in $PROJECT_ROOT"
    echo "Please run this script from the project root or scripts directory."
    exit 1
fi

SERVICE_NAME="mastergroup-api"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
USER=$(whoami)
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"
UVICORN="$PROJECT_ROOT/venv/bin/uvicorn"

echo "🚀 Setting up ${SERVICE_NAME} service..."
echo "📂 Project Root: $PROJECT_ROOT"
echo "👤 User: $USER"

# Check if venv exists
if [ ! -f "$UVICORN" ]; then
    echo "❌ Error: Virtual environment not found at $PROJECT_ROOT/venv"
    echo "Please run 'python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt' first."
    exit 1
fi

# Generate service file
echo "📝 Generating service file..."
sudo tee ${SERVICE_FILE} > /dev/null <<EOF
[Unit]
Description=MasterGroup Recommendation API
After=network.target postgresql.service redis-server.service

[Service]
Type=simple
User=${USER}
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=${PROJECT_ROOT}/.env
ExecStart=${UVICORN} src.main:app --host 0.0.0.0 --port 8001 --workers 4
Restart=always
RestartSec=5
StandardOutput=append:/var/log/${SERVICE_NAME}.log
StandardError=append:/var/log/${SERVICE_NAME}.err

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Created service file at ${SERVICE_FILE}"

# Create log files with permissions
echo "📝 Setting up logs..."
sudo touch /var/log/${SERVICE_NAME}.log /var/log/${SERVICE_NAME}.err
sudo chown ${USER}:${USER} /var/log/${SERVICE_NAME}.log /var/log/${SERVICE_NAME}.err

# Reload and enable
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}

echo "✅ Service enabled and started!"
echo "📊 Checking status..."
sudo systemctl status ${SERVICE_NAME} --no-pager | head -n 20

echo ""
echo "🎉 Auto-start setup complete!"
echo "The API will now start automatically when the system reboots."
echo "Logs are available at:"
echo "  - /var/log/${SERVICE_NAME}.log"
echo "  - /var/log/${SERVICE_NAME}.err"
