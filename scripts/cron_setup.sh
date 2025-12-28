#!/bin/bash
# =============================================================================
# LOCAL ML PIPELINE - CRON SETUP
# =============================================================================
# 
# This script sets up automated daily/weekly training jobs.
# 
# Usage:
#   ./scripts/cron_setup.sh install   # Install cron jobs
#   ./scripts/cron_setup.sh remove    # Remove cron jobs
#   ./scripts/cron_setup.sh status    # Show current cron jobs
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_PATH="${PYTHON_PATH:-/opt/mastergroup-ml/venv/bin/python}"

# Cron job definitions - using absolute paths to avoid shell issues
DAILY_JOB="0 2 * * * cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python scripts/local_ml_pipeline.py --sync-days 7 >> /opt/mastergroup-ml/logs/ml_cron.log 2>&1"
WEEKLY_JOB="0 3 * * 0 cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python scripts/local_ml_pipeline.py --sync-days 365 >> /opt/mastergroup-ml/logs/ml_cron_weekly.log 2>&1"

install_cron() {
    echo "Installing cron jobs..."
    
    # Create logs directory
    mkdir -p /opt/mastergroup-ml/logs
    
    # Create new crontab with proper shell settings
    cat > /tmp/new_cron << 'CRONTAB'
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin:/opt/mastergroup-ml/venv/bin

# MasterGroup Local ML Pipeline - Daily (2 AM UTC / 7 AM PKT)
0 2 * * * cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python scripts/local_ml_pipeline.py --sync-days 7 >> /opt/mastergroup-ml/logs/ml_cron.log 2>&1

# MasterGroup Local ML Pipeline - Weekly Full (Sunday 3 AM UTC)
0 3 * * 0 cd /opt/mastergroup-ml && /opt/mastergroup-ml/venv/bin/python scripts/local_ml_pipeline.py --sync-days 365 >> /opt/mastergroup-ml/logs/ml_cron_weekly.log 2>&1
CRONTAB
    
    # Get existing crontab, remove old ML pipeline entries, and add new ones
    crontab -l 2>/dev/null | grep -v "local_ml_pipeline\|MasterGroup Local ML Pipeline\|SHELL=/bin/bash\|PATH=.*mastergroup" > /tmp/other_cron 2>/dev/null || true
    cat /tmp/other_cron /tmp/new_cron | crontab -
    
    rm -f /tmp/current_cron /tmp/new_cron /tmp/other_cron
    
    echo "✅ Cron jobs installed!"
    echo ""
    echo "Daily job: 2:00 AM every day (sync last 7 days)"
    echo "Weekly job: 3:00 AM every Sunday (full retrain)"
}

remove_cron() {
    echo "Removing cron jobs..."
    
    crontab -l 2>/dev/null | grep -v "local_ml_pipeline.py" | grep -v "MasterGroup Local ML Pipeline" > /tmp/current_cron
    crontab /tmp/current_cron
    rm /tmp/current_cron
    
    echo "✅ Cron jobs removed"
}

status_cron() {
    echo "Current cron jobs:"
    echo "=================="
    crontab -l 2>/dev/null | grep -A1 "local_ml_pipeline" || echo "No local_ml_pipeline jobs found"
}

case "$1" in
    install)
        install_cron
        ;;
    remove)
        remove_cron
        ;;
    status)
        status_cron
        ;;
    *)
        echo "Usage: $0 {install|remove|status}"
        exit 1
        ;;
esac
