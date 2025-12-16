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
PYTHON_PATH="${PYTHON_PATH:-python3}"

# Cron job definitions
DAILY_JOB="0 2 * * * cd $PROJECT_DIR && $PYTHON_PATH scripts/local_ml_pipeline.py --sync-days 7 >> logs/pipeline.log 2>&1"
WEEKLY_JOB="0 3 * * 0 cd $PROJECT_DIR && $PYTHON_PATH scripts/local_ml_pipeline.py --sync-days 365 >> logs/pipeline_weekly.log 2>&1"

install_cron() {
    echo "Installing cron jobs..."
    
    # Create logs directory
    mkdir -p "$PROJECT_DIR/logs"
    
    # Get current crontab
    crontab -l 2>/dev/null > /tmp/current_cron
    
    # Add our jobs if not already present
    if ! grep -q "local_ml_pipeline.py" /tmp/current_cron; then
        echo "# MasterGroup Local ML Pipeline - Daily (2 AM)" >> /tmp/current_cron
        echo "$DAILY_JOB" >> /tmp/current_cron
        echo "# MasterGroup Local ML Pipeline - Weekly Full (Sunday 3 AM)" >> /tmp/current_cron
        echo "$WEEKLY_JOB" >> /tmp/current_cron
        
        crontab /tmp/current_cron
        echo "✅ Cron jobs installed!"
        echo ""
        echo "Daily job: 2:00 AM every day (sync last 7 days)"
        echo "Weekly job: 3:00 AM every Sunday (full retrain)"
    else
        echo "⚠️ Cron jobs already installed"
    fi
    
    rm /tmp/current_cron
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
