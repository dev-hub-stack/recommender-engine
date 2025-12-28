#!/bin/bash
# Script to clear Redis cache and restart the API service on EC2
# This will fix the Punjab/PUNJAB duplication issue by clearing cached data

echo "=================================================="
echo "🔧 MasterGroup ML - Cache Clear & Service Restart"
echo "=================================================="

# Change to application directory
cd /opt/mastergroup-ml || exit 1

# Check Redis status
echo -e "\n📊 Checking Redis Status..."
sudo systemctl status redis-server --no-pager | head -10

# Flush all Redis cache
echo -e "\n🗑️  Flushing Redis Cache..."
redis-cli FLUSHALL
if [ $? -eq 0 ]; then
    echo "✅ Redis cache cleared successfully!"
else
    echo "❌ Failed to clear Redis cache"
    exit 1
fi

# Restart the API service
echo -e "\n🔄 Restarting MasterGroup ML API Service..."
sudo systemctl restart mastergroup-ml
sleep 3

# Check service status
echo -e "\n📈 Service Status:"
sudo systemctl status mastergroup-ml --no-pager | head -15

# Verify the service is running
if sudo systemctl is-active --quiet mastergroup-ml; then
    echo -e "\n✅ Service restarted successfully!"
    echo -e "\n🎯 FIXES APPLIED:"
    echo "   • Geographic API error fixed (set_cache_data → set_to_cache)"
    echo "   • Province normalization fixed (Punjab/PUNJAB merged)"
    echo "   • Cache cleared - fresh data will be loaded"
else
    echo -e "\n❌ Service failed to start. Check logs:"
    sudo journalctl -u mastergroup-ml -n 50 --no-pager
    exit 1
fi

echo -e "\n=================================================="
echo "✅ Cache cleared and service restarted!"
echo "=================================================="
