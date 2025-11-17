#!/bin/bash
# Recommendation Engine Service - Quick Deployment Script

set -e  # Exit on any error

echo "🚀 Recommendation Engine Service - Quick Deployment"
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if we're in the right directory
if [ ! -f "setup_recommendation_tables.sql" ]; then
    echo "❌ Error: setup_recommendation_tables.sql not found. Are you in the recommendation-engine-service directory?"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    if [ -f "../.env" ]; then
        echo "📋 Using existing .env from parent directory..."
        ln -sf "../.env" ".env"
    elif [ -f ".env.example" ]; then
        echo "📋 Copying .env.example to .env..."
        cp .env.example .env
        echo "⚠️  Please edit .env with your actual database configuration"
    else
        echo "❌ Error: No .env file found and no .env.example template"
        exit 1
    fi
fi

# Verify database setup
echo "🔍 Verifying database setup..."
if python3 verify_database_setup.py; then
    echo "✅ Database verification successful!"
else
    echo "❌ Database verification failed. Please check your configuration."
    exit 1
fi

# Test service imports
echo "🧪 Testing service imports..."
if python3 -c "
import sys
sys.path.append('src')
try:
    from recommendation_service_complete import RecommendationEngine
    print('✓ Service imports successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"; then
    echo "✅ Service ready!"
else
    echo "❌ Service import test failed"
    exit 1
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "Next steps:"
echo "1. Review and update .env file with your settings"
echo "2. Start the recommendation service"
echo "3. Monitor logs for any issues"
echo ""
echo "To start the service:"
echo "  source venv/bin/activate"
echo "  python3 src/recommendation_service_complete.py"
echo ""
