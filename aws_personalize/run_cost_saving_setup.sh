#!/bin/bash
#
# AWS Personalize Cost-Saving Setup - All-in-One Script
# 
# This script automates the entire setup process for the cost-saving
# batch inference workflow.
#

set -e  # Exit on error

echo "======================================================================"
echo "  AWS PERSONALIZE COST-SAVING SETUP"
echo "======================================================================"
echo ""
echo "This will set up the batch inference workflow to reduce AWS costs"
echo "from ~\$432/month to ~\$7.50/month (98% reduction!)"
echo ""
echo "Steps:"
echo "  1. Create database tables"
echo "  2. Generate batch input files (from PostgreSQL → S3)"
echo "  3. Start AWS Personalize training & batch jobs"
echo ""
echo "⏱️  Total time: ~4-6 hours (mostly waiting for AWS)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Check for required environment variables
echo ""
echo "Checking environment variables..."
if [ -z "$PG_HOST" ] || [ -z "$PG_DATABASE" ]; then
    echo "⚠️  PostgreSQL environment variables not set!"
    echo "Please set: PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD"
    exit 1
fi

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "⚠️  AWS credentials not set!"
    echo "Please set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION"
    exit 1
fi

echo "✅ Environment OK"
echo ""

# Navigate to aws_personalize directory
cd "$(dirname "$0")"

# Step 1: Create database tables
echo "======================================================================"
echo "STEP 1: Creating Database Tables"
echo "======================================================================"
python setup_offline_tables.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to create database tables"
    exit 1
fi
echo ""

# Step 2: Generate batch input files
echo "======================================================================"
echo "STEP 2: Generating Batch Input Files"
echo "======================================================================"
python generate_batch_inputs.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to generate batch inputs"
    exit 1
fi
echo ""

# Step 3: Train models and run batch inference
echo "======================================================================"
echo "STEP 3: Training Models & Running Batch Inference"
echo "======================================================================"
echo ""
echo "⚠️  This step will take ~2-4 hours (AWS will run batch jobs asynchronously)"
echo ""
read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python train_hybrid_model.py
    
    echo ""
    echo "======================================================================"
    echo "✅ SETUP COMPLETE!"
    echo "======================================================================"
    echo ""
    echo "📊 Next Steps:"
    echo "  1. Wait for batch jobs to complete (~2-4 hours)"
    echo "  2. Check AWS Console for batch job status"
    echo "  3. Once complete, run: python load_batch_results.py"
    echo ""
    echo "🔔 Set a reminder to check back in 4 hours!"
    echo ""
else
    echo ""
    echo "Training skipped. Run manually later:"
    echo "  python train_hybrid_model.py"
    echo ""
fi

echo "======================================================================"
echo "📚 For more details, see: COST_SAVING_GUIDE.md"
echo "======================================================================"
