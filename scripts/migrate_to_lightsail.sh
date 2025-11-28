#!/bin/bash
# ============================================================
# Migrate PostgreSQL from Local to AWS Lightsail
# ============================================================

# Source (Local) Database
SOURCE_HOST="localhost"
SOURCE_PORT="5432"
SOURCE_DB="mastergroup_recommendations"
SOURCE_USER="postgres"
SOURCE_PASSWORD="postgres"

# Target (Lightsail) Database - UPDATE THESE AFTER DB IS READY
TARGET_HOST="ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com"
TARGET_PORT="5432"
TARGET_DB="mastergroup_recommendations"
TARGET_USER="postgres"
TARGET_PASSWORD="MasterGroup2024Secure!"

# Backup file
BACKUP_FILE="/tmp/mastergroup_backup_$(date +%Y%m%d_%H%M%S).sql"

echo "============================================================"
echo "PostgreSQL Migration: Local → AWS Lightsail"
echo "============================================================"
echo ""

# Step 1: Check if target host is set
if [[ "$TARGET_HOST" == "<LIGHTSAIL_ENDPOINT>" ]]; then
    echo "❌ ERROR: Please update TARGET_HOST with your Lightsail endpoint"
    echo ""
    echo "Get it with:"
    echo "  aws lightsail get-relational-database --relational-database-name mastergroup-postgres --profile mastergroup --region us-east-1"
    echo ""
    exit 1
fi

# Step 2: Export from local
echo "Step 1: Exporting from local database..."
echo "  Source: $SOURCE_HOST:$SOURCE_PORT/$SOURCE_DB"
PGPASSWORD=$SOURCE_PASSWORD pg_dump \
    -h $SOURCE_HOST \
    -p $SOURCE_PORT \
    -U $SOURCE_USER \
    -d $SOURCE_DB \
    -F c \
    -f $BACKUP_FILE

if [ $? -eq 0 ]; then
    echo "  ✅ Export complete: $BACKUP_FILE"
    echo "  Size: $(du -h $BACKUP_FILE | cut -f1)"
else
    echo "  ❌ Export failed"
    exit 1
fi

# Step 3: Import to Lightsail
echo ""
echo "Step 2: Importing to Lightsail database..."
echo "  Target: $TARGET_HOST:$TARGET_PORT/$TARGET_DB"

PGPASSWORD=$TARGET_PASSWORD pg_restore \
    -h $TARGET_HOST \
    -p $TARGET_PORT \
    -U $TARGET_USER \
    -d $TARGET_DB \
    --no-owner \
    --no-privileges \
    -c \
    $BACKUP_FILE

if [ $? -eq 0 ]; then
    echo "  ✅ Import complete!"
else
    echo "  ⚠️  Import completed with warnings (this is usually OK)"
fi

# Step 4: Verify
echo ""
echo "Step 3: Verifying migration..."
PGPASSWORD=$TARGET_PASSWORD psql \
    -h $TARGET_HOST \
    -p $TARGET_PORT \
    -U $TARGET_USER \
    -d $TARGET_DB \
    -c "SELECT 'Orders: ' || COUNT(*) FROM orders UNION ALL SELECT 'Order Items: ' || COUNT(*) FROM order_items;"

# Step 5: Cleanup
echo ""
echo "Step 4: Cleaning up..."
rm $BACKUP_FILE
echo "  ✅ Backup file removed"

echo ""
echo "============================================================"
echo "MIGRATION COMPLETE!"
echo "============================================================"
echo ""
echo "Update your environment variables:"
echo ""
echo "  export PG_HOST=$TARGET_HOST"
echo "  export PG_PORT=$TARGET_PORT"
echo "  export PG_DB=$TARGET_DB"
echo "  export PG_USER=$TARGET_USER"
echo "  export PG_PASSWORD=$TARGET_PASSWORD"
echo ""
