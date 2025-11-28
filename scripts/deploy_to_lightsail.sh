#!/bin/bash
# ============================================================
# Deploy Backend to AWS Lightsail Container Service
# ============================================================

set -e

SERVICE_NAME="mastergroup-api"
CONTAINER_NAME="api"
AWS_PROFILE="mastergroup"
AWS_REGION="us-east-1"

echo "============================================================"
echo "Deploying to AWS Lightsail Container Service"
echo "============================================================"
echo ""

# Step 1: Build Docker image
echo "Step 1: Building Docker image..."
docker build -t mastergroup-api:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi
echo "✅ Docker image built"

# Step 2: Check container service status
echo ""
echo "Step 2: Checking container service status..."
STATUS=$(aws lightsail get-container-services \
    --service-name $SERVICE_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --query 'containerServices[0].state' \
    --output text)

echo "  Container service status: $STATUS"

if [ "$STATUS" != "READY" ] && [ "$STATUS" != "RUNNING" ]; then
    echo "⏳ Waiting for container service to be ready..."
    while [ "$STATUS" != "READY" ] && [ "$STATUS" != "RUNNING" ]; do
        sleep 30
        STATUS=$(aws lightsail get-container-services \
            --service-name $SERVICE_NAME \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --query 'containerServices[0].state' \
            --output text)
        echo "  Status: $STATUS"
    done
fi

# Step 3: Push image to Lightsail
echo ""
echo "Step 3: Pushing image to Lightsail..."
aws lightsail push-container-image \
    --service-name $SERVICE_NAME \
    --label $CONTAINER_NAME \
    --image mastergroup-api:latest \
    --profile $AWS_PROFILE \
    --region $AWS_REGION

if [ $? -ne 0 ]; then
    echo "❌ Image push failed"
    exit 1
fi
echo "✅ Image pushed to Lightsail"

# Step 4: Get the image name
echo ""
echo "Step 4: Getting pushed image reference..."
IMAGE=$(aws lightsail get-container-images \
    --service-name $SERVICE_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION \
    --query 'containerImages[0].image' \
    --output text)

echo "  Image: $IMAGE"

# Step 5: Update deployment config with image
echo ""
echo "Step 5: Creating deployment..."

# Create deployment JSON with actual image
cat > /tmp/deployment.json << EOF
{
    "containers": {
        "api": {
            "image": "$IMAGE",
            "ports": {
                "8001": "HTTP"
            },
            "environment": {
                "PG_HOST": "ls-49a54a36b814758103dcc97a4c41b7f8bd563888.cijig8im8oxl.us-east-1.rds.amazonaws.com",
                "PG_PORT": "5432",
                "PG_DB": "mastergroup_recommendations",
                "PG_USER": "postgres",
                "PG_PASSWORD": "MasterGroup2024Secure!",
                "REDIS_HOST": "localhost",
                "REDIS_PORT": "6379",
                "AWS_REGION": "us-east-1",
                "PERSONALIZE_CAMPAIGN_ARN": "arn:aws:personalize:us-east-1:657020414783:campaign/mastergroup-campaign",
                "PERSONALIZE_TRACKING_ID": "6b8748e4-4cbe-412e-8247-b6978d2814ac"
            }
        }
    },
    "publicEndpoint": {
        "containerName": "api",
        "containerPort": 8001,
        "healthCheck": {
            "path": "/health",
            "intervalSeconds": 30,
            "timeoutSeconds": 5,
            "successCodes": "200",
            "healthyThreshold": 2,
            "unhealthyThreshold": 3
        }
    }
}
EOF

aws lightsail create-container-service-deployment \
    --service-name $SERVICE_NAME \
    --containers file:///tmp/deployment.json \
    --public-endpoint file:///tmp/deployment.json \
    --profile $AWS_PROFILE \
    --region $AWS_REGION

echo ""
echo "============================================================"
echo "DEPLOYMENT STARTED!"
echo "============================================================"
echo ""
echo "Check status:"
echo "  aws lightsail get-container-services --service-name $SERVICE_NAME --profile $AWS_PROFILE --region $AWS_REGION"
echo ""
echo "Your API will be available at:"
echo "  https://$SERVICE_NAME.jq1azhq0wwj68.us-east-1.cs.amazonlightsail.com/"
echo ""
