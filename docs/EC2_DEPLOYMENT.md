# 🚀 EC2 Deployment Guide

> **Last Updated:** December 16, 2025  
> **Status:** ✅ ACTIVE  
> **Instance IP:** 3.209.80.206

## Instance Details

| Setting | Value |
|---------|-------|
| **Instance ID** | i-026fdc18662d467dc |
| **Instance Type** | t3.medium |
| **RAM** | 4 GB |
| **vCPU** | 2 |
| **Storage** | 40 GB gp3 |
| **Region** | us-east-1 |
| **OS** | Ubuntu 22.04 LTS |
| **Cost** | ~$30/month (on-demand) |

## Access

```bash
# SSH Key Location (local)
/Users/clustox_1/Documents/MasterGroup-RecommendationSystem/recommendation-engine-service/mastergroup-ec2-key.pem

# SSH Command
ssh -i mastergroup-ec2-key.pem ubuntu@3.209.80.206
```

## Installed Components

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.11 | ✅ |
| Redis | 6.0.16 | ✅ Running |
| PostgreSQL Client | 14 | ✅ |
| Git | Latest | ✅ |

## Directory Structure

```
/opt/mastergroup-ml/
├── venv/                    # Python virtual environment
├── src/                     # API source code
│   ├── main.py             # FastAPI application
│   └── services/           # Business logic
├── scripts/
│   └── local_ml_pipeline.py # ML training pipeline
├── models/                  # Trained ML models
├── data/                    # Training data exports
└── .env                     # Environment configuration
```

## Services

### API Service
```bash
# Start API
cd /opt/mastergroup-ml
source venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8001 --workers 2

# Check status
curl http://3.209.80.206:8001/health
```

### Redis
```bash
sudo systemctl status redis-server
redis-cli ping  # Should return PONG
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/shopify/recommendations` | POST | Personalized recommendations |
| `/api/v1/shopify/similar/{product_id}` | GET | Similar products |
| `/api/v1/shopify/popular` | GET | Popular products by location |
| `/api/v1/recommendations/popular` | GET | Dashboard popular products |
| `/api/v1/analytics/product-categories` | GET | Product categories |

## ML Pipeline Execution

```bash
# Run ML Pipeline (takes ~3 minutes on t3.medium)
cd /opt/mastergroup-ml
source venv/bin/activate
python scripts/local_ml_pipeline.py
```

### Pipeline Output
- **Users with recommendations:** 79,623
- **Products with similar items:** 6,906
- **Execution time:** ~3 minutes

## Security Group

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | 0.0.0.0/0 | SSH |
| 80 | TCP | 0.0.0.0/0 | HTTP |
| 8001 | TCP | 0.0.0.0/0 | API |

## Logs

```bash
# API logs
tail -f /tmp/api.log

# ML Pipeline logs
tail -f /tmp/ml_pipeline.log
```

## Maintenance Commands

```bash
# Update code from GitHub
cd /opt/mastergroup-ml
git pull origin dev

# Restart API
pkill -f uvicorn
source venv/bin/activate
nohup uvicorn src.main:app --host 0.0.0.0 --port 8001 --workers 2 > /tmp/api.log 2>&1 &

# Re-train models
python scripts/local_ml_pipeline.py
```

## Migration from Lightsail

The EC2 instance was created to replace the Lightsail instance (44.201.11.243) due to:
1. **Memory limitations** - Lightsail micro only has 1GB RAM
2. **Account restrictions** - Could not upgrade Lightsail plan size
3. **ML Training** - Needed 4GB+ RAM for training 79k users

### Migration Checklist
- [x] EC2 instance created
- [x] Security group configured
- [x] Python environment set up
- [x] Code deployed from GitHub
- [x] .env copied from Lightsail
- [x] Redis installed
- [x] ML pipeline executed
- [x] API running and tested
- [ ] Systemd service for auto-start
- [ ] Update GitHub Actions deploy.yml
- [ ] Terminate Lightsail instance
