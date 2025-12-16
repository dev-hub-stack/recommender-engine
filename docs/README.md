# 📚 MasterGroup Recommendation Engine Documentation

> **Version:** 2.0 (Local ML)  
> **Last Updated:** December 16, 2025  
> **Status:** ✅ Production

## Documentation Index

### Core Documentation

| Document | Description |
|----------|-------------|
| [BACKEND_SETUP_GUIDE.md](./BACKEND_SETUP_GUIDE.md) | Complete backend setup from scratch |
| [EC2_DEPLOYMENT.md](./EC2_DEPLOYMENT.md) | EC2 instance setup and configuration |
| [LOCAL_ML_PIPELINE.md](./LOCAL_ML_PIPELINE.md) | ML pipeline architecture overview |
| [LOCAL_ML_SETUP.md](./LOCAL_ML_SETUP.md) | Local ML quick start guide |

### ML & Models

| Document | Description |
|----------|-------------|
| [ML_MODELS_DOCUMENTATION.md](./ML_MODELS_DOCUMENTATION.md) | **Detailed model documentation with accuracy metrics** |
| [DATA_SYNC_AND_TRAINING.md](./DATA_SYNC_AND_TRAINING.md) | **Data fetching from Master Group & daily training** |

### Integration

| Document | Description |
|----------|-------------|
| [SHOPIFY_INTEGRATION.md](./SHOPIFY_INTEGRATION.md) | Shopify API endpoints and configuration |
| [SHOPIFY_TESTING_GUIDE.md](./SHOPIFY_TESTING_GUIDE.md) | **Step-by-step Shopify testing guide** |

### AWS & Infrastructure

| Document | Description |
|----------|-------------|
| [AWS_PERSONALIZE_SHUTDOWN.md](./AWS_PERSONALIZE_SHUTDOWN.md) | AWS Personalize shutdown documentation |

---

## Quick Links

### Production API
- **EC2 API:** http://3.209.80.206:8001
- **Health Check:** http://3.209.80.206:8001/health
- **API Docs:** http://3.209.80.206:8001/docs

### SSH Access
```bash
ssh -i mastergroup-ec2-key.pem ubuntu@3.209.80.206
```

### Run ML Pipeline
```bash
cd /opt/mastergroup-ml && source venv/bin/activate && python scripts/local_ml_pipeline.py
```

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Data Sources:                                                │
│  ├── Master Group POS API → Retail purchases                │
│  └── Master Group OE API  → Enterprise orders                │
│                                                               │
│  Storage:                                                     │
│  ├── PostgreSQL (AWS Lightsail DB) → Raw order data          │
│  ├── Redis (EC2) → Recommendation cache                      │
│  └── File System → Trained models                            │
│                                                               │
│  ML Models:                                                   │
│  ├── SVD (Collaborative Filtering) → Personalized recs      │
│  ├── Item Similarity (Cosine) → Similar products             │
│  └── Popularity → Trending items fallback                    │
│                                                               │
│  Consumers:                                                   │
│  ├── Analytics Dashboard → mastergroup-analytics-dashboard  │
│  └── Shopify Store → masterverse-project.myshopify.com       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Statistics

| Metric | Value |
|--------|-------|
| **Users with recommendations** | 79,623 |
| **Products with similarities** | 6,906 |
| **Total interactions** | 1,971,527 |
| **Training time** | ~3 minutes |
| **API latency** | 10-50ms |
| **Monthly cost** | ~$30 (EC2) |
| **Cost savings** | $170/month (vs AWS Personalize) |

---

## Contact

For questions about this system, refer to the documentation or check the source code at:
- GitHub: https://github.com/dev-hub-stack/recommender-engine
