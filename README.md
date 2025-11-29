# MasterGroup Recommendation Engine

A production-ready recommendation system powered by AWS Personalize and collaborative filtering algorithms.

## Features

- **AWS Personalize Integration** - User personalization, similar items, batch inference
- **Collaborative Filtering** - Product co-purchase analysis, customer similarity
- **Real-time API** - FastAPI-based REST endpoints for recommendations
- **Analytics Dashboard** - Geographic insights, RFM segmentation, trending products
- **Automated Data Sync** - Scheduled sync from MasterGroup ERP/POS systems

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/dev-hub-stack/recommender-engine.git
cd recommender-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database and API credentials
```

### 3. Setup Database

```bash
python scripts/setup_database.py
```

This will:
- Run all database migrations
- Create required tables and indexes
- Set up stored procedures
- Seed the admin user

### 4. Start the Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001
```

### 5. Access Dashboard

- **URL:** http://localhost:8001
- **Email:** admin@mastergroup.com
- **Password:** MG@2024#Secure!Pass

⚠️ Change the password after first login!

## Project Structure

```
recommendation-engine-service/
├── src/
│   └── main.py              # FastAPI application
├── services/
│   ├── sync_service.py      # Data sync from MasterGroup API
│   └── scheduler.py         # Background job scheduler
├── aws_personalize/
│   ├── personalize_service.py
│   ├── run_batch_inference.py
│   └── load_batch_results.py
├── migrations/
│   └── versions/            # Alembic migrations
├── scripts/
│   └── setup_database.py    # Database setup script
├── config/
│   └── master_group_api.py  # Configuration
└── tests/
```

## API Endpoints

### Recommendations

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/personalize/recommendations/{user_id}` | User recommendations |
| `GET /api/v1/personalize/recommendations/similar/{product_id}` | Similar products |
| `GET /api/v1/personalize/recommendations/by-location` | Location-based recommendations |

### Analytics

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/analytics/collaborative-products` | Top collaborative products |
| `GET /api/v1/analytics/collaborative-pairs` | Product pair analysis |
| `GET /api/v1/analytics/customer-similarity` | Similar customers |

See [API_COLLECTION.md](API_COLLECTION.md) for complete API documentation.

## Database Migrations

The project uses Alembic for database migrations:

```bash
# Apply all migrations
alembic upgrade head

# Create new migration
alembic revision -m "description"

# Rollback
alembic downgrade -1
```

See [DATABASE_SETUP.md](DATABASE_SETUP.md) for complete database documentation.

## AWS Personalize

The system uses AWS Personalize for ML-powered recommendations:

- **User Personalization** - Personalized product recommendations
- **Similar Items** - Product-to-product recommendations
- **Batch Inference** - Daily batch processing (cost-effective)

See [aws_personalize/README.md](aws_personalize/README.md) for setup guide.

## CI/CD

### Backend (This repo)
- Push to `dev` branch triggers auto-deploy to Lightsail
- GitHub Actions workflow in `.github/workflows/deploy.yml`

### Frontend (Dashboard)
- Main branch auto-deploys to Netlify

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete system architecture.

## Development

```bash
# Run with auto-reload
uvicorn src.main:app --reload --port 8001

# Run tests
pytest

# Check code style
flake8 src/
```

## License

Proprietary - MasterGroup © 2024
