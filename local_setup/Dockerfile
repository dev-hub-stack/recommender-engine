FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY aws_personalize/ ./aws_personalize/
COPY custom_ml/ ./custom_ml/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Run the API
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
