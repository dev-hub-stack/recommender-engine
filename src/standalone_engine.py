#!/usr/bin/env python3
"""
Standalone Recommendation Engine - Simplified version for testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
from datetime import datetime
from typing import List, Optional

app = FastAPI(
    title="Master Group Recommendation Engine - Standalone",
    description="Simplified Recommendation Engine for testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

startup_time = time.time()

# Sample data for testing
SAMPLE_PRODUCTS = [
    {
        "product_id": "MAT_MOLTYO",
        "name": "Molty Ortho Mattress",
        "price": 45000.0,
        "category": "Bedding",
        "company": "Master Molty"
    },
    {
        "product_id": "PIL_COMFORT",
        "name": "Comfort Pillow Set",
        "price": 3500.0,
        "category": "Bedding",
        "company": "Master Textiles"
    },
    {
        "product_id": "BED_FRAME",
        "name": "Wooden Bed Frame",
        "price": 18000.0,
        "category": "Furniture",
        "company": "Master Furniture"
    }
]

class RecommendationRequest(BaseModel):
    user_id: str
    algorithm: Optional[str] = "popularity_based"
    limit: Optional[int] = 5

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    
    return {
        "status": "healthy",
        "service": "recommendation-engine-standalone",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime,
        "models_loaded": True,
        "algorithms_available": ["popularity_based", "collaborative_filtering"]
    }

@app.get("/api/v1/products")
async def get_products():
    """Get sample products"""
    return {
        "success": True,
        "products": SAMPLE_PRODUCTS,
        "total": len(SAMPLE_PRODUCTS)
    }

@app.post("/api/v1/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get sample recommendations"""
    
    # Simulate processing time
    processing_start = time.time()
    
    # Generate sample recommendations
    recommendations = []
    for i, product in enumerate(SAMPLE_PRODUCTS[:request.limit]):
        recommendations.append({
            "product_id": product["product_id"],
            "product_name": product["name"],
            "price": product["price"],
            "category": product["category"],
            "company": product["company"],
            "confidence_score": 0.85 - (i * 0.1),
            "algorithm_used": request.algorithm,
            "reason": f"Popular product in {product['category']} category"
        })
    
    processing_time = (time.time() - processing_start) * 1000
    
    return {
        "success": True,
        "user_id": request.user_id,
        "algorithm": request.algorithm,
        "recommendations": recommendations,
        "total_recommendations": len(recommendations),
        "processing_time_ms": round(processing_time, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/cross-selling")
async def get_cross_selling(request: dict):
    """Get sample cross-selling recommendations"""
    
    cross_sell_recs = [
        {
            "product_id": "PIL_COMFORT",
            "product_name": "Comfort Pillow Set",
            "company": "Master Textiles",
            "price": 3500.0,
            "confidence_score": 0.78,
            "cross_sell_reason": "Customers who buy mattresses often purchase pillows",
            "potential_revenue": 875.0
        }
    ]
    
    return {
        "success": True,
        "user_id": request.get("user_id"),
        "cross_selling_recommendations": cross_sell_recs,
        "total_recommendations": len(cross_sell_recs),
        "estimated_additional_revenue": sum(r["potential_revenue"] for r in cross_sell_recs),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting Master Group Recommendation Engine (Standalone)")
    uvicorn.run(app, host="0.0.0.0", port=8001)