"""
Popularity-based Recommendation Engine
Implements demographic and location-based popularity algorithms for new users
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import structlog

import sys
import os
# Removed shared dependency for local execution

try:
    from models.recommendation import (
        Recommendation, RecommendationAlgorithm, RecommendationContext, 
        RecommendationMetadata, RecommendationRequest, RecommendationResponse
    )
    from models.customer import CustomerSegment
except ModuleNotFoundError:
    from src.models.recommendation import (
        Recommendation, RecommendationAlgorithm, RecommendationContext, 
        RecommendationMetadata, RecommendationRequest, RecommendationResponse
    )
    from src.models.customer import CustomerSegment

logger = structlog.get_logger()


class CustomerSegmentationEngine:
    """Handles customer segmentation for 300k-500k PKR income bracket targeting"""
    
    def __init__(self):
        self.segments = {}
        self.segment_profiles = {}
        
    def create_income_segments(self) -> Dict[str, CustomerSegment]:
        """Create predefined customer segments based on income brackets"""
        segments = {
            'premium_karachi': CustomerSegment(
                segment_id='premium_karachi',
                segment_name='Premium Karachi Customers',
                characteristics={
                    'income_range': '300k-500k PKR',
                    'city': 'Karachi',
                    'lifestyle': 'urban_professional',
                    'purchase_behavior': 'quality_focused'
                },
                target_income_range=(300000, 500000),
                geographic_focus=['Karachi']
            ),
            'premium_lahore': CustomerSegment(
                segment_id='premium_lahore',
                segment_name='Premium Lahore Customers',
                characteristics={
                    'income_range': '300k-500k PKR',
                    'city': 'Lahore',
                    'lifestyle': 'urban_professional',
                    'purchase_behavior': 'quality_focused'
                },
                target_income_range=(300000, 500000),
                geographic_focus=['Lahore']
            ),
            'premium_islamabad': CustomerSegment(
                segment_id='premium_islamabad',
                segment_name='Premium Islamabad Customers',
                characteristics={
                    'income_range': '300k-500k PKR',
                    'city': 'Islamabad',
                    'lifestyle': 'urban_professional',
                    'purchase_behavior': 'quality_focused'
                },
                target_income_range=(300000, 500000),
                geographic_focus=['Islamabad']
            ),
            'general_metro': CustomerSegment(
                segment_id='general_metro',
                segment_name='General Metropolitan Customers',
                characteristics={
                    'income_range': 'mixed',
                    'city': 'metropolitan',
                    'lifestyle': 'mixed',
                    'purchase_behavior': 'value_conscious'
                },
                target_income_range=(200000, 600000),
                geographic_focus=['Karachi', 'Lahore', 'Islamabad']
            )
        }
        
        self.segments = segments
        return segments
    
    def assign_customer_segment(self, customer_data: Dict) -> str:
        """
        Assign customer to appropriate segment based on demographics
        
        Args:
            customer_data: Dictionary with customer info (city, income, etc.)
            
        Returns:
            Segment ID
        """
        city = customer_data.get('city', '').lower()
        income = customer_data.get('income', 0)
        
        # Check if customer fits premium income bracket
        if 300000 <= income <= 500000:
            if 'karachi' in city:
                return 'premium_karachi'
            elif 'lahore' in city:
                return 'premium_lahore'
            elif 'islamabad' in city:
                return 'premium_islamabad'
        
        # Default to general metropolitan
        return 'general_metro'


class TrendingProductAnalyzer:
    """Analyzes product trends based on sales velocity and category performance"""
    
    def __init__(self, trend_window_days: int = 30):
        self.trend_window_days = trend_window_days
        self.trending_products = {}
        self.category_trends = {}
        
    def calculate_sales_velocity(self, sales_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate sales velocity for products
        
        Args:
            sales_data: DataFrame with columns ['product_id', 'sale_date', 'quantity', 'amount']
            
        Returns:
            Dictionary mapping product_id to sales velocity (sales per day)
        """
        logger.info("Calculating sales velocity", n_sales=len(sales_data))
        
        # Filter to recent sales within trend window
        cutoff_date = datetime.now() - timedelta(days=self.trend_window_days)
        recent_sales = sales_data[sales_data['sale_date'] >= cutoff_date]
        
        # Calculate velocity for each product
        velocity_data = {}
        
        for product_id in recent_sales['product_id'].unique():
            product_sales = recent_sales[recent_sales['product_id'] == product_id]
            
            # Calculate metrics
            total_quantity = product_sales['quantity'].sum()
            total_amount = product_sales['amount'].sum()
            days_active = (product_sales['sale_date'].max() - product_sales['sale_date'].min()).days + 1
            
            # Sales velocity (quantity per day)
            velocity = total_quantity / max(days_active, 1)
            
            # Revenue velocity (amount per day)
            revenue_velocity = total_amount / max(days_active, 1)
            
            velocity_data[product_id] = {
                'sales_velocity': velocity,
                'revenue_velocity': float(revenue_velocity),
                'total_sales': int(total_quantity),
                'total_revenue': float(total_amount),
                'days_active': days_active
            }
        
        logger.info("Sales velocity calculated", n_products=len(velocity_data))
        return velocity_data
    
    def identify_trending_products(self, sales_data: pd.DataFrame, 
                                 product_data: pd.DataFrame,
                                 top_n: int = 50) -> List[Dict]:
        """
        Identify trending products based on multiple factors
        
        Args:
            sales_data: Sales transaction data
            product_data: Product information with categories
            top_n: Number of top trending products to return
            
        Returns:
            List of trending product dictionaries
        """
        logger.info("Identifying trending products", top_n=top_n)
        
        # Calculate sales velocity
        velocity_data = self.calculate_sales_velocity(sales_data)
        
        # Calculate category performance
        category_performance = self.analyze_category_trends(sales_data, product_data)
        
        # Score products based on multiple factors
        trending_scores = {}
        
        for product_id, velocity_info in velocity_data.items():
            # Get product category
            product_info = product_data[product_data['product_id'] == product_id]
            if product_info.empty:
                continue
                
            category = product_info.iloc[0]['category_id']
            
            # Calculate trending score
            base_score = velocity_info['sales_velocity']
            
            # Boost score based on category performance
            category_boost = category_performance.get(category, {}).get('growth_rate', 1.0)
            
            # Revenue factor
            revenue_factor = min(velocity_info['revenue_velocity'] / 1000, 5.0)  # Cap at 5x
            
            # Recency factor (more recent sales get higher score)
            recent_sales = sales_data[
                (sales_data['product_id'] == product_id) & 
                (sales_data['sale_date'] >= datetime.now() - timedelta(days=7))
            ]
            recency_factor = 1 + (len(recent_sales) / max(velocity_info['total_sales'], 1))
            
            trending_score = base_score * category_boost * revenue_factor * recency_factor
            
            trending_scores[product_id] = {
                'product_id': product_id,
                'trending_score': trending_score,
                'sales_velocity': velocity_info['sales_velocity'],
                'revenue_velocity': velocity_info['revenue_velocity'],
                'category': category,
                'category_growth': category_boost,
                'recency_factor': recency_factor
            }
        
        # Sort by trending score and return top N
        sorted_products = sorted(
            trending_scores.values(), 
            key=lambda x: x['trending_score'], 
            reverse=True
        )
        
        self.trending_products = {p['product_id']: p for p in sorted_products[:top_n]}
        
        logger.info("Trending products identified", 
                   n_trending=len(self.trending_products))
        
        return sorted_products[:top_n]
    
    def analyze_category_trends(self, sales_data: pd.DataFrame, 
                              product_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze category-level trends and performance
        
        Args:
            sales_data: Sales transaction data
            product_data: Product information
            
        Returns:
            Dictionary mapping category_id to trend metrics
        """
        # Merge sales with product data to get categories
        sales_with_categories = sales_data.merge(
            product_data[['product_id', 'category_id']], 
            on='product_id', 
            how='left'
        )
        
        # Calculate category metrics
        category_trends = {}
        
        for category in sales_with_categories['category_id'].unique():
            if pd.isna(category):
                continue
                
            category_sales = sales_with_categories[
                sales_with_categories['category_id'] == category
            ]
            
            # Calculate growth rate (comparing last 15 days vs previous 15 days)
            recent_cutoff = datetime.now() - timedelta(days=15)
            previous_cutoff = datetime.now() - timedelta(days=30)
            
            recent_sales = category_sales[category_sales['sale_date'] >= recent_cutoff]
            previous_sales = category_sales[
                (category_sales['sale_date'] >= previous_cutoff) & 
                (category_sales['sale_date'] < recent_cutoff)
            ]
            
            recent_revenue = recent_sales['amount'].sum()
            previous_revenue = previous_sales['amount'].sum()
            
            growth_rate = 1.0
            if previous_revenue > 0:
                growth_rate = recent_revenue / previous_revenue
            
            category_trends[category] = {
                'growth_rate': growth_rate,
                'recent_revenue': float(recent_revenue),
                'previous_revenue': float(previous_revenue),
                'total_products': category_sales['product_id'].nunique(),
                'avg_sales_velocity': category_sales.groupby('product_id')['quantity'].sum().mean()
            }
        
        self.category_trends = category_trends
        return category_trends


class PopularityBasedEngine:
    """Main popularity-based recommendation engine"""
    
    def __init__(self, min_sales_threshold: int = 5, 
                 popularity_weight: float = 0.4,
                 trend_weight: float = 0.3,
                 segment_weight: float = 0.3):
        self.min_sales_threshold = min_sales_threshold
        self.popularity_weight = popularity_weight
        self.trend_weight = trend_weight
        self.segment_weight = segment_weight
        
        self.segmentation_engine = CustomerSegmentationEngine()
        self.trend_analyzer = TrendingProductAnalyzer()
        
        self.popularity_scores = {}
        self.segment_preferences = {}
        self.is_trained = False
        self.model_version = "1.0.0"
        
    def train(self, sales_data: pd.DataFrame, product_data: pd.DataFrame, 
              customer_data: pd.DataFrame) -> Dict[str, any]:
        """
        Train popularity-based recommendation model
        
        Args:
            sales_data: Sales transaction data
            product_data: Product information
            customer_data: Customer demographic data
            
        Returns:
            Training metrics
        """
        start_time = datetime.now()
        logger.info("Training popularity-based recommendation engine")
        
        # Initialize customer segments
        segments = self.segmentation_engine.create_income_segments()
        
        # Calculate overall popularity scores
        self.popularity_scores = self._calculate_popularity_scores(sales_data, product_data)
        
        # Identify trending products
        trending_products = self.trend_analyzer.identify_trending_products(
            sales_data, product_data
        )
        
        # Calculate segment-specific preferences
        self.segment_preferences = self._calculate_segment_preferences(
            sales_data, product_data, customer_data
        )
        
        self.is_trained = True
        training_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            'training_time_seconds': training_time,
            'n_products_analyzed': len(self.popularity_scores),
            'n_trending_products': len(trending_products),
            'n_segments': len(segments),
            'avg_popularity_score': np.mean(list(self.popularity_scores.values())),
            'model_version': self.model_version
        }
        
        logger.info("Popularity-based training completed", **metrics)
        return metrics
    
    def _calculate_popularity_scores(self, sales_data: pd.DataFrame, 
                                   product_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall popularity scores for products"""
        logger.info("Calculating popularity scores")
        
        # Calculate basic popularity metrics
        product_sales = sales_data.groupby('product_id').agg({
            'quantity': 'sum',
            'amount': 'sum',
            'sale_date': ['count', 'max', 'min']
        }).reset_index()
        
        product_sales.columns = [
            'product_id', 'total_quantity', 'total_revenue', 
            'transaction_count', 'last_sale', 'first_sale'
        ]
        
        # Calculate popularity score
        popularity_scores = {}
        
        for _, row in product_sales.iterrows():
            product_id = row['product_id']
            
            # Skip products with insufficient sales
            if row['total_quantity'] < self.min_sales_threshold:
                continue
            
            # Normalize metrics
            quantity_score = min(row['total_quantity'] / 100, 10)  # Cap at 10
            revenue_score = min(row['total_revenue'] / 10000, 10)  # Cap at 10
            frequency_score = min(row['transaction_count'] / 10, 10)  # Cap at 10
            
            # Recency score (more recent = higher score)
            days_since_last_sale = (datetime.now() - row['last_sale']).days
            recency_score = max(0, 10 - (days_since_last_sale / 30))  # Decay over 30 days
            
            # Combined popularity score
            popularity_score = (
                quantity_score * 0.3 + 
                revenue_score * 0.3 + 
                frequency_score * 0.2 + 
                recency_score * 0.2
            )
            
            popularity_scores[product_id] = popularity_score
        
        logger.info("Popularity scores calculated", n_products=len(popularity_scores))
        return popularity_scores
    
    def _calculate_segment_preferences(self, sales_data: pd.DataFrame,
                                     product_data: pd.DataFrame,
                                     customer_data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate product preferences for each customer segment"""
        logger.info("Calculating segment preferences")
        
        # Merge sales with customer data
        sales_with_customers = sales_data.merge(
            customer_data[['customer_id', 'city', 'income_bracket']], 
            on='customer_id', 
            how='left'
        )
        
        segment_preferences = defaultdict(lambda: defaultdict(float))
        
        # Assign customers to segments and calculate preferences
        for _, row in sales_with_customers.iterrows():
            customer_info = {
                'city': row.get('city', ''),
                'income': self._parse_income_bracket(row.get('income_bracket', ''))
            }
            
            segment_id = self.segmentation_engine.assign_customer_segment(customer_info)
            product_id = row['product_id']
            quantity = row['quantity']
            
            # Add to segment preferences
            segment_preferences[segment_id][product_id] += quantity
        
        # Normalize preferences within each segment
        normalized_preferences = {}
        for segment_id, preferences in segment_preferences.items():
            total_purchases = sum(preferences.values())
            if total_purchases > 0:
                normalized_preferences[segment_id] = {
                    product_id: count / total_purchases 
                    for product_id, count in preferences.items()
                }
            else:
                normalized_preferences[segment_id] = {}
        
        logger.info("Segment preferences calculated", 
                   n_segments=len(normalized_preferences))
        return normalized_preferences
    
    def _parse_income_bracket(self, income_bracket: str) -> float:
        """Parse income bracket string to numeric value"""
        if not income_bracket or pd.isna(income_bracket):
            return 350000  # Default to middle of target range
        
        # Extract numbers from bracket like "300k-500k PKR"
        import re
        numbers = re.findall(r'(\d+)k?', str(income_bracket).lower())
        if len(numbers) >= 2:
            # Take average of range
            min_income = float(numbers[0]) * 1000
            max_income = float(numbers[1]) * 1000
            return (min_income + max_income) / 2
        elif len(numbers) == 1:
            return float(numbers[0]) * 1000
        
        return 350000  # Default
    
    def get_recommendations(self, request: RecommendationRequest,
                          customer_data: Optional[Dict] = None) -> RecommendationResponse:
        """
        Generate popularity-based recommendations
        
        Args:
            request: Recommendation request
            customer_data: Optional customer demographic data
            
        Returns:
            Recommendation response
        """
        start_time = datetime.now()
        
        if not self.is_trained:
            return RecommendationResponse(
                recommendations=[],
                total_count=0,
                algorithm_used=RecommendationAlgorithm.POPULARITY_BASED,
                fallback_applied=True,
                processing_time_ms=0,
                cache_hit=False
            )
        
        # Determine customer segment
        segment_id = 'general_metro'  # Default
        if customer_data:
            segment_id = self.segmentation_engine.assign_customer_segment(customer_data)
        
        # Get segment preferences
        segment_prefs = self.segment_preferences.get(segment_id, {})
        
        # Get trending products
        trending_products = self.trend_analyzer.trending_products
        
        # Calculate combined scores
        combined_scores = {}
        
        # Combine popularity, trending, and segment preferences
        all_products = set(self.popularity_scores.keys()) | set(trending_products.keys()) | set(segment_prefs.keys())
        
        for product_id in all_products:
            if product_id in request.exclude_products:
                continue
            
            # Get individual scores
            popularity_score = self.popularity_scores.get(product_id, 0)
            trending_score = trending_products.get(product_id, {}).get('trending_score', 0)
            segment_score = segment_prefs.get(product_id, 0) * 10  # Scale up segment preference
            
            # Combine scores
            combined_score = (
                popularity_score * self.popularity_weight +
                trending_score * self.trend_weight +
                segment_score * self.segment_weight
            )
            
            if combined_score > 0:
                combined_scores[product_id] = combined_score
        
        # Sort and select top recommendations
        sorted_products = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:request.num_recommendations]
        
        # Create recommendation objects
        recommendations = []
        for product_id, score in sorted_products:
            metadata = RecommendationMetadata(
                confidence_score=min(score / 10, 1.0),  # Normalize to 0-1
                explanation=f"Popular in {segment_id} segment",
                fallback_used=False,
                model_version=self.model_version,
                processing_time_ms=0,  # Will be set below
                ab_test_group=None
            )
            
            rec = Recommendation(
                user_id=request.user_id,
                product_id=product_id,
                score=score,
                algorithm=RecommendationAlgorithm.POPULARITY_BASED,
                context=request.context,
                metadata=metadata,
                timestamp=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=6)  # Shorter expiry for popularity
            )
            recommendations.append(rec)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update processing time in metadata
        for rec in recommendations:
            rec.metadata.processing_time_ms = processing_time
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            algorithm_used=RecommendationAlgorithm.POPULARITY_BASED,
            fallback_applied=False,
            processing_time_ms=processing_time,
            cache_hit=False
        )
    
    def get_segment_recommendations(self, segment_id: str, 
                                 num_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Get recommendations specifically for a customer segment
        
        Args:
            segment_id: Customer segment identifier
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of (product_id, score) tuples
        """
        if not self.is_trained or segment_id not in self.segment_preferences:
            return []
        
        segment_prefs = self.segment_preferences[segment_id]
        
        # Sort by preference score
        sorted_prefs = sorted(
            segment_prefs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_prefs[:num_recommendations]
    
    def get_trending_recommendations(self, num_recommendations: int = 10) -> List[Dict]:
        """
        Get current trending product recommendations
        
        Args:
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of trending product dictionaries
        """
        if not self.is_trained:
            return []
        
        trending_list = list(self.trend_analyzer.trending_products.values())
        return trending_list[:num_recommendations]