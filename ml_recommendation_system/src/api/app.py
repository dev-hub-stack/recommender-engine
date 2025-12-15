"""
Flask API Application
REST API for serving recommendations
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from .model_loader import ModelLoader
from .recommender import Recommender

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(model_dir: str = None):
    """
    Create and configure Flask application
    
    Args:
        model_dir: Directory containing trained models
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Configuration
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # Load models on startup
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'production')
    
    model_loader = ModelLoader()
    models_loaded = model_loader.load_models(model_dir)
    
    if not models_loaded:
        logger.error("Failed to load models. API will not function properly.")
    
    # Initialize recommender
    try:
        recommender = Recommender()
        logger.info("✅ Recommender initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize recommender: {e}")
        recommender = None
    
    # Error handler
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found',
            'message': str(error)
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(error)
        }), 500
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'success': True,
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': model_loader.is_loaded()
        })
    
    # Model info endpoint
    @app.route('/api/v1/model/info', methods=['GET'])
    def model_info():
        """Get model information"""
        return jsonify({
            'success': True,
            'model_info': model_loader.get_model_info()
        })
    
    # Location-based recommendations endpoint (NEW - PRIMARY)
    @app.route('/api/v1/recommendations/location', methods=['GET'])
    def get_location_recommendations():
        """
        Get personalized recommendations for a location
        
        Query Parameters:
            city: Customer city (optional)
            state: Customer state (optional)
            country: Customer country (optional)
            limit: Number of recommendations (default: 10)
            exclude_purchased: Exclude already purchased products (default: true)
        """
        if recommender is None:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized'
            }), 500
        
        try:
            # Get query parameters
            city = request.args.get('city', '')
            state = request.args.get('state', '')
            country = request.args.get('country', '')
            limit = request.args.get('limit', 10, type=int)
            exclude_purchased = request.args.get('exclude_purchased', 'true').lower() == 'true'
            
            # Validate limit
            if limit < 1 or limit > 100:
                return jsonify({
                    'success': False,
                    'error': 'Invalid limit. Must be between 1 and 100'
                }), 400
            
            # Create location ID using same logic as training
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from pipeline.data_processor import DataProcessor
            processor = DataProcessor()
            location_id = processor.create_location_id(city, state, country)
            
            # Generate recommendations
            recommendations = recommender.get_location_recommendations(
                location_id,
                n_recommendations=limit,
                exclude_purchased=exclude_purchased
            )
            
            return jsonify({
                'success': True,
                'location_id': location_id,
                'location_params': {
                    'city': city,
                    'state': state,
                    'country': country
                },
                'algorithm': 'location_based_collaborative_filtering',
                'n_recommendations': len(recommendations),
                'recommendations': recommendations
            })
            
        except Exception as e:
            logger.error(f"Error generating location recommendations: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # User recommendations endpoint (DEPRECATED - kept for backward compatibility)
    @app.route('/api/v1/recommendations/user/<customer_id>', methods=['GET'])
    def get_user_recommendations(customer_id):
        """
        DEPRECATED: Use /api/v1/recommendations/location instead
        Get personalized recommendations for a customer
        
        Query Parameters:
            limit: Number of recommendations (default: 10)
            exclude_purchased: Exclude already purchased products (default: true)
        """
        if recommender is None:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized'
            }), 500
        
        try:
            # Get query parameters
            limit = request.args.get('limit', 10, type=int)
            exclude_purchased = request.args.get('exclude_purchased', 'true').lower() == 'true'
            
            # Validate limit
            if limit < 1 or limit > 100:
                return jsonify({
                    'success': False,
                    'error': 'Invalid limit. Must be between 1 and 100'
                }), 400
            
            # Generate recommendations (now uses location-based internally)
            recommendations = recommender.get_user_recommendations(
                customer_id,
                n_recommendations=limit,
                exclude_purchased=exclude_purchased
            )
            
            return jsonify({
                'success': True,
                'customer_id': customer_id,
                'algorithm': 'collaborative_filtering',
                'n_recommendations': len(recommendations),
                'recommendations': recommendations
            })
            
        except Exception as e:
            logger.error(f"Error generating recommendations for {customer_id}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Similar products endpoint
    @app.route('/api/v1/recommendations/similar-products/<product_id>', methods=['GET'])
    def get_similar_products(product_id):
        """
        Get products similar to a given product
        
        Query Parameters:
            limit: Number of similar products (default: 10)
        """
        if recommender is None:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized'
            }), 500
        
        try:
            # Get query parameters
            limit = request.args.get('limit', 10, type=int)
            
            # Validate limit
            if limit < 1 or limit > 100:
                return jsonify({
                    'success': False,
                    'error': 'Invalid limit. Must be between 1 and 100'
                }), 400
            
            # Get similar products
            similar_products = recommender.get_similar_products(
                product_id,
                n_recommendations=limit
            )
            
            if not similar_products:
                return jsonify({
                    'success': False,
                    'error': f'Product {product_id} not found'
                }), 404
            
            return jsonify({
                'success': True,
                'product_id': product_id,
                'algorithm': 'item_based_collaborative_filtering',
                'n_similar_products': len(similar_products),
                'similar_products': similar_products
            })
            
        except Exception as e:
            logger.error(f"Error finding similar products for {product_id}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Popular products endpoint
    @app.route('/api/v1/recommendations/popular', methods=['GET'])
    def get_popular_products():
        """
        Get most popular products
        
        Query Parameters:
            limit: Number of products (default: 10)
        """
        if recommender is None:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized'
            }), 500
        
        try:
            # Get query parameters
            limit = request.args.get('limit', 10, type=int)
            
            # Validate limit
            if limit < 1 or limit > 100:
                return jsonify({
                    'success': False,
                    'error': 'Invalid limit. Must be between 1 and 100'
                }), 400
            
            # Get popular products
            popular_products = recommender.get_popular_products(n_recommendations=limit)
            
            return jsonify({
                'success': True,
                'algorithm': 'popularity_based',
                'n_products': len(popular_products),
                'products': popular_products
            })
            
        except Exception as e:
            logger.error(f"Error getting popular products: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Customer purchase history endpoint
    @app.route('/api/v1/customer/<customer_id>/history', methods=['GET'])
    def get_customer_history(customer_id):
        """Get customer's purchase history"""
        if recommender is None:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized'
            }), 500
        
        try:
            history = recommender.get_customer_purchase_history(customer_id)
            
            if not history:
                return jsonify({
                    'success': False,
                    'error': f'Customer {customer_id} not found or has no purchase history'
                }), 404
            
            return jsonify({
                'success': True,
                'customer_id': customer_id,
                'n_products': len(history),
                'purchase_history': history
            })
            
        except Exception as e:
            logger.error(f"Error getting history for {customer_id}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Cart-based recommendations endpoint (for Shopify checkout)
    @app.route('/api/v1/recommendations/cart', methods=['POST'])
    def get_cart_recommendations():
        """
        Get recommendations based on cart contents and location
        "Frequently Bought Together" for checkout page
        
        Request Body:
            {
                "location": {
                    "city": "Lahore",
                    "state": "Punjab",
                    "country": "Pakistan"
                },
                "cart_items": [
                    {"sku": "005868", "quantity": 2},
                    {"sku": "Rose-Gold-Wedding-Bundle", "quantity": 1}
                ],
                "limit": 5
            }
        """
        if recommender is None:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized'
            }), 500
        
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'Missing request body'
                }), 400
            
            # Validate required fields
            if 'location' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing location in request body'
                }), 400
            
            if 'cart_items' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing cart_items in request body'
                }), 400
            
            location = data['location']
            cart_items = data['cart_items']
            limit = data.get('limit', 5)
            
            # Validate limit
            if limit < 1 or limit > 20:
                return jsonify({
                    'success': False,
                    'error': 'Invalid limit. Must be between 1 and 20'
                }), 400
            
            # Create location ID
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from pipeline.data_processor import DataProcessor
            processor = DataProcessor()
            location_id = processor.create_location_id(
                location.get('city', ''),
                location.get('state', ''),
                location.get('country', '')
            )
            
            # Extract SKUs from cart items
            cart_skus = []
            for item in cart_items:
                if isinstance(item, dict) and 'sku' in item:
                    cart_skus.append(item['sku'])
                elif isinstance(item, str):
                    cart_skus.append(item)
            
            if not cart_skus:
                return jsonify({
                    'success': False,
                    'error': 'No valid SKUs found in cart_items'
                }), 400
            
            # Generate cart-based recommendations
            recommendations = recommender.get_cart_recommendations(
                location_id=location_id,
                cart_skus=cart_skus,
                n_recommendations=limit
            )
            
            return jsonify({
                'success': True,
                'location_id': location_id,
                'location': location,
                'cart_items_count': len(cart_skus),
                'cart_skus': cart_skus,
                'algorithm': 'cart_based_collaborative_filtering',
                'n_recommendations': len(recommendations),
                'recommendations': recommendations
            })
            
        except Exception as e:
            logger.error(f"Error generating cart recommendations: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Shopify cart recommendations endpoint (with product enrichment)
    @app.route('/api/v1/recommendations/shopify-cart', methods=['POST'])
    def get_shopify_cart_recommendations():
        """
        Get recommendations with full Shopify product details
        Accepts Shopify cart payload, returns Shopify variant format
        
        Request Body:
            {
                "location": {"city": "LAHORE"},
                "items": [
                    {
                        "id": 50639911452978,
                        "sku": "beauty-rest-Single-78*42-5",
                        "quantity": 1,
                        "price": 1092500
                    }
                ],
                "total_price": 1092500,
                "item_count": 1,
                "currency": "PKR"
            }
            
        Response:
            [
                {
                    "id": 50639911485746,
                    "title": "Single - 78*42 / 6",
                    "sku": "beauty-rest-Single-78*42-6",
                    "price": 1358500,
                    "available": true,
                    ...
                }
            ]
        """
        if recommender is None:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized'
            }), 500
        
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'Missing request body'
                }), 400
            
            # Extract location from Shopify cart
            location = data.get('location', {})
            items = data.get('items', [])
            limit = data.get('limit', 5)
            
            if not items:
                return jsonify({
                    'success': False,
                    'error': 'Missing items in request body'
                }), 400
            
            # Create location ID
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from pipeline.data_processor import DataProcessor
            processor = DataProcessor()
            location_id = processor.create_location_id(
                location.get('city', ''),
                location.get('state', ''),
                location.get('country', '')
            )
            
            # Extract SKUs from Shopify cart items
            cart_skus = []
            for item in items:
                if isinstance(item, dict) and 'sku' in item:
                    cart_skus.append(item['sku'])
            
            if not cart_skus:
                return jsonify({
                    'success': False,
                    'error': 'No valid SKUs found in cart items'
                }), 400
            
            logger.info(f"Shopify cart request: {len(cart_skus)} items, location={location_id}")
            
            # Step 1: Get MORE ML recommendations than requested (to account for filtering)
            # Request 4x the limit to ensure we have enough after filtering
            ml_limit = min(limit * 4, 50)  # Get 4x but cap at 50
            ml_recommendations = recommender.get_cart_recommendations(
                location_id=location_id,
                cart_skus=cart_skus,
                n_recommendations=ml_limit
            )
            
            if not ml_recommendations:
                logger.warning("No ML recommendations generated")
                return jsonify([])
            
            # Step 2: Extract recommended SKUs
            recommended_skus = [rec['product_id'] for rec in ml_recommendations]
            logger.info(f"ML recommended {len(recommended_skus)} SKUs: {recommended_skus}")
            
            # TESTING MODE: Swap ML SKUs with real Shopify SKUs for testing
            testing_mode = os.getenv('SHOPIFY_TESTING_MODE', 'false').lower() == 'true'
            if testing_mode:
                # Load Shopify test SKUs from environment (comma-separated)
                test_skus_env = os.getenv('SHOPIFY_TEST_SKUS', '')
                if test_skus_env:
                    shopify_test_skus = [sku.strip() for sku in test_skus_env.split(',') if sku.strip()]
                    
                    # Replace ML SKUs with Shopify SKUs (keep same order/scores)
                    original_skus = recommended_skus.copy()
                    recommended_skus = shopify_test_skus[:len(recommended_skus)]
                    
                    logger.info(f"🧪 TESTING MODE: Swapped {len(original_skus)} ML SKUs with Shopify staging SKUs")
                    logger.info(f"   Original: {original_skus[:3]}...")
                    logger.info(f"   Swapped:  {recommended_skus[:3]}...")
                else:
                    logger.warning("⚠️ TESTING MODE enabled but SHOPIFY_TEST_SKUS not configured in .env")
                    logger.warning("   Add SHOPIFY_TEST_SKUS to .env with comma-separated SKUs")
            
            # Step 3: Fetch product details from Shopify (ONE API call for all SKUs)
            try:
                # Import from the correct module path
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                from shopify.shopify_client import get_shopify_client
                
                shopify_client = get_shopify_client()
                shopify_variants = shopify_client.get_variants_by_skus(recommended_skus)
                logger.info(f"Shopify returned {len(shopify_variants)} variants")
            except Exception as e:
                logger.error(f"Failed to fetch from Shopify: {e}")
                import traceback
                traceback.print_exc()
                # Return empty array if Shopify fails
                return jsonify([])
            
            # Step 4: Build final response (only include products found in Shopify)
            final_recommendations = []
            
            # In testing mode, iterate through swapped SKUs
            # In production mode, iterate through ML recommendations
            if testing_mode:
                # Use the swapped Shopify SKUs directly
                for sku in recommended_skus:
                    if sku in shopify_variants:
                        final_recommendations.append(shopify_variants[sku])
                        if len(final_recommendations) >= limit:
                            break
            else:
                # Normal mode: check ML recommended SKUs
                for rec in ml_recommendations:
                    sku = rec['product_id']
                    if sku in shopify_variants:
                        final_recommendations.append(shopify_variants[sku])
                        if len(final_recommendations) >= limit:
                            break
                    else:
                        logger.debug(f"SKU {sku} not found in Shopify (skipping)")
            
            logger.info(f"✅ Returning {len(final_recommendations)} enriched recommendations (from {len(ml_recommendations)} ML recommendations)")
            
            # Return array of Shopify variant objects (no wrapper)
            return jsonify(final_recommendations)
            
        except Exception as e:
            logger.error(f"Error in Shopify cart recommendations: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Batch recommendations endpoint
    @app.route('/api/v1/recommendations/batch', methods=['POST'])
    def get_batch_recommendations():
        """
        Get recommendations for multiple locations
        
        Request Body:
            {
                "locations": [
                    {"city": "Lahore", "state": "Punjab", "country": "Pakistan"},
                    {"city": "Karachi", "state": "Sindh", "country": "Pakistan"}
                ],
                "limit": 10
            }
            
        OR (deprecated format):
            {
                "customer_ids": ["id1", "id2", ...],
                "limit": 10
            }
        """
        if recommender is None:
            return jsonify({
                'success': False,
                'error': 'Recommender not initialized'
            }), 500
        
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'Missing request body'
                }), 400
            
            limit = data.get('limit', 10)
            
            # Support new location-based format
            if 'locations' in data:
                locations = data['locations']
                
                if not isinstance(locations, list):
                    return jsonify({
                        'success': False,
                        'error': 'locations must be a list'
                    }), 400
                
                if len(locations) > 1000:
                    return jsonify({
                        'success': False,
                        'error': 'Maximum 1000 locations per batch request'
                    }), 400
                
                # Create location IDs
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                from pipeline.data_processor import DataProcessor
                processor = DataProcessor()
                location_ids = []
                
                for loc in locations:
                    location_id = processor.create_location_id(
                        loc.get('city', ''),
                        loc.get('state', ''),
                        loc.get('country', '')
                    )
                    location_ids.append(location_id)
                
                # Generate batch recommendations
                results = recommender.batch_recommendations(location_ids, limit)
                
                return jsonify({
                    'success': True,
                    'n_locations': len(location_ids),
                    'recommendations': results
                })
            
            # Support deprecated customer_ids format
            elif 'customer_ids' in data:
                customer_ids = data['customer_ids']
                
                if not isinstance(customer_ids, list):
                    return jsonify({
                        'success': False,
                        'error': 'customer_ids must be a list'
                    }), 400
                
                if len(customer_ids) > 1000:
                    return jsonify({
                        'success': False,
                        'error': 'Maximum 1000 customers per batch request'
                    }), 400
                
                # Generate batch recommendations
                results = recommender.batch_recommendations(customer_ids, limit)
                
                return jsonify({
                    'success': True,
                    'n_customers': len(customer_ids),
                    'recommendations': results
                })
            
            else:
                return jsonify({
                    'success': False,
                    'error': 'Missing locations or customer_ids in request body'
                }), 400
            
        except Exception as e:
            logger.error(f"Error in batch recommendations: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=True)
