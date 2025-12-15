"""
Run the Recommendation API server
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api import create_app


def main():
    """Run the API server"""
    
    print("\n" + "="*70)
    print("ML RECOMMENDATION SYSTEM - API SERVER")
    print("="*70)
    print("\nStarting API server...")
    print("API will be available at: http://localhost:8000")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /api/v1/model/info")
    print("  GET  /api/v1/recommendations/user/<customer_id>")
    print("  GET  /api/v1/recommendations/similar-products/<product_id>")
    print("  GET  /api/v1/recommendations/popular")
    print("  GET  /api/v1/customer/<customer_id>/history")
    print("  POST /api/v1/recommendations/batch")
    print("\nPress CTRL+C to stop the server")
    print("="*70 + "\n")
    
    # Create and run app
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=False)


if __name__ == '__main__':
    main()
