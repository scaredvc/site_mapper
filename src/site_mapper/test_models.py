#!/usr/bin/env python3
"""
AI Model Testing Script

Simple script to test different AI models on any website for link classification.
Just add your API keys and run!

Usage:
    # From project root (Windows CMD):
    python src\site_mapper\test_models.py --url https://example.com

    # Or as a module
    python -m site_mapper.test_models --url https://example.com
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional

# Ensure 'src' is on sys.path so we can import site_mapper
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from site_mapper.generalized_crawler import crawl_website  # type: ignore
except Exception as e:
    raise ImportError("generalized_crawler not found; please restore it or adjust the tester.")

from site_mapper.generalized_classifier import (
    create_classifier,
    check_api_keys,
)
from site_mapper.analysis.metrics import compute_model_performance
from site_mapper.analysis.reporting import (
    print_overview,
    print_performance_table,
    print_distributions,
)


def print_banner():
    """Print simple banner"""
    print("AI MODEL TESTING")
    print("=" * 60)


def check_setup():
    """Check if everything is set up"""
    print("Checking setup...")
    status = check_api_keys()
    available = [m for m, ok in status.items() if ok]
    if not available:
        print("No API keys found. Set at least OPENAI_API_KEY in your environment or .env")
        return False
    print(f"Found {len(available)} available model backends: {', '.join(available)}")
    return True


def test_models_on_website(url: str, model_ids: List[str], max_links: int = 20) -> Dict[str, Any]:
    """
    Test multiple AI models on a website
    
    Args:
        url: Website URL to test
        model_ids: List of model IDs to test
        max_links: Maximum number of links to process
        
    Returns:
        Dictionary with test results
    """
    print(f"\nTesting {len(model_ids)} models on: {url}")
    print(f"Processing up to {max_links} links per model")
    
    # Crawl the website
    print("\nCrawling website...")
    crawl_result = crawl_website(url, max_links=max_links)
    
    if not crawl_result or 'filtered_links' not in crawl_result:
        print("Failed to crawl the website")
        return None
    
    links = crawl_result['filtered_links']
    screenshot_path = crawl_result['crawl_metadata']['screenshot_path']
    
    print(f"Found {len(links)} links to test")
    
    # Test each model
    results = {}
    performance = {}
    
    for model_id in model_ids:
        print(f"\nTesting {model_id}...")
        
        try:
            # Create classifier
            classifier = create_classifier(model_id)
            
            # Test on links
            model_results = []
            for i, link in enumerate(links, 1):
                text = link.get('text', '').strip()
                print(f"   Link {i}/{len(links)}: '{text[:30]}...'")
                
                try:
                    result = classifier.classify_link(link, screenshot_path)
                    model_results.append(result)
                    
                    # Minimal output: classification only
                    classification = result.get('classification', 'error')
                    print(f"      {classification}")
                    
                    # Small delay to be respectful to APIs
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"      Error: {e}")
                    model_results.append({
                        "classification": "error",
                        "confidence": 0.0,
                        "model": model_id,
                        "processing_time": 0.0,
                        "cost": 0.0,
                        "success": False,
                        "error": str(e)
                    })
            
            results[model_id] = model_results
            
            perf = compute_model_performance(model_results)
            performance[model_id] = perf
            print(f"   {model_id}: {perf['successful_tests']}/{perf['total_tests']} successful, ${perf['total_cost']:.4f} cost")
            
        except Exception as e:
            print(f"   ❌ {model_id}: Failed to initialize - {e}")
            continue
    # Generate report
    return {
        "url": url,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "models_tested": model_ids,
        "total_links": len(links),
        "results": results,
        "performance": performance,
        "crawl_metadata": crawl_result['crawl_metadata']
    }


def print_results(report: Dict[str, Any]):
    """Print test results"""
    if not report:
        print("No results to display")
        return
    print_overview(report)
    print_performance_table(report['performance'])
    print_distributions(report['results'])


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Test AI models on any website for link classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test (from project root)
  python src/site_mapper/test_models.py --url https://example.com
  
  # As a module
  python -m site_mapper.test_models --url https://example.com
  
  # Test a specific model
  python src/site_mapper/test_models.py --url https://example.com --model gpt-4o-mini
  
  # Test with more links
  python src/site_mapper/test_models.py --url https://example.com --max-links 50
  
  # Save results
  python src/site_mapper/test_models.py --url https://example.com --output results.json
        """
    )
    
    parser.add_argument('--url', required=True, help='Website URL to test')
    parser.add_argument('--models', help='Comma-separated list of models to test (e.g., gpt-4o-mini)')
    parser.add_argument('--max-links', type=int, default=20, help='Maximum links to process')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    print_banner()

    # Check setup
    if not check_setup():
        return 1

    try:
        # Determine models to test from available keys
        available_models = [model for model, has_key in check_api_keys().items() if has_key]
        if args.models:
            model_ids = [m.strip() for m in args.models.split(',') if m.strip()]
            model_ids = [m for m in model_ids if m in available_models]
            if not model_ids:
                print("None of the specified models are available")
                print(f"Available: {', '.join(available_models)}")
                return 1
        else:
            model_ids = available_models
            if not model_ids:
                print("No models available for testing")
                return 1

        # Run the test
        results = test_models_on_website(args.url, model_ids, args.max_links)
        
        if results:
            print_results(results)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Results saved to {args.output}")
            
            print("Test completed successfully")
            return 0
        else:
            print("Test failed")
            return 1
            
    except KeyboardInterrupt:
        print("Test interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during testing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
