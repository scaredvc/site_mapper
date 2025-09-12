#!/usr/bin/env python3
"""
AI Classification Experiment Runner

This script compares models for classifying links into:
- content_link
- filter_link
- navigation_link

Usage:
    python scripts/run_classifier.py --test-labels tests/test_labels.csv --crawl-data data/crawl_data.json
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add src to path so we can import our modules
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(repo_root, 'src'))

from site_mapper.generalized_classifier import create_classifier, check_api_keys


def load_test_data(test_labels_file: str, crawl_data_file: str) -> List[Dict[str, Any]]:
    """
    Load test data from CSV labels and JSON crawl data.
    
    Args:
        test_labels_file: Path to CSV file with manual labels
        crawl_data_file: Path to JSON file with crawl data
        
    Returns:
        List of test cases with both labels and link data
    """
    # Load crawl data
    with open(crawl_data_file, 'r', encoding='utf-8') as f:
        crawl_data = json.load(f)
    
    # Create a lookup dictionary for link data by URL
    link_lookup = {}
    for link in crawl_data.get('all_links', []):
        link_lookup[link.get('absolute_url', '')] = link
    
    # Load test labels
    test_cases = []
    with open(test_labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('absolute_url', '').strip()
            correct_label = row.get('correct_label', '').strip().lower()
            
            if not url or not correct_label:
                print(f"Skipping incomplete row: {row}")
                continue
                
            if correct_label not in ['content_link', 'filter_link', 'navigation_link']:
                print(f"Skipping invalid label '{correct_label}' for {url}")
                continue
            
            # Find matching link data
            if url in link_lookup:
                test_case = {
                    'url': url,
                    'text': row.get('text', ''),
                    'correct_label': correct_label,
                    'notes': row.get('notes', ''),
                    'link_data': link_lookup[url],
                    'screenshot_path': crawl_data.get('crawl_metadata', {}).get('screenshot_path')
                }
                test_cases.append(test_case)
            else:
                print(f"Warning: No link data found for {url}")
    
    print(f"Loaded {len(test_cases)} test cases")
    return test_cases


def run_classification_experiment(test_cases: List[Dict[str, Any]], 
                                models: List[str] = None) -> Dict[str, Any]:
    """
    Run the classification experiment with both models.
    
    Args:
        test_cases: List of test cases with labels and link data
        models: List of model names to test (default: both)
        
    Returns:
        Dictionary with experiment results
    """
    if models is None:
        # Use all available models based on API keys
        available_models = [m for m, has_key in check_api_keys().items() if has_key]
        if not available_models:
            print("No API keys found. Please set OPENAI_API_KEY or GOOGLE_API_KEY")
            return None
        # Map back to CLI aliases for consistency
        model_map = {'gpt-4o': 'gpt4o', 'gpt-3.5-turbo': 'gpt35', 'gpt-4o-mini': 'gpt4o-mini'}
        models = [model_map.get(m, m) for m in available_models]
    
    # Initialize classifiers
    classifiers = {}
    for model_name in models:
        try:
            # Map CLI aliases to generalized model IDs
            aliases = {
                'gpt4o': 'gpt-4o',
                'gpt35': 'gpt-3.5-turbo', 
                'gpt4o-mini': 'gpt-4o-mini'
            }
            resolved = aliases.get(model_name.lower(), model_name)
            
            classifier = create_classifier(resolved)
            classifiers[model_name] = classifier
            print(f"Initialized {model_name} classifier")
        except Exception as e:
            print(f"Failed to initialize {model_name}: {e}")
            return None
    
    # Run experiments
    results = {
        'experiment_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_test_cases': len(test_cases),
            'models_tested': list(classifiers.keys())
        },
        'test_results': [],
        'model_stats': {}
    }
    
    print(f"\nStarting classification experiment with {len(test_cases)} test cases...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nProcessing test case {i}/{len(test_cases)}: {test_case['url']}")
        
        result = {
            'test_case': {
                'url': test_case['url'],
                'text': test_case['text'],
                'correct_label': test_case['correct_label'],
                'notes': test_case['notes']
            },
            'classifications': {}
        }
        
        # Test each model
        for model_name, classifier in classifiers.items():
            print(f"  Testing with {model_name}...")
            
            try:
                classification_result = classifier.classify_link(
                    test_case['link_data'], 
                    test_case['screenshot_path']
                )
                
                result['classifications'][model_name] = {
                    'predicted_label': classification_result.get('classification', 'error'),
                    'confidence': classification_result.get('confidence', 0.0),
                    'log_probability': classification_result.get('log_probability', None),  # Add log probability data
                    'processing_time': classification_result.get('processing_time', 0.0),
                    'cost': classification_result.get('cost', 0.0),
                    'success': classification_result.get('success', False),
                    'error': classification_result.get('error', None)
                }
                
                # Check if prediction is correct
                predicted = classification_result.get('classification', 'error')
                correct = test_case['correct_label']
                is_correct = predicted == correct
                
                result['classifications'][model_name]['is_correct'] = is_correct
                
                # Display result with log probability and uncertainty information
                confidence = classification_result.get('confidence', 0.0)
                log_prob = classification_result.get('log_probability', None)
                is_uncertain = classification_result.get('is_uncertain', False)
                log_prob_str = f" (log_prob: {log_prob:.3f})" if log_prob is not None else ""
                uncertainty_str = " ⚠️ UNCERTAIN" if is_uncertain else ""
                print(f"    Result: {predicted} (correct: {correct}) - {'✓' if is_correct else '✗'}")
                print(f"      Confidence: {confidence:.3f}{log_prob_str}{uncertainty_str}")
                if not classification_result.get('success', False):
                    err = classification_result.get('error', '')
                    if err:
                        print(f"      Error detail: {err}")
                
            except Exception as e:
                print(f"    Error with {model_name}: {e}")
                result['classifications'][model_name] = {
                    'predicted_label': 'error',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'cost': 0.0,
                    'success': False,
                    'error': str(e),
                    'is_correct': False
                }
        
        results['test_results'].append(result)
        
        # Add a small delay to be respectful to APIs
        time.sleep(0.5)
    
    # Calculate model statistics
    for model_name in classifiers.keys():
        model_results = [r['classifications'][model_name] for r in results['test_results']]
        
        correct_predictions = sum(1 for r in model_results if r.get('is_correct', False))
        total_predictions = len(model_results)
        successful_predictions = sum(1 for r in model_results if r.get('success', False))
        
        total_time = sum(r.get('processing_time', 0) for r in model_results)
        total_cost = sum(r.get('cost', 0) for r in model_results)
        
        # Calculate log probability statistics
        log_probs = [r.get('log_probability') for r in model_results if r.get('log_probability') is not None]
        avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else None
        log_prob_std = None
        if len(log_probs) > 1:
            variance = sum((lp - avg_log_prob) ** 2 for lp in log_probs) / (len(log_probs) - 1)
            log_prob_std = variance ** 0.5
        
        # Calculate uncertainty statistics
        uncertain_predictions = [r for r in model_results if r.get('is_uncertain', False)]
        uncertainty_rate = len(uncertain_predictions) / total_predictions if total_predictions > 0 else 0
        
        results['model_stats'][model_name] = {
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
            'success_rate': successful_predictions / total_predictions if total_predictions > 0 else 0,
            'total_correct': correct_predictions,
            'total_predictions': total_predictions,
            'total_time': total_time,
            'total_cost': total_cost,
            'avg_time_per_prediction': total_time / total_predictions if total_predictions > 0 else 0,
            'avg_log_probability': avg_log_prob,
            'log_probability_std': log_prob_std,
            'log_probability_count': len(log_probs),
            'uncertain_predictions': len(uncertain_predictions),
            'uncertainty_rate': uncertainty_rate
        }
    
    # Get final stats from classifiers
    for model_name, classifier in classifiers.items():
        results['model_stats'][model_name]['classifier_stats'] = classifier.get_stats()
    
    return results


def analyze_results(results: Dict[str, Any]) -> None:
    """
    Analyze and display experiment results.
    
    Args:
        results: Results dictionary from run_classification_experiment
    """
    print("\n" + "="*80)
    print("AI BAKE-OFF RESULTS")
    print("="*80)
    
    metadata = results['experiment_metadata']
    print(f"Experiment Date: {metadata['timestamp']}")
    print(f"Total Test Cases: {metadata['total_test_cases']}")
    print(f"Models Tested: {', '.join(metadata['models_tested'])}")
    
    print("\n" + "-"*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("-"*50)
    
    # Create comparison table
    models = list(results['model_stats'].keys())
    if len(models) >= 2:
        print(f"{'Metric':<25} {models[0]:<15} {models[1]:<15}")
        print("-" * 55)
        
        for metric in ['accuracy', 'success_rate', 'total_correct', 'total_time', 'total_cost']:
            values = []
            for model in models:
                value = results['model_stats'][model].get(metric, 0)
                if metric in ['accuracy', 'success_rate']:
                    values.append(f"{value:.3f}")
                elif metric == 'total_time':
                    values.append(f"{value:.2f}s")
                elif metric == 'total_cost':
                    values.append(f"${value:.4f}")
                else:
                    values.append(str(value))
            
            print(f"{metric.replace('_', ' ').title():<25} {values[0]:<15} {values[1]:<15}")
    
    print("\n" + "-"*50)
    print("DETAILED MODEL STATISTICS")
    print("-"*50)
    
    for model_name, stats in results['model_stats'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {stats['accuracy']:.3f} ({stats['total_correct']}/{stats['total_predictions']})")
        print(f"  Success Rate: {stats['success_rate']:.3f}")
        print(f"  Total Time: {stats['total_time']:.2f} seconds")
        print(f"  Total Cost: ${stats['total_cost']:.4f}")
        print(f"  Avg Time per Prediction: {stats['avg_time_per_prediction']:.2f} seconds")
        
        # Log probability analysis
        if 'avg_log_probability' in stats:
            print(f"  Avg Log Probability: {stats['avg_log_probability']:.3f}")
        if 'log_probability_std' in stats:
            print(f"  Log Probability Std Dev: {stats['log_probability_std']:.3f}")
        
        # Uncertainty analysis
        if 'uncertain_predictions' in stats:
            print(f"  Uncertain Predictions: {stats['uncertain_predictions']}/{stats['total_predictions']} ({stats['uncertainty_rate']:.1%})")
    
    # Failure analysis
    print("\n" + "-"*50)
    print("FAILURE ANALYSIS")
    print("-"*50)
    
    for model_name in models:
        print(f"\n{model_name.upper()} - Incorrect Predictions:")
        incorrect_cases = []
        
        for result in results['test_results']:
            classification = result['classifications'].get(model_name, {})
            if not classification.get('is_correct', False) and classification.get('success', False):
                incorrect_cases.append({
                    'url': result['test_case']['url'],
                    'text': result['test_case']['text'],
                    'correct': result['test_case']['correct_label'],
                    'predicted': classification.get('predicted_label', 'error')
                })
        
        if incorrect_cases:
            for case in incorrect_cases[:5]:  # Show first 5 failures
                print(f"  URL: {case['url']}")
                print(f"  Text: '{case['text']}'")
                print(f"  Correct: {case['correct']}, Predicted: {case['predicted']}")
                print()
        else:
            print("  No incorrect predictions!")


def save_results(results: Dict[str, Any], output_file: str = "bakeoff_results.json") -> None:
    """
    Save experiment results to a JSON file.
    
    Args:
        results: Results dictionary
        output_file: Path to save results
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


def main():
    """Main function to run the AI bake-off experiment."""
    parser = argparse.ArgumentParser(description='Run AI bake-off experiment')
    parser.add_argument('--test-labels', required=True, 
                       help='Path to CSV file with test labels')
    parser.add_argument('--crawl-data', required=True,
                       help='Path to JSON file with crawl data')
    parser.add_argument('--models', nargs='+', choices=['gpt4o', 'gpt35', 'gpt4o-mini'],
                       default=None,
                       help='Models to test (default: all available based on API keys)')
    parser.add_argument('--output', default='bakeoff_results.json',
                       help='Output file for results (default: bakeoff_results.json)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.test_labels):
        print(f"Error: Test labels file not found: {args.test_labels}")
        return 1
    
    if not os.path.exists(args.crawl_data):
        print(f"Error: Crawl data file not found: {args.crawl_data}")
        return 1
    
    # Check for API keys for requested models
    api_status = check_api_keys()
    aliases = {'gpt4o': 'gpt-4o', 'gpt35': 'gpt-3.5-turbo', 'gpt4o-mini': 'gpt-4o-mini'}
    
    missing_keys = []
    for model in args.models:
        resolved = aliases.get(model.lower(), model)
        if not api_status.get(resolved, False):
            missing_keys.append(f"{model} (needs API key for {resolved})")
    
    if missing_keys:
        print(f"Error: Missing API keys for: {', '.join(missing_keys)}")
        print("Please set OPENAI_API_KEY and/or GOOGLE_API_KEY in environment variables")
        return 1
    
    try:
        # Load test data
        print("Loading test data...")
        test_cases = load_test_data(args.test_labels, args.crawl_data)
        
        if not test_cases:
            print("Error: No valid test cases found")
            return 1
        
        # Run experiment
        print("Starting AI bake-off experiment...")
        results = run_classification_experiment(test_cases, args.models)
        
        if results is None:
            print("Error: Experiment failed")
            return 1
        
        # Analyze and display results
        analyze_results(results)
        
        # Save results
        save_results(results, args.output)
        
        print("\nExperiment completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running experiment: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
