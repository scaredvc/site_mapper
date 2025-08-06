import json
import pandas as pd
from pathlib import Path

def flatten_link_features(link_data):
    """
    Convert a nested JSON link structure into a flat dictionary of features.
    This is the key to converting nested JSON to CSV format.
    """
    # Get the nested analysis data
    analysis = link_data.get('analysis', {})
    archive_analysis = analysis.get('analyze_archive_it_link', {})
    
    features = {
        # Basic Features - Direct from top level
        'url': link_data['absolute_url'],
        'link_text': link_data['text'],
        'is_external': link_data['is_external'],
        
        # DOM Features - From analysis
        'dom_path': analysis.get('dom_hierarchy', ''),
        # Convert array of classes to a string with ',' separator
        'css_classes': ','.join(analysis.get('css_classes', [])),
        # Check if link is in navigation based on parent elements
        'in_navigation': any(
            parent in ['nav', 'header', 'menu'] 
            for parent in analysis.get('parent_elements', [])
        ),
        
        # Position Features
        'position_on_page': analysis.get('link_position', ''),
        
        # URL Structure Features - From archive_analysis
        'has_query_params': archive_analysis.get('has_query_params', False),
        'has_sort_param': archive_analysis.get('has_sort_param', False),
        'has_filter_param': archive_analysis.get('has_filter_param', False),
        'has_page_param': archive_analysis.get('has_page_param', False),
        'has_show_param': archive_analysis.get('has_show_param', False),
        
        # Content Type Features - Derived from URL patterns
        'is_detail_page': (
            '/organizations/' in link_data['absolute_url'] or 
            '/collections/' in link_data['absolute_url']
        ),
        'is_list_page': '/explore' in link_data['absolute_url'],
        
        # Trap Detection Features
        'in_faceted_search': archive_analysis.get('in_faceted_search_ui', False),
        'number_of_issues': len(archive_analysis.get('potential_issues', [])),
        # Convert array of issues to a string
        'issue_types': ','.join(archive_analysis.get('potential_issues', [])),
        
        # URL Complexity Features
        'path_depth': len(archive_analysis.get('path_segments', [])),
        'path_components': ','.join(archive_analysis.get('path_segments', [])),
    }
    
    return features

def create_training_dataset(json_path, output_csv_path):
    """
    Create a machine learning ready dataset from crawler results.
    """
    # Load the JSON data
    print(f"Loading data from {json_path}")
    with open(json_path, 'r') as f:
        crawl_data = json.load(f)
    
    # Process all links
    all_links = []
    for source_url, links in crawl_data.items():
        for link in links:
            # Get flattened features
            features = flatten_link_features(link)
            # Add source page for context
            features['source_page'] = source_url
            all_links.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_links)
    
    # Create initial labels based on our heuristics
    df['is_good_link'] = (
        # Positive indicators
        (df['is_detail_page'] | df['in_navigation']) &
        # Negative indicators (no trap characteristics)
        ~(
            df['has_sort_param'] |
            df['has_filter_param'] |
            df['has_page_param'] |
            df['has_show_param'] |
            df['in_faceted_search'] |
            (df['number_of_issues'] > 0)
        )
    )
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total links processed: {len(df)}")
    print(f"Good links: {df['is_good_link'].sum()}")
    print(f"Bad links: {(~df['is_good_link']).sum()}")
    print("\nPage Types:")
    print(f"Detail pages: {df['is_detail_page'].sum()}")
    print(f"List pages: {df['is_list_page'].sum()}")
    print("\nPotential Issues:")
    print(f"Links with query parameters: {df['has_query_params'].sum()}")
    print(f"Links in faceted search: {df['in_faceted_search'].sum()}")
    print(f"Links with potential issues: {(df['number_of_issues'] > 0).sum()}")
    
    # Show sample of the data
    print("\nSample of the dataset (first 5 rows):")
    print(df[['url', 'is_good_link', 'is_detail_page', 'number_of_issues']].head())
    
    return df

if __name__ == "__main__":
    # Set up paths
    results_dir = Path("results")
    json_path = results_dir / "crawl_results_final.json"
    output_csv_path = results_dir / "training_data.csv"
    
    # Create the dataset
    df = create_training_dataset(json_path, output_csv_path)