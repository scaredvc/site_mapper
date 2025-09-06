"""
Focused Crawler for AI Bake-off Data Collection

This module provides specialized crawling functionality for collecting
test data from the archive-it.org explore page for the AI classification experiment.
"""

import json
import os
from typing import List, Dict, Any
from playwright.sync_api import sync_playwright, Browser

from site_mapper.crawler import crawl_page
from site_mapper.outlink_analyzers import (
    dom_hierarchy, bounding_box, css_classes, link_position, 
    parent_elements, computed_styles
)


def crawl_archive_it_explore(target_url: str = "https://archive-it.org/explore?show=Collections",
                           output_dir: str = "bakeoff_data") -> Dict[str, Any]:
    """
    Crawl the archive-it.org explore page to collect test data for the AI bake-off.
    
    Args:
        target_url: The URL to crawl (default: archive-it.org explore page)
        output_dir: Directory to save collected data and screenshots
        
    Returns:
        Dictionary containing all collected link data and metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    screenshot_dir = os.path.join(output_dir, "screenshots")
    
    # Define analysis functions to get comprehensive link data
    analyzers = [
        dom_hierarchy,
        bounding_box,
        css_classes,
        link_position,
        parent_elements,
        computed_styles
    ]
    
    print(f"Starting focused crawl of: {target_url}")
    print(f"Output directory: {output_dir}")
    
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)  # Keep visible for debugging
        
        try:
            # Crawl the target page with screenshot capture
            result = crawl_page(
                browser=browser,
                url=target_url,
                analysis_functions=analyzers,
                capture_screenshot=True,
                screenshot_dir=screenshot_dir
            )
            
            print(f"Found {result['outlinks_count']} total links")
            
            # Filter and categorize links for better analysis
            filtered_links = filter_links_for_analysis(result['outlinks'])
            
            # Save the complete data
            output_data = {
                'crawl_metadata': {
                    'target_url': target_url,
                    'crawl_timestamp': result.get('timestamp', 'unknown'),
                    'total_links_found': result['outlinks_count'],
                    'filtered_links_count': len(filtered_links),
                    'screenshot_path': result.get('screenshot_path')
                },
                'all_links': result['outlinks'],
                'filtered_links': filtered_links
            }
            
            # Save to JSON file
            output_file = os.path.join(output_dir, "crawl_data.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"Data saved to: {output_file}")
            print(f"Screenshot saved to: {result.get('screenshot_path', 'N/A')}")
            
            return output_data
            
        finally:
            browser.close()


def filter_links_for_analysis(links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter and categorize links to focus on the most relevant ones for classification.
    
    Args:
        links: List of all extracted links
        
    Returns:
        List of filtered links with additional categorization metadata
    """
    filtered = []
    
    for link in links:
        # Skip empty or invalid links
        if not link.get('text', '').strip() and not link.get('href', '').strip():
            continue
            
        # Skip external links (focus on archive-it.org internal links)
        if link.get('is_external', False):
            continue
            
        # Skip javascript and mailto links
        href = link.get('href', '').lower()
        if href.startswith(('javascript:', 'mailto:', 'tel:')):
            continue
            
        # Add categorization hints based on URL patterns and text
        link['categorization_hints'] = analyze_link_patterns(link)
        
        filtered.append(link)
    
    return filtered


def analyze_link_patterns(link: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze link patterns to provide hints for manual classification.
    
    Args:
        link: Link data dictionary
        
    Returns:
        Dictionary with categorization hints
    """
    hints = {
        'likely_content': [],
        'likely_filter': [],
        'ambiguous': []
    }
    
    text = link.get('text', '').lower()
    url = link.get('absolute_url', '').lower()
    analysis = link.get('analysis', {})
    css_classes = analysis.get('css_classes', [])
    dom_hierarchy = analysis.get('dom_hierarchy', '').lower()
    
    # Content link indicators
    if '/collections/' in url:
        hints['likely_content'].append('URL contains /collections/')
    if any(word in text for word in ['collection', 'archive', 'digital', 'web']):
        hints['likely_content'].append('Text suggests content')
    if len(text) > 20:  # Longer text often indicates content titles
        hints['likely_content'].append('Long descriptive text')
    
    # Filter link indicators
    if any(word in text for word in ['filter', 'sort', 'order', 'view', 'page']):
        hints['likely_filter'].append('Text suggests filtering/sorting')
    if any(word in url for word in ['filter', 'sort', 'order', 'page=']):
        hints['likely_filter'].append('URL suggests filtering/sorting')
    if 'facet' in dom_hierarchy or 'filter' in dom_hierarchy:
        hints['likely_filter'].append('DOM structure suggests filter')
    if any(cls in css_classes for cls in ['filter', 'facet', 'sort', 'pagination']):
        hints['likely_filter'].append('CSS classes suggest filter')
    
    # Pagination indicators
    if text.isdigit() or text in ['next', 'previous', 'prev', 'first', 'last']:
        hints['likely_filter'].append('Pagination control')
    
    # Ambiguous cases
    if not hints['likely_content'] and not hints['likely_filter']:
        hints['ambiguous'].append('No clear indicators')
    if hints['likely_content'] and hints['likely_filter']:
        hints['ambiguous'].append('Conflicting indicators')
    
    return hints


def create_test_labels_template(crawl_data: Dict[str, Any], output_file: str = "test_labels.csv"):
    """
    Create a CSV template for manual labeling of test links.
    
    Args:
        crawl_data: Data from crawl_archive_it_explore()
        output_file: Path to save the CSV template
    """
    import csv
    
    filtered_links = crawl_data.get('filtered_links', [])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'absolute_url', 'text', 'correct_label', 'notes', 
            'likely_content_hints', 'likely_filter_hints', 'ambiguous_hints'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for link in filtered_links:
            hints = link.get('categorization_hints', {})
            writer.writerow({
                'absolute_url': link.get('absolute_url', ''),
                'text': link.get('text', ''),
                'correct_label': '',  # To be filled manually
                'notes': '',  # Optional notes
                'likely_content_hints': '; '.join(hints.get('likely_content', [])),
                'likely_filter_hints': '; '.join(hints.get('likely_filter', [])),
                'ambiguous_hints': '; '.join(hints.get('ambiguous', []))
            })
    
    print(f"Test labels template created: {output_file}")
    print(f"Please manually fill in the 'correct_label' column with 'content_link' or 'filter_link'")
    print(f"Select 50-100 diverse links for your test set")


if __name__ == "__main__":
    # Run the focused crawl
    data = crawl_archive_it_explore()
    
    # Create the test labels template
    create_test_labels_template(data)
    
    print("\nData collection complete!")
    print("Next steps:")
    print("1. Review the crawl_data.json file")
    print("2. Fill in the test_labels.csv with correct classifications")
    print("3. Run the AI bake-off experiment")
