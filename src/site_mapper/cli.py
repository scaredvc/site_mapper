import argparse
import logging
from pathlib import Path
from typing import List

from site_mapper.crawler import crawl_site
from site_mapper.config import load_config
from site_mapper.output_handler import OutputHandler
from site_mapper.outlink_analyzers import (
    dom_hierarchy,
    bounding_box,
    css_classes,
    link_position,
    parent_elements
)

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_default_analyzers() -> List:
    """Get list of default analyzer functions"""
    return [
        dom_hierarchy,
        bounding_box,
        css_classes,
        link_position,
        parent_elements
    ]

def main():
    parser = argparse.ArgumentParser(description='Site Mapper - Web Crawler and Link Analyzer')
    parser.add_argument('--url', required=True, help='Starting URL to crawl')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output-dir', help='Directory to save results (overrides config file)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                      help='Logging level (overrides config file)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.log_level:
        config['log_level'] = args.log_level

    # Setup logging
    setup_logging(config.get('log_level', 'INFO'))
    
    # Initialize output handler
    output_dir = config.get('output_dir', './results')
    output_handler = OutputHandler(output_dir)
    
    try:
        # Start crawling
        logging.info(f"Starting crawl of {args.url}")
        logging.info(f"Results will be saved to {output_dir}")
        
        link_graph = crawl_site(
            args.url,
            scope_rules=config,
            analysis_functions=get_default_analyzers(),
            output_handler=output_handler
        )
        
        logging.info(f"Crawl completed. Processed {len(link_graph)} pages")
        
    except Exception as e:
        logging.error(f"Crawl failed: {e}")
        raise

if __name__ == "__main__":
    main()