import json
import time
import logging
from collections import deque
from urllib.parse import urljoin, urlparse
from typing import Any, Callable, Dict, List, Optional

from playwright.sync_api import Page, sync_playwright, Browser, TimeoutError as PlaywrightTimeout

from site_mapper.outlink_analyzers import *
from site_mapper.output_handler import OutputHandler

class CrawlerError(Exception):
    """Base class for crawler exceptions"""
    pass

class NetworkError(CrawlerError):
    """Raised when network issues occur"""
    pass

class RateLimiter:
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.last_request_time = 0

    def wait(self):
        """Wait appropriate amount of time between requests"""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        self.last_request_time = time.time()

def is_url_in_scope(url: str, scope_rules: dict) -> bool:
    """Check if a URL is in scope based on scope rules"""
    allowed_hosts = scope_rules.get('allowed_hosts', [])
    if not allowed_hosts:
        return True
    parsed_url = urlparse(url)
    return parsed_url.hostname in allowed_hosts

def extract_outlinks_with_analysis(page: Page, base_url: str, analysis_functions: Optional[List[Callable]] = None) -> List[Dict[str, Any]]:
    """Extract all outlinks from the page and run analysis functions on each link element."""
    if analysis_functions is None:
        analysis_functions = []

    try:
        # Find all link elements
        link_elements = page.locator('a[href]').all()

        outlinks = []
        for element in link_elements:
            try:
                # Get basic link information
                href = element.get_attribute('href')
                text = element.inner_text().strip()

                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)

                parsed_base = urlparse(base_url)
                parsed_link = urlparse(absolute_url)

                link_data = {
                    'href': href,
                    'absolute_url': absolute_url,
                    'text': text,
                    'is_external': parsed_base.hostname != parsed_link.hostname,
                    'analysis': {}
                }

                # Run analysis functions on this link element
                for analysis_func in analysis_functions:
                    try:
                        func_name = analysis_func.__name__
                        link_data['analysis'][func_name] = analysis_func(page, element.element_handle())
                    except Exception as e:
                        logging.error(f"Error running {analysis_func.__name__} on link {absolute_url}: {e}")
                        link_data['analysis'][func_name] = None

                outlinks.append(link_data)

            except Exception as e:
                logging.error(f"Error processing link element: {e}")
                continue

        return outlinks
    except Exception as e:
        logging.error(f"Error extracting outlinks: {e}")
        raise NetworkError(f"Failed to extract outlinks: {e}")

def crawl_page(browser: Browser, url: str, analysis_functions: Optional[List[Callable]] = None, max_retries: int = 3) -> Dict[str, Any]:
    """Crawl a page and extract outlinks with optional analysis functions."""
    retry_count = 0
    while retry_count < max_retries:
        page = browser.new_page()
        try:
            # Set up request blocking for wayback.archive-it.org
            def handle_route(route):
                if "wayback.archive-it.org" in route.request.url:
                    logging.info(f"Blocked request to: {route.request.url}")
                    route.abort()
                else:
                    route.continue_()

            # Enable request interception
            page.route("**/*", handle_route)

            logging.info(f"Attempting to crawl: {url}")
            page.goto(url, timeout=30000)  # 30 second timeout

            # Extract outlinks with analysis
            outlinks = extract_outlinks_with_analysis(page, url, analysis_functions)

            return {
                'url': url,
                'outlinks': outlinks,
                'outlinks_count': len(outlinks)
            }

        except PlaywrightTimeout:
            retry_count += 1
            logging.warning(f"Timeout crawling {url}. Attempt {retry_count} of {max_retries}")
            if retry_count == max_retries:
                raise NetworkError(f"Failed to crawl {url} after {max_retries} attempts")
            time.sleep(retry_count * 2)  # Exponential backoff

        except Exception as e:
            logging.error(f"Error crawling {url}: {e}")
            raise NetworkError(f"Failed to crawl {url}: {e}")

        finally:
            page.close()

def log_link_analysis(url: str, link_data: Dict[str, Any]):
    """Log interesting information about analyzed links"""
    analysis = link_data['analysis'].get('analyze_archive_it_link', {})
    if not analysis:
        return
        
    # If there are potential issues, log them
    if analysis.get('potential_issues'):
        issues = ', '.join(analysis['potential_issues'])
        logging.info(f"Potential crawler trap at {url}: {issues}")
        
    # Log interesting URL patterns
    if analysis.get('path_segments'):
        path = '/'.join(analysis['path_segments'])
        if 'organization' in path or 'collection' in path:
            logging.info(f"Found detail page: {url}")
        elif 'explore' in path or 'browse' in path:
            logging.info(f"Found list page: {url}")

def crawl_site(seed_url: str, scope_rules: Dict[str, Any], analysis_functions: Optional[List[Callable]] = None, 
               output_handler: Optional[OutputHandler] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Crawl a site starting from seed_url and collect link analysis.
    
    Args:
        seed_url: Starting URL for crawl
        scope_rules: Dictionary containing crawl rules
        analysis_functions: List of functions to analyze links
        output_handler: Optional OutputHandler for saving results
    """
    # Ensure our Archive-It analyzer is included
    if analysis_functions is None:
        analysis_functions = []
    if analyze_archive_it_link not in analysis_functions:
        analysis_functions.append(analyze_archive_it_link)
    
    link_graph = {}
    visited = set()
    frontier_set = set()
    frontier_queue = deque()
    rate_limiter = RateLimiter(delay=scope_rules.get('delay', 1.0))

    add_to_frontier = lambda url: frontier_set.add(url) or frontier_queue.append(url)
    
    def pop_from_frontier():
        url = frontier_queue.popleft() if frontier_queue else None
        if url:
            frontier_set.remove(url)
        return url

    # Seed our crawl
    add_to_frontier(seed_url)
    
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)

        try:
            while page_url := pop_from_frontier():
                if len(visited) >= scope_rules.get('page_limit', 100):
                    logging.info("Reached page limit, stopping crawl")
                    break
                
                if page_url in visited:
                    continue

                logging.info(f"Crawling {page_url}")
                visited.add(page_url)

                # Rate limiting
                rate_limiter.wait()

                try:
                    result = crawl_page(
                        browser, 
                        page_url, 
                        analysis_functions,
                        max_retries=scope_rules.get('max_retries', 3)
                    )
                    logging.info(f"Found {result['outlinks_count']} outlinks on {page_url}")

                    # Add new URLs to the frontier
                    for link in result['outlinks']:
                        # Log analysis results for this link
                        log_link_analysis(link['absolute_url'], link)
                        
                        if (link['absolute_url'] not in visited and 
                            link['absolute_url'] not in frontier_set):
                            if is_url_in_scope(link['absolute_url'], scope_rules):
                                add_to_frontier(link['absolute_url'])

                    # Add the outlink results
                    link_graph[page_url] = result['outlinks']

                    # Save intermediate results if handler provided
                    if output_handler:
                        output_handler.save_json(link_graph, "crawl_results_intermediate.json")

                except NetworkError as e:
                    logging.error(f"Network error crawling {page_url}: {e}")
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error crawling {page_url}: {e}")
                    continue

        finally:
            browser.close()

    # Save final results if handler provided
    if output_handler:
        output_handler.save_json(link_graph, "crawl_results_final.json")
        output_handler.save_csv(link_graph, "crawl_results_final.csv")

    return link_graph