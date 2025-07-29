import json
from collections import deque
from pprint import pprint
from urllib.parse import urljoin, urlparse
from typing import Any, Callable

from playwright.sync_api import Page, sync_playwright, Browser

from site_mapper.outlink_analyzers import *


def is_url_in_scope(url: str, scope_rules: dict) -> bool:
    """Check if a URL is in scope based on scope rules"""
    allowed_hosts = scope_rules.get('allowed_hosts', [])
    if not allowed_hosts:
        return True
    parsed_url = urlparse(url)
    return parsed_url.hostname in allowed_hosts


def extract_outlinks_with_analysis(page: Page, base_url: str, analysis_functions: list[Callable] = None) -> list[dict[str, Any]]:
    """
    Extract all outlinks from the page and run analysis functions on each link element.

    Returns:
        List of dictionaries containing link data and analysis results
    """
    if analysis_functions is None:
        analysis_functions = []

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
                    print(f"Error running {analysis_func.__name__} on link {absolute_url}: {e}")
                    link_data['analysis'][func_name] = None

            outlinks.append(link_data)

        except Exception as e:
            print(f"Error processing link element: {e}")
            continue

    return outlinks


def crawl_page(browser: Browser, url: str, analysis_functions: list[Callable] = None):
    """
    Crawl a page and extract outlinks with optional analysis functions.

    Args:
        browser: Playwright browser instance
        url: URL to crawl
        analysis_functions: List of functions to run on each outlink element

    Returns:
        Dictionary containing page content and extracted outlinks with analysis
    """
    page = browser.new_page()
    try:
        page.goto(url)

        # Extract outlinks with analysis
        outlinks = extract_outlinks_with_analysis(page, url, analysis_functions)

        return {
            'url': url,
            'outlinks': outlinks,
            'outlinks_count': len(outlinks)
        }
    finally:
        page.close()


def crawl_site(seed_url, scope_rules, analysis_functions: list[Callable] = None):
    link_graph = {}  # dict of visited URLs to a list of outlinks with analysis
    visited = set()
    frontier_set = set()
    frontier_queue = deque()
    add_to_frontier = lambda url: frontier_set.add(url) or frontier_queue.append(url)
    def pop_from_frontier():
        url = frontier_queue.popleft() if frontier_queue else None
        if url:
            frontier_set.remove(url)
        return url

    # seed our crawl with the seed_url
    add_to_frontier(seed_url)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)

        while page_url := pop_from_frontier():
            if len(visited) >= scope_rules.get('page_limit', 100):
                break
            if page_url in visited:
                continue

            print(f"Crawling {page_url}")
            visited.add(page_url)

            try:
                result = crawl_page(browser, page_url, analysis_functions)
                print(f"Found {result['outlinks_count']} outlinks")

                # Add new URLs to the frontier (you can add filtering logic here)
                for link in result['outlinks']:
                    link["should_be_crawled"] = True
                    if link['absolute_url'] not in visited and link['absolute_url'] not in frontier_set:
                        if is_url_in_scope(link['absolute_url'], scope_rules):
                            add_to_frontier(link['absolute_url'])

                # Add the outlink results for analysis
                link_graph[page_url] = result['outlinks']

            except Exception as e:
                print(f"Error crawling {page_url}: {e}")

        browser.close()

    return link_graph


# Example of how to use with custom analysis functions
if __name__ == "__main__":
    # Define custom analysis functions


    # Use custom analysis functions
    analyzers = [
        dom_hierarchy,
        bounding_box,
        css_classes,
        link_position,
        parent_elements
    ]

    scope_rules = {
        "page_limit": 4,
        "allowed_hosts": ["archive-it.org"],
    }

    link_graph = crawl_site("https://archive-it.org", scope_rules, analyzers)
    print(json.dumps(link_graph, indent=4))