"""
Generalized crawler wrapper

Provides a simple `crawl_website(url, max_links)` function used by testing tools.
Internally it uses Playwright and the existing `crawler.crawl_page` plus the
standard analyzers in `outlink_analyzers`.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from playwright.sync_api import sync_playwright

from site_mapper import crawler as base_crawler
from site_mapper.outlink_analyzers import (
    dom_hierarchy,
    bounding_box,
    css_classes,
    link_position,
    parent_elements,
)


def crawl_website(url: str, max_links: int = 20, requests_per_min: int = 30) -> Dict[str, Any]:
    """Crawl a single page and return links and screenshot metadata.

    This is intentionally minimal for testing purposes. It crawls just the
    provided URL, extracts links with analyzers, captures a screenshot, and
    returns a structure compatible with existing testers.
    """
    analyzers = [dom_hierarchy, bounding_box, css_classes, link_position, parent_elements]

    # Simple rate limiting: enforce a minimum interval between page fetches
    # requests_per_min = 30 -> min_interval ≈ 2 seconds
    if requests_per_min <= 0:
        requests_per_min = 1
    min_interval_seconds = 60.0 / float(requests_per_min)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            # Since this minimal crawler only visits one page, we enforce a delay
            # upfront to respect the configured rate. In future multi-page flows
            # this would be applied between successive requests.
            import time
            time.sleep(min_interval_seconds)
            page_result = base_crawler.crawl_page(
                browser=browser,
                url=url,
                analysis_functions=analyzers,
                capture_screenshot=True,
                screenshot_dir="screenshots",
            )
        finally:
            browser.close()

    filtered_links = page_result.get("outlinks", [])[: max_links or 20]
    screenshot_path = page_result.get("screenshot_path")

    return {
        "filtered_links": filtered_links,
        "crawl_metadata": {"screenshot_path": screenshot_path},
    }