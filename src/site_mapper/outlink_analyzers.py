from playwright.sync_api import Page
from urllib.parse import urlparse, parse_qs


def dom_hierarchy(page: Page, element) -> str:
    """Get the DOM hierarchy path to an element as a string"""
    return page.evaluate("""
        (element) => {
            const path = [];
            let current = element;
            while (current && current.nodeType === Node.ELEMENT_NODE) {
                let selector = current.tagName.toLowerCase();
                if (current.id) {
                    selector += '#' + current.id;
                } else if (current.className) {
                    selector += '.' + current.className.split(' ').join('.');
                }
                path.unshift(selector);
                current = current.parentElement;
            }
            return path.join(' > ');
        }
    """, element)


def bounding_box(page: Page, element) -> dict[str, float]:
    """Get the bounding box of an element"""
    return page.evaluate("""
        (element) => {
            const rect = element.getBoundingClientRect();
            return {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            };
        }
    """, element)


def css_classes(page: Page, element) -> list[str]:
    """Get CSS classes of an element"""
    return page.evaluate("(element) => Array.from(element.classList)", element)


def computed_styles(page: Page, element, properties: list[str] = ['color', 'background-color', 'font-size', 'font-weight']) -> dict[str, str]:
    """Get computed CSS styles for specified properties"""
    return page.evaluate("""
        (element, properties) => {
            const styles = window.getComputedStyle(element);
            const result = {};
            properties.forEach(prop => {
                result[prop] = styles.getPropertyValue(prop);
            });
            return result;
        }
    """, element, properties)


def link_position(page: Page, element) -> str:
    """Custom function to determine link position in page"""
    return page.evaluate("""
            (element) => {
                const rect = element.getBoundingClientRect();
                const windowHeight = window.innerHeight;
                const windowWidth = window.innerWidth;
                
                let position = [];
                if (rect.top < windowHeight / 3) position.push('top');
                else if (rect.top > windowHeight * 2/3) position.push('bottom');
                else position.push('middle');
                
                if (rect.left < windowWidth / 3) position.push('left');
                else if (rect.left > windowWidth * 2/3) position.push('right');
                else position.push('center');
                
                return position.join('-');
            }
        """, element)


def parent_elements(page: Page, element) -> list[str]:
    """Get the tag names of parent elements"""
    return page.evaluate("""
            (element) => {
                const parents = [];
                let current = element.parentElement;
                let depth = 0;
                while (current && depth < 5) {  // Limit to 5 levels up
                    parents.push(current.tagName.toLowerCase());
                    current = current.parentElement;
                    depth++;
                }
                return parents;
            }
        """, element)


def analyze_archive_it_link(page: Page, element) -> dict:
    """
    Step 1: Basic analysis of Archive-It links to identify potential crawler traps.
    
    This analyzer focuses on two main aspects:
    1. Is the link part of a faceted search interface?
    2. Does the URL suggest redundant content?
    """
    # Get the href attribute
    href = page.evaluate("(element) => element.getAttribute('href')", element)
    
    # Parse the URL
    parsed_url = urlparse(href)
    query_params = parse_qs(parsed_url.query)
    
    # Check if link is in a faceted search section
    in_faceted_search = page.evaluate("""
        (element) => {
            // Common class names and attributes for faceted search elements
            return Boolean(
                element.closest('.faceted-search, .filters, .sorting, [data-testid*="facet"]')
            );
        }
    """, element)
    
    # Analyze URL characteristics
    analysis = {
        # Faceted search detection
        'in_faceted_search_ui': in_faceted_search,
        
        # URL analysis
        'has_query_params': bool(query_params),
        'query_params': list(query_params.keys()),
        
        # Common patterns that might indicate redundant content
        'has_sort_param': 'sort' in query_params,
        'has_filter_param': 'filter' in query_params,
        'has_page_param': 'page' in query_params,
        'has_show_param': 'show' in query_params,
        
        # Path analysis
        'path': parsed_url.path,
        'path_segments': parsed_url.path.strip('/').split('/')
    }
    
    # Add a human-readable explanation of potential issues
    reasons = []
    if analysis['in_faceted_search_ui']:
        reasons.append("Link is part of faceted search interface")
    if analysis['has_sort_param']:
        reasons.append("URL contains sorting parameter")
    if analysis['has_filter_param']:
        reasons.append("URL contains filter parameter")
    if analysis['has_page_param']:
        reasons.append("URL contains pagination parameter")
    if analysis['has_show_param']:
        reasons.append("URL contains show/display parameter")
        
    analysis['potential_issues'] = reasons
    
    return analysis