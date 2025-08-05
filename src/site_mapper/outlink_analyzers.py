from playwright.sync_api import Page


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