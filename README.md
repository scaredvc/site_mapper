# site-mapper

## Setup

Modify settings in crawler.py then

```shell
uv run python -m site_mapper.crawler
```

## Settings

See the scope_rules dictionary at the bottom of crawler.py.

For quick testing, it's set to stop the crawl after 4 pages. You can change that by
modifying the `scope_rules["page_limit"]` key.

After the page_limit is reached the crawler will print a report of every page it 
visited, along with the outlinks it found on each of those pages.