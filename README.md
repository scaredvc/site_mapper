# Site Mapper

Minimal tools to crawl a page, extract links, and classify them with OpenAI.

## Setup (Windows CMD)

```cmd
pip install -U pip
pip install python-dotenv openai pillow playwright
python -m playwright install chromium
set OPENAI_API_KEY=YOUR_KEY
```

Optional: rate limit requests (default 60 req/min)
```cmd
set CLASSIFIER_RPM=30
```

## Quick start

Classify up to N links on a page and print a simple report:
```cmd
python src\site_mapper\test_models.py --url https://example.com --models gpt-4o-mini --max-links 50
```

### Options (flags)

- `--url` (required): Page to crawl (single page only).
- `--models` (optional): Comma-separated model IDs. Defaults to all with keys.
  - Supported: `gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o`
- `--max-links` (optional): Max links to analyze from the page. Default: 20.
- `--output` (optional): Save full JSON report to a file.

### Examples

- Run with 50 links on archive-it:
```cmd
python src\site_mapper\test_models.py --url https://archive-it.org --models gpt-4o-mini --max-links 50
```

- Compare multiple models:
```cmd
python src\site_mapper\test_models.py --url https://example.com --models gpt-3.5-turbo,gpt-4o-mini,gpt-4o --max-links 30
```

- Save results to a file:
```cmd
python src\site_mapper\test_models.py --url https://example.com --models gpt-4o --max-links 40 --output results\classification_results.json
```

## Notes

- Supported models: `gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o`.
- Screenshots saved to `screenshots/` (add `/screenshots/` to `.gitignore`).
- Costs are estimated from token usage via `src/site_mapper/pricing.py`.
- The crawler (`generalized_crawler.py`) uses Playwright and standard analyzers.