# Financial News Scrapers

This package contains web scrapers for major financial news and data sources:

- **Money Control** - Indian financial news and stock data
- **Google Finance** - Stock prices, charts, and financial news
- **CNBC Network 18** - Financial news, expert opinions, and market analysis

## Features

- Fetch top financial news articles
- Get stock-specific news and data
- Retrieve market sentiment and trends
- Extract expert opinions and market alerts
- Aggregate data from multiple sources

## Usage

```python
from ml_training.data.scrapers import MoneyControlScraper, GoogleFinanceScraper, CnbcTv18Scraper, FinancialNewsAggregator

# Initialize scrapers
money_control = MoneyControlScraper()
google_finance = GoogleFinanceScraper()
cnbc_tv18 = CnbcTv18Scraper()
news_aggregator = FinancialNewsAggregator()

# Get top news
top_news = money_control.get_top_news(limit=5)

# Get stock data
reliance_data = google_finance.get_stock_data("RELIANCE")

# Get expert opinions
expert_opinions = cnbc_tv18.get_expert_opinions(limit=3)

# Get aggregated news from all sources
aggregated_news = await news_aggregator.get_all_top_news(limit=10)
```

## Robustness Features

These scrapers include several features to ensure reliable operation:

- **Error Handling**: Gracefully handles HTTP errors (404, 410, 403, 429)
- **Retry Logic**: Implements exponential backoff for transient errors
- **User-Agent Rotation**: Rotates user agents to avoid being blocked
- **Mock Data Fallbacks**: Provides mock data when live data is unavailable
- **Caching**: Caches results to reduce requests to source websites
- **Defensive Parsing**: Safely extracts data from HTML with proper error handling

## Testing

Run the test script to verify scraper functionality:

```bash
cd ml_training/data/scrapers
python test_scrapers_updated.py
```

## Notes

- These scrapers are for educational and research purposes only
- Be respectful of the websites' robots.txt and terms of service
- Consider implementing rate limiting for production use
