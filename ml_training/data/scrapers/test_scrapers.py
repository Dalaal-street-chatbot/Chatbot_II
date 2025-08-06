#!/usr/bin/env python3
"""
Test script for web scrapers
Run this script to test the web scrapers and view sample data
"""

import asyncio
import json
import os
import sys
from pprint import pprint

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ml_training.data.scrapers.financial_news_aggregator import FinancialNewsAggregator
from ml_training.data.scrapers.moneycontrol_scraper import MoneyControlScraper
from ml_training.data.scrapers.google_finance_scraper import GoogleFinanceScraper
from ml_training.data.scrapers.cnbc_tv18_scraper import CnbcTv18Scraper

async def test_scrapers():
    """Test all scrapers and display sample data"""
    
    print("=" * 80)
    print("🔍 TESTING FINANCIAL NEWS SCRAPERS")
    print("=" * 80)
    
    # Test Money Control scraper
    print("\n\n📊 Testing Money Control Scraper")
    print("-" * 40)
    
    print("\nGetting top news...")
    top_news = moneycontrol_scraper.get_top_news(limit=3)
    print(f"Retrieved {len(top_news)} articles")
    pprint(top_news[0] if top_news else "No news found")
    
    print("\nGetting stock news for RELIANCE...")
    stock_news = moneycontrol_scraper.get_stock_news("RELIANCE", limit=2)
    print(f"Retrieved {len(stock_news)} articles")
    pprint(stock_news[0] if stock_news else "No news found")
    
    print("\nGetting market sentiment...")
    sentiment = moneycontrol_scraper.get_market_sentiment()
    print(f"Market sentiment: {sentiment.get('sentiment', 'N/A')}")
    print(f"Sentiment score: {sentiment.get('sentiment_score', 'N/A')}")
    print(f"Advances: {sentiment.get('advances', 'N/A')}")
    print(f"Declines: {sentiment.get('declines', 'N/A')}")
    
    # Test Google Finance scraper
    print("\n\n📈 Testing Google Finance Scraper")
    print("-" * 40)
    
    print("\nGetting stock data for RELIANCE...")
    stock_data = google_finance_scraper.get_stock_data("RELIANCE")
    pprint(stock_data)
    
    print("\nGetting NIFTY index data...")
    nifty_data = google_finance_scraper.get_index_data("NIFTY")
    pprint(nifty_data)
    
    print("\nGetting stock news for INFY...")
    infy_news = google_finance_scraper.get_stock_news("INFY", limit=2)
    print(f"Retrieved {len(infy_news)} articles")
    pprint(infy_news[0] if infy_news else "No news found")
    
    print("\nGetting market trends...")
    trends = google_finance_scraper.get_market_trends()
    print("Top Gainers:")
    pprint(trends.get("gainers", [])[:2])
    print("Top Losers:")
    pprint(trends.get("losers", [])[:2])
    
    # Test CNBC TV18 scraper
    print("\n\n📺 Testing CNBC TV18 Scraper")
    print("-" * 40)
    
    print("\nGetting top news...")
    cnbc_top_news = cnbc_tv18_scraper.get_top_news(limit=3)
    print(f"Retrieved {len(cnbc_top_news)} articles")
    pprint(cnbc_top_news[0] if cnbc_top_news else "No news found")
    
    print("\nGetting expert opinions...")
    expert_opinions = cnbc_tv18_scraper.get_expert_opinions(limit=2)
    print(f"Retrieved {len(expert_opinions)} opinions")
    pprint(expert_opinions[0] if expert_opinions else "No opinions found")
    
    print("\nGetting market alerts...")
    alerts = cnbc_tv18_scraper.get_market_alerts()
    print(f"Retrieved {len(alerts)} alerts")
    pprint(alerts[0] if alerts else "No alerts found")
    
    print("\nGetting banking sector news...")
    banking_news = cnbc_tv18_scraper.get_sector_news("banking", limit=2)
    print(f"Retrieved {len(banking_news)} articles")
    pprint(banking_news[0] if banking_news else "No news found")
    
    # Test Financial News Aggregator
    print("\n\n🔄 Testing Financial News Aggregator")
    print("-" * 40)
    
    print("\nGetting all top news...")
    all_news = await financial_news_aggregator.get_all_top_news(limit=5)
    print(f"Retrieved {len(all_news)} articles from all sources")
    pprint(all_news[0] if all_news else "No news found")
    
    print("\nGetting aggregated stock news for TCS...")
    tcs_news = await financial_news_aggregator.get_stock_news("TCS", limit=3)
    print(f"Retrieved {len(tcs_news)} articles for TCS")
    pprint(tcs_news[0] if tcs_news else "No news found")
    
    print("\nGetting aggregated market sentiment...")
    agg_sentiment = await financial_news_aggregator.get_market_sentiment()
    print(f"Overall sentiment: {agg_sentiment.get('overall_sentiment', 'N/A')}")
    print(f"Overall score: {agg_sentiment.get('overall_score', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✅ Testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_scrapers())
