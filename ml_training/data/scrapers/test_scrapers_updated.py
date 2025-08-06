#!/usr/bin/env python3
"""
Test script for web scrapers - Updated version with better error handling
"""

import sys
import os
import time
from pprint import pprint
import traceback

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import scrapers
try:
    from ml_training.data.scrapers.moneycontrol_scraper import MoneyControlScraper
    from ml_training.data.scrapers.google_finance_scraper import GoogleFinanceScraper
    from ml_training.data.scrapers.cnbc_tv18_scraper import CnbcTv18Scraper
except ImportError as e:
    print(f"Failed to import scrapers: {e}")
    sys.exit(1)

def test_moneycontrol_scraper():
    """Test the MoneyControl scraper"""
    print("\n\n📊 Testing Money Control Scraper")
    print("-" * 40)
    
    try:
        money_control = MoneyControlScraper()
        
        print("\n📰 Getting top news...")
        top_news = money_control.get_top_news(limit=3)
        print(f"✅ Found {len(top_news)} news articles")
        if top_news:
            pprint(top_news[0])
        
        print("\n🏢 Getting stock news for RELIANCE...")
        stock_news = money_control.get_stock_news("RELIANCE", limit=2)
        print(f"✅ Found {len(stock_news)} stock news articles")
        if stock_news:
            pprint(stock_news[0])
        
        print("\n🌡️ Getting market sentiment...")
        sentiment = money_control.get_market_sentiment()
        print("✅ Got market sentiment")
        pprint(sentiment)
    except Exception as e:
        print(f"❌ Error with Money Control scraper: {str(e)}")
        traceback.print_exc()

def test_google_finance_scraper():
    """Test the Google Finance scraper"""
    print("\n\n📊 Testing Google Finance Scraper")
    print("-" * 40)
    
    try:
        google_finance = GoogleFinanceScraper()
        
        print("📈 Getting stock data for RELIANCE...")
        stock_data = google_finance.get_stock_data("RELIANCE")
        print("✅ Got stock data")
        pprint(stock_data)
        
        print("\n📉 Getting index data for NIFTY...")
        nifty_data = google_finance.get_index_data("NIFTY")
        print("✅ Got index data")
        pprint(nifty_data)
        
        print("\n📰 Getting stock news for INFY...")
        infy_news = google_finance.get_stock_news("INFY", limit=2)
        print(f"✅ Found {len(infy_news)} news articles")
        if infy_news:
            pprint(infy_news[0])
        
        print("\n🔍 Getting market trends...")
        trends = google_finance.get_market_trends()
        print("✅ Got market trends")
        pprint(trends)
    except Exception as e:
        print(f"❌ Error with Google Finance scraper: {str(e)}")
        traceback.print_exc()

def test_cnbc_tv18_scraper():
    """Test the CNBC TV18 scraper"""
    print("\n\n📊 Testing CNBC TV18 Scraper")
    print("-" * 40)
    
    try:
        cnbc_tv18 = CnbcTv18Scraper()
        
        print("📰 Getting top news...")
        cnbc_top_news = cnbc_tv18.get_top_news(limit=3)
        print(f"✅ Found {len(cnbc_top_news)} news articles")
        if cnbc_top_news:
            pprint(cnbc_top_news[0])
        
        print("\n👨‍💼 Getting expert opinions...")
        expert_opinions = cnbc_tv18.get_expert_opinions(limit=2)
        print(f"✅ Found {len(expert_opinions)} expert opinions")
        if expert_opinions:
            pprint(expert_opinions[0])
        
        print("\n⚠️ Getting market alerts...")
        alerts = cnbc_tv18.get_market_alerts()
        print(f"✅ Found {len(alerts)} market alerts")
        if alerts:
            pprint(alerts[0])
        
        print("\n🏦 Getting banking sector news...")
        banking_news = cnbc_tv18.get_sector_news("banking", limit=2)
        print(f"✅ Found {len(banking_news)} sector news articles")
        if banking_news:
            pprint(banking_news[0])
    except Exception as e:
        print(f"❌ Error with CNBC TV18 scraper: {str(e)}")
        traceback.print_exc()

def main():
    print("=" * 80)
    print("🔍 TESTING FINANCIAL NEWS SCRAPERS")
    print("=" * 80)
    
    test_moneycontrol_scraper()
    time.sleep(2)  # Pause between tests
    
    test_google_finance_scraper()
    time.sleep(2)  # Pause between tests
    
    test_cnbc_tv18_scraper()
    
    print("\n\n✅ All tests completed")

if __name__ == "__main__":
    main()
