"""
Financial News Aggregator - Combines data from multiple financial news sources
and provides a unified interface for accessing market news, alerts, and sentiments.
"""

import asyncio
import pandas as pd
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Import scrapers
from ml_training.data.scrapers.moneycontrol_scraper import moneycontrol_scraper
from ml_training.data.scrapers.google_finance_scraper import google_finance_scraper
from ml_training.data.scrapers.cnbc_tv18_scraper import cnbc_tv18_scraper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialNewsAggregator:
    """Aggregates financial news and data from multiple sources"""
    
    def __init__(self):
        """Initialize financial news aggregator"""
        self.sources = {
            "moneycontrol": moneycontrol_scraper,
            "google_finance": google_finance_scraper,
            "cnbc_tv18": cnbc_tv18_scraper
        }
        self.cache = {}
        self.cache_duration = 1800  # 30 minutes
        logger.info("Financial News Aggregator initialized")
    
    async def get_all_top_news(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Get top financial news from all sources
        
        Args:
            limit: Maximum number of news articles to return per source
            
        Returns:
            List of news articles from all sources
        """
        cache_key = f"all_top_news_{limit}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info("Returning cached all top news")
                return cache_data
        
        all_news = []
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
            # Create futures
            futures = {
                "moneycontrol": executor.submit(self.sources["moneycontrol"].get_top_news, limit),
                "google_finance": executor.submit(lambda: []),  # Google Finance doesn't have a generic top news method
                "cnbc_tv18": executor.submit(self.sources["cnbc_tv18"].get_top_news, limit)
            }
            
            # Collect results
            for source, future in futures.items():
                try:
                    news = future.result()
                    all_news.extend(news)
                except Exception as e:
                    logger.error(f"Error getting top news from {source}: {e}")
        
        # Sort by timestamp, newest first
        all_news.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Remove duplicates based on title similarity
        unique_news = self._remove_duplicate_news(all_news)
        
        # Cache the results
        self.cache[cache_key] = (datetime.now(), unique_news)
        logger.info(f"Fetched {len(unique_news)} total top news articles from all sources")
        
        return unique_news
    
    async def get_stock_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get news for a specific stock from all sources
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            limit: Maximum number of news articles to return per source
            
        Returns:
            List of news articles for the stock
        """
        cache_key = f"stock_news_{symbol}_{limit}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info(f"Returning cached news for {symbol}")
                return cache_data
        
        stock_news = []
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
            # Create futures
            futures = {
                "moneycontrol": executor.submit(self.sources["moneycontrol"].get_stock_news, symbol, limit),
                "google_finance": executor.submit(self.sources["google_finance"].get_stock_news, symbol, limit),
                "cnbc_tv18": executor.submit(lambda: [])  # CNBC TV18 doesn't have a dedicated stock news method
            }
            
            # Collect results
            for source, future in futures.items():
                try:
                    news = future.result()
                    stock_news.extend(news)
                except Exception as e:
                    logger.error(f"Error getting stock news for {symbol} from {source}: {e}")
        
        # Sort by timestamp, newest first
        stock_news.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Remove duplicates
        unique_news = self._remove_duplicate_news(stock_news)
        
        # Cache the results
        self.cache[cache_key] = (datetime.now(), unique_news)
        logger.info(f"Fetched {len(unique_news)} total news articles for {symbol} from all sources")
        
        return unique_news
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get aggregated market sentiment from multiple sources
        
        Returns:
            Aggregated market sentiment data
        """
        cache_key = "market_sentiment"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info("Returning cached market sentiment")
                return cache_data
        
        # Get sentiment from Money Control
        try:
            moneycontrol_sentiment = self.sources["moneycontrol"].get_market_sentiment()
        except Exception as e:
            logger.error(f"Error getting market sentiment from Money Control: {e}")
            moneycontrol_sentiment = {
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "advances": 0,
                "declines": 0
            }
        
        # Get market trends from Google Finance
        try:
            google_trends = self.sources["google_finance"].get_market_trends()
            
            # Calculate sentiment from Google trends
            google_gainers_count = len(google_trends.get("gainers", []))
            google_losers_count = len(google_trends.get("losers", []))
            
            total_movers = google_gainers_count + google_losers_count
            google_sentiment = "neutral"
            google_sentiment_score = 0.0
            
            if total_movers > 0:
                google_sentiment_score = (google_gainers_count - google_losers_count) / total_movers
                
                if google_sentiment_score > 0.1:
                    google_sentiment = "bullish"
                elif google_sentiment_score < -0.1:
                    google_sentiment = "bearish"
        
        except Exception as e:
            logger.error(f"Error getting market trends from Google Finance: {e}")
            google_sentiment = "neutral"
            google_sentiment_score = 0.0
            google_gainers_count = 0
            google_losers_count = 0
        
        # Get market alerts from CNBC TV18 for additional sentiment context
        try:
            cnbc_alerts = self.sources["cnbc_tv18"].get_market_alerts()
            
            # Simple sentiment analysis on alerts
            cnbc_sentiment = "neutral"
            cnbc_sentiment_score = 0.0
            
            positive_terms = ["rise", "gain", "up", "higher", "surge", "jump", "rally", "bullish", "positive"]
            negative_terms = ["fall", "drop", "down", "lower", "plunge", "dip", "bearish", "negative"]
            
            positive_count = 0
            negative_count = 0
            
            for alert in cnbc_alerts:
                text = alert.get("alert_text", "").lower()
                for term in positive_terms:
                    if term in text:
                        positive_count += 1
                for term in negative_terms:
                    if term in text:
                        negative_count += 1
            
            if positive_count > negative_count:
                cnbc_sentiment = "bullish"
                cnbc_sentiment_score = min(0.5, (positive_count - negative_count) / max(1, len(cnbc_alerts)))
            elif negative_count > positive_count:
                cnbc_sentiment = "bearish"
                cnbc_sentiment_score = min(0.5, (negative_count - positive_count) / max(1, len(cnbc_alerts)))
        
        except Exception as e:
            logger.error(f"Error analyzing CNBC TV18 alerts for sentiment: {e}")
            cnbc_sentiment = "neutral"
            cnbc_sentiment_score = 0.0
        
        # Aggregate sentiment from all sources
        # Weight: MoneyControl (0.5), Google Finance (0.3), CNBC TV18 (0.2)
        combined_sentiment_score = (
            moneycontrol_sentiment.get("sentiment_score", 0) * 0.5 + 
            google_sentiment_score * 0.3 + 
            cnbc_sentiment_score * 0.2
        )
        
        combined_sentiment = "neutral"
        if combined_sentiment_score > 0.1:
            combined_sentiment = "bullish"
        elif combined_sentiment_score < -0.1:
            combined_sentiment = "bearish"
        
        # Build the aggregated sentiment result
        aggregated_sentiment = {
            "timestamp": datetime.now().isoformat(),
            "overall_sentiment": combined_sentiment,
            "overall_score": round(combined_sentiment_score, 2),
            "sources": {
                "moneycontrol": {
                    "sentiment": moneycontrol_sentiment.get("sentiment"),
                    "score": moneycontrol_sentiment.get("sentiment_score"),
                    "advances": moneycontrol_sentiment.get("advances"),
                    "declines": moneycontrol_sentiment.get("declines")
                },
                "google_finance": {
                    "sentiment": google_sentiment,
                    "score": round(google_sentiment_score, 2),
                    "gainers_count": google_gainers_count,
                    "losers_count": google_losers_count
                },
                "cnbc_tv18": {
                    "sentiment": cnbc_sentiment,
                    "score": round(cnbc_sentiment_score, 2),
                    "alert_count": len(cnbc_alerts) if 'cnbc_alerts' in locals() else 0
                }
            }
        }
        
        # Cache the results
        self.cache[cache_key] = (datetime.now(), aggregated_sentiment)
        logger.info("Aggregated market sentiment from all sources")
        
        return aggregated_sentiment
    
    async def get_market_alerts(self) -> List[Dict[str, Any]]:
        """Get all market alerts from CNBC TV18
        
        Returns:
            List of market alerts
        """
        try:
            alerts = self.sources["cnbc_tv18"].get_market_alerts()
            logger.info(f"Fetched {len(alerts)} market alerts from CNBC TV18")
            return alerts
        except Exception as e:
            logger.error(f"Error getting market alerts: {e}")
            return []
    
    async def get_expert_opinions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get expert opinions from CNBC TV18
        
        Args:
            limit: Maximum number of opinions to return
            
        Returns:
            List of expert opinions
        """
        try:
            opinions = self.sources["cnbc_tv18"].get_expert_opinions(limit)
            logger.info(f"Fetched {len(opinions)} expert opinions from CNBC TV18")
            return opinions
        except Exception as e:
            logger.error(f"Error getting expert opinions: {e}")
            return []
    
    async def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get stock data from Google Finance
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            
        Returns:
            Stock data
        """
        try:
            data = self.sources["google_finance"].get_stock_data(symbol)
            logger.info(f"Fetched stock data for {symbol} from Google Finance")
            return data
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_index_data(self, index: str = "NIFTY") -> Dict[str, Any]:
        """Get index data from Google Finance
        
        Args:
            index: Index name (e.g., "NIFTY" or "SENSEX")
            
        Returns:
            Index data
        """
        try:
            data = self.sources["google_finance"].get_index_data(index)
            logger.info(f"Fetched data for {index} index from Google Finance")
            return data
        except Exception as e:
            logger.error(f"Error getting data for {index} index: {e}")
            return {
                "symbol": index,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_sector_news(self, sector: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get news for a specific sector from CNBC TV18
        
        Args:
            sector: Sector name (e.g., "banking", "auto", "technology")
            limit: Maximum number of news articles to return
            
        Returns:
            List of news articles for the sector
        """
        try:
            news = self.sources["cnbc_tv18"].get_sector_news(sector, limit)
            logger.info(f"Fetched {len(news)} news articles for {sector} sector from CNBC TV18")
            return news
        except Exception as e:
            logger.error(f"Error getting news for {sector} sector: {e}")
            return []
    
    def clear_all_caches(self):
        """Clear caches of all scrapers and the aggregator"""
        for source_name, source in self.sources.items():
            try:
                source.clear_cache()
                logger.info(f"Cleared cache for {source_name}")
            except Exception as e:
                logger.error(f"Error clearing cache for {source_name}: {e}")
        
        self.cache = {}
        logger.info("Cleared aggregator cache")
    
    def _remove_duplicate_news(self, news_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate news articles based on title similarity
        
        Args:
            news_articles: List of news articles
            
        Returns:
            List of unique news articles
        """
        if not news_articles:
            return []
        
        unique_articles = []
        titles = []
        
        for article in news_articles:
            title = article.get("title", "").lower()
            is_duplicate = False
            
            # Check if this title is similar to any we've already seen
            for existing_title in titles:
                if self._calculate_similarity(title, existing_title) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                titles.append(title)
        
        return unique_articles
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Jaccard similarity
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Tokenize strings into words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1) + len(words2) - intersection
        
        if union == 0:
            return 0
        
        return intersection / union

# Create singleton instance
financial_news_aggregator = FinancialNewsAggregator()
