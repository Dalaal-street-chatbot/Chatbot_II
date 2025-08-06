"""
Enhanced News Service - Integrates web scrapers for real-time market news and sentiment
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.news_service import NewsService
from ml_training.data.scrapers import (
    MoneyControlScraper, 
    GoogleFinanceScraper, 
    CnbcTv18Scraper,
    FinancialNewsAggregator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedNewsService(NewsService):
    """Enhanced news service that combines NewsAPI with web scrapers"""
    
    def __init__(self):
        """Initialize the enhanced news service"""
        try:
            super().__init__()
            self.news_aggregator = FinancialNewsAggregator()
            self.money_control = MoneyControlScraper()
            self.google_finance = GoogleFinanceScraper()
            self.cnbc_tv18 = CnbcTv18Scraper()
            self.cache = {}
            self.cache_duration = 900  # 15 minutes
            logger.info("Enhanced News Service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Enhanced News Service: {e}")
            raise
    
    async def get_enhanced_financial_news(
        self, 
        query: Optional[str] = None, 
        page_size: int = 10,
        use_scrapers: bool = True
    ) -> Dict[str, Any]:
        """Get enhanced financial news from multiple sources
        
        Args:
            query: Optional search query
            page_size: Number of articles to return
            use_scrapers: Whether to use web scrapers in addition to NewsAPI
            
        Returns:
            Dictionary with news articles
        """
        cache_key = f"enhanced_news_{query}_{page_size}_{use_scrapers}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info("Returning cached enhanced financial news")
                return cache_data
        
        # Get news from NewsAPI
        api_news = self.get_financial_news(query, page_size)
        
        if use_scrapers:
            try:
                # Get news from scrapers
                scraped_news = await self.news_aggregator.get_all_top_news(limit=page_size)
                
                # Combine results
                combined_articles = api_news['articles'] + self._convert_scraped_news(scraped_news)
                
                # Sort by date, newest first
                combined_articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
                
                # Limit to requested page size
                combined_articles = combined_articles[:page_size]
                
                result = {
                    'status': 'success',
                    'source': 'combined',
                    'total_results': len(combined_articles),
                    'articles': combined_articles
                }
            except Exception as e:
                logger.error(f"Error fetching scraped news: {e}")
                # Fallback to API news only
                result = api_news
                result['source'] = 'newsapi_only'
        else:
            result = api_news
            result['source'] = 'newsapi_only'
        
        # Cache results
        self.cache[cache_key] = (datetime.now(), result)
        return result
    
    async def get_enhanced_company_news(
        self, 
        company: str, 
        page_size: int = 5,
        use_scrapers: bool = True
    ) -> Dict[str, Any]:
        """Get enhanced company news from multiple sources
        
        Args:
            company: Company name or symbol
            page_size: Number of articles to return
            use_scrapers: Whether to use web scrapers in addition to NewsAPI
            
        Returns:
            Dictionary with company news
        """
        cache_key = f"enhanced_company_news_{company}_{page_size}_{use_scrapers}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info(f"Returning cached enhanced news for {company}")
                return cache_data
        
        # Get news from NewsAPI
        api_news = self.get_company_news(company, page_size)
        
        if use_scrapers:
            try:
                # Get news from scrapers
                scraped_news = await self.news_aggregator.get_stock_news(company, limit=page_size)
                
                # Combine results
                combined_articles = api_news['articles'] + self._convert_scraped_news(scraped_news)
                
                # Sort by date, newest first
                combined_articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
                
                # Limit to requested page size
                combined_articles = combined_articles[:page_size]
                
                result = {
                    'status': 'success',
                    'company': company,
                    'source': 'combined',
                    'total_results': len(combined_articles),
                    'articles': combined_articles
                }
            except Exception as e:
                logger.error(f"Error fetching scraped company news: {e}")
                # Fallback to API news only
                result = api_news
                result['source'] = 'newsapi_only'
        else:
            result = api_news
            result['source'] = 'newsapi_only'
        
        # Cache results
        self.cache[cache_key] = (datetime.now(), result)
        return result
    
    async def get_enhanced_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment analysis from scrapers and APIs
        
        Returns:
            Market sentiment data
        """
        try:
            cache_key = "market_sentiment"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_duration:
                    return {
                        "status": "success",
                        "source": "cache",
                        "sentiment": cache_data
                    }
                    
            # Get sentiment from news aggregator
            sentiment = await self.news_aggregator.get_market_sentiment()
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), sentiment)
            
            return {
                "status": "success",
                "source": "scrapers",
                "sentiment": sentiment
            }
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {
                "status": "error",
                "message": str(e),
                "sentiment": {"sentiment": "neutral", "sentiment_score": 0.0}
            }
            
    async def get_expert_insights(self, limit: int = 5) -> Dict[str, Any]:
        """Get expert opinions and market insights
        
        Args:
            limit: Maximum number of insights to return
            
        Returns:
            Expert insights data
        """
        try:
            cache_key = f"expert_insights_{limit}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_duration:
                    return {
                        "status": "success",
                        "source": "cache",
                        "totalResults": len(cache_data),
                        "insights": cache_data
                    }
            
            # Get expert opinions from CNBC TV18
            expert_opinions = self.cnbc_tv18.get_expert_opinions(limit=limit)
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), expert_opinions)
            
            return {
                "status": "success",
                "source": "scrapers",
                "totalResults": len(expert_opinions),
                "insights": expert_opinions
            }
        except Exception as e:
            logger.error(f"Error getting expert insights: {e}")
            return {
                "status": "error",
                "message": str(e),
                "totalResults": 0,
                "insights": []
            }
            
    async def get_market_alerts(self) -> Dict[str, Any]:
        """Get important market alerts
        
        Returns:
            Market alerts data
        """
        try:
            cache_key = "market_alerts"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_duration:
                    return {
                        "status": "success",
                        "source": "cache",
                        "totalResults": len(cache_data),
                        "alerts": cache_data
                    }
            
            # Get market alerts from CNBC TV18
            alerts = self.cnbc_tv18.get_market_alerts()
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), alerts)
            
            return {
                "status": "success",
                "source": "scrapers",
                "totalResults": len(alerts),
                "alerts": alerts
            }
        except Exception as e:
            logger.error(f"Error getting market alerts: {e}")
            return {
                "status": "error",
                "message": str(e),
                "totalResults": 0,
                "alerts": []
            }
            
    async def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get stock data for a specific symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stock data
        """
        try:
            cache_key = f"stock_data_{symbol}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_duration:
                    return {
                        "status": "success",
                        "source": "cache",
                        "data": cache_data
                    }
            
            # Try Google Finance first
            stock_data = self.google_finance.get_stock_data(symbol)
            
            # If no price, try Money Control as backup
            if not stock_data.get("price"):
                try:
                    # Use get_stock_news instead of get_stock_quote since that's what's available
                    mc_news = self.money_control.get_stock_news(symbol, limit=1)
                    if mc_news:
                        # Add the stock news to our data
                        stock_data["latest_news"] = mc_news
                except Exception as mc_error:
                    logger.warning(f"Error getting Money Control data for {symbol}: {mc_error}")
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), stock_data)
            
            return {
                "status": "success",
                "source": "scrapers",
                "data": stock_data
            }
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "data": {}
            }
            
    async def get_market_dashboard(self) -> Dict[str, Any]:
        """Get a complete market dashboard with data from all sources
        
        Returns:
            Market dashboard data
        """
        try:
            # Run all these tasks in parallel
            top_news_task = asyncio.create_task(self.news_aggregator.get_all_top_news(limit=5))
            sentiment_task = asyncio.create_task(self.news_aggregator.get_market_sentiment())
            
            # These are synchronous, so we'll run them directly
            market_trends = self.google_finance.get_market_trends()
            
            # Use get_market_sentiment from MoneyControlScraper instead of get_market_summary
            market_summary = self.money_control.get_market_sentiment()
            
            expert_opinions = self.cnbc_tv18.get_expert_opinions(limit=3)
            alerts = self.cnbc_tv18.get_market_alerts()
            
            # Wait for async tasks to complete
            top_news = await top_news_task
            sentiment = await sentiment_task
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "top_news": top_news,
                "market_sentiment": sentiment,
                "market_trends": market_trends,
                "market_summary": market_summary,
                "expert_opinions": expert_opinions,
                "market_alerts": alerts
            }
        except Exception as e:
            logger.error(f"Error fetching market dashboard: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    async def get_market_sentiment_analysis(self) -> Dict[str, Any]:
        """Get comprehensive market sentiment analysis
        
        Returns:
            Dictionary with market sentiment analysis
        """
        cache_key = "market_sentiment"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info("Returning cached market sentiment")
                return cache_data
        
        try:
            # Get market sentiment from aggregator
            sentiment = await self.news_aggregator.get_market_sentiment()
            
            result = {
                'status': 'success',
                'timestamp': sentiment.get('timestamp', datetime.now().isoformat()),
                'overall_sentiment': sentiment.get('overall_sentiment', 'neutral'),
                'overall_score': sentiment.get('overall_score', 0.0),
                'sources': sentiment.get('sources', {}),
                'market_summary': self._generate_market_summary(sentiment)
            }
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {e}")
            result = {
                'status': 'error',
                'message': 'Failed to fetch market sentiment',
                'overall_sentiment': 'neutral',
                'overall_score': 0.0
            }
        
        # Cache results
        self.cache[cache_key] = (datetime.now(), result)
        return result
    
    async def get_expert_opinions(self, limit: int = 5) -> Dict[str, Any]:
        """Get expert opinions on the market
        
        Args:
            limit: Number of opinions to return
            
        Returns:
            Dictionary with expert opinions
        """
        cache_key = f"expert_opinions_{limit}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info("Returning cached expert opinions")
                return cache_data
        
        try:
            # Get expert opinions from aggregator
            opinions = await self.news_aggregator.get_expert_opinions(limit)
            
            result = {
                'status': 'success',
                'total_results': len(opinions),
                'opinions': opinions
            }
        except Exception as e:
            logger.error(f"Error fetching expert opinions: {e}")
            result = {
                'status': 'error',
                'message': 'Failed to fetch expert opinions',
                'opinions': []
            }
        
        # Cache results
        self.cache[cache_key] = (datetime.now(), result)
        return result
    
    async def get_realtime_market_alerts(self) -> Dict[str, Any]:
        """Get latest real-time market alerts
        
        Returns:
            Dictionary with real-time market alerts
        """
        cache_key = "realtime_market_alerts"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration / 3:  # Shorter cache for alerts
                logger.info("Returning cached real-time market alerts")
                return cache_data
        
        try:
            # Get market alerts from aggregator
            alerts = await self.news_aggregator.get_market_alerts()
            
            result = {
                'status': 'success',
                'total_results': len(alerts),
                'alerts': alerts
            }
        except Exception as e:
            logger.error(f"Error fetching market alerts: {e}")
            result = {
                'status': 'error',
                'message': 'Failed to fetch market alerts',
                'alerts': []
            }
        
        # Cache results
        self.cache[cache_key] = (datetime.now(), result)
        return result
    
    async def get_sector_news(self, sector: str, limit: int = 5) -> Dict[str, Any]:
        """Get news for a specific sector
        
        Args:
            sector: Sector name (e.g., "banking", "auto", "technology")
            limit: Number of articles to return
            
        Returns:
            Dictionary with sector news
        """
        cache_key = f"sector_news_{sector}_{limit}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info(f"Returning cached news for {sector} sector")
                return cache_data
        
        try:
            # Get sector news from aggregator
            news = await self.news_aggregator.get_sector_news(sector, limit)
            
            result = {
                'status': 'success',
                'sector': sector,
                'total_results': len(news),
                'articles': self._convert_scraped_news(news)
            }
        except Exception as e:
            logger.error(f"Error fetching news for {sector} sector: {e}")
            result = {
                'status': 'error',
                'message': f'Failed to fetch news for {sector} sector',
                'articles': []
            }
        
        # Cache results
        self.cache[cache_key] = (datetime.now(), result)
        return result
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
        logger.info("Enhanced News Service cache cleared")
    
    def _convert_scraped_news(self, scraped_news: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert scraped news to NewsAPI format
        
        Args:
            scraped_news: List of news articles from scrapers
            
        Returns:
            List of articles in NewsAPI format
        """
        articles = []
        
        for article in scraped_news:
            # Map scraped news fields to NewsAPI format
            formatted_article = {
                'source': {'id': article.get('source', 'scraper').lower(), 'name': article.get('source', 'Financial Scraper')},
                'author': article.get('expert_name', 'Financial Reporter'),
                'title': article.get('title', ''),
                'description': article.get('summary', ''),
                'url': article.get('url', ''),
                'urlToImage': article.get('image_url', ''),
                'publishedAt': article.get('timestamp', datetime.now().isoformat()),
                'content': article.get('summary', '')
            }
            
            articles.append(formatted_article)
        
        return articles
    
    def _generate_market_summary(self, sentiment: Dict[str, Any]) -> str:
        """Generate a market summary based on sentiment data
        
        Args:
            sentiment: Market sentiment data
            
        Returns:
            Market summary text
        """
        overall = sentiment.get('overall_sentiment', 'neutral')
        score = sentiment.get('overall_score', 0.0)
        
        # Get source-specific data
        mc_data = sentiment.get('sources', {}).get('moneycontrol', {})
        advances = mc_data.get('advances', 0)
        declines = mc_data.get('declines', 0)
        
        if overall == 'bullish':
            summary = f"The market is currently bullish (sentiment score: {score:.2f}). "
            summary += f"Advances ({advances}) are outpacing declines ({declines}). "
            summary += "Investors are showing confidence in the overall market direction."
        elif overall == 'bearish':
            summary = f"The market is currently bearish (sentiment score: {score:.2f}). "
            summary += f"Declines ({declines}) are outpacing advances ({advances}). "
            summary += "Investors are showing caution in the current market conditions."
        else:
            summary = f"The market is currently neutral (sentiment score: {score:.2f}). "
            summary += f"Advances ({advances}) and declines ({declines}) are relatively balanced. "
            summary += "Investors are showing mixed sentiment in the current market."
        
        return summary

# Create singleton instance
enhanced_news_service = EnhancedNewsService()
