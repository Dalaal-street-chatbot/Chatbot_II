import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import config

class NewsService:
    """Financial news service using News API"""
    
    def __init__(self):
        if not config.NEWS_API:
            raise ValueError("NEWS_API environment variable is required")
        
        self.newsapi = NewsApiClient(api_key=config.NEWS_API)
        self.financial_keywords = [
            'stock market', 'share market', 'sensex', 'nifty', 
            'BSE', 'NSE', 'trading', 'investment', 'mutual funds',
            'IPO', 'FII', 'DII', 'rupee', 'economy', 'GDP'
        ]
    
    def get_financial_news(
        self, 
        query: Optional[str] = None, 
        page_size: int = 10
    ) -> Dict[str, Any]:
        """Get latest financial news"""
        try:
            # Use specific query or general financial terms
            search_query = query or 'indian stock market OR sensex OR nifty'
            
            articles = self.newsapi.get_everything(
                q=search_query,
                language='en',
                sort_by='publishedAt',
                page_size=page_size,
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            )
            
            return {
                'status': 'success',
                'total_results': articles['totalResults'],
                'articles': self._format_articles(articles['articles'])
            }
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return {
                'status': 'error',
                'message': 'Failed to fetch news',
                'articles': []
            }
    
    def get_company_news(self, company: str, page_size: int = 5) -> Dict[str, Any]:
        """Get news for a specific company"""
        try:
            query = f'{company} AND (stock OR share OR market OR trading)'
            
            articles = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                page_size=page_size,
                from_param=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            )
            
            return {
                'status': 'success',
                'company': company,
                'total_results': articles['totalResults'],
                'articles': self._format_articles(articles['articles'])
            }
            
        except Exception as e:
            print(f"Error fetching company news for {company}: {e}")
            return {
                'status': 'error',
                'message': f'Failed to fetch news for {company}',
                'articles': []
            }
    
    def get_market_headlines(self) -> Dict[str, Any]:
        """Get top market headlines from business sources"""
        try:
            headlines = self.newsapi.get_top_headlines(
                category='business',
                country='in',
                page_size=15
            )
            
            return {
                'status': 'success',
                'total_results': headlines['totalResults'],
                'articles': self._format_articles(headlines['articles'])
            }
            
        except Exception as e:
            print(f"Error fetching headlines: {e}")
            return {
                'status': 'error',
                'message': 'Failed to fetch headlines',
                'articles': []
            }
    
    def _format_articles(self, articles: List[Dict]) -> List[Dict[str, Any]]:
        """Format news articles for consistent output"""
        formatted = []
        
        for article in articles:
            formatted.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'image_url': article.get('urlToImage', '')
            })
        
        return formatted
    
    def search_news_by_sentiment(self, sentiment: str = 'positive') -> Dict[str, Any]:
        """Search for news with specific sentiment (positive/negative)"""
        sentiment_keywords = {
            'positive': 'rally OR surge OR gains OR bullish OR growth OR profit',
            'negative': 'fall OR crash OR bearish OR loss OR decline OR drop'
        }
        
        query = f'indian stock market AND ({sentiment_keywords.get(sentiment, sentiment_keywords["positive"])})'
        
        return self.get_financial_news(query=query, page_size=8)
    
    def get_stock_news(
        self, 
        symbol: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Get news for a specific stock
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            limit: Maximum number of articles to return
            
        Returns:
            Dictionary with news articles
        """
        try:
            # Use company name and symbol for better results
            company_name = self._get_company_name(symbol)
            search_query = f"{company_name} OR {symbol} stock"
            
            return self.get_financial_news(search_query, limit)
            
        except Exception as e:
            print(f"Error fetching stock news: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'articles': []
            }
    
    def get_stock_sentiment(
        self, 
        symbol: str
    ) -> Dict[str, Any]:
        """Get sentiment analysis for a specific stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with sentiment analysis
        """
        try:
            # Use simplified sentiment logic based on news headlines
            company_name = self._get_company_name(symbol)
            news = self.get_stock_news(symbol, limit=10)
            
            if news['status'] != 'success':
                return {
                    'status': 'error',
                    'message': news.get('message', 'Failed to get news'),
                    'sentiment': 'neutral',
                    'score': 0.5
                }
            
            # Simple rule-based sentiment analysis
            positive_keywords = ['rise', 'gain', 'jump', 'surge', 'rally', 'bullish', 'up']
            negative_keywords = ['fall', 'drop', 'decline', 'crash', 'bearish', 'down', 'loss']
            
            positive_count = 0
            negative_count = 0
            
            for article in news['articles']:
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                
                for keyword in positive_keywords:
                    if keyword in title or keyword in description:
                        positive_count += 1
                
                for keyword in negative_keywords:
                    if keyword in title or keyword in description:
                        negative_count += 1
            
            # Calculate sentiment score (0.0 to 1.0)
            total = positive_count + negative_count
            if total > 0:
                score = positive_count / total
            else:
                score = 0.5  # Neutral
                
            # Determine sentiment label
            if score > 0.6:
                sentiment = 'positive'
            elif score < 0.4:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'status': 'success',
                'symbol': symbol,
                'company_name': company_name,
                'sentiment': sentiment,
                'score': score,
                'positive_mentions': positive_count,
                'negative_mentions': negative_count,
                'total_articles': len(news['articles'])
            }
            
        except Exception as e:
            print(f"Error analyzing stock sentiment: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'sentiment': 'neutral',
                'score': 0.5
            }
    
    def get_market_news(self) -> Dict[str, Any]:
        """Get general market news
        
        Returns:
            Dictionary with market news
        """
        return self.get_financial_news(query="indian stock market OR sensex OR nifty", page_size=10)
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol"""
        # Basic mapping of some major Indian companies
        company_names = {
            'RELIANCE': 'Reliance Industries',
            'TCS': 'Tata Consultancy Services',
            'HDFCBANK': 'HDFC Bank',
            'INFY': 'Infosys',
            'ICICIBANK': 'ICICI Bank',
            'HINDUNILVR': 'Hindustan Unilever',
            'KOTAKBANK': 'Kotak Mahindra Bank',
            'ITC': 'ITC Limited',
            'SBIN': 'State Bank of India',
            'BAJFINANCE': 'Bajaj Finance',
            'ASIANPAINT': 'Asian Paints'
        }
        
        return company_names.get(symbol.upper(), symbol)
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment
        
        Returns:
            Dictionary with market sentiment
        """
        try:
            # Get news for major indices
            sensex_news = self.get_financial_news("sensex", 5)
            nifty_news = self.get_financial_news("nifty", 5)
            
            # Combine news
            all_articles = sensex_news.get('articles', []) + nifty_news.get('articles', [])
            
            # Simple rule-based sentiment analysis
            positive_keywords = ['rise', 'gain', 'jump', 'surge', 'rally', 'bullish', 'up']
            negative_keywords = ['fall', 'drop', 'decline', 'crash', 'bearish', 'down', 'loss']
            
            positive_count = 0
            negative_count = 0
            
            for article in all_articles:
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                
                for keyword in positive_keywords:
                    if keyword in title or keyword in description:
                        positive_count += 1
                
                for keyword in negative_keywords:
                    if keyword in title or keyword in description:
                        negative_count += 1
            
            # Calculate sentiment score
            total = positive_count + negative_count
            if total > 0:
                score = positive_count / total
            else:
                score = 0.5  # Neutral
                
            # Determine sentiment label
            if score > 0.6:
                sentiment = 'positive'
            elif score < 0.4:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'status': 'success',
                'overall_sentiment': sentiment,
                'overall_score': score,
                'positive_mentions': positive_count,
                'negative_mentions': negative_count,
                'total_articles': len(all_articles)
            }
            
        except Exception as e:
            print(f"Error analyzing market sentiment: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'overall_sentiment': 'neutral',
                'overall_score': 0.5
            }

# Create global instance
news_service = NewsService()
