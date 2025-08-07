#!/usr/bin/env python3
"""
Comprehensive Financial Data Service
Integrates Tavily search, Upstox API, NSE/BSE data, and database storage
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
import json

from .indian_stock_database import indian_stock_db, StockData, HistoricalData, NewsData
from .upstox_integration import upstox_service
from config.api_config import config

logger = logging.getLogger(__name__)

class ComprehensiveFinancialService:
    """Main service for all financial data operations"""
    
    def __init__(self):
        self.db_service = indian_stock_db
        self.upstox = upstox_service
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        await self.db_service.__aenter__()
        await self.upstox.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        await self.db_service.__aexit__(exc_type, exc_val, exc_tb)
        await self.upstox.__aexit__(exc_type, exc_val, exc_tb)
    
    async def setup_upstox_authentication(self, api_key: str, api_secret: str, redirect_uri: str) -> str:
        """Setup Upstox authentication and return authorization URL"""
        # Update configuration
        self.config.UPSTOX_API_KEY = api_key
        self.config.UPSTOX_API_SECRET = api_secret
        self.config.UPSTOX_REDIRECT_URI = redirect_uri
        
        # Get authorization URL
        auth_url = self.upstox.get_authorization_url()
        logger.info(f"Upstox authorization URL generated: {auth_url}")
        return auth_url
    
    async def complete_upstox_authentication(self, authorization_code: str) -> bool:
        """Complete Upstox OAuth flow with authorization code"""
        try:
            auth_response = await self.upstox.exchange_code_for_token(authorization_code)
            if auth_response:
                # Store token in database service
                self.db_service.upstox_access_token = auth_response.access_token
                logger.info("Upstox authentication completed successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Upstox authentication failed: {e}")
            return False
    
    async def get_comprehensive_stock_data(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Get comprehensive stock data from all available sources"""
        try:
            # Get real-time data
            stock_data = await self.db_service.get_stock_data_with_fallback(symbol, exchange)
            
            # Get historical data (last 30 days)
            historical_data = await self.db_service.get_historical_data(symbol, days=30, exchange=exchange)
            
            # Get related news
            news_data = await self.db_service.search_financial_news(f"{symbol} stock", max_results=5)
            
            # Get Upstox-specific data if available
            upstox_data = None
            if self.upstox.is_token_valid():
                instrument_key = f"{exchange}_EQ|{symbol}"
                upstox_data = await self.upstox.get_full_market_quote(instrument_key)
            
            return {
                'symbol': symbol,
                'exchange': exchange,
                'current_data': stock_data.__dict__ if stock_data else None,
                'historical_data': [h.__dict__ for h in historical_data],
                'news': [n.__dict__ for n in news_data],
                'upstox_data': upstox_data,
                'last_updated': datetime.now().isoformat(),
                'data_sources': self._get_active_data_sources()
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive stock data: {e}")
            return {'error': str(e)}
    
    async def search_and_analyze_stock(self, query: str) -> Dict[str, Any]:
        """Search for stocks and provide comprehensive analysis"""
        try:
            # Search for financial information using Tavily
            search_results = await self.db_service.search_financial_news(query, max_results=10)
            
            # Extract potential stock symbols from search results
            potential_symbols = self._extract_stock_symbols(search_results)
            
            # Get data for identified symbols
            stock_analyses = {}
            for symbol in potential_symbols[:5]:  # Limit to top 5 symbols
                stock_data = await self.get_comprehensive_stock_data(symbol)
                if stock_data and not stock_data.get('error'):
                    stock_analyses[symbol] = stock_data
            
            # Perform sentiment analysis on news
            sentiment_analysis = self._analyze_news_sentiment(search_results)
            
            return {
                'query': query,
                'search_results': [s.__dict__ for s in search_results],
                'identified_symbols': potential_symbols,
                'stock_analyses': stock_analyses,
                'sentiment_analysis': sentiment_analysis,
                'recommendation': self._generate_recommendation(stock_analyses, sentiment_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in search and analysis: {e}")
            return {'error': str(e)}
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market overview"""
        try:
            # Get major indices and stocks
            major_symbols = [
                'NIFTY50', 'SENSEX', 'BANKNIFTY',  # Indices
                'RELIANCE', 'TCS', 'INFY', 'HDFC', 'SBIN', 'ITC',  # Major stocks
                'BAJFINANCE', 'LT', 'WIPRO', 'HCLTECH'
            ]
            
            market_data = {}
            for symbol in major_symbols:
                try:
                    data = await self.db_service.get_stock_data_with_fallback(symbol)
                    if data:
                        market_data[symbol] = {
                            'price': data.price,
                            'change': data.change,
                            'change_percent': data.change_percent,
                            'volume': data.volume,
                            'timestamp': data.timestamp.isoformat()
                        }
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
            
            # Get market news
            market_news = await self.db_service.search_financial_news("Indian stock market today", max_results=10)
            
            # Get sector analysis
            sector_analysis = await self._get_sector_analysis()
            
            # Calculate market summary statistics
            market_stats = self._calculate_market_statistics(market_data)
            
            return {
                'market_data': market_data,
                'market_news': [n.__dict__ for n in market_news],
                'sector_analysis': sector_analysis,
                'market_statistics': market_stats,
                'market_sentiment': self._analyze_news_sentiment(market_news),
                'last_updated': datetime.now().isoformat(),
                'data_sources': self._get_active_data_sources()
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {'error': str(e)}
    
    async def get_portfolio_analysis(self) -> Dict[str, Any]:
        """Get portfolio analysis using Upstox data"""
        if not self.upstox.is_token_valid():
            return {'error': 'Upstox authentication required for portfolio analysis'}
        
        try:
            # Get positions and holdings
            positions = await self.upstox.get_positions()
            holdings = await self.upstox.get_holdings()
            funds = await self.upstox.get_funds_and_margin()
            
            # Calculate portfolio statistics
            portfolio_stats = self._calculate_portfolio_statistics(positions, holdings)
            
            # Get current market data for portfolio instruments
            portfolio_instruments = []
            if positions:
                portfolio_instruments.extend([p.instrument_token for p in positions])
            if holdings:
                portfolio_instruments.extend([h.instrument_token for h in holdings])
            
            current_market_data = {}
            for instrument in set(portfolio_instruments):
                try:
                    market_data = await self.upstox.get_market_quote(instrument)
                    if market_data:
                        current_market_data[instrument] = market_data
                except Exception as e:
                    logger.error(f"Error fetching market data for {instrument}: {e}")
            
            return {
                'positions': [p.__dict__ for p in positions] if positions else [],
                'holdings': [h.__dict__ for h in holdings] if holdings else [],
                'funds_and_margin': funds,
                'portfolio_statistics': portfolio_stats,
                'current_market_data': current_market_data,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
            return {'error': str(e)}
    
    async def execute_trade_analysis(self, symbol: str, quantity: int, 
                                   transaction_type: str, order_type: str = "MARKET") -> Dict[str, Any]:
        """Analyze a potential trade before execution"""
        try:
            # Get comprehensive stock data
            stock_analysis = await self.get_comprehensive_stock_data(symbol)
            
            # Get current market quote
            current_quote = None
            if self.upstox.is_token_valid():
                instrument_key = f"NSE_EQ|{symbol}"
                current_quote = await self.upstox.get_full_market_quote(instrument_key)
            
            # Calculate trade impact
            trade_impact = self._calculate_trade_impact(
                stock_analysis, quantity, transaction_type, current_quote
            )
            
            # Risk analysis
            risk_analysis = self._perform_risk_analysis(stock_analysis, quantity, transaction_type)
            
            # Generate recommendation
            recommendation = self._generate_trade_recommendation(
                stock_analysis, trade_impact, risk_analysis
            )
            
            return {
                'symbol': symbol,
                'quantity': quantity,
                'transaction_type': transaction_type,
                'order_type': order_type,
                'stock_analysis': stock_analysis,
                'current_quote': current_quote,
                'trade_impact': trade_impact,
                'risk_analysis': risk_analysis,
                'recommendation': recommendation,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in trade analysis: {e}")
            return {'error': str(e)}
    
    def _extract_stock_symbols(self, news_data: List[NewsData]) -> List[str]:
        """Extract potential stock symbols from news data"""
        # Common Indian stock symbols
        common_symbols = [
            'RELIANCE', 'TCS', 'INFY', 'HDFC', 'SBIN', 'ITC', 'BAJFINANCE',
            'LT', 'WIPRO', 'HCLTECH', 'MARUTI', 'BHARTIARTL', 'ASIANPAINT',
            'NESTLEIND', 'KOTAKBANK', 'HINDUNILVR', 'ICICIBANK', 'AXISBANK'
        ]
        
        found_symbols = []
        for news_item in news_data:
            content = f"{news_item.title} {news_item.content}".upper()
            for symbol in common_symbols:
                if symbol in content and symbol not in found_symbols:
                    found_symbols.append(symbol)
        
        return found_symbols[:10]  # Return top 10
    
    def _analyze_news_sentiment(self, news_data: List[NewsData]) -> Dict[str, Any]:
        """Basic sentiment analysis of news data"""
        if not news_data:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        # Simple keyword-based sentiment analysis
        positive_keywords = ['gains', 'up', 'rise', 'profit', 'growth', 'positive', 'bullish']
        negative_keywords = ['falls', 'down', 'loss', 'decline', 'negative', 'bearish', 'crash']
        
        positive_count = 0
        negative_count = 0
        total_words = 0
        
        for news_item in news_data:
            content = f"{news_item.title} {news_item.content}".lower()
            words = content.split()
            total_words += len(words)
            
            for word in words:
                if word in positive_keywords:
                    positive_count += 1
                elif word in negative_keywords:
                    negative_count += 1
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = positive_count / max(total_words, 1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = negative_count / max(total_words, 1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': min(confidence * 10, 1.0),  # Scale confidence
            'positive_signals': positive_count,
            'negative_signals': negative_count,
            'analysis_basis': f'{len(news_data)} news items'
        }
    
    def _generate_recommendation(self, stock_analyses: Dict[str, Any], 
                               sentiment: Dict[str, Any]) -> Dict[str, str]:
        """Generate investment recommendation"""
        if not stock_analyses:
            return {'recommendation': 'HOLD', 'reason': 'Insufficient data for analysis'}
        
        # Simple recommendation logic
        if sentiment['sentiment'] == 'positive' and sentiment['confidence'] > 0.7:
            return {
                'recommendation': 'BUY',
                'reason': f"Positive market sentiment ({sentiment['confidence']:.2f} confidence)"
            }
        elif sentiment['sentiment'] == 'negative' and sentiment['confidence'] > 0.7:
            return {
                'recommendation': 'SELL',
                'reason': f"Negative market sentiment ({sentiment['confidence']:.2f} confidence)"
            }
        else:
            return {
                'recommendation': 'HOLD',
                'reason': 'Mixed or neutral market sentiment'
            }
    
    async def _get_sector_analysis(self) -> Dict[str, Any]:
        """Get sector-wise analysis"""
        sectors = {
            'Banking': ['HDFC', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH'],
            'Energy': ['RELIANCE', 'ONGC'],
            'Auto': ['MARUTI', 'BAJAJ-AUTO'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND']
        }
        
        sector_performance = {}
        for sector, symbols in sectors.items():
            sector_data = []
            for symbol in symbols:
                try:
                    data = await self.db_service.get_stock_data_with_fallback(symbol)
                    if data:
                        sector_data.append(data.change_percent)
                except:
                    continue
            
            if sector_data:
                avg_change = sum(sector_data) / len(sector_data)
                sector_performance[sector] = {
                    'average_change': round(avg_change, 2),
                    'stocks_analyzed': len(sector_data),
                    'trend': 'positive' if avg_change > 0 else 'negative'
                }
        
        return sector_performance
    
    def _calculate_market_statistics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market summary statistics"""
        if not market_data:
            return {}
        
        changes = [data['change_percent'] for data in market_data.values() 
                  if 'change_percent' in data]
        
        if not changes:
            return {}
        
        positive_changes = [c for c in changes if c > 0]
        negative_changes = [c for c in changes if c < 0]
        
        return {
            'total_stocks': len(changes),
            'gainers': len(positive_changes),
            'losers': len(negative_changes),
            'unchanged': len(changes) - len(positive_changes) - len(negative_changes),
            'average_change': round(sum(changes) / len(changes), 2),
            'market_trend': 'bullish' if len(positive_changes) > len(negative_changes) else 'bearish'
        }
    
    def _calculate_portfolio_statistics(self, positions, holdings) -> Dict[str, Any]:
        """Calculate portfolio statistics"""
        total_pnl = 0
        total_value = 0
        
        if positions:
            for pos in positions:
                total_pnl += pos.pnl
                total_value += pos.quantity * pos.last_price
        
        if holdings:
            for holding in holdings:
                total_pnl += holding.pnl
                total_value += holding.quantity * holding.last_price
        
        return {
            'total_portfolio_value': round(total_value, 2),
            'total_pnl': round(total_pnl, 2),
            'total_positions': len(positions) if positions else 0,
            'total_holdings': len(holdings) if holdings else 0,
            'portfolio_return_percent': round((total_pnl / max(total_value - total_pnl, 1)) * 100, 2)
        }
    
    def _calculate_trade_impact(self, stock_analysis, quantity, transaction_type, current_quote):
        """Calculate potential trade impact"""
        if not stock_analysis.get('current_data'):
            return {'error': 'No current stock data available'}
        
        current_price = stock_analysis['current_data']['price']
        trade_value = current_price * quantity
        
        return {
            'trade_value': round(trade_value, 2),
            'estimated_brokerage': round(min(trade_value * 0.0005, 20), 2),  # Upstox rates
            'transaction_type': transaction_type,
            'quantity': quantity,
            'price_per_share': current_price
        }
    
    def _perform_risk_analysis(self, stock_analysis, quantity, transaction_type):
        """Perform basic risk analysis"""
        if not stock_analysis.get('historical_data'):
            return {'risk_level': 'unknown', 'reason': 'No historical data available'}
        
        # Calculate volatility from historical data
        prices = [h['close'] for h in stock_analysis['historical_data'] if 'close' in h]
        if len(prices) < 2:
            return {'risk_level': 'unknown', 'reason': 'Insufficient historical data'}
        
        # Simple volatility calculation
        price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        avg_volatility = sum(price_changes) / len(price_changes)
        
        if avg_volatility > 0.05:  # 5% average daily change
            risk_level = 'high'
        elif avg_volatility > 0.02:  # 2% average daily change
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'volatility': round(avg_volatility * 100, 2),
            'analysis_period_days': len(prices),
            'recommendation': f'Risk level is {risk_level} based on {len(prices)} days of data'
        }
    
    def _generate_trade_recommendation(self, stock_analysis, trade_impact, risk_analysis):
        """Generate trade recommendation"""
        recommendation = {
            'action': 'REVIEW',
            'confidence': 'medium',
            'reasons': []
        }
        
        # Risk-based recommendation
        if risk_analysis['risk_level'] == 'high':
            recommendation['reasons'].append('High volatility detected - consider smaller position size')
        elif risk_analysis['risk_level'] == 'low':
            recommendation['reasons'].append('Low volatility - stable stock for investment')
        
        # News sentiment impact
        if stock_analysis.get('news'):
            sentiment = self._analyze_news_sentiment(
                [NewsData(**n) for n in stock_analysis['news']]
            )
            if sentiment['sentiment'] == 'positive':
                recommendation['reasons'].append('Positive news sentiment')
                recommendation['action'] = 'PROCEED'
            elif sentiment['sentiment'] == 'negative':
                recommendation['reasons'].append('Negative news sentiment')
                recommendation['action'] = 'CAUTION'
        
        return recommendation
    
    def _get_active_data_sources(self) -> List[str]:
        """Get list of active data sources"""
        sources = ['database_cache']
        
        if self.config.TAVILY_API_KEY:
            sources.append('tavily_search')
        
        if self.upstox.is_token_valid():
            sources.append('upstox_api')
        
        sources.append('nse_unofficial')
        
        return sources

# Global comprehensive service instance
financial_service = ComprehensiveFinancialService()
