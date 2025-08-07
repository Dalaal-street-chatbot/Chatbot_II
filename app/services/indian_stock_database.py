#!/usr/bin/env python3
"""
Indian Stock Market Database Service
Integrates Upstox API, NSE/BSE APIs, and Tavily search for comprehensive stock data
"""

import asyncio
import aiohttp
import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from config.api_config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Stock data model"""
    symbol: str
    exchange: str
    price: float
    change: float
    change_percent: float
    volume: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    timestamp: datetime
    source: str

@dataclass
class HistoricalData:
    """Historical stock data model"""
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str

@dataclass
class NewsData:
    """Financial news data model"""
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    sentiment: Optional[str] = None
    relevance_score: Optional[float] = None

class IndianStockDatabaseService:
    """Comprehensive Indian stock market database service"""
    
    def __init__(self):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.upstox_access_token: Optional[str] = None
        self.rate_limit_tracker = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect('stock_data.db')
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_quotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    price REAL NOT NULL,
                    change_amount REAL,
                    change_percent REAL,
                    volume INTEGER,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER,
                    source TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date, source)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    url TEXT UNIQUE,
                    published_at DATETIME,
                    source TEXT NOT NULL,
                    sentiment TEXT,
                    relevance_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_stock_symbol_timestamp 
                ON stock_quotes(symbol, timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_historical_symbol_date 
                ON historical_data(symbol, date)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, service: str) -> bool:
        """Check if rate limit allows request"""
        now = time.time()
        if service not in self.rate_limit_tracker:
            self.rate_limit_tracker[service] = []
        
        # Remove old entries
        self.rate_limit_tracker[service] = [
            t for t in self.rate_limit_tracker[service] 
            if now - t < 60  # Last minute
        ]
        
        # Check if under limit
        if len(self.rate_limit_tracker[service]) < self.config.MAX_REQUESTS_PER_MINUTE:
            self.rate_limit_tracker[service].append(now)
            return True
        return False
    
    async def search_financial_news(self, query: str, max_results: int = 10) -> List[NewsData]:
        """Search financial news using Tavily API"""
        if not self._check_rate_limit('tavily'):
            logger.warning("Rate limit exceeded for Tavily API")
            return []
        
        try:
            url = f"{self.config.TAVILY_BASE_URL}/search"
            payload = {
                "api_key": self.config.TAVILY_API_KEY,
                "query": f"{query} Indian stock market financial news",
                "search_depth": "advanced",
                "include_domains": [
                    "economictimes.indiatimes.com",
                    "moneycontrol.com", 
                    "business-standard.com",
                    "livemint.com",
                    "financialexpress.com"
                ],
                "max_results": max_results
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    news_items = []
                    
                    for item in data.get('results', []):
                        news_data = NewsData(
                            title=item.get('title', ''),
                            content=item.get('content', ''),
                            url=item.get('url', ''),
                            published_at=datetime.now(),  # Tavily doesn't provide timestamp
                            source='tavily_search',
                            relevance_score=item.get('score', 0.0)
                        )
                        news_items.append(news_data)
                        
                        # Store in database
                        await self._store_news_data(news_data)
                    
                    return news_items
                else:
                    logger.error(f"Tavily API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching financial news: {e}")
            return []
    
    async def get_upstox_stock_data(self, symbol: str, exchange: str = "NSE") -> Optional[StockData]:
        """Get stock data from Upstox API"""
        if not self.upstox_access_token:
            logger.warning("Upstox access token not available")
            return None
            
        if not self._check_rate_limit('upstox'):
            logger.warning("Rate limit exceeded for Upstox API")
            return None
        
        try:
            # Format instrument token for Upstox
            instrument_token = f"{exchange}_EQ|{symbol}"
            url = f"{self.config.UPSTOX_BASE_URL}/market-quote/ltp"
            
            params = {
                'instrument_key': instrument_token
            }
            
            headers = {
                'Authorization': f'Bearer {self.upstox_access_token}',
                'Accept': 'application/json'
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    quote_data = data.get('data', {}).get(instrument_token, {})
                    
                    stock_data = StockData(
                        symbol=symbol,
                        exchange=exchange,
                        price=quote_data.get('last_price', 0.0),
                        change=quote_data.get('net_change', 0.0),
                        change_percent=quote_data.get('percent_change', 0.0),
                        volume=quote_data.get('volume', 0),
                        open_price=quote_data.get('ohlc', {}).get('open', 0.0),
                        high_price=quote_data.get('ohlc', {}).get('high', 0.0),
                        low_price=quote_data.get('ohlc', {}).get('low', 0.0),
                        close_price=quote_data.get('ohlc', {}).get('close', 0.0),
                        timestamp=datetime.now(),
                        source='upstox'
                    )
                    
                    await self._store_stock_data(stock_data)
                    return stock_data
                    
                else:
                    logger.error(f"Upstox API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Upstox stock data: {e}")
            return None
    
    async def get_nse_stock_data(self, symbol: str) -> Optional[StockData]:
        """Get stock data from NSE (unofficial API)"""
        if not self._check_rate_limit('nse'):
            logger.warning("Rate limit exceeded for NSE API")
            return None
        
        try:
            # NSE unofficial API endpoint
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    price_info = data.get('priceInfo', {})
                    
                    stock_data = StockData(
                        symbol=symbol,
                        exchange='NSE',
                        price=float(price_info.get('lastPrice', 0)),
                        change=float(price_info.get('change', 0)),
                        change_percent=float(price_info.get('pChange', 0)),
                        volume=int(data.get('totalTradedVolume', 0)),
                        open_price=float(price_info.get('open', 0)),
                        high_price=float(price_info.get('intraDayHighLow', {}).get('max', 0)),
                        low_price=float(price_info.get('intraDayHighLow', {}).get('min', 0)),
                        close_price=float(price_info.get('previousClose', 0)),
                        timestamp=datetime.now(),
                        source='nse_unofficial'
                    )
                    
                    await self._store_stock_data(stock_data)
                    return stock_data
                    
                else:
                    logger.error(f"NSE API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching NSE stock data: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, days: int = 30, exchange: str = "NSE") -> List[HistoricalData]:
        """Get historical stock data"""
        try:
            # Try Upstox first if available
            if self.upstox_access_token:
                return await self._get_upstox_historical_data(symbol, days, exchange)
            
            # Fallback to database cached data
            return await self._get_cached_historical_data(symbol, days)
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    async def _get_upstox_historical_data(self, symbol: str, days: int, exchange: str) -> List[HistoricalData]:
        """Get historical data from Upstox"""
        if not self._check_rate_limit('upstox_historical'):
            return []
        
        try:
            instrument_token = f"{exchange}_EQ|{symbol}"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.config.UPSTOX_BASE_URL}/historical-candle/{instrument_token}/day/{end_date.strftime('%Y-%m-%d')}/{start_date.strftime('%Y-%m-%d')}"
            
            headers = {
                'Authorization': f'Bearer {self.upstox_access_token}',
                'Accept': 'application/json'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    candles = data.get('data', {}).get('candles', [])
                    
                    historical_data = []
                    for candle in candles:
                        if len(candle) >= 6:
                            hist_data = HistoricalData(
                                symbol=symbol,
                                date=datetime.fromisoformat(candle[0]),
                                open=float(candle[1]),
                                high=float(candle[2]),
                                low=float(candle[3]),
                                close=float(candle[4]),
                                volume=int(candle[5]),
                                source='upstox'
                            )
                            historical_data.append(hist_data)
                            await self._store_historical_data(hist_data)
                    
                    return historical_data
                else:
                    logger.error(f"Upstox historical API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching Upstox historical data: {e}")
            return []
    
    async def get_stock_data_with_fallback(self, symbol: str, exchange: str = "NSE") -> Optional[StockData]:
        """Get stock data with multiple API fallbacks"""
        # Try Upstox first
        data = await self.get_upstox_stock_data(symbol, exchange)
        if data:
            return data
        
        # Try NSE unofficial API
        if exchange == "NSE":
            data = await self.get_nse_stock_data(symbol)
            if data:
                return data
        
        # Try cached data as last resort
        return await self._get_cached_stock_data(symbol)
    
    async def _store_stock_data(self, data: StockData):
        """Store stock data in database"""
        try:
            conn = sqlite3.connect('stock_data.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO stock_quotes 
                (symbol, exchange, price, change_amount, change_percent, volume, 
                 open_price, high_price, low_price, close_price, timestamp, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.exchange, data.price, data.change, data.change_percent,
                data.volume, data.open_price, data.high_price, data.low_price,
                data.close_price, data.timestamp, data.source
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing stock data: {e}")
    
    async def _store_historical_data(self, data: HistoricalData):
        """Store historical data in database"""
        try:
            conn = sqlite3.connect('stock_data.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO historical_data 
                (symbol, date, open_price, high_price, low_price, close_price, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.date, data.open, data.high, data.low,
                data.close, data.volume, data.source
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
    
    async def _store_news_data(self, data: NewsData):
        """Store news data in database"""
        try:
            conn = sqlite3.connect('stock_data.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO news_data 
                (title, content, url, published_at, source, sentiment, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.title, data.content, data.url, data.published_at,
                data.source, data.sentiment, data.relevance_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing news data: {e}")
    
    async def _get_cached_stock_data(self, symbol: str) -> Optional[StockData]:
        """Get latest cached stock data"""
        try:
            conn = sqlite3.connect('stock_data.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, exchange, price, change_amount, change_percent, volume,
                       open_price, high_price, low_price, close_price, timestamp, source
                FROM stock_quotes 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (symbol,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return StockData(
                    symbol=row[0], exchange=row[1], price=row[2], change=row[3],
                    change_percent=row[4], volume=row[5], open_price=row[6],
                    high_price=row[7], low_price=row[8], close_price=row[9],
                    timestamp=datetime.fromisoformat(row[10]), source=row[11]
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached stock data: {e}")
            return None
    
    async def _get_cached_historical_data(self, symbol: str, days: int) -> List[HistoricalData]:
        """Get cached historical data"""
        try:
            conn = sqlite3.connect('stock_data.db')
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT symbol, date, open_price, high_price, low_price, close_price, volume, source
                FROM historical_data 
                WHERE symbol = ? AND date >= ?
                ORDER BY date DESC
            ''', (symbol, start_date))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                HistoricalData(
                    symbol=row[0], date=datetime.fromisoformat(row[1]),
                    open=row[2], high=row[3], low=row[4], close=row[5],
                    volume=row[6], source=row[7]
                )
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Error getting cached historical data: {e}")
            return []
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        try:
            # Get data for major indices
            major_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'SBIN']
            market_data = {}
            
            for symbol in major_stocks:
                data = await self.get_stock_data_with_fallback(symbol)
                if data:
                    market_data[symbol] = asdict(data)
            
            # Get latest financial news
            news = await self.search_financial_news("market summary today", max_results=5)
            
            return {
                'stocks': market_data,
                'news': [asdict(n) for n in news],
                'last_updated': datetime.now().isoformat(),
                'data_sources': ['upstox', 'nse_unofficial', 'tavily_search']
            }
            
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {}

# Global service instance
indian_stock_db = IndianStockDatabaseService()
