import yfinance as yf
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config

class MarketDataService:
    """Enhanced market data service with multiple data sources"""
    
    def __init__(self):
        self.indian_stock_api_key = config.INDIAN_STOCK_API_KEY
        self.indian_stock_base_url = config.INDIAN_STOCK_API_BASE_URL
        self.upstox_access_token = config.UPSTOX_ACCESS_TOKEN
    
    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price for Indian stocks"""
        try:
            # Try Yahoo Finance first
            stock = yf.Ticker(f"{symbol}.NS")  # NSE suffix for Indian stocks
            info = stock.info
            
            if 'currentPrice' in info:
                return {
                    'symbol': symbol,
                    'price': info['currentPrice'],
                    'currency': 'INR',
                    'change': info.get('regularMarketChange', 0),
                    'changePercent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('volume', 0),
                    'marketCap': info.get('marketCap', 0),
                    'source': 'Yahoo Finance'
                }
            else:
                # Fallback to Indian Stock API
                return self.get_indian_stock_data(symbol)
                
        except Exception as e:
            print(f"Error fetching stock price for {symbol}: {e}")
            return {'error': f'Could not fetch data for {symbol}'}
    
    def get_indian_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get stock data from Indian Stock API"""
        try:
            headers = {
                'X-API-Key': self.indian_stock_api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                f"{self.indian_stock_base_url}/quote",
                params={'symbol': symbol},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'price': data.get('price', 0),
                    'currency': 'INR',
                    'change': data.get('change', 0),
                    'changePercent': data.get('pChange', 0),
                    'volume': data.get('volume', 0),
                    'source': 'Indian Stock API'
                }
            else:
                return {'error': f'API error: {response.status_code}'}
                
        except Exception as e:
            print(f"Error fetching Indian stock data: {e}")
            return {'error': 'Failed to fetch Indian stock data'}
    
    def get_market_indices(self) -> Dict[str, Any]:
        """Get major Indian market indices"""
        indices = {
            'NIFTY50': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK',
            'NIFTYIT': '^CNXIT'
        }
        
        results = {}
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.history(period="1d")
                if not info.empty:
                    latest = info.iloc[-1]
                    results[name] = {
                        'price': latest['Close'],
                        'change': latest['Close'] - latest['Open'],
                        'volume': latest['Volume']
                    }
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                results[name] = {'error': 'Data not available'}
        
        return results

# Legacy function for backward compatibility
def get_stock_price(symbol: str) -> dict:
    """Legacy function - use MarketDataService.get_stock_price instead"""
    service = MarketDataService()
    return service.get_stock_price(symbol)

# Create global instance
market_service = MarketDataService()