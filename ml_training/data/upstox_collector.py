import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import config

class UpstoxDataCollector:
    """Collect and preprocess data from Upstox API for ML training"""
    
    def __init__(self):
        self.access_token = config.UPSTOX_ACCESS_TOKEN
        self.api_key = config.UPSTOX_API_KEY
        self.base_url = "https://api.upstox.com/v2"
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
    
    def collect_market_data(self) -> pd.DataFrame:
        """Collect market data from Upstox API or generate mock data for training"""
        
        # Check if we have valid API credentials
        if not config.UPSTOX_ACCESS_TOKEN:
            print("⚠️ No Upstox access token found, generating mock data for training...")
            return self._generate_mock_market_data()
        
        try:
            # Try to collect real data first
            real_data = self._collect_real_market_data()
            if not real_data.empty:
                return real_data
        except Exception as e:
            print(f"⚠️ Failed to collect real market data: {e}")
            print("Generating mock data for training...")
        
        return self._generate_mock_market_data()
    
    def _generate_mock_market_data(self) -> pd.DataFrame:
        """Generate realistic mock market data for training purposes"""
        
        # Popular Indian stocks for training
        stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ITC', 'LT', 'SBIN', 'BHARTIARTL',
            'ASIANPAINT', 'MARUTI', 'BAJFINANCE', 'HCLTECH', 'KOTAKBANK', 'HINDUNILVR',
            'AXISBANK', 'ICICIBANK', 'WIPRO', 'ULTRACEMCO', 'NESTLEIND', 'TITAN'
        ]
        
        all_data = []
        base_date = datetime.now() - timedelta(days=30)
        
        for stock in stocks:
            # Generate 30 days of mock data
            for i in range(30):
                current_date = base_date + timedelta(days=i)
                
                # Generate realistic price movement
                base_price = np.random.uniform(100, 3000)
                price_change = np.random.normal(0, base_price * 0.02)  # 2% volatility
                
                high = base_price + abs(price_change) + np.random.uniform(0, base_price * 0.01)
                low = base_price - abs(price_change) - np.random.uniform(0, base_price * 0.01)
                
                record = {
                    'symbol': stock,
                    'timestamp': current_date.isoformat(),
                    'open': base_price,
                    'high': high,
                    'low': low,
                    'close': base_price + price_change,
                    'volume': np.random.randint(100000, 10000000),
                    'change': price_change,
                    'change_percent': (price_change / base_price) * 100,
                    'market_cap': np.random.randint(10000, 500000) * 1000000,
                    'pe_ratio': np.random.uniform(10, 50),
                    'pb_ratio': np.random.uniform(0.5, 5),
                    'dividend_yield': np.random.uniform(0, 5),
                    'sector': np.random.choice(['Banking', 'IT', 'Energy', 'FMCG', 'Auto', 'Pharma']),
                    'market_sentiment': np.random.choice(['bullish', 'bearish', 'neutral']),
                    'data_source': 'mock_training_data'
                }
                all_data.append(record)
        
        return pd.DataFrame(all_data)
    
    def _collect_real_market_data(self) -> pd.DataFrame:
        """Collect historical market data for training"""
        symbols = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ITC', 'LT', 'SBIN', 'BHARTIARTL'
        ]
        days = 30
        all_data = []
        
        for symbol in symbols:
            try:
                # Get historical data
                historical_data = self._get_historical_data(symbol, days)
                
                # Get current quote
                current_quote = self._get_current_quote(symbol)
                
                # Get option chain if available
                option_data = self._get_option_chain(symbol)
                
                # Combine all data
                for record in historical_data:
                    combined_record = {
                        'symbol': symbol,
                        'timestamp': record['timestamp'],
                        'open': record['open'],
                        'high': record['high'],
                        'low': record['low'],
                        'close': record['close'],
                        'volume': record['volume'],
                        'current_price': current_quote.get('last_price', 0),
                        'bid': current_quote.get('bid_price', 0),
                        'ask': current_quote.get('ask_price', 0),
                        'option_volume': option_data.get('total_volume', 0),
                        'option_oi': option_data.get('total_oi', 0)
                    }
                    all_data.append(combined_record)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error collecting data for {symbol}: {e}")
                continue
        
        return pd.DataFrame(all_data)
    
    def _get_historical_data(self, symbol: str, days: int) -> List[Dict]:
        """Get historical OHLCV data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            params = {
                'instrument_key': f'NSE_EQ|{symbol}',
                'interval': '1day',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(
                f"{self.base_url}/historical-candle/NSE_EQ%7C{symbol}/1day/{end_date.strftime('%Y-%m-%d')}/{start_date.strftime('%Y-%m-%d')}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                candles = data.get('data', {}).get('candles', [])
                
                historical_data = []
                for candle in candles:
                    historical_data.append({
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                
                return historical_data
            else:
                return []
                
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return []
    
    def _get_current_quote(self, symbol: str) -> Dict:
        """Get current market quote"""
        try:
            response = requests.get(
                f"{self.base_url}/market-quote/ltp",
                params={'instrument_key': f'NSE_EQ|{symbol}'},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {}).get(f'NSE_EQ|{symbol}', {})
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting current quote: {e}")
            return {}
    
    def _get_option_chain(self, symbol: str) -> Dict:
        """Get option chain data"""
        try:
            response = requests.get(
                f"{self.base_url}/option/chain",
                params={'instrument_key': f'NSE_EQ|{symbol}'},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                option_data = data.get('data', [])
                
                total_volume = sum(opt.get('volume', 0) for opt in option_data)
                total_oi = sum(opt.get('oi', 0) for opt in option_data)
                
                return {
                    'total_volume': total_volume,
                    'total_oi': total_oi
                }
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting option chain: {e}")
            return {}
    
    def collect_trading_signals(self) -> pd.DataFrame:
        """Collect trading signals and patterns"""
        signals_data = []
        
        # Popular Indian stocks for signal collection
        symbols = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
            'ICICIBANK', 'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ASIANPAINT'
        ]
        
        for symbol in symbols:
            try:
                # Get market depth
                depth_data = self._get_market_depth(symbol)
                
                # Calculate technical indicators
                technical_data = self._calculate_technical_indicators(symbol)
                
                signal_record = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'bid_depth': depth_data.get('bid_depth', 0),
                    'ask_depth': depth_data.get('ask_depth', 0),
                    'spread': depth_data.get('spread', 0),
                    'rsi': technical_data.get('rsi', 50),
                    'macd': technical_data.get('macd', 0),
                    'bollinger_position': technical_data.get('bollinger_position', 0)
                }
                
                signals_data.append(signal_record)
                time.sleep(0.3)
                
            except Exception as e:
                print(f"Error collecting signals for {symbol}: {e}")
                continue
        
        return pd.DataFrame(signals_data)
    
    def _get_market_depth(self, symbol: str) -> Dict:
        """Get market depth data"""
        try:
            response = requests.get(
                f"{self.base_url}/market-quote/depth",
                params={'instrument_key': f'NSE_EQ|{symbol}'},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                depth = data.get('data', {}).get(f'NSE_EQ|{symbol}', {})
                
                bid_depth = sum(level.get('quantity', 0) for level in depth.get('bid', []))
                ask_depth = sum(level.get('quantity', 0) for level in depth.get('ask', []))
                
                bid_price = depth.get('bid', [{}])[0].get('price', 0) if depth.get('bid') else 0
                ask_price = depth.get('ask', [{}])[0].get('price', 0) if depth.get('ask') else 0
                spread = ask_price - bid_price if ask_price and bid_price else 0
                
                return {
                    'bid_depth': bid_depth,
                    'ask_depth': ask_depth,
                    'spread': spread
                }
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting market depth: {e}")
            return {}
    
    def _calculate_technical_indicators(self, symbol: str) -> Dict:
        """Calculate basic technical indicators"""
        try:
            # Get recent price data
            historical_data = self._get_historical_data(symbol, 20)
            
            if len(historical_data) < 14:
                return {}
            
            closes = [float(record['close']) for record in historical_data]
            
            # RSI calculation
            rsi = self._calculate_rsi(closes)
            
            # Simple MACD
            macd = self._calculate_macd(closes)
            
            # Bollinger Bands position
            bollinger_position = self._calculate_bollinger_position(closes)
            
            return {
                'rsi': rsi,
                'macd': macd,
                'bollinger_position': bollinger_position
            }
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> float:
        """Calculate MACD"""
        if len(prices) < 26:
            return 0
        
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        return ema12 - ema26
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_bollinger_position(self, prices: List[float], period: int = 20) -> float:
        """Calculate position relative to Bollinger Bands"""
        if len(prices) < period:
            return 0
        
        recent_prices = prices[-period:]
        mean = sum(recent_prices) / len(recent_prices)
        variance = sum((x - mean) ** 2 for x in recent_prices) / len(recent_prices)
        std_dev = variance ** 0.5
        
        current_price = prices[-1]
        upper_band = mean + (2 * std_dev)
        lower_band = mean - (2 * std_dev)
        
        if upper_band == lower_band:
            return 0
        
        # Position between bands (-1 to 1)
        position = (current_price - mean) / (std_dev * 2)
        return max(-1, min(1, position))

# Create global instance
upstox_collector = UpstoxDataCollector()
