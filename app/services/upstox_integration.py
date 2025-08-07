#!/usr/bin/env python3
"""
Upstox API Integration Service
Handles authentication and trading operations with Upstox API v2
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from config.api_config import config

logger = logging.getLogger(__name__)

@dataclass
class UpstoxAuthResponse:
    """Upstox authentication response"""
    access_token: str
    token_type: str
    expires_in: int
    scope: str
    
@dataclass
class UpstoxPosition:
    """Upstox position data"""
    instrument_token: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    product: str
    
@dataclass
class UpstoxHolding:
    """Upstox holding data"""
    instrument_token: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    product: str

class UpstoxService:
    """Comprehensive Upstox API integration service"""
    
    def __init__(self):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.sandbox_mode = config.UPSTOX_SANDBOX
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def get_authorization_url(self, state: str = "dalaal_street_bot") -> str:
        """Get Upstox authorization URL for OAuth flow"""
        if not self.config.UPSTOX_API_KEY or not self.config.UPSTOX_REDIRECT_URI:
            raise ValueError("Upstox API key and redirect URI must be configured")
        
        base_url = f"{self.config.UPSTOX_BASE_URL}/login/authorization/dialog"
        params = {
            'response_type': 'code',
            'client_id': self.config.UPSTOX_API_KEY,
            'redirect_uri': self.config.UPSTOX_REDIRECT_URI,
            'state': state
        }
        
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{param_string}"
    
    async def exchange_code_for_token(self, authorization_code: str) -> Optional[UpstoxAuthResponse]:
        """Exchange authorization code for access token"""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/login/authorization/token"
            
            payload = {
                'code': authorization_code,
                'client_id': self.config.UPSTOX_API_KEY,
                'client_secret': self.config.UPSTOX_API_SECRET,
                'redirect_uri': self.config.UPSTOX_REDIRECT_URI,
                'grant_type': 'authorization_code'
            }
            
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            async with self.session.post(url, data=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    auth_response = UpstoxAuthResponse(
                        access_token=data['access_token'],
                        token_type=data['token_type'],
                        expires_in=data['expires_in'],
                        scope=data.get('scope', '')
                    )
                    
                    # Store token and expiration
                    self.access_token = auth_response.access_token
                    self.token_expires_at = datetime.now() + timedelta(seconds=auth_response.expires_in)
                    
                    logger.info("Successfully obtained Upstox access token")
                    return auth_response
                    
                else:
                    error_data = await response.json()
                    logger.error(f"Token exchange failed: {error_data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            return None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get standard headers for Upstox API calls"""
        if not self.access_token:
            raise ValueError("Access token not available. Please authenticate first.")
        
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json',
            'Api-Version': '2.0'
        }
    
    async def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """Get user profile information"""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/user/profile"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {})
                else:
                    logger.error(f"Profile fetch failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None
    
    async def get_market_quote(self, instrument_key: str) -> Optional[Dict[str, Any]]:
        """Get market quote for an instrument"""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/market-quote/ltp"
            headers = self._get_headers()
            params = {'instrument_key': instrument_key}
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {})
                else:
                    logger.error(f"Market quote fetch failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching market quote: {e}")
            return None
    
    async def get_full_market_quote(self, instrument_key: str) -> Optional[Dict[str, Any]]:
        """Get full market quote with OHLC, bid/ask, etc."""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/market-quote/quotes"
            headers = self._get_headers()
            params = {'instrument_key': instrument_key}
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {})
                else:
                    logger.error(f"Full market quote fetch failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching full market quote: {e}")
            return None
    
    async def get_historical_candles(self, instrument_key: str, interval: str, 
                                   to_date: str, from_date: str) -> Optional[List[Dict]]:
        """
        Get historical candle data
        
        Args:
            instrument_key: Instrument identifier (e.g., NSE_EQ|INE669E01016)
            interval: Candle interval (minute, day, week, month)
            to_date: End date (YYYY-MM-DD)
            from_date: Start date (YYYY-MM-DD)
        """
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('candles', [])
                else:
                    logger.error(f"Historical candles fetch failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching historical candles: {e}")
            return None
    
    async def get_intraday_candles(self, instrument_key: str, interval: str) -> Optional[List[Dict]]:
        """Get intraday candle data"""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/historical-candle/intraday/{instrument_key}/{interval}"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('candles', [])
                else:
                    logger.error(f"Intraday candles fetch failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching intraday candles: {e}")
            return None
    
    async def get_positions(self) -> Optional[List[UpstoxPosition]]:
        """Get current positions"""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/portfolio/short-term-positions"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    positions = []
                    
                    for pos_data in data.get('data', []):
                        position = UpstoxPosition(
                            instrument_token=pos_data.get('instrument_token', ''),
                            quantity=pos_data.get('quantity', 0),
                            average_price=pos_data.get('average_price', 0.0),
                            last_price=pos_data.get('last_price', 0.0),
                            pnl=pos_data.get('pnl', 0.0),
                            product=pos_data.get('product', '')
                        )
                        positions.append(position)
                    
                    return positions
                else:
                    logger.error(f"Positions fetch failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return None
    
    async def get_holdings(self) -> Optional[List[UpstoxHolding]]:
        """Get current holdings"""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/portfolio/long-term-holdings"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    holdings = []
                    
                    for holding_data in data.get('data', []):
                        holding = UpstoxHolding(
                            instrument_token=holding_data.get('instrument_token', ''),
                            quantity=holding_data.get('quantity', 0),
                            average_price=holding_data.get('average_price', 0.0),
                            last_price=holding_data.get('last_price', 0.0),
                            pnl=holding_data.get('pnl', 0.0),
                            product=holding_data.get('product', '')
                        )
                        holdings.append(holding)
                    
                    return holdings
                else:
                    logger.error(f"Holdings fetch failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            return None
    
    async def get_funds_and_margin(self) -> Optional[Dict[str, Any]]:
        """Get available funds and margin"""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/user/get-funds-and-margin"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {})
                else:
                    logger.error(f"Funds fetch failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching funds: {e}")
            return None
    
    async def place_order(self, quantity: int, product: str, validity: str,
                         price: float, instrument_token: str, order_type: str,
                         transaction_type: str, disclosed_quantity: int = 0,
                         trigger_price: float = 0.0, is_amo: bool = False) -> Optional[Dict[str, Any]]:
        """
        Place a new order
        
        Args:
            quantity: Order quantity
            product: Product type (I, D, CO, OCO)
            validity: Order validity (DAY, IOC)
            price: Order price
            instrument_token: Instrument identifier
            order_type: Order type (MARKET, LIMIT, SL, SL-M)
            transaction_type: BUY or SELL
            disclosed_quantity: Disclosed quantity for iceberg orders
            trigger_price: Trigger price for stop-loss orders
            is_amo: After Market Order flag
        """
        if self.sandbox_mode:
            logger.info("Sandbox mode: Order placement simulated")
            return {
                'order_id': f'SANDBOX_{int(time.time())}',
                'status': 'success',
                'message': 'Order placed successfully in sandbox mode'
            }
        
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/order/place"
            headers = self._get_headers()
            headers['Content-Type'] = 'application/json'
            
            payload = {
                'quantity': quantity,
                'product': product,
                'validity': validity,
                'price': price,
                'instrument_token': instrument_token,
                'order_type': order_type,
                'transaction_type': transaction_type,
                'disclosed_quantity': disclosed_quantity,
                'trigger_price': trigger_price,
                'is_amo': is_amo
            }
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {})
                else:
                    error_data = await response.json()
                    logger.error(f"Order placement failed: {error_data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def get_order_book(self) -> Optional[List[Dict[str, Any]]]:
        """Get order book"""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/order/retrieve-all"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                else:
                    logger.error(f"Order book fetch failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Cancel an order"""
        if self.sandbox_mode:
            logger.info(f"Sandbox mode: Order {order_id} cancellation simulated")
            return {
                'order_id': order_id,
                'status': 'success',
                'message': 'Order cancelled successfully in sandbox mode'
            }
        
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/order/cancel"
            headers = self._get_headers()
            headers['Content-Type'] = 'application/json'
            
            payload = {'order_id': order_id}
            
            async with self.session.delete(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {})
                else:
                    error_data = await response.json()
                    logger.error(f"Order cancellation failed: {error_data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return None
    
    def is_token_valid(self) -> bool:
        """Check if access token is still valid"""
        if not self.access_token or not self.token_expires_at:
            return False
        return datetime.now() < self.token_expires_at
    
    async def get_instrument_search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Search for instruments"""
        try:
            url = f"{self.config.UPSTOX_BASE_URL}/search/instruments"
            headers = self._get_headers()
            params = {'query': query}
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                else:
                    logger.error(f"Instrument search failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error searching instruments: {e}")
            return None

# Global Upstox service instance
upstox_service = UpstoxService()
