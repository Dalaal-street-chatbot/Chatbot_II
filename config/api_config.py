#!/usr/bin/env python3
"""
API Configuration for Dalaal Street Chatbot
Secure storage and management of API keys and endpoints
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class APIConfiguration:
    """Configuration class for all API services"""
    
    # Tavily API for web search
    TAVILY_API_KEY: str = "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"
    TAVILY_BASE_URL: str = "https://api.tavily.com"
    
    # Upstox API Configuration
    UPSTOX_API_KEY: str = "73971cc2-2f15-4f63-9bf7-e0d34cd5b6d2"
    UPSTOX_API_SECRET: str = "rn82ezezd5"
    UPSTOX_ACCESS_TOKEN: str = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzQkNYVUEiLCJqdGkiOiI2ODk0ODUxOWUzYTY5NzQ0NTJiNzJhMjciLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzU0NTYzODY1LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NTQ2MDQwMDB9.dpGn9iNzoR6YxKY979fg30-jA-aVKqErwn97rdCprXU"
    UPSTOX_REDIRECT_URI: str = "https://localhost:8080/callback"
    UPSTOX_BASE_URL: str = "https://api.upstox.com/v2"
    UPSTOX_SANDBOX: bool = False  # Using production credentials
    
    # Alternative Indian Stock APIs
    NSE_API_BASE_URL: str = "https://www.nseindia.com/api"
    BSE_API_BASE_URL: str = "https://api.bseindia.com"
    
    # Global DataFeeds API (Alternative)
    GFDL_API_KEY: Optional[str] = None
    GFDL_BASE_URL: str = "https://globaldatafeeds.in/api"
    
    # ICICI Direct Breeze API
    BREEZE_API_KEY: Optional[str] = None
    BREEZE_API_SECRET: Optional[str] = None
    BREEZE_SESSION_TOKEN: Optional[str] = None
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///stock_data.db"
    REDIS_URL: Optional[str] = None  # For caching
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 60
    CACHE_DURATION_SECONDS: int = 300  # 5 minutes
    
    @classmethod
    def from_environment(cls) -> 'APIConfiguration':
        """Load configuration from environment variables"""
        return cls(
            TAVILY_API_KEY=os.getenv('TAVILY_API_KEY', cls.TAVILY_API_KEY),
            UPSTOX_API_KEY=os.getenv('UPSTOX_API_KEY', cls.UPSTOX_API_KEY),
            UPSTOX_API_SECRET=os.getenv('UPSTOX_API_SECRET', cls.UPSTOX_API_SECRET),
            UPSTOX_ACCESS_TOKEN=os.getenv('UPSTOX_ACCESS_TOKEN', cls.UPSTOX_ACCESS_TOKEN),
            UPSTOX_REDIRECT_URI=os.getenv('UPSTOX_REDIRECT_URI', cls.UPSTOX_REDIRECT_URI),
            UPSTOX_SANDBOX=os.getenv('UPSTOX_SANDBOX', 'False').lower() == 'true',
            GFDL_API_KEY=os.getenv('GFDL_API_KEY'),
            BREEZE_API_KEY=os.getenv('BREEZE_API_KEY'),
            BREEZE_API_SECRET=os.getenv('BREEZE_API_SECRET'),
            BREEZE_SESSION_TOKEN=os.getenv('BREEZE_SESSION_TOKEN'),
            DATABASE_URL=os.getenv('DATABASE_URL', cls.DATABASE_URL),
            REDIS_URL=os.getenv('REDIS_URL'),
        )
    
    def get_upstox_auth_url(self) -> str:
        """Generate Upstox OAuth authorization URL"""
        if not self.UPSTOX_API_KEY or not self.UPSTOX_REDIRECT_URI:
            raise ValueError("Upstox API key and redirect URI must be configured")
            
        params = {
            'response_type': 'code',
            'client_id': self.UPSTOX_API_KEY,
            'redirect_uri': self.UPSTOX_REDIRECT_URI,
            'state': 'dalaal_street_chatbot'
        }
        
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.UPSTOX_BASE_URL}/login/authorization/dialog?{param_string}"
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate API configuration"""
        return {
            'tavily_configured': bool(self.TAVILY_API_KEY),
            'upstox_configured': bool(self.UPSTOX_API_KEY and self.UPSTOX_API_SECRET),
            'database_configured': bool(self.DATABASE_URL),
            'cache_configured': bool(self.REDIS_URL),
        }

# Global configuration instance
config = APIConfiguration.from_environment()

# Export commonly used configurations
TAVILY_API_KEY = config.TAVILY_API_KEY
UPSTOX_SANDBOX = config.UPSTOX_SANDBOX
DATABASE_URL = config.DATABASE_URL
