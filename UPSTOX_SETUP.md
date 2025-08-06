# Upstox API Setup Guide

## Getting Started with Upstox API

### 1. Create Upstox Account
1. Visit [Upstox Developer Portal](https://upstox.com/developer/)
2. Sign up for a developer account
3. Create a new app to get your API credentials

### 2. Get API Credentials
After creating your app, you'll receive:
- **API Key**: Your application's unique identifier
- **API Secret**: Keep this secure and never share publicly
- **Redirect URI**: Configure this in your app settings

### 3. Generate Access Token
Upstox uses OAuth2 for authentication. You need to:
1. Get an authorization code using the authorization URL
2. Exchange the authorization code for an access token
3. Use the access token for API calls

### Authorization URL Format:
```
https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id=YOUR_API_KEY&redirect_uri=YOUR_REDIRECT_URI
```

### 4. Environment Configuration
Add these to your `.env` file:
```bash
UPSTOX_API_KEY=your_api_key_here
UPSTOX_API_SECRET=your_api_secret_here
UPSTOX_ACCESS_TOKEN=your_access_token_here
UPSTOX_REDIRECT_URI=your_redirect_uri_here
```

### 5. Alternative APIs for Fallback

#### Indian Stock API
- Website: [Indian Stock API](https://indianstockapi.com/)
- Free tier available with limited requests
- Add to .env: `INDIAN_STOCK_API_KEY=your_key_here`

#### Alpha Vantage (Alternative)
- Website: [Alpha Vantage](https://www.alphavantage.co/)
- Free tier: 5 requests per minute
- Add to .env: `ALPHA_VANTAGE_API_KEY=your_key_here`

### 6. Testing Your Setup
Run the test script to verify your API integration:
```bash
python test_upstox.py
```

### 7. Important Notes
- Upstox API has rate limits - implement proper caching
- Access tokens expire and need to be refreshed
- Test with paper trading first before live data
- Keep your API credentials secure and never commit them to version control

### 8. Rate Limits
- **Upstox**: 25 requests per second
- **Indian Stock API**: Varies by plan
- **yfinance**: No official limits but subject to rate limiting

### 9. Symbol Formats
Different APIs use different symbol formats:
- **Upstox**: `NSE_EQ|INE002A01018` (instrument key format)
- **Yahoo Finance**: `RELIANCE.NS` (symbol + exchange)
- **Indian Stock API**: `RELIANCE` (simple symbol)

Our service automatically maps between these formats.
