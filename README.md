# Dalaal Street Chatbot 🤖📈

An advanced AI-powered financial chatbot specializing in Indian stock markets, powered by multiple AI services including Groq, Azure OpenAI, Google AI, and more.

## 🚀 Features

### AI Services Integration
- **Groq AI** - Primary NLP engine for fast, intelligent responses
- **Azure OpenAI** - Enterprise-grade AI for complex financial analysis
- **Google AI (Gemini)** - Market predictions and forecasting
- **Google Cloud Services** - Comprehensive cloud AI capabilities:
  - **Vertex AI** - Financial analysis and predictions
  - **Dialogflow** - Specialized financial conversations
  - **Cloud Vision API** - Chart and document analysis
  - **BigQuery** - Financial data warehousing
- **Codestral AI** - Financial algorithm and code generation
- **DeepSeek AI** - Deep reasoning and step-by-step analysis
- **Ollama** - Local AI for privacy-sensitive queries

### Financial Data Sources
- **Yahoo Finance** - Real-time stock prices and market data
- **Indian Stock API** - Specialized Indian market data
- **Upstox API** - Professional trading data and analytics
- **News API** - Latest financial news and market sentiment

### Core Capabilities
- 💬 Intelligent chat with context awareness
- 📊 Real-time stock prices and market data
- 📰 Financial news analysis and sentiment
- 📈 Market indices tracking (NIFTY, SENSEX, BANKNIFTY)
- 🔍 Deep financial analysis and insights
- 💻 Trading algorithm generation
- 🎯 Market predictions and forecasting
- 📱 Multi-platform support (Web, API)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)
- API keys for various services

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/Dalaal-street-chatbot.git
cd Dalaal-street-chatbot
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Setup**
Copy `.env` file with all API keys (already configured)

   For Google Cloud services integration, see [GOOGLE_CLOUD_SETUP.md](GOOGLE_CLOUD_SETUP.md) for detailed instructions.

4. **Run the backend**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install dependencies**
```bash
npm install
```

2. **Start development server**
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## 📡 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### 1. Chat with AI
```http
POST /chat
```

**Request Body:**
```json
{
  "message": "What's the current price of Reliance?",
  "context": {},
  "session_id": "user_session_123"
}
```

**Response:**
```json
{
  "response": "The current price of Reliance Industries is ₹2,847.50, showing a gain of 1.2% today...",
  "intent": "stock_price",
  "entities": ["RELIANCE"],
  "confidence": 0.95,
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### 2. Get Stock Data
```http
POST /stock
```

**Request Body:**
```json
{
  "symbol": "RELIANCE",
  "exchange": "NSE"
}
```

**Response:**
```json
{
  "symbol": "RELIANCE",
  "price": 2847.50,
  "currency": "INR",
  "change": 33.75,
  "change_percent": 1.2,
  "volume": 1250000,
  "market_cap": 1925000000000,
  "source": "Yahoo Finance"
}
```

#### 3. Get Market Indices
```http
GET /indices
```

**Response:**
```json
{
  "indices": {
    "NIFTY50": {
      "price": 23175.20,
      "change": 145.30,
      "volume": 125000000
    },
    "SENSEX": {
      "price": 76520.10,
      "change": 285.50,
      "volume": 89000000
    }
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### 4. Get Financial News
```http
POST /news
```

**Request Body:**
```json
{
  "query": "stock market",
  "company": "Reliance",
  "page_size": 10
}
```

**Response:**
```json
{
  "status": "success",
  "total_results": 150,
  "articles": [
    {
      "title": "Reliance Industries Reports Strong Q3 Results",
      "description": "Company beats estimates with 15% revenue growth...",
      "url": "https://example.com/news/1",
      "source": "Economic Times",
      "published_at": "2025-01-27T09:00:00Z",
      "image_url": "https://example.com/image.jpg"
    }
  ]
}
```

#### 5. Get AI Analysis
```http
POST /analysis
```

**Request Body:**
```json
{
  "symbol": "NIFTY",
  "query": "Should I invest in banking stocks?",
  "time_period": "3mo"
}
```

**Response:**
```json
{
  "analysis": "Based on current market conditions and banking sector performance...",
  "data": {
    "stock_data": {},
    "market_indices": {},
    "recent_news": {}
  },
  "recommendations": [
    "Consider diversification across banking sub-sectors",
    "Monitor RBI policy announcements"
  ],
  "confidence": 0.85
}
```

### Google Cloud Services Endpoints

#### 1. GCloud Chat with AI
```http
POST /gcloud/chat
```

**Request Body:**
```json
{
  "message": "Analyze the technical indicators for Reliance",
  "session_id": "user_session_123",
  "context": {"previous_analysis": "sentiment"}
}
```

**Response:**
```json
{
  "status": "success",
  "response": "Based on technical analysis, Reliance shows bullish signals...",
  "intent": "technical_analysis",
  "symbols": ["RELIANCE"],
  "technical_analysis": {
    "RELIANCE": {
      "signals": [
        {"indicator": "RSI", "value": 62.5, "signal": "bullish"},
        {"indicator": "MACD", "value": 3.2, "signal": "bullish"},
        {"indicator": "Bollinger Bands", "signal": "neutral"}
      ],
      "overall_signal": "Bullish"
    }
  }
}
```

#### 2. Image Analysis
```http
POST /gcloud/image-analysis
```

**Request Body:**
```json
{
  "base64_image": "base64_encoded_image_data",
  "session_id": "user_session_123",
  "context": {"analysis_type": "chart"}
}
```

**Response:**
```json
{
  "status": "success",
  "content_type": "chart",
  "analysis": {
    "chart_type": "candlestick",
    "time_frame": "daily",
    "symbols": ["NIFTY"],
    "chart_analysis": "This chart shows a bullish breakout pattern with increased volume..."
  },
  "response": "Here's my analysis of your financial chart: This shows a bullish breakout pattern with strong support at 21,500..."
}
```

#### 3. Market Insights
```http
POST /gcloud/market-insights
```

**Request Body:**
```json
{
  "symbols": ["RELIANCE", "HDFCBANK", "TCS"]
}
```

**Response:**
```json
{
  "status": "success",
  "insights": {
    "news": {
      "status": "success",
      "news_items": [...]
    },
    "sentiment": {
      "status": "success",
      "sentiments": {
        "RELIANCE": {"sentiment": "bullish", "score": 0.78},
        "HDFCBANK": {"sentiment": "neutral", "score": 0.52},
        "TCS": {"sentiment": "bearish", "score": 0.35}
      }
    },
    "technical_analysis": {
      "status": "success",
      "analyses": {...}
    },
    "predictions": {
      "status": "success",
      "predictions": {...}
    }
  }
}
```

## 🔧 Configuration

### Environment Variables

The `.env` file contains all necessary API keys and configurations:

```env
# Primary AI Service
GROQ_API_KEY=your_groq_api_key

# Financial Data APIs
NEWS_API=your_news_api_key
INDIAN_STOCK_API_KEY=your_indian_stock_api_key
UPSTOX_ACCESS_TOKEN=your_upstox_token

# Additional AI Services
AZURE_OPENAI_API_KEY=your_azure_key
GOOGLE_AI_API_KEY=your_google_key
CODESTRAL_API_KEY=your_codestral_key
DEEPSEEK_AI_R1_API=your_deepseek_key

# Google Cloud Services
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
VERTEXAI_PROJECT_ID=your-project-id
VERTEXAI_LOCATION=us-central1
DIALOGFLOW_PROJECT_ID=your-dialogflow-project-id

# Local AI
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

## 🎯 Usage Examples

### Basic Chat
```python
import requests

response = requests.post("http://localhost:8000/api/v1/chat", json={
    "message": "What's happening in the stock market today?"
})
print(response.json()["response"])
```

### Get Stock Price
```python
response = requests.post("http://localhost:8000/api/v1/stock", json={
    "symbol": "TCS"
})
print(f"TCS Price: ₹{response.json()['price']}")
```

### Get Market Analysis
```python
response = requests.post("http://localhost:8000/api/v1/analysis", json={
    "query": "Analyze the IT sector performance",
    "time_period": "1mo"
})
print(response.json()["analysis"])
```

### Google Cloud Market Insights
```python
import requests

response = requests.post("http://localhost:8000/api/v1/gcloud/market-insights", json={
    "symbols": ["RELIANCE", "HDFCBANK", "TCS"]
})

result = response.json()
if result["status"] == "success":
    insights = result["insights"]
    
    # Print sentiment analysis
    if "sentiment" in insights and insights["sentiment"]["status"] == "success":
        print("Market Sentiment:")
        for symbol, data in insights["sentiment"]["sentiments"].items():
            score = data["score"] * 100
            print(f"- {symbol}: {data['sentiment'].title()} ({score:.1f}%)")
    
    # Print technical signals
    if "technical_analysis" in insights and insights["technical_analysis"]["status"] == "success":
        print("\nTechnical Analysis:")
        for symbol, analysis in insights["technical_analysis"]["analyses"].items():
            print(f"{symbol}: {analysis.get('overall_signal', 'Neutral')}")
            
    # Print price predictions
    if "predictions" in insights and insights["predictions"]["status"] == "success":
        print("\nPrice Predictions (5-day):")
        for symbol, predictions in insights["predictions"]["predictions"].items():
            if predictions and len(predictions) > 0:
                last_day = predictions[-1]
                print(f"{symbol}: ₹{last_day['predicted_price']:.2f}")
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │────│   FastAPI       │────│   AI Services   │
│                 │    │   Backend       │    │   (Groq, Azure, │
│   - Chat UI     │    │                 │    │    Google, etc.) │
│   - Charts      │    │   - API Routes  │    │                 │
│   - Dashboard   │    │   - Data        │    │   - NLP         │
└─────────────────┘    │     Processing  │    │   - Analysis    │
                       └─────────────────┘    │   - Generation  │
                                             └─────────────────┘
                                                      │
                                             ┌─────────────────┐
                                             │   Data Sources  │
                                             │                 │
                                             │   - Yahoo Finance│
                                             │   - News API    │
                                             │   - Upstox      │
                                             │   - Indian APIs │
                                             └─────────────────┘
```

## 🔒 Security & Privacy

- **API Key Security**: All API keys are stored in environment variables
- **Local AI Option**: Ollama provides offline AI capabilities for sensitive queries
- **Rate Limiting**: Built-in rate limiting for all external API calls
- **Data Privacy**: No user data is stored permanently

## 🚀 Deployment

### Using Docker
```bash
docker build -t dalaal-street-bot .
docker run -p 8000:8000 --env-file .env dalaal-street-bot
```

### Using Azure App Service
1. Configure environment variables in Azure portal
2. Deploy using Azure CLI or GitHub Actions
3. Update CORS settings for your domain

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Acknowledgments

- **Groq** for ultra-fast AI inference
- **Azure OpenAI** for enterprise AI capabilities
- **Google AI** for advanced language models
- **Yahoo Finance** for reliable market data
- **News API** for financial news
- **Upstox** for professional trading data

## 📞 Support

For support, email support@dalaalstreetbot.com or create an issue on GitHub.

---

**Disclaimer**: This chatbot provides information for educational purposes only. Always consult with qualified financial advisors before making investment decisions.<!--
**Dalaal-street-chatbot/Dalaal-street-chatbot** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
