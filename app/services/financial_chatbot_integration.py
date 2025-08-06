"""
Financial Chatbot Integration with Real APIs

This module integrates all available real APIs and services:
- Groq AI for intelligent responses
- News API for real market news
- Upstox API for live Indian stock data
- Scrapers for additional market data
"""

import os
import logging
import asyncio
import re
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Import real service modules
from app.services.groq_service import GroqAIService
from app.services.news_service import NewsService
from config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialChatbotIntegration:
    """
    Integration class to combine all available real APIs
    for the financial chatbot application
    """
    
    def __init__(self):
        """Initialize the Financial Chatbot Integration with real APIs"""
        try:
            # Initialize real API services
            self.groq_ai = GroqAIService() if config.GROQ_API_KEY else None
            self.news_service = NewsService() if config.NEWS_API else None
            
            # Initialize other available services
            self.upstox_access_token = config.UPSTOX_ACCESS_TOKEN
            self.indian_stock_api_key = config.INDIAN_STOCK_API_KEY
            self.indian_stock_api_base_url = config.INDIAN_STOCK_API_BASE_URL
            
            logger.info("Financial Chatbot Integration initialized with real APIs")
        except Exception as e:
            logger.error(f"Error during Financial Chatbot Integration initialization: {e}")
            # Initialize with None to handle gracefully
            self.groq_ai = None
            self.news_service = None
            self.upstox_access_token = None
    
    async def process_user_message(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message using real APIs and services
        
        Args:
            message: The user's text message
            session_id: Optional session identifier
            user_data: Optional user profile data
            
        Returns:
            Processed chatbot response using real data
        """
        try:
            # Analyze user intent using simple keyword matching
            intent, entities = self._analyze_user_intent(message)
            
            # Route to appropriate handler based on intent
            if intent == "stock_price":
                return await self._handle_stock_price_query(message, entities, session_id)
            elif intent == "market_news":
                return await self._handle_market_news_query(message, entities, session_id)
            elif intent == "company_news":
                return await self._handle_company_news_query(message, entities, session_id)
            elif intent == "market_analysis":
                return await self._handle_market_analysis_query(message, entities, session_id)
            elif intent == "greeting":
                return await self._handle_greeting(message, session_id)
            else:
                # Use Groq AI for general financial queries
                return await self._handle_general_query(message, session_id)
                
        except Exception as e:
            logger.error(f"Error processing user message: {str(e)}")
            return {
                "status": "success",
                "session_id": session_id if session_id else "unknown",
                "response": f"I'm sorry, I encountered an error while processing your message. Please try again.",
                "intent": "error",
                "entities": [],
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_user_intent(self, message: str) -> tuple[str, List[str]]:
        """Simple intent analysis using keywords"""
        message_lower = message.lower()
        entities = []
        
        # Extract potential stock symbols (3-5 uppercase letters)
        import re
        stock_symbols = re.findall(r'\b[A-Z]{3,5}\b', message)
        entities.extend(stock_symbols)
        
        # Extract company names (common Indian companies)
        companies = ['reliance', 'tcs', 'infosys', 'hdfc', 'icici', 'bajaj', 'maruti', 'bharti', 'wipro', 'adani']
        for company in companies:
            if company in message_lower:
                entities.append(company.upper())
        
        # Intent classification
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "greeting", entities
        elif any(word in message_lower for word in ['price', 'quote', 'value', 'cost']) and (stock_symbols or any(company in message_lower for company in companies)):
            return "stock_price", entities
        elif any(word in message_lower for word in ['news', 'updates', 'headlines']) and (stock_symbols or any(company in message_lower for company in companies)):
            return "company_news", entities
        elif any(word in message_lower for word in ['news', 'market', 'updates', 'headlines']):
            return "market_news", entities
        elif any(word in message_lower for word in ['analysis', 'forecast', 'prediction', 'trend']):
            return "market_analysis", entities
        else:
            return "general", entities
    
    async def _handle_stock_price_query(self, message: str, entities: List[str], session_id: Optional[str]) -> Dict[str, Any]:
        """Handle stock price queries using Upstox API"""
        session_id = session_id or "unknown"
        try:
            # If we have stock symbols, get their prices
            if entities and self.upstox_access_token:
                stock_data = await self._get_upstox_stock_data(entities[0])
                if stock_data:
                    response = f"📈 {entities[0]} Stock Information:\n"
                    response += f"Current Price: ₹{stock_data.get('price', 'N/A')}\n"
                    response += f"Change: {stock_data.get('change', 'N/A')}\n"
                    response += f"Volume: {stock_data.get('volume', 'N/A')}"
                else:
                    response = f"I couldn't fetch current data for {entities[0]}. Let me get some general market information instead."
                    # Fallback to news
                    if self.news_service:
                        news_data = self.news_service.get_financial_news(query=entities[0])
                        if news_data['status'] == 'success' and news_data['articles']:
                            response += f"\n\nLatest news about {entities[0]}:\n"
                            response += news_data['articles'][0]['title']
            else:
                response = "Please specify a stock symbol (e.g., 'RELIANCE price' or 'TCS quote')"
            
            return {
                "status": "success",
                "response": response,
                "intent": "stock_price",
                "entities": entities,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling stock price query: {e}")
            return await self._handle_general_query(message, session_id)
    
    async def _handle_market_news_query(self, message: str, entities: List[str], session_id: Optional[str]) -> Dict[str, Any]:
        """Handle market news queries using News API"""
        session_id = session_id or "unknown"
        try:
            if self.news_service:
                news_data = self.news_service.get_financial_news(page_size=5)
                
                if news_data['status'] == 'success' and news_data['articles']:
                    response = "📰 Latest Market News:\n\n"
                    for i, article in enumerate(news_data['articles'][:3], 1):
                        response += f"{i}. {article['title']}\n"
                        if article.get('description'):
                            response += f"   {article['description'][:100]}...\n\n"
                else:
                    response = "I couldn't fetch the latest market news right now. Please try again later."
            else:
                response = "News service is currently unavailable."
            
            return {
                "status": "success",
                "response": response,
                "intent": "market_news", 
                "entities": entities,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling market news query: {e}")
            return await self._handle_general_query(message, session_id)
    
    async def _handle_company_news_query(self, message: str, entities: List[str], session_id: Optional[str]) -> Dict[str, Any]:
        """Handle company-specific news queries"""
        session_id = session_id or "unknown"
        try:
            if entities and self.news_service:
                company = entities[0]
                news_data = self.news_service.get_company_news(company, page_size=3)
                
                if news_data['status'] == 'success' and news_data['articles']:
                    response = f"📈 Latest News for {company}:\n\n"
                    for i, article in enumerate(news_data['articles'], 1):
                        response += f"{i}. {article['title']}\n"
                        if article.get('description'):
                            response += f"   {article['description'][:100]}...\n\n"
                else:
                    response = f"I couldn't find recent news for {company}. Let me get general market news instead."
                    return await self._handle_market_news_query(message, entities, session_id)
            else:
                response = "Please specify a company name for news (e.g., 'Reliance news' or 'TCS updates')"
            
            return {
                "status": "success",
                "response": response,
                "intent": "company_news",
                "entities": entities,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling company news query: {e}")
            return await self._handle_general_query(message, session_id)
    
    async def _handle_market_analysis_query(self, message: str, entities: List[str], session_id: str) -> Dict[str, Any]:
        """Handle market analysis queries using Groq AI with real market data"""
        try:
            # Get recent market news for context
            market_context = ""
            if self.news_service:
                news_data = self.news_service.get_financial_news(page_size=3)
                if news_data['status'] == 'success':
                    market_context = "Recent market developments: "
                    for article in news_data['articles']:
                        market_context += f"{article['title']}. "
            
            # Use Groq AI for analysis
            if self.groq_ai:
                system_prompt = f"""
                You are an expert Indian stock market analyst. Provide insightful analysis based on:
                1. Current market context: {market_context}
                2. User's specific query: {message}
                
                Provide actionable insights while reminding users to do their own research.
                """
                
                response = await self.groq_ai.generate_response(
                    message, 
                    context={"market_news": market_context},
                    system_prompt=system_prompt
                )
            else:
                response = "Market analysis service is currently unavailable. Please try again later."
            
            return {
                "status": "success",
                "response": response,
                "intent": "market_analysis",
                "entities": entities,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling market analysis query: {e}")
            return await self._handle_general_query(message, session_id)
    
    async def _handle_greeting(self, message: str, session_id: str) -> Dict[str, Any]:
        """Handle greeting messages"""
        response = """Hello! 👋 I'm the Dalaal Street financial chatbot!

I can help you with:
📈 Real-time stock prices and quotes
📰 Latest market news and updates
🏢 Company-specific news and analysis
📊 Market analysis and insights

What would you like to know about the markets today?"""
        
        return {
            "status": "success",
            "response": response,
            "intent": "greeting",
            "entities": [],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_general_query(self, message: str, session_id: str) -> Dict[str, Any]:
        """Handle general queries using Groq AI"""
        try:
            if self.groq_ai:
                response = await self.groq_ai.generate_response(message)
            else:
                response = f"I understand you're asking about: '{message}'. I'm here to help with financial questions! Could you please be more specific about what market information you need?"
            
            return {
                "status": "success",
                "response": response,
                "intent": "general",
                "entities": [],
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling general query: {e}")
            return {
                "status": "success",
                "response": "I'm here to help with your financial questions! Try asking about stock prices, market news, or company information.",
                "intent": "general",
                "entities": [],
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_upstox_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch real stock data from Upstox API"""
        try:
            if not self.upstox_access_token:
                return None
                
            import requests
            
            headers = {
                'Authorization': f'Bearer {self.upstox_access_token}',
                'Accept': 'application/json'
            }
            
            # Upstox API endpoint for market quotes
            url = f'https://api.upstox.com/v2/market-quote/ltp?instrument_key=NSE_EQ|{symbol}'
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    quote_data = data.get('data', {})
                    return {
                        'symbol': symbol,
                        'price': quote_data.get('last_price'),
                        'change': quote_data.get('net_change'),
                        'volume': quote_data.get('volume')
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Upstox data for {symbol}: {e}")
            return None
    
    async def _process_with_fallback(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process message using fallback logic when Google Cloud services are unavailable
        """
        # Simple keyword-based intent detection
        message_lower = message.lower()
        
        # Greeting intents
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return {
                "status": "success",
                "response": "Hello! I'm the Dalaal Street chatbot. I can help you with stock information, market news, and financial analysis. What would you like to know?",
                "intent": "greeting",
                "entities": [],
                "session_id": session_id or "fallback_session",
                "timestamp": datetime.now().isoformat()
            }
        
        # Stock price queries
        elif any(word in message_lower for word in ['price', 'stock', 'share', 'value']):
            response = "I can help you with stock information! Currently running in limited mode. Please specify which stock you'd like to know about (e.g., 'AAPL price' or 'Tesla stock')."
            
            return {
                "status": "success",
                "response": response,
                "intent": "stock_price",
                "entities": [],
                "session_id": session_id or "fallback_session",
                "timestamp": datetime.now().isoformat()
            }
        
        # Market news queries
        elif any(word in message_lower for word in ['news', 'market', 'updates', 'headlines']):
            response = "I can provide market news and updates. Currently gathering the latest information..."
            
            return {
                "status": "success",
                "response": response,
                "intent": "market_news",
                "entities": [],
                "session_id": session_id or "fallback_session",
                "timestamp": datetime.now().isoformat()
            }
        
        # Help queries
        elif any(word in message_lower for word in ['help', 'what can you do', 'features', 'capabilities']):
            return {
                "status": "success",
                "response": """I'm the Dalaal Street financial chatbot! I can help you with:

📈 Stock prices and information
📰 Market news and updates  
📊 Financial analysis
💡 Investment insights
🔍 Company research

Currently running in limited mode, but I'm still here to assist you! What would you like to explore?""",
                "intent": "help",
                "entities": [],
                "session_id": session_id or "fallback_session",
                "timestamp": datetime.now().isoformat()
            }
        
        # Default response
        else:
            return {
                "status": "success",
                "response": f"I understand you're asking about: '{message}'. I'm currently running in limited mode but I'm here to help! Could you please be more specific about what financial information you need?",
                "intent": "default",
                "entities": [],
                "session_id": session_id or "fallback_session",
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_image_message(
        self, 
        image_data: bytes, 
        session_id: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an image message containing financial charts or documents
        
        Args:
            image_data: The binary image data
            session_id: Optional session identifier
            user_data: Optional user profile data
            
        Returns:
            Processed image analysis response
        """
        try:
            # First, determine if it's a chart or document using Vision API
            vision_analysis = await self.cloud_services.analyze_chart_image(image_data)
            
            if "chart_type" in vision_analysis and vision_analysis["chart_type"] != "unknown":
                # It's a chart, use Vertex AI to analyze it
                chart_analysis = await self.vertex_ai.analyze_financial_chart(image_data)
                
                return {
                    "status": "success",
                    "session_id": session_id,
                    "content_type": "chart",
                    "analysis": chart_analysis,
                    "response": "Here's my analysis of your financial chart:\n\n" + chart_analysis.get("chart_analysis", ""),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # It's likely a document, use Vision API to extract text
                document_analysis = await self.cloud_services.extract_financial_document(image_data)
                
                # Further analyze the extracted text with Vertex AI
                if document_analysis["status"] == "success" and "full_text" in document_analysis:
                    text_analysis = await self.vertex_ai.analyze_financial_document(document_analysis["full_text"])
                    
                    return {
                        "status": "success",
                        "session_id": session_id,
                        "content_type": "document",
                        "document_extraction": document_analysis,
                        "text_analysis": text_analysis,
                        "response": "I've analyzed your financial document. Here are the key insights:\n\n" + 
                                    self._format_document_analysis(text_analysis),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "error",
                        "response": "I couldn't properly analyze the image. Please ensure it contains clear financial information.",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }
        
        except Exception as e:
            logger.error(f"Error processing image message: {str(e)}")
            return {
                "status": "error",
                "session_id": session_id if session_id else "unknown",
                "response": f"I'm sorry, I couldn't process your image: {str(e)}",
                "fallback_response": "I'm sorry, I encountered an error while processing your image. Please try again."
            }
    
    async def get_market_insights(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comprehensive market insights using all available services
        
        Args:
            symbols: Optional list of stock symbols to focus on
            
        Returns:
            Comprehensive market insights
        """
        try:
            insights: Dict[str, Any] = {
                "news": None,
                "sentiment": None,
                "technical_analysis": None,
                "predictions": None
            }
            
            # Get market news
            if symbols and len(symbols) > 0:
                news_tasks = [self.news_service.get_stock_news(symbol) for symbol in symbols]
                news_results = await asyncio.gather(*news_tasks)
                
                # Combine news results
                combined_news = []
                for result in news_results:
                    if result["status"] == "success":
                        combined_news.extend(result.get("news_items", []))
                
                # Sort by date (descending)
                combined_news.sort(key=lambda x: x.get("date", ""), reverse=True)
                
                insights["news"] = {
                    "status": "success",
                    "news_items": combined_news[:10],  # Limit to top 10 news items
                    "symbols": symbols
                }
            else:
                # Get general market news
                insights["news"] = await self.news_service.get_market_news()
            
            # Get sentiment analysis
            if symbols and len(symbols) > 0:
                sentiment_tasks = [self.news_service.get_stock_sentiment(symbol) for symbol in symbols]
                sentiment_results = await asyncio.gather(*sentiment_tasks)
                
                # Format sentiment results
                sentiments = {}
                for i, symbol in enumerate(symbols):
                    if sentiment_results[i]["status"] == "success":
                        sentiments[symbol] = {
                            "sentiment": sentiment_results[i]["sentiment"],
                            "score": sentiment_results[i]["score"]
                        }
                
                insights["sentiment"] = {
                    "status": "success",
                    "sentiments": sentiments,
                    "symbols": symbols
                }
            else:
                # Get general market sentiment
                insights["sentiment"] = await self.news_service.get_market_sentiment()
            
            # Get technical analysis if symbols provided
            if symbols and len(symbols) > 0:
                technical_tasks = []
                for symbol in symbols:
                    # Get historical data first (mock for now)
                    history = self._get_mock_stock_history(symbol, 60)
                    technical_tasks.append(self.vertex_ai.generate_technical_indicators(
                        symbol, {"history": history}
                    ))
                
                technical_results = await asyncio.gather(*technical_tasks)
                
                # Format technical analysis results
                technical_analyses = {}
                for i, symbol in enumerate(symbols):
                    if technical_results[i]["status"] == "success":
                        technical_analyses[symbol] = technical_results[i]["technical_analysis"]
                
                insights["technical_analysis"] = {
                    "status": "success",
                    "analyses": technical_analyses,
                    "symbols": symbols
                }
            
            # Get price predictions if symbols provided
            if symbols and len(symbols) > 0:
                prediction_tasks = []
                for symbol in symbols:
                    # Get historical data first (mock for now)
                    history = self._get_mock_stock_history(symbol, 60)
                    prediction_tasks.append(self.vertex_ai.predict_stock_price(
                        symbol, history, 5
                    ))
                
                prediction_results = await asyncio.gather(*prediction_tasks)
                
                # Format prediction results
                predictions = {}
                for i, symbol in enumerate(symbols):
                    if prediction_results[i]["status"] == "success":
                        predictions[symbol] = prediction_results[i]["predictions"]
                
                insights["predictions"] = {
                    "status": "success",
                    "predictions": predictions,
                    "symbols": symbols
                }
            
            return {
                "status": "success",
                "insights": insights,
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting market insights: {str(e)}")
            return {
                "status": "error",
                "response": f"I'm sorry, I couldn't get market insights at the moment: {str(e)}"
            }
    
    # ===== ENHANCEMENT METHODS =====
    
    async def _enhance_stock_response(self, response: Dict[str, Any]) -> None:
        """Enhance stock price/info response with additional data"""
        if "symbols" not in response or not response["symbols"]:
            return
        
        try:
            symbols = response["symbols"]
            
            # Add technical indicators
            technical_tasks = []
            for symbol in symbols:
                # Get historical data first (mock for now)
                history = self._get_mock_stock_history(symbol, 30)
                technical_tasks.append(self.vertex_ai.generate_technical_indicators(
                    symbol, {"history": history}
                ))
            
            technical_results = await asyncio.gather(*technical_tasks)
            
            # Format technical analysis results
            technical_analyses = {}
            for i, symbol in enumerate(symbols):
                if technical_results[i]["status"] == "success":
                    technical_analyses[symbol] = technical_results[i]["technical_analysis"]
            
            response["technical_analysis"] = technical_analyses
            
            # Add recent news mentions
            news_tasks = [self.news_service.get_stock_news(symbol, limit=3) for symbol in symbols]
            news_results = await asyncio.gather(*news_tasks)
            
            # Format news results
            news_mentions = {}
            for i, symbol in enumerate(symbols):
                if news_results[i]["status"] == "success":
                    news_mentions[symbol] = news_results[i]["news_items"]
            
            response["news_mentions"] = news_mentions
            
            # Enhance response text with technical signals
            if "response" in response and technical_analyses:
                for symbol, analysis in technical_analyses.items():
                    if "signals" in analysis and analysis["signals"]:
                        signal_text = "\n\nTechnical Signals for " + symbol + ":\n"
                        for signal in analysis["signals"][:3]:  # Show top 3 signals
                            signal_text += f"- {signal['indicator']}: {signal['signal'].capitalize()}\n"
                        signal_text += f"Overall: {analysis.get('overall_signal', 'Neutral')}"
                        
                        response["response"] += signal_text
        
        except Exception as e:
            logger.error(f"Error enhancing stock response: {str(e)}")
            # Don't modify the response if enhancement fails
    
    async def _enhance_news_response(self, response: Dict[str, Any]) -> None:
        """Enhance market news response with sentiment analysis"""
        if "news_items" not in response or not response["news_items"]:
            return
        
        try:
            news_items = response["news_items"]
            
            # Extract text for sentiment analysis
            news_texts = [item.get("title", "") + " " + item.get("summary", "") for item in news_items]
            
            # Get sentiment analysis
            sentiment_analysis = await self.vertex_ai.analyze_financial_sentiment(news_texts)
            
            if sentiment_analysis["status"] == "success":
                response["sentiment_analysis"] = sentiment_analysis["sentiment_analysis"]
                
                # Add sentiment summary to response
                sentiment = sentiment_analysis["sentiment_analysis"]["overall_sentiment"].capitalize()
                score = sentiment_analysis["sentiment_analysis"]["overall_score"] * 100
                
                response["response"] += f"\n\nOverall sentiment from these news items: {sentiment} ({score:.1f}%)"
        
        except Exception as e:
            logger.error(f"Error enhancing news response: {str(e)}")
            # Don't modify the response if enhancement fails
    
    async def _enhance_sentiment_response(self, response: Dict[str, Any]) -> None:
        """Enhance sentiment response with additional analysis"""
        if "sentiment_data" not in response:
            return
        
        try:
            sentiment_data = response["sentiment_data"]
            symbols = response.get("symbols", [])
            
            # Add recent news mentions for context
            if symbols:
                news_tasks = [self.news_service.get_stock_news(symbol, limit=2) for symbol in symbols]
                news_results = await asyncio.gather(*news_tasks)
                
                # Format news results
                supporting_news = {}
                for i, symbol in enumerate(symbols):
                    if news_results[i]["status"] == "success":
                        supporting_news[symbol] = news_results[i]["news_items"]
                
                response["supporting_news"] = supporting_news
                
                # Add news headlines to response
                if supporting_news and "response" in response:
                    response["response"] += "\n\nRecent headlines affecting sentiment:"
                    for symbol, news in supporting_news.items():
                        if news:
                            response["response"] += f"\n\n{symbol}:"
                            for item in news:
                                response["response"] += f"\n- {item['title']}"
        
        except Exception as e:
            logger.error(f"Error enhancing sentiment response: {str(e)}")
            # Don't modify the response if enhancement fails
    
    async def _enhance_market_analysis_response(self, response: Dict[str, Any]) -> None:
        """Enhance market analysis response with predictions"""
        symbols = response.get("symbols", [])
        if not symbols:
            return
        
        try:
            # Get price predictions
            prediction_tasks = []
            for symbol in symbols:
                # Get historical data first (mock for now)
                history = self._get_mock_stock_history(symbol, 60)
                prediction_tasks.append(self.vertex_ai.predict_stock_price(
                    symbol, history, 5
                ))
            
            prediction_results = await asyncio.gather(*prediction_tasks)
            
            # Format prediction results
            predictions = {}
            for i, symbol in enumerate(symbols):
                if prediction_results[i]["status"] == "success":
                    predictions[symbol] = prediction_results[i]["predictions"]
            
            response["predictions"] = predictions
            
            # Add prediction summary to response
            if predictions and "response" in response:
                response["response"] += "\n\nPrice Predictions:"
                for symbol, prediction_data in predictions.items():
                    if prediction_data:
                        # Show prediction for day 1 and day 5
                        first_day = prediction_data[0] if prediction_data else None
                        last_day = prediction_data[-1] if len(prediction_data) > 1 else None
                        
                        response["response"] += f"\n\n{symbol}:"
                        if first_day:
                            response["response"] += f"\n- Tomorrow: ${first_day['predicted_price']:.2f} (Confidence: {first_day['confidence']*100:.1f}%)"
                        if last_day:
                            response["response"] += f"\n- 5 Day: ${last_day['predicted_price']:.2f} (Confidence: {last_day['confidence']*100:.1f}%)"
        
        except Exception as e:
            logger.error(f"Error enhancing market analysis response: {str(e)}")
            # Don't modify the response if enhancement fails
    
    async def _enhance_comparison_response(self, response: Dict[str, Any]) -> None:
        """Enhance stock comparison response with additional analysis"""
        symbols = response.get("symbols", [])
        if not symbols or len(symbols) < 2:
            return
        
        try:
            # Get sentiment analysis for each stock
            sentiment_tasks = [self.news_service.get_stock_sentiment(symbol) for symbol in symbols]
            sentiment_results = await asyncio.gather(*sentiment_tasks)
            
            # Format sentiment results
            sentiments = {}
            for i, symbol in enumerate(symbols):
                if sentiment_results[i]["status"] == "success":
                    sentiments[symbol] = {
                        "sentiment": sentiment_results[i]["sentiment"],
                        "score": sentiment_results[i]["score"]
                    }
            
            response["sentiments"] = sentiments
            
            # Add sentiment comparison to response
            if sentiments and "response" in response:
                response["response"] += "\n\nMarket Sentiment:"
                for symbol, sentiment_data in sentiments.items():
                    sentiment = sentiment_data["sentiment"].capitalize()
                    score = sentiment_data["score"] * 100
                    response["response"] += f"\n{symbol}: {sentiment} ({score:.1f}%)"
        
        except Exception as e:
            logger.error(f"Error enhancing comparison response: {str(e)}")
            # Don't modify the response if enhancement fails
    
    # ===== HELPER METHODS =====
    
    def _format_document_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format document analysis results into readable text"""
        if "financial_analysis" not in analysis:
            return "No financial data could be extracted from this document."
        
        financial_analysis = analysis["financial_analysis"]
        
        formatted_text = ""
        
        # Extract company info
        if "company_info" in financial_analysis:
            company_info = financial_analysis["company_info"]
            if "name" in company_info:
                formatted_text += f"Company: {company_info['name']}\n"
            if "ticker" in company_info:
                formatted_text += f"Ticker: {company_info['ticker']}\n"
            if "reporting_period" in company_info:
                formatted_text += f"Period: {company_info['reporting_period']}\n"
            formatted_text += "\n"
        
        # Extract financial metrics
        if "financial_metrics" in financial_analysis:
            metrics = financial_analysis["financial_metrics"]
            formatted_text += "Key Financial Metrics:\n"
            for key, value in metrics.items():
                formatted_text += f"- {key.replace('_', ' ').title()}: {value}\n"
            formatted_text += "\n"
        
        # Extract highlights
        if "highlights" in financial_analysis and financial_analysis["highlights"]:
            formatted_text += "Highlights:\n"
            for highlight in financial_analysis["highlights"]:
                formatted_text += f"- {highlight}\n"
            formatted_text += "\n"
        
        # Extract outlook
        if "outlook" in financial_analysis and financial_analysis["outlook"]:
            formatted_text += f"Outlook: {financial_analysis['outlook']}\n"
        
        return formatted_text
    
    def _get_mock_stock_history(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Generate mock stock history data for demo purposes"""
        import random
        import numpy as np
        from datetime import datetime, timedelta
        
        # Seed with symbol to get consistent results
        random.seed(hash(symbol) % 10000)
        np.random.seed(hash(symbol) % 10000)
        
        # Generate start price
        base_price = random.uniform(50, 500)
        
        # Generate price movements using random walk
        price_changes = np.random.normal(0, base_price * 0.015, days)
        
        # Calculate cumulative prices
        prices = [base_price]
        for change in price_changes:
            new_price = max(0.1, prices[-1] + change)  # Ensure price doesn't go negative
            prices.append(new_price)
        
        # Generate historical data
        history = []
        end_date = datetime.now()
        
        for i in range(days):
            date = (end_date - timedelta(days=days-i)).strftime('%Y-%m-%d')
            close_price = prices[i]
            
            # Generate other price points
            open_price = close_price * random.uniform(0.99, 1.01)
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.02)
            low_price = min(open_price, close_price) * random.uniform(0.98, 1.0)
            
            # Generate volume
            volume = int(random.uniform(100000, 10000000))
            
            history.append({
                "date": date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
        
        return history


# Initialize the singleton instance
financial_chatbot = FinancialChatbotIntegration()
