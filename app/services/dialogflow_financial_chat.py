"""
Dialogflow Financial Chat Integration Module

This module provides specialized integration with Google's Dialogflow:
1. Financial intent recognition
2. Entity extraction (stock symbols, monetary values, etc.)
3. Market query handling
4. Portfolio management conversation flows
"""

import os
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio

import google.cloud.dialogflow_v2 as dialogflow
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToDict

from config.settings import config
from app.services.news_service import news_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DialogflowFinancialChat:
    """
    Dialogflow integration for financial conversations
    """
    
    def __init__(self):
        """Initialize the Dialogflow Financial Chat service"""
        # Initialize credentials and clients based on environment variables
        self.project_id = config.GOOGLE_CLOUD_PROJECT_ID
        self.session_clients = {}
        self.sessions = {}
        self.contexts_client = None
        self.intents_client = None
        
        # Check if required environment variables are set
        if not self.project_id:
            logger.warning("GOOGLE_CLOUD_PROJECT_ID is not set. Dialogflow features may not work.")
        
        # Path to service account credentials file
        self.credentials_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
        self._credentials = None
        
        # Initialize services on-demand
        self._initialized = False
        logger.info("Dialogflow Financial Chat service initialized")
    
    def initialize(self):
        """Initialize Dialogflow clients"""
        if not self._initialized:
            try:
                # Load credentials
                if self.credentials_path and os.path.exists(self.credentials_path):
                    self._credentials = service_account.Credentials.from_service_account_file(
                        self.credentials_path
                    )
                    logger.info("Using service account credentials from file")
                else:
                    # Fall back to default credentials
                    self._credentials = None
                    logger.info("Using default credentials")
                
                # Initialize Dialogflow clients
                self.contexts_client = dialogflow.ContextsClient(credentials=self._credentials)
                self.intents_client = dialogflow.IntentsClient(credentials=self._credentials)
                
                self._initialized = True
                logger.info("Dialogflow clients initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Dialogflow clients: {str(e)}")
                raise
    
    def get_session_client(self, session_id: str):
        """Get or create a session client for the specified session ID"""
        if session_id not in self.session_clients:
            try:
                self.initialize()
                self.session_clients[session_id] = dialogflow.SessionsClient(credentials=self._credentials)
                # Initialize session tracking
                self.sessions[session_id] = {
                    "created_at": datetime.now(),
                    "last_activity": datetime.now(),
                    "active_contexts": []
                }
                logger.info(f"Created new session client for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to create session client: {str(e)}")
                raise
        else:
            # Update last activity
            self.sessions[session_id]["last_activity"] = datetime.now()
        
        return self.session_clients[session_id]
    
    async def detect_financial_intent(
        self, 
        text: str, 
        session_id: Optional[str] = None,
        language_code: str = "en"
    ) -> Dict[str, Any]:
        """
        Detect financial intent from user text
        
        Args:
            text: User input text
            session_id: Optional session ID (will generate if not provided)
            language_code: Language code (default: 'en')
            
        Returns:
            Detected intent, entities, and response
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            session_client = self.get_session_client(session_id)
            session_path = session_client.session_path(self.project_id, session_id)
            
            # Create text input
            text_input = dialogflow.TextInput(text=text, language_code=language_code)
            query_input = dialogflow.QueryInput(text=text_input)
            
            # Send request
            response = session_client.detect_intent(
                request={"session": session_path, "query_input": query_input}
            )
            
            # Process response
            query_result = response.query_result
            intent = query_result.intent.display_name
            
            # Update active contexts
            self.sessions[session_id]["active_contexts"] = [
                context.name.split('/')[-1] for context in query_result.output_contexts
            ]
            
            # Extract parameters/entities
            parameters = {}
            if query_result.parameters:
                parameters = MessageToDict(query_result.parameters)
            
            # Get response message
            messages = []
            for msg in query_result.fulfillment_messages:
                if msg.text.text:
                    messages.extend(msg.text.text)
            
            return {
                "status": "success",
                "session_id": session_id,
                "intent": intent,
                "confidence": query_result.intent_detection_confidence,
                "parameters": parameters,
                "messages": messages,
                "action": query_result.action
            }
        except Exception as e:
            logger.error(f"Error detecting financial intent: {str(e)}")
            return {
                "status": "error",
                "session_id": session_id,
                "response": f"Failed to detect intent: {str(e)}",
            }
    
    async def process_financial_chat(
        self, 
        text: str, 
        session_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a financial chat message with enhanced context
        
        Args:
            text: User input text
            session_id: Optional session ID (will generate if not provided)
            user_context: Additional user context information
            
        Returns:
            Processed response with financial information
        """
        try:
            # Detect intent
            intent_response = await self.detect_financial_intent(text, session_id)
            
            if intent_response["status"] != "success":
                return intent_response
            
            session_id = intent_response["session_id"]
            intent = intent_response["intent"]
            parameters = intent_response["parameters"]
            
            # Update session with the user context
            if user_context and session_id in self.sessions:
                if "user_context" not in self.sessions[session_id]:
                    self.sessions[session_id]["user_context"] = {}
                self.sessions[session_id]["user_context"].update(user_context)
            
            # Process specific financial intents with additional data enrichment
            if intent == "StockPrice" or intent == "StockInfo":
                return await self._handle_stock_intent(intent_response, session_id)
            
            elif intent == "MarketNews":
                return await self._handle_market_news_intent(intent_response, session_id)
            
            elif intent == "StockComparison":
                return await self._handle_stock_comparison_intent(intent_response, session_id)
            
            elif intent == "MarketSentiment":
                return await self._handle_sentiment_intent(intent_response, session_id)
            
            elif intent == "PortfolioManagement":
                return await self._handle_portfolio_intent(intent_response, session_id)
            
            # Generic response for other intents
            return {
                "status": "success",
                "session_id": session_id,
                "intent": intent,
                "confidence": intent_response["confidence"],
                "response": intent_response["messages"][0] if intent_response["messages"] else "I'm not sure how to respond to that.",
                "response_type": "text"
            }
        
        except Exception as e:
            logger.error(f"Error processing financial chat: {str(e)}")
            return {
                "status": "error",
                "session_id": session_id if session_id else "unknown",
                "response": f"Failed to process chat: {str(e)}",
            }
    
    async def create_context(
        self,
        session_id: str,
        context_name: str,
        parameters: Dict[str, Any],
        lifespan_count: int = 5
    ) -> Dict[str, Any]:
        """
        Create a context for the current session
        
        Args:
            session_id: Session identifier
            context_name: Name of the context to create
            parameters: Parameters to store in the context
            lifespan_count: Number of turns the context remains active
            
        Returns:
            Status of context creation
        """
        self.initialize()
        
        try:
            # Create context path
            parent = self.contexts_client.session_path(self.project_id, session_id)
            context_id = f"{parent}/contexts/{context_name}"
            
            # Create context
            context = dialogflow.Context(
                name=context_id,
                lifespan_count=lifespan_count,
                parameters=parameters
            )
            
            # Create the context
            created_context = self.contexts_client.create_context(
                parent=parent, context=context
            )
            
            # Update session tracking
            if session_id in self.sessions:
                if context_name not in self.sessions[session_id]["active_contexts"]:
                    self.sessions[session_id]["active_contexts"].append(context_name)
            
            return {
                "status": "success",
                "context_name": context_name,
                "lifespan": lifespan_count,
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"Error creating context: {str(e)}")
            return {
                "status": "error",
                "response": f"Failed to create context: {str(e)}",
                "session_id": session_id
            }
    
    async def delete_context(
        self,
        session_id: str,
        context_name: str
    ) -> Dict[str, Any]:
        """
        Delete a context from the current session
        
        Args:
            session_id: Session identifier
            context_name: Name of the context to delete
            
        Returns:
            Status of context deletion
        """
        self.initialize()
        
        try:
            # Create context path
            context_path = self.contexts_client.context_path(
                self.project_id, session_id, context_name
            )
            
            # Delete the context
            self.contexts_client.delete_context(name=context_path)
            
            # Update session tracking
            if session_id in self.sessions and context_name in self.sessions[session_id]["active_contexts"]:
                self.sessions[session_id]["active_contexts"].remove(context_name)
            
            return {
                "status": "success",
                "context_name": context_name,
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"Error deleting context: {str(e)}")
            return {
                "status": "error",
                "response": f"Failed to delete context: {str(e)}",
                "session_id": session_id
            }
    
    async def cleanup_old_sessions(self, max_age_hours: int = 12):
        """
        Cleanup old sessions to prevent memory leaks
        
        Args:
            max_age_hours: Maximum age of inactive sessions in hours
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            now = datetime.now()
            sessions_to_remove = []
            
            for session_id, session_data in self.sessions.items():
                last_activity = session_data["last_activity"]
                age_hours = (now - last_activity).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            # Remove old sessions
            for session_id in sessions_to_remove:
                if session_id in self.session_clients:
                    del self.session_clients[session_id]
                if session_id in self.sessions:
                    del self.sessions[session_id]
            
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
            return len(sessions_to_remove)
        
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {str(e)}")
            return 0
    
    # ===== INTENT HANDLERS =====
    
    async def _handle_stock_intent(
        self, 
        intent_response: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Handle stock price or information intent"""
        try:
            parameters = intent_response["parameters"]
            intent = intent_response["intent"]
            
            # Extract stock symbols
            stock_symbols = parameters.get("stock-symbol", [])
            if isinstance(stock_symbols, str):
                stock_symbols = [stock_symbols]
            
            if not stock_symbols:
                return {
                    "status": "success",
                    "session_id": session_id,
                    "intent": intent,
                    "response": "Which stock would you like information about?",
                    "response_type": "text",
                    "needs_followup": True
                }
            
            # In a real implementation, we would fetch actual stock data here
            # For now, generate a mock response
            stock_data = {}
            for symbol in stock_symbols:
                stock_data[symbol] = self._generate_mock_stock_data(symbol)
            
            # Create response based on intent type
            if intent == "StockPrice":
                response = self._format_stock_price_response(stock_data)
            else:  # StockInfo
                response = self._format_stock_info_response(stock_data)
            
            return {
                "status": "success",
                "session_id": session_id,
                "intent": intent,
                "symbols": stock_symbols,
                "stock_data": stock_data,
                "response": response,
                "response_type": "enhanced_stock",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling stock intent: {str(e)}")
            return {
                "status": "error",
                "session_id": session_id,
                "response": f"Failed to process stock information: {str(e)}",
            }
    
    async def _handle_market_news_intent(
        self, 
        intent_response: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Handle market news intent"""
        try:
            parameters = intent_response["parameters"]
            
            # Extract parameters
            stock_symbols = parameters.get("stock-symbol", [])
            if isinstance(stock_symbols, str):
                stock_symbols = [stock_symbols]
            
            category = parameters.get("news-category", "general")
            date_period = parameters.get("date-period", "today")
            
            # Use the news service to get actual news
            try:
                if stock_symbols:
                    # Get news for specific stocks
                    news_items = []
                    for symbol in stock_symbols:
                        stock_news = await news_service.get_stock_news(symbol)
                        if stock_news["status"] == "success":
                            news_items.extend(stock_news["news_items"])
                else:
                    # Get general market news
                    market_news = await news_service.get_market_news(category)
                    news_items = market_news.get("news_items", [])
            except Exception as e:
                logger.error(f"Error fetching news: {str(e)}")
                news_items = []  # Fallback to empty list
            
            # Format the response
            if news_items:
                # Limit to top 5 news items
                top_news = news_items[:5]
                
                # Create response text
                if stock_symbols:
                    symbols_text = ", ".join(stock_symbols)
                    response = f"Here's the latest news for {symbols_text}:\n\n"
                else:
                    response = f"Here's the latest {category} market news:\n\n"
                
                for i, item in enumerate(top_news, 1):
                    response += f"{i}. {item['title']}\n"
                    if 'source' in item:
                        response += f"   Source: {item['source']}\n"
                    if 'summary' in item:
                        response += f"   {item['summary'][:100]}...\n"
                    response += "\n"
            else:
                if stock_symbols:
                    symbols_text = ", ".join(stock_symbols)
                    response = f"I couldn't find any recent news for {symbols_text}."
                else:
                    response = "I couldn't find any recent market news."
            
            return {
                "status": "success",
                "session_id": session_id,
                "intent": "MarketNews",
                "news_items": news_items[:5] if news_items else [],
                "response": response,
                "response_type": "news",
                "symbols": stock_symbols,
                "category": category,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling market news intent: {str(e)}")
            return {
                "status": "error",
                "session_id": session_id,
                "response": f"Failed to process market news: {str(e)}",
            }
    
    async def _handle_stock_comparison_intent(
        self, 
        intent_response: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Handle stock comparison intent"""
        try:
            parameters = intent_response["parameters"]
            
            # Extract stock symbols
            stock_symbols = parameters.get("stock-symbol", [])
            if isinstance(stock_symbols, str):
                stock_symbols = [stock_symbols]
            
            if len(stock_symbols) < 2:
                return {
                    "status": "success",
                    "session_id": session_id,
                    "intent": "StockComparison",
                    "response": "Please specify at least two stocks to compare.",
                    "response_type": "text",
                    "needs_followup": True
                }
            
            # In a real implementation, we would fetch actual stock data here
            # For now, generate mock data
            stock_data = {}
            for symbol in stock_symbols:
                stock_data[symbol] = self._generate_mock_stock_data(symbol)
            
            # Format comparison response
            response = f"Comparison of {', '.join(stock_symbols)}:\n\n"
            
            # Compare prices
            response += "Current Prices:\n"
            for symbol, data in stock_data.items():
                response += f"{symbol}: ${data['price']:.2f} ({data['change_percent']:.2f}%)\n"
            
            response += "\nKey Metrics:\n"
            for symbol, data in stock_data.items():
                response += f"{symbol} - P/E: {data['pe_ratio']:.2f}, Div: {data['dividend_yield']:.2f}%\n"
            
            response += "\nPerformance (YTD):\n"
            for symbol, data in stock_data.items():
                response += f"{symbol}: {data['ytd_performance']:.2f}%\n"
            
            return {
                "status": "success",
                "session_id": session_id,
                "intent": "StockComparison",
                "symbols": stock_symbols,
                "comparison_data": stock_data,
                "response": response,
                "response_type": "comparison",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling stock comparison intent: {str(e)}")
            return {
                "status": "error",
                "session_id": session_id,
                "response": f"Failed to compare stocks: {str(e)}",
            }
    
    async def _handle_sentiment_intent(
        self, 
        intent_response: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Handle market sentiment intent"""
        try:
            parameters = intent_response["parameters"]
            
            # Extract parameters
            stock_symbols = parameters.get("stock-symbol", [])
            if isinstance(stock_symbols, str):
                stock_symbols = [stock_symbols]
            
            market_index = parameters.get("market-index", "")
            
            # Use news service to get sentiment data
            if stock_symbols:
                # Get sentiment for specific stocks
                sentiment_data = {}
                for symbol in stock_symbols:
                    try:
                        stock_sentiment = await news_service.get_stock_sentiment(symbol)
                        if stock_sentiment["status"] == "success":
                            sentiment_data[symbol] = {
                                "sentiment": stock_sentiment["sentiment"],
                                "score": stock_sentiment["score"],
                                "sources": stock_sentiment.get("sources", 0)
                            }
                    except Exception as e:
                        logger.error(f"Error fetching sentiment for {symbol}: {str(e)}")
                        sentiment_data[symbol] = {
                            "sentiment": "neutral",
                            "score": 0.5,
                            "sources": 0
                        }
            else:
                # Get general market sentiment
                try:
                    market_sentiment = await news_service.get_market_sentiment()
                    sentiment_data = {
                        "market": {
                            "sentiment": market_sentiment["sentiment"],
                            "score": market_sentiment["score"],
                            "sources": market_sentiment.get("sources", 0)
                        }
                    }
                except Exception as e:
                    logger.error(f"Error fetching market sentiment: {str(e)}")
                    sentiment_data = {
                        "market": {
                            "sentiment": "neutral",
                            "score": 0.5,
                            "sources": 0
                        }
                    }
            
            # Format sentiment response
            if stock_symbols:
                response = "Current sentiment analysis:\n\n"
                for symbol, data in sentiment_data.items():
                    sentiment_text = data["sentiment"].capitalize()
                    score = data["score"] * 100
                    response += f"{symbol}: {sentiment_text} sentiment ({score:.1f}%)"
                    if data["sources"] > 0:
                        response += f" based on {data['sources']} sources\n"
                    else:
                        response += "\n"
            else:
                market_data = sentiment_data.get("market", {"sentiment": "neutral", "score": 0.5})
                sentiment_text = market_data["sentiment"].capitalize()
                score = market_data["score"] * 100
                response = f"Overall market sentiment: {sentiment_text} ({score:.1f}%)"
                if market_data.get("sources", 0) > 0:
                    response += f"\nBased on {market_data['sources']} news sources"
            
            return {
                "status": "success",
                "session_id": session_id,
                "intent": "MarketSentiment",
                "sentiment_data": sentiment_data,
                "symbols": stock_symbols,
                "market_index": market_index if market_index else "general",
                "response": response,
                "response_type": "sentiment",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error handling sentiment intent: {str(e)}")
            return {
                "status": "error",
                "session_id": session_id,
                "response": f"Failed to analyze sentiment: {str(e)}",
            }
    
    async def _handle_portfolio_intent(
        self, 
        intent_response: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Handle portfolio management intent"""
        try:
            parameters = intent_response["parameters"]
            
            # Get portfolio action
            action = parameters.get("portfolio-action", "")
            stock_symbols = parameters.get("stock-symbol", [])
            if isinstance(stock_symbols, str):
                stock_symbols = [stock_symbols]
                
            quantity = parameters.get("number", 0)
            
            # Check if we have user context with portfolio information
            user_portfolio = None
            if session_id in self.sessions and "user_context" in self.sessions[session_id]:
                user_portfolio = self.sessions[session_id]["user_context"].get("portfolio", None)
            
            # Mock portfolio if not found in context
            if not user_portfolio:
                user_portfolio = self._generate_mock_portfolio()
                # Save in session
                if session_id in self.sessions:
                    if "user_context" not in self.sessions[session_id]:
                        self.sessions[session_id]["user_context"] = {}
                    self.sessions[session_id]["user_context"]["portfolio"] = user_portfolio
            
            # Process the portfolio action
            if action == "view":
                return self._handle_view_portfolio(user_portfolio, session_id)
            elif action == "add" and stock_symbols:
                return self._handle_add_to_portfolio(user_portfolio, stock_symbols, quantity, session_id)
            elif action == "remove" and stock_symbols:
                return self._handle_remove_from_portfolio(user_portfolio, stock_symbols, quantity, session_id)
            elif action == "analyze":
                return self._handle_analyze_portfolio(user_portfolio, session_id)
            else:
                return {
                    "status": "success",
                    "session_id": session_id,
                    "intent": "PortfolioManagement",
                    "response": "What would you like to do with your portfolio? You can view, add stocks, remove stocks, or analyze it.",
                    "response_type": "text",
                    "needs_followup": True
                }
        
        except Exception as e:
            logger.error(f"Error handling portfolio intent: {str(e)}")
            return {
                "status": "error",
                "session_id": session_id,
                "response": f"Failed to manage portfolio: {str(e)}",
            }
    
    # ===== HELPER METHODS =====
    
    def _generate_mock_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock stock data for demo purposes"""
        import random
        
        # Seed with symbol to get consistent results
        random.seed(hash(symbol) % 10000)
        
        price = random.uniform(50, 500)
        change = random.uniform(-5, 5)
        change_percent = (change / price) * 100
        
        return {
            "symbol": symbol,
            "price": price,
            "change": change,
            "change_percent": change_percent,
            "volume": random.randint(100000, 10000000),
            "pe_ratio": random.uniform(10, 30),
            "market_cap": random.uniform(1, 200) * 1000000000,
            "dividend_yield": random.uniform(0, 3),
            "ytd_performance": random.uniform(-15, 30),
            "last_updated": datetime.now().isoformat()
        }
    
    def _format_stock_price_response(self, stock_data: Dict[str, Dict[str, Any]]) -> str:
        """Format stock price response"""
        if len(stock_data) == 1:
            symbol = list(stock_data.keys())[0]
            data = stock_data[symbol]
            
            change_text = f"+${data['change']:.2f} (+{data['change_percent']:.2f}%)" if data['change'] >= 0 else f"-${abs(data['change']):.2f} ({data['change_percent']:.2f}%)"
            
            return f"{symbol} is currently trading at ${data['price']:.2f}, {change_text}."
        else:
            response = "Current stock prices:\n\n"
            for symbol, data in stock_data.items():
                change_text = f"+{data['change_percent']:.2f}%" if data['change'] >= 0 else f"{data['change_percent']:.2f}%"
                response += f"{symbol}: ${data['price']:.2f} ({change_text})\n"
            return response
    
    def _format_stock_info_response(self, stock_data: Dict[str, Dict[str, Any]]) -> str:
        """Format stock info response"""
        if len(stock_data) == 1:
            symbol = list(stock_data.keys())[0]
            data = stock_data[symbol]
            
            response = f"{symbol} Stock Information:\n\n"
            response += f"Price: ${data['price']:.2f}\n"
            response += f"Change: {'+' if data['change'] >= 0 else ''}{data['change']:.2f} ({data['change_percent']:.2f}%)\n"
            response += f"Volume: {data['volume']:,}\n"
            response += f"P/E Ratio: {data['pe_ratio']:.2f}\n"
            response += f"Market Cap: ${data['market_cap'] / 1000000000:.2f}B\n"
            response += f"Dividend Yield: {data['dividend_yield']:.2f}%\n"
            response += f"YTD Performance: {data['ytd_performance']:.2f}%"
            
            return response
        else:
            response = "Stock Information:\n\n"
            for symbol, data in stock_data.items():
                response += f"--- {symbol} ---\n"
                response += f"Price: ${data['price']:.2f} ({'+' if data['change'] >= 0 else ''}{data['change_percent']:.2f}%)\n"
                response += f"P/E: {data['pe_ratio']:.2f}, Market Cap: ${data['market_cap'] / 1000000000:.2f}B\n\n"
            return response
    
    def _generate_mock_portfolio(self) -> Dict[str, Any]:
        """Generate mock portfolio for demo purposes"""
        import random
        
        portfolio = {
            "holdings": [
                {"symbol": "AAPL", "quantity": 10, "avg_price": 150.50},
                {"symbol": "MSFT", "quantity": 5, "avg_price": 280.75},
                {"symbol": "AMZN", "quantity": 2, "avg_price": 3200.30},
                {"symbol": "GOOGL", "quantity": 3, "avg_price": 2700.10}
            ],
            "cash_balance": random.uniform(5000, 15000),
            "total_invested": 0,
            "total_value": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Calculate totals
        total_invested = 0
        total_value = 0
        
        for holding in portfolio["holdings"]:
            holding_invested = holding["quantity"] * holding["avg_price"]
            total_invested += holding_invested
            
            # Get current price from mock data
            current_data = self._generate_mock_stock_data(holding["symbol"])
            holding["current_price"] = current_data["price"]
            holding["current_value"] = holding["quantity"] * holding["current_price"]
            holding["profit_loss"] = holding["current_value"] - holding_invested
            holding["profit_loss_percent"] = (holding["profit_loss"] / holding_invested) * 100
            
            total_value += holding["current_value"]
        
        portfolio["total_invested"] = total_invested
        portfolio["total_value"] = total_value
        portfolio["total_profit_loss"] = total_value - total_invested
        portfolio["total_profit_loss_percent"] = (portfolio["total_profit_loss"] / total_invested) * 100
        
        return portfolio
    
    def _handle_view_portfolio(
        self, 
        portfolio: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Handle view portfolio action"""
        # Format portfolio view response
        response = "Your Portfolio Summary:\n\n"
        
        # Add holdings table
        response += "Holdings:\n"
        for holding in portfolio["holdings"]:
            symbol = holding["symbol"]
            quantity = holding["quantity"]
            current_value = holding["current_value"]
            pl_percent = holding["profit_loss_percent"]
            pl_sign = "+" if pl_percent >= 0 else ""
            
            response += f"{symbol}: {quantity} shares, ${current_value:.2f}, {pl_sign}{pl_percent:.2f}%\n"
        
        # Add summary
        response += f"\nCash Balance: ${portfolio['cash_balance']:.2f}\n"
        response += f"Total Portfolio Value: ${portfolio['total_value']:.2f}\n"
        
        # Add performance
        pl_sign = "+" if portfolio["total_profit_loss"] >= 0 else ""
        response += f"Total Profit/Loss: {pl_sign}${portfolio['total_profit_loss']:.2f} ({pl_sign}{portfolio['total_profit_loss_percent']:.2f}%)"
        
        return {
            "status": "success",
            "session_id": session_id,
            "intent": "PortfolioManagement",
            "subintent": "view",
            "portfolio": portfolio,
            "response": response,
            "response_type": "portfolio",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_add_to_portfolio(
        self, 
        portfolio: Dict[str, Any], 
        symbols: List[str],
        quantity: int,
        session_id: str
    ) -> Dict[str, Any]:
        """Handle add to portfolio action"""
        if quantity <= 0:
            quantity = 1  # Default to 1 if not specified
        
        added_symbols = []
        failed_symbols = []
        
        for symbol in symbols:
            # Get current price from mock data
            stock_data = self._generate_mock_stock_data(symbol)
            current_price = stock_data["price"]
            
            # Check if user has enough cash
            cost = current_price * quantity
            if cost > portfolio["cash_balance"]:
                failed_symbols.append({"symbol": symbol, "reason": "insufficient_funds"})
                continue
            
            # Check if symbol already exists in portfolio
            existing_holding = None
            for holding in portfolio["holdings"]:
                if holding["symbol"] == symbol:
                    existing_holding = holding
                    break
            
            if existing_holding:
                # Update existing holding
                old_avg_price = existing_holding["avg_price"]
                old_quantity = existing_holding["quantity"]
                new_quantity = old_quantity + quantity
                
                # Calculate new average price
                existing_holding["avg_price"] = ((old_avg_price * old_quantity) + (current_price * quantity)) / new_quantity
                existing_holding["quantity"] = new_quantity
                existing_holding["current_price"] = current_price
                existing_holding["current_value"] = existing_holding["quantity"] * current_price
                existing_holding["profit_loss"] = existing_holding["current_value"] - (existing_holding["avg_price"] * existing_holding["quantity"])
                existing_holding["profit_loss_percent"] = (existing_holding["profit_loss"] / (existing_holding["avg_price"] * existing_holding["quantity"])) * 100
            else:
                # Add new holding
                new_holding = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_price": current_price,
                    "current_price": current_price,
                    "current_value": current_price * quantity,
                    "profit_loss": 0,
                    "profit_loss_percent": 0
                }
                portfolio["holdings"].append(new_holding)
            
            # Deduct from cash balance
            portfolio["cash_balance"] -= cost
            
            added_symbols.append({
                "symbol": symbol,
                "quantity": quantity,
                "price": current_price,
                "cost": cost
            })
        
        # Recalculate portfolio totals
        total_invested = 0
        total_value = 0
        
        for holding in portfolio["holdings"]:
            holding_invested = holding["quantity"] * holding["avg_price"]
            total_invested += holding_invested
            total_value += holding["current_value"]
        
        portfolio["total_invested"] = total_invested
        portfolio["total_value"] = total_value
        portfolio["total_profit_loss"] = total_value - total_invested
        portfolio["total_profit_loss_percent"] = (portfolio["total_profit_loss"] / total_invested) * 100 if total_invested > 0 else 0
        portfolio["last_updated"] = datetime.now().isoformat()
        
        # Format response
        if added_symbols:
            if len(added_symbols) == 1:
                symbol_data = added_symbols[0]
                response = f"Added {symbol_data['quantity']} shares of {symbol_data['symbol']} at ${symbol_data['price']:.2f} per share (Total: ${symbol_data['cost']:.2f})."
            else:
                response = "Added to your portfolio:\n\n"
                for symbol_data in added_symbols:
                    response += f"{symbol_data['quantity']} shares of {symbol_data['symbol']} at ${symbol_data['price']:.2f} (Total: ${symbol_data['cost']:.2f})\n"
            
            response += f"\nRemaining cash balance: ${portfolio['cash_balance']:.2f}"
        else:
            response = "Couldn't add any of the requested stocks to your portfolio."
        
        # Add failed symbols to response
        if failed_symbols:
            response += "\n\nFailed to add:"
            for fail in failed_symbols:
                if fail["reason"] == "insufficient_funds":
                    response += f"\n{fail['symbol']}: Insufficient funds"
        
        return {
            "status": "success",
            "session_id": session_id,
            "intent": "PortfolioManagement",
            "subintent": "add",
            "added": added_symbols,
            "failed": failed_symbols,
            "portfolio": portfolio,
            "response": response,
            "response_type": "portfolio_update",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_remove_from_portfolio(
        self, 
        portfolio: Dict[str, Any], 
        symbols: List[str],
        quantity: int,
        session_id: str
    ) -> Dict[str, Any]:
        """Handle remove from portfolio action"""
        removed_symbols = []
        failed_symbols = []
        
        for symbol in symbols:
            # Find symbol in portfolio
            symbol_index = None
            holding = None
            
            for i, h in enumerate(portfolio["holdings"]):
                if h["symbol"] == symbol:
                    symbol_index = i
                    holding = h
                    break
            
            if holding is None:
                failed_symbols.append({"symbol": symbol, "reason": "not_in_portfolio"})
                continue
            
            # Handle quantity
            if quantity <= 0 or quantity >= holding["quantity"]:
                # Remove entire position
                sell_quantity = holding["quantity"]
                sell_value = holding["current_value"]
                portfolio["holdings"].pop(symbol_index)
            else:
                # Remove partial position
                sell_quantity = quantity
                sell_value = sell_quantity * holding["current_price"]
                
                # Update holding
                holding["quantity"] -= sell_quantity
                holding["current_value"] = holding["quantity"] * holding["current_price"]
                holding["profit_loss"] = holding["current_value"] - (holding["avg_price"] * holding["quantity"])
                holding["profit_loss_percent"] = (holding["profit_loss"] / (holding["avg_price"] * holding["quantity"])) * 100 if holding["quantity"] > 0 else 0
            
            # Add to cash balance
            portfolio["cash_balance"] += sell_value
            
            removed_symbols.append({
                "symbol": symbol,
                "quantity": sell_quantity,
                "value": sell_value
            })
        
        # Recalculate portfolio totals
        total_invested = 0
        total_value = 0
        
        for holding in portfolio["holdings"]:
            holding_invested = holding["quantity"] * holding["avg_price"]
            total_invested += holding_invested
            total_value += holding["current_value"]
        
        portfolio["total_invested"] = total_invested
        portfolio["total_value"] = total_value
        portfolio["total_profit_loss"] = total_value - total_invested
        portfolio["total_profit_loss_percent"] = (portfolio["total_profit_loss"] / total_invested) * 100 if total_invested > 0 else 0
        portfolio["last_updated"] = datetime.now().isoformat()
        
        # Format response
        if removed_symbols:
            if len(removed_symbols) == 1:
                symbol_data = removed_symbols[0]
                response = f"Sold {symbol_data['quantity']} shares of {symbol_data['symbol']} for ${symbol_data['value']:.2f}."
            else:
                response = "Removed from your portfolio:\n\n"
                for symbol_data in removed_symbols:
                    response += f"{symbol_data['quantity']} shares of {symbol_data['symbol']} for ${symbol_data['value']:.2f}\n"
            
            response += f"\nUpdated cash balance: ${portfolio['cash_balance']:.2f}"
        else:
            response = "Couldn't remove any of the requested stocks from your portfolio."
        
        # Add failed symbols to response
        if failed_symbols:
            response += "\n\nFailed to remove:"
            for fail in failed_symbols:
                if fail["reason"] == "not_in_portfolio":
                    response += f"\n{fail['symbol']}: Not in your portfolio"
        
        return {
            "status": "success",
            "session_id": session_id,
            "intent": "PortfolioManagement",
            "subintent": "remove",
            "removed": removed_symbols,
            "failed": failed_symbols,
            "portfolio": portfolio,
            "response": response,
            "response_type": "portfolio_update",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_analyze_portfolio(
        self, 
        portfolio: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Handle analyze portfolio action"""
        # Get mock analysis data
        diversification = self._analyze_portfolio_diversification(portfolio)
        risk_level = self._analyze_portfolio_risk(portfolio)
        recommendations = self._generate_portfolio_recommendations(portfolio)
        
        # Format response
        response = "Portfolio Analysis:\n\n"
        
        # Add performance metrics
        pl_sign = "+" if portfolio["total_profit_loss"] >= 0 else ""
        response += f"Performance: {pl_sign}{portfolio['total_profit_loss_percent']:.2f}%\n"
        
        # Add diversification
        response += "\nDiversification:\n"
        for sector, percentage in diversification["sectors"].items():
            response += f"- {sector}: {percentage:.1f}%\n"
        
        # Add risk assessment
        response += f"\nRisk Level: {risk_level['overall_risk']}\n"
        response += f"Volatility: {risk_level['volatility']}\n"
        
        # Add recommendations
        response += "\nRecommendations:\n"
        for rec in recommendations[:3]:  # Show top 3 recommendations
            response += f"- {rec}\n"
        
        return {
            "status": "success",
            "session_id": session_id,
            "intent": "PortfolioManagement",
            "subintent": "analyze",
            "portfolio": portfolio,
            "analysis": {
                "diversification": diversification,
                "risk_level": risk_level,
                "recommendations": recommendations
            },
            "response": response,
            "response_type": "portfolio_analysis",
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_portfolio_diversification(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio diversification"""
        import random
        
        # Mock sector allocation
        sectors = ["Technology", "Healthcare", "Consumer Cyclical", "Financial Services", "Industrials"]
        sector_allocation = {}
        
        # Assign random sectors to holdings
        for holding in portfolio["holdings"]:
            # Use symbol to seed random choice for consistency
            random.seed(hash(holding["symbol"]))
            sector = random.choice(sectors)
            
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            
            sector_allocation[sector] += holding["current_value"]
        
        # Convert to percentages
        sector_percentages = {}
        if portfolio["total_value"] > 0:
            for sector, value in sector_allocation.items():
                sector_percentages[sector] = (value / portfolio["total_value"]) * 100
        
        # Overall diversification score (0-100)
        num_sectors = len(sector_allocation)
        max_allocation = max(sector_percentages.values()) if sector_percentages else 0
        
        diversification_score = 0
        if num_sectors > 0:
            # More sectors = better diversification
            sector_factor = min(num_sectors / len(sectors), 1.0) * 50
            # Lower max allocation = better diversification
            allocation_factor = (100 - max_allocation) / 2
            diversification_score = sector_factor + allocation_factor
        
        return {
            "score": diversification_score,
            "sectors": sector_percentages,
            "largest_sector": max(sector_percentages.items(), key=lambda x: x[1])[0] if sector_percentages else "None"
        }
    
    def _analyze_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio risk level"""
        import random
        
        # Mock volatility data for holdings
        volatilities = {}
        for holding in portfolio["holdings"]:
            # Use symbol to seed random choice for consistency
            random.seed(hash(holding["symbol"]) + 1)
            volatilities[holding["symbol"]] = random.uniform(0.5, 2.5)
        
        # Calculate weighted average volatility
        weighted_volatility = 0
        if portfolio["total_value"] > 0:
            for holding in portfolio["holdings"]:
                weight = holding["current_value"] / portfolio["total_value"]
                weighted_volatility += weight * volatilities[holding["symbol"]]
        
        # Determine risk level
        if weighted_volatility < 0.8:
            risk_level = "Low"
        elif weighted_volatility < 1.3:
            risk_level = "Moderate"
        elif weighted_volatility < 1.8:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        return {
            "overall_risk": risk_level,
            "volatility": f"{weighted_volatility:.2f}",
            "holding_volatilities": volatilities
        }
    
    def _generate_portfolio_recommendations(self, portfolio: Dict[str, Any]) -> List[str]:
        """Generate recommendations for portfolio optimization"""
        import random
        
        # Placeholder recommendations
        recommendations = [
            "Consider adding more diversification to your portfolio",
            "Your technology exposure is high, consider reducing to manage risk",
            "Add some defensive stocks to balance your portfolio",
            "Consider adding a bond allocation for stability",
            "Look for opportunities to rebalance overweight sectors",
            "Consider taking profits from your best performers",
            "Add international exposure to diversify geographically"
        ]
        
        # Shuffle recommendations (but consistently based on portfolio)
        random.seed(hash(portfolio["last_updated"]))
        random.shuffle(recommendations)
        
        return recommendations


# Initialize the singleton instance
dialogflow_financial_chat = DialogflowFinancialChat()
