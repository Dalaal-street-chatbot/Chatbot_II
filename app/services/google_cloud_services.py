"""
Google Cloud Services Integration Module

This module provides integration with various Google Cloud services:
1. Vertex AI for advanced machine learning capabilities
2. DialogFlow for conversational AI
3. Cloud Vision API for image processing and recognition
4. BigQuery for data analytics
"""

import os
import logging
import re
from typing import Dict, List, Optional, Any, Union
from google.oauth2 import service_account
from google.cloud import vision, bigquery
import google.cloud.dialogflow_v2 as dialogflow
import google.cloud.aiplatform as aiplatform
from datetime import datetime
import json
import base64
import requests

from config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleCloudServices:
    """
    Google Cloud Services Integration class for financial and market analysis
    """
    
    def __init__(self):
        """Initialize the Google Cloud Services integration"""
        # Initialize credentials and clients based on environment variables
        self.project_id = config.GOOGLE_CLOUD_PROJECT_ID
        self.location = config.GOOGLE_CLOUD_LOCATION
        self.google_ai_api_key = config.GOOGLE_AI_API_KEY
        self.cloud_vision_api_key = config.CLOUD_VISION_API
        
        # Check if required environment variables are set
        if not self.project_id:
            logger.warning("GOOGLE_CLOUD_PROJECT_ID is not set. Some Google Cloud features may not work.")
        
        if not self.location:
            logger.warning("GOOGLE_CLOUD_LOCATION is not set. Using default: 'us-central1'")
            self.location = "us-central1"
        
        # Check for API keys
        if not self.google_ai_api_key:
            logger.warning("GOOGLE_AI_API_KEY is not set. Google AI features may not work.")
            
        if not self.cloud_vision_api_key:
            logger.warning("CLOUD_VISION_API is not set. Vision API features may not work.")
        
        # Initialize services when needed to prevent unnecessary authentication
        self._vision_client = None
        self._dialogflow_client = None
        self._bigquery_client = None
        self._vertex_ai_initialized = False
        
        logger.info("Google Cloud Services integration initialized")
    
    # ===== VERTEX AI SERVICES =====
    
    def initialize_vertex_ai(self) -> None:
        """Initialize Vertex AI services"""
        if not self._vertex_ai_initialized:
            try:
                # Initialize Vertex AI with the specified project and location
                aiplatform.init(
                    project=self.project_id,
                    location=self.location,
                )
                self._vertex_ai_initialized = True
                logger.info("Vertex AI initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI: {str(e)}")
                raise
    
    async def analyze_market_trend(self, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze market trends using Vertex AI
        
        Args:
            market_data: List of market data points to analyze
            
        Returns:
            Analysis results including trends, predictions, and recommendations
        """
        self.initialize_vertex_ai()
        
        try:
            # Convert data to proper format for model
            formatted_data = self._format_data_for_prediction(market_data)
            
            # Create feature vector
            endpoint = aiplatform.Endpoint(f"projects/{self.project_id}/locations/{self.location}/endpoints/market_trend_endpoint")
            prediction = endpoint.predict(instances=formatted_data)
            
            # Process prediction results
            results = self._process_market_prediction(prediction)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "analysis": results
            }
        except Exception as e:
            logger.error(f"Error analyzing market trend: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze market trends: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_market_forecast(self, symbol: str, timeframe: str = "1w") -> Dict[str, Any]:
        """
        Generate market forecast for a specific stock using Vertex AI
        
        Args:
            symbol: Stock symbol to forecast
            timeframe: Timeframe for the forecast (e.g., 1d, 1w, 1m, 3m)
            
        Returns:
            Forecast results including predicted price ranges and confidence scores
        """
        self.initialize_vertex_ai()
        
        try:
            # Format request for forecasting model
            request_data = {
                "symbol": symbol,
                "timeframe": timeframe
            }
            
            # Use PaLM API for text generation
            model = aiplatform.TextGenerationModel.from_pretrained("text-bison")
            prompt = f"""
            Generate a market forecast for {symbol} over the next {timeframe} timeframe.
            Include:
            1. Price prediction (range)
            2. Key factors affecting the price
            3. Confidence level
            4. Potential market events to watch
            Format as JSON.
            """
            
            response = model.predict(prompt=prompt, temperature=0.2, max_output_tokens=1024)
            
            # Parse the response and format it
            forecast = self._parse_forecast_response(response.text, symbol, timeframe)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "forecast": forecast
            }
        except Exception as e:
            logger.error(f"Error generating forecast for {symbol}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate forecast for {symbol}: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    # ===== DIALOGFLOW SERVICES =====
    
    @property
    def dialogflow_client(self):
        """Lazily initialize the DialogFlow client"""
        if self._dialogflow_client is None:
            try:
                self._dialogflow_client = dialogflow.SessionsClient()
                logger.info("DialogFlow client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DialogFlow client: {str(e)}")
                raise
        return self._dialogflow_client
    
    async def detect_intent(self, session_id: str, text: str, language_code: str = "en") -> Dict[str, Any]:
        """
        Detect intent from user query using DialogFlow
        
        Args:
            session_id: Unique session identifier
            text: User query text
            language_code: Language code (default: 'en')
            
        Returns:
            Detected intent and response
        """
        try:
            session = self.dialogflow_client.session_path(self.project_id, session_id)
            text_input = dialogflow.TextInput(text=text, language_code=language_code)
            query_input = dialogflow.QueryInput(text=text_input)
            
            response = self.dialogflow_client.detect_intent(
                request={"session": session, "query_input": query_input}
            )
            
            query_result = response.query_result
            
            return {
                "status": "success",
                "intent": query_result.intent.display_name,
                "confidence": query_result.intent_detection_confidence,
                "parameters": dict(query_result.parameters),
                "fulfillment_text": query_result.fulfillment_text
            }
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to detect intent: {str(e)}"
            }
    
    async def handle_finance_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Handle financial queries using DialogFlow
        
        Args:
            session_id: Unique session identifier
            query: User's financial query
            
        Returns:
            Response to the financial query
        """
        try:
            # Detect intent
            intent_result = await self.detect_intent(session_id, query)
            
            if intent_result["status"] != "success":
                return intent_result
            
            # Process the intent
            if intent_result["intent"] == "StockPrice":
                symbol = intent_result["parameters"].get("stock-symbol", "")
                if symbol:
                    # Implement stock price lookup logic here
                    return {
                        "status": "success",
                        "response_type": "stock_price",
                        "symbol": symbol,
                        "message": f"Looking up price for {symbol}..."
                    }
            elif intent_result["intent"] == "MarketNews":
                # Implement market news lookup logic here
                return {
                    "status": "success",
                    "response_type": "market_news",
                    "message": "Here are the latest market news..."
                }
            
            # Default: return the fulfillment text from DialogFlow
            return {
                "status": "success",
                "response_type": "text",
                "message": intent_result["fulfillment_text"]
            }
        except Exception as e:
            logger.error(f"Error handling finance query: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to process financial query: {str(e)}"
            }
    
    # ===== CLOUD VISION API SERVICES =====
    
    @property
    def vision_client(self):
        """Lazily initialize the Vision client"""
        if self._vision_client is None:
            try:
                if self.cloud_vision_api_key:
                    # Create client with API key
                    import os
                    
                    # Set the API key as environment variable temporarily
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''  # Clear any existing credentials
                    
                    # Create client with explicit API key
                    client_options = {"api_key": self.cloud_vision_api_key}
                    self._vision_client = vision.ImageAnnotatorClient(client_options=client_options)
                    logger.info("Vision client initialized with API key")
                else:
                    # Fallback to default credentials
                    self._vision_client = vision.ImageAnnotatorClient()
                    logger.info("Vision client initialized with default credentials")
            except Exception as e:
                logger.error(f"Failed to initialize Vision client: {str(e)}")
                # Return None so methods can handle the missing client gracefully
                self._vision_client = None
        return self._vision_client
    
    async def analyze_chart_image(self, image_data: Union[str, bytes]) -> Dict[str, Any]:
        """
        Analyze stock chart image using Cloud Vision API
        
        Args:
            image_data: Image data (file path, URL, or bytes)
            
        Returns:
            Analysis results including detected patterns, text, and trends
        """
        try:
            # Check if Vision client is available
            vision_client = self.vision_client
            if vision_client is None:
                return {
                    "status": "error",
                    "message": "Cloud Vision API is not available. Please check your credentials.",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Prepare image
            if isinstance(image_data, str):
                if image_data.startswith("http"):
                    # Download from URL
                    response = requests.get(image_data)
                    content = response.content
                else:
                    # Read from file path
                    with open(image_data, "rb") as image_file:
                        content = image_file.read()
            else:
                # Use provided bytes
                content = image_data
            
            image = vision.Image(content=content)
            
            # Detect text in image
            text_response = vision_client.text_detection(image=image)
            texts = text_response.text_annotations
            
            # Detect objects and labels in image
            label_response = vision_client.label_detection(image=image)
            labels = label_response.label_annotations
            
            # Process chart data
            chart_data = self._extract_chart_data(texts, labels)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "chart_data": chart_data
            }
        except Exception as e:
            logger.error(f"Error analyzing chart image: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze chart image: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def extract_financial_document(self, document_data: Union[str, bytes]) -> Dict[str, Any]:
        """
        Extract financial information from documents using Cloud Vision API
        
        Args:
            document_data: Path to the financial document or raw document bytes
            
        Returns:
            Extracted financial information
        """
        try:
            # Check if Vision client is available
            vision_client = self.vision_client
            if vision_client is None:
                return {
                    "status": "error",
                    "message": "Cloud Vision API is not available. Please check your credentials.",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Process input based on type
            if isinstance(document_data, str):
                # It's a file path
                with open(document_data, "rb") as document_file:
                    content = document_file.read()
            else:
                # It's already bytes
                content = document_data
            
            image = vision.Image(content=content)
            
            # Detect text
            text_response = vision_client.document_text_detection(image=image)
            document = text_response.full_text_annotation
            
            # Extract financial data
            financial_data = self._extract_financial_data(document.text)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "financial_data": financial_data
            }
        except Exception as e:
            logger.error(f"Error extracting financial document: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to extract financial document: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    # ===== BIGQUERY SERVICES =====
    
    @property
    def bigquery_client(self):
        """Lazily initialize the BigQuery client"""
        if self._bigquery_client is None:
            try:
                self._bigquery_client = bigquery.Client(project=self.project_id)
                logger.info("BigQuery client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize BigQuery client: {str(e)}")
                raise
        return self._bigquery_client
    
    async def query_market_data(self, query: str) -> Dict[str, Any]:
        """
        Query market data using BigQuery
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results
        """
        try:
            # Run the query
            query_job = self.bigquery_client.query(query)
            results = query_job.result()
            
            # Convert results to a list of dictionaries
            rows = []
            for row in results:
                rows.append(dict(row.items()))
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "results": rows,
                "row_count": len(rows)
            }
        except Exception as e:
            logger.error(f"Error querying market data: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to query market data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_market_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get historical market data for a symbol using BigQuery
        
        Args:
            symbol: Stock symbol
            days: Number of days of history to retrieve
            
        Returns:
            Historical market data
        """
        try:
            # Build SQL query
            query = f"""
            SELECT date, open, high, low, close, volume
            FROM `{self.project_id}.market_data.historical_prices`
            WHERE symbol = @symbol
            AND date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
            ORDER BY date ASC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol)
                ]
            )
            
            # Run the query
            query_job = self.bigquery_client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Convert results to a list of dictionaries
            history = []
            for row in results:
                history.append({
                    "date": row.date.isoformat(),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": int(row.volume)
                })
            
            return {
                "status": "success",
                "symbol": symbol,
                "days": days,
                "history": history,
                "count": len(history)
            }
        except Exception as e:
            logger.error(f"Error getting market history for {symbol}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to get market history for {symbol}: {str(e)}",
                "symbol": symbol
            }
    
    # ===== HELPER METHODS =====
    
    def _format_data_for_prediction(self, market_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format market data for prediction models"""
        formatted_data = []
        for item in market_data:
            # Extract relevant features and format them for the model
            formatted_item = {
                "price": float(item.get("price", 0)),
                "volume": int(item.get("volume", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "timestamp": item.get("timestamp", datetime.now().isoformat())
            }
            formatted_data.append(formatted_item)
        return formatted_data
    
    def _process_market_prediction(self, prediction: Any) -> Dict[str, Any]:
        """Process market prediction results"""
        try:
            # Example processing logic - adjust based on actual model output
            predictions = prediction.predictions
            
            # Extract relevant information from predictions
            results = {
                "trend_direction": predictions[0][0],  # e.g., "up", "down", "sideways"
                "confidence": predictions[0][1],       # e.g., 0.85 (85% confidence)
                "predicted_range": {
                    "low": predictions[0][2],
                    "high": predictions[0][3]
                },
                "indicators": {
                    "rsi": predictions[0][4],
                    "macd": predictions[0][5],
                    "moving_average": predictions[0][6]
                }
            }
            return results
        except Exception as e:
            logger.error(f"Error processing prediction: {str(e)}")
            return {
                "error": f"Failed to process prediction: {str(e)}"
            }
    
    def _parse_forecast_response(self, response_text: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Parse forecast response from the model"""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                forecast_data = json.loads(json_str)
            else:
                # Fallback to manual parsing if JSON extraction fails
                forecast_data = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "price_prediction": {
                        "low": None,
                        "high": None
                    },
                    "key_factors": [],
                    "confidence": "medium",
                    "events_to_watch": []
                }
                
                # Extract price prediction
                price_match = re.search(r"Price prediction.*?(\d+\.?\d*).*?(\d+\.?\d*)", response_text)
                if price_match:
                    forecast_data["price_prediction"]["low"] = float(price_match.group(1))
                    forecast_data["price_prediction"]["high"] = float(price_match.group(2))
                
                # Extract key factors
                factors_match = re.search(r"Key factors:(.*?)(?:Confidence|$)", response_text, re.DOTALL)
                if factors_match:
                    factors_text = factors_match.group(1).strip()
                    factors = [f.strip() for f in factors_text.split("\n") if f.strip()]
                    forecast_data["key_factors"] = factors
                
                # Extract confidence
                confidence_match = re.search(r"Confidence.*?(high|medium|low)", response_text, re.IGNORECASE)
                if confidence_match:
                    forecast_data["confidence"] = confidence_match.group(1).lower()
                
                # Extract events to watch
                events_match = re.search(r"events to watch:(.*?)(?:$)", response_text, re.DOTALL | re.IGNORECASE)
                if events_match:
                    events_text = events_match.group(1).strip()
                    events = [e.strip() for e in events_text.split("\n") if e.strip()]
                    forecast_data["events_to_watch"] = events
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error parsing forecast response: {str(e)}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "error": f"Failed to parse forecast: {str(e)}",
                "raw_response": response_text[:500]  # Include partial response for debugging
            }
    
    def _extract_chart_data(self, texts: List[Any], labels: List[Any]) -> Dict[str, Any]:
        """Extract data from chart image"""
        # Extract text content
        text_content = texts[0].description if texts else ""
        
        # Look for stock symbol and price data
        symbol = None
        prices = []
        dates = []
        
        # Simple pattern matching for stock symbols (e.g., AAPL, MSFT)
        symbol_match = re.search(r'\b([A-Z]{2,5})\b', text_content)
        if symbol_match:
            symbol = symbol_match.group(1)
        
        # Extract price values (numbers with decimal points)
        price_matches = re.finditer(r'(\d+\.\d+)', text_content)
        for match in price_matches:
            prices.append(float(match.group(1)))
        
        # Extract dates (simple date formats)
        date_matches = re.finditer(r'(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', text_content)
        for match in date_matches:
            dates.append(match.group(1))
        
        # Extract chart type from labels
        chart_type = None
        for label in labels:
            if label.description.lower() in ["line chart", "bar chart", "candlestick", "stock chart", "graph"]:
                chart_type = label.description
                break
        
        return {
            "symbol": symbol,
            "chart_type": chart_type,
            "text_content": text_content[:1000] if len(text_content) > 1000 else text_content,
            "extracted_prices": prices[:10],  # Limit to first 10 prices
            "extracted_dates": dates[:10],    # Limit to first 10 dates
            "labels": [label.description for label in labels[:5]]  # Top 5 labels
        }
    
    def _extract_financial_data(self, text: str) -> Dict[str, Any]:
        """Extract financial data from document text"""
        # Initialize financial data dictionary
        financial_data = {
            "company_name": None,
            "ticker_symbol": None,
            "financial_figures": {},
            "dates": [],
            "key_metrics": {}
        }
        
        # Extract company name
        company_match = re.search(r'(?:Company|Corporation|Corp|Inc|Ltd):\s*([A-Za-z0-9\s]+)', text)
        if company_match:
            financial_data["company_name"] = company_match.group(1).strip()
        
        # Extract ticker symbol
        ticker_match = re.search(r'(?:Ticker|Symbol):\s*([A-Z]{1,5})', text)
        if ticker_match:
            financial_data["ticker_symbol"] = ticker_match.group(1)
        
        # Extract dates
        date_matches = re.finditer(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})', text)
        for match in date_matches:
            financial_data["dates"].append(match.group(1))
        
        # Extract key financial figures (revenue, profit, etc.)
        revenue_match = re.search(r'Revenue:?\s*\$?([\d,.]+)(?:\s*million|\s*billion)?', text)
        if revenue_match:
            revenue = revenue_match.group(1).replace(",", "")
            financial_data["financial_figures"]["revenue"] = float(revenue)
        
        profit_match = re.search(r'(?:Net Income|Profit):?\s*\$?([\d,.]+)(?:\s*million|\s*billion)?', text)
        if profit_match:
            profit = profit_match.group(1).replace(",", "")
            financial_data["financial_figures"]["profit"] = float(profit)
        
        eps_match = re.search(r'EPS:?\s*\$?([\d,.]+)', text)
        if eps_match:
            eps = eps_match.group(1).replace(",", "")
            financial_data["financial_figures"]["eps"] = float(eps)
        
        # Extract key metrics (P/E, ROI, etc.)
        pe_match = re.search(r'P/E(?:\s*Ratio)?:?\s*([\d,.]+)', text)
        if pe_match:
            pe = pe_match.group(1).replace(",", "")
            financial_data["key_metrics"]["pe_ratio"] = float(pe)
        
        roi_match = re.search(r'ROI:?\s*([\d,.]+)%?', text)
        if roi_match:
            roi = roi_match.group(1).replace(",", "")
            financial_data["key_metrics"]["roi"] = float(roi)
        
        return financial_data


# Initialize the singleton instance
google_cloud_services = GoogleCloudServices()
