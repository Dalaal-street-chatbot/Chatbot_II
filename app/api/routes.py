from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models.schemas import (
    ChatRequest, ChatResponse, StockRequest, StockResponse,
    NewsRequest, NewsResponse, MarketIndicesResponse,
    AnalysisRequest, AnalysisResponse
)
from app.services.groq_service import groq_service
from app.services.enhanced_news_service import enhanced_news_service
from app.services.comprehensive_chat import comprehensive_chat
from app.services.financial_chatbot_integration import financial_chatbot
from app.services.real_time_stock_service import real_time_stock_service
from market_data import market_service
from pydantic import BaseModel
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Main chat endpoint with AI integration"""
    try:
        # Check if the request contains special indicators for Google Cloud services
        lower_message = request.message.lower()
        
        # For Google Cloud specialized requests, use the financial_chatbot
        if any(keyword in lower_message for keyword in ['vertex ai', 'dialogflow', 'cloud vision', 'bigquery', 
                                                       'chart analysis', 'technical indicators',
                                                       'comprehensive analysis', 'complete market view']):
            # Use the fully integrated financial chatbot for Google Cloud services
            result = await financial_chatbot.process_user_message(
                message=request.message,
                session_id=request.session_id or "user_session",
                user_data=request.context or {}
            )
            
            return ChatResponse(
                response=result.get("response", "Sorry, I couldn't process that request."),
                intent=result.get("intent", "unknown"),
                entities=result.get("symbols", []),
                confidence=0.9
            )
        
        # For general requests, use the comprehensive chat service
        result = await comprehensive_chat.process_comprehensive_query(
            user_message=request.message,
            session_context=request.context
        )
        
        return ChatResponse(
            response=result.get("response", "Sorry, I couldn't process that request."),
            intent=result.get("intent", "general_info"),
            entities=result.get("entities", []),
            confidence=result.get("confidence", 0.0)
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        
        # Try to fallback to just the Google Cloud financial chatbot
        try:
            result = await financial_chatbot.process_user_message(
                message=request.message,
                session_id="error_fallback",
                user_data={}
            )
            
            return ChatResponse(
                response=result.get("response", "I apologize, but I'm experiencing some technical difficulties. Please try again later."),
                intent="error",
                entities=[],
                confidence=0.5
            )
        except Exception as fallback_error:
            print(f"Error in fallback handler: {fallback_error}")
            
            return ChatResponse(
                response="I apologize, but I'm experiencing some technical difficulties. Please try again later.",
                intent="error",
                entities=[],
                confidence=0.0
            )

@router.post("/stock", response_model=StockResponse)
async def get_stock_data(request: StockRequest):
    """Get real-time stock price and data"""
    try:
        symbol = request.symbol.upper()
        
        # Get real-time stock data
        stock_data = await real_time_stock_service.get_real_time_data(symbol)
        
        if stock_data:
            return StockResponse(
                symbol=stock_data.symbol,
                price=round(stock_data.close, 2),
                change=round(stock_data.change, 2),
                change_percent=round(stock_data.change_percent, 2),
                volume=stock_data.volume,
                market_cap=0,  # Not available in current StockData model
                source="Real-time API"
            )
        else:
            # Fallback to market_service if real-time service fails
            logger.warning(f"Real-time service failed for {symbol}, using fallback")
            fallback_data = market_service.get_stock_price(symbol)
            
            if 'error' in fallback_data:
                raise HTTPException(status_code=404, detail=fallback_data['error'])
            
            return StockResponse(**fallback_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indices", response_model=MarketIndicesResponse)
async def get_market_indices():
    """Get major market indices with real-time data"""
    try:
        # List of major Indian indices
        indian_indices = [
            ("^NSEI", "NIFTY 50"),
            ("^BSESN", "SENSEX"),
            ("^NSEBANK", "BANK NIFTY"),
            ("^NSEIT", "NIFTY IT"),
            ("^NSEAUTO", "NIFTY AUTO"),
            ("^NSEFMCG", "NIFTY FMCG")
        ]
        
        indices_dict = {}
        
        # Try to get real-time data for each index
        for symbol, name in indian_indices:
            try:
                index_data = await real_time_stock_service.get_real_time_data(symbol)
                if index_data:
                    indices_dict[name] = {
                        "value": round(index_data.close, 2),
                        "change": round(index_data.change, 2),
                        "change_percent": round(index_data.change_percent, 2)
                    }
                else:
                    # Add fallback data if real-time fails
                    indices_dict[name] = {
                        "value": 20000.0,  # Placeholder
                        "change": 0.0,
                        "change_percent": 0.0
                    }
            except Exception as e:
                logger.warning(f"Failed to get real-time data for {symbol}: {e}")
                # Add fallback data
                indices_dict[name] = {
                    "value": 20000.0,  # Placeholder
                    "change": 0.0,
                    "change_percent": 0.0
                }
        
        # If no real-time data available, fall back to market_service
        if not indices_dict:
            fallback_indices = market_service.get_market_indices()
            return MarketIndicesResponse(indices=fallback_indices)
        
        return MarketIndicesResponse(indices=indices_dict)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/news", response_model=NewsResponse)
async def get_financial_news(request: NewsRequest):
    """Get financial news"""
    try:
        # Default page size if not provided
        page_size = request.page_size if request.page_size is not None else 5
        
        if request.company:
            news_data = await enhanced_news_service.get_enhanced_company_news(request.company, page_size=page_size)
        elif request.query:
            news_data = await enhanced_news_service.get_enhanced_financial_news(query=request.query, page_size=page_size)
        else:
            # Get top news with market sentiment
            news_data = await enhanced_news_service.get_enhanced_financial_news(page_size=page_size)
        
        if news_data['status'] == 'error':
            raise HTTPException(status_code=500, detail=news_data['message'])
        
        return NewsResponse(**news_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analysis", response_model=AnalysisResponse)
async def get_financial_analysis(request: AnalysisRequest):
    """Get AI-powered financial analysis"""
    try:
        context = {}
        
        # Get relevant data for analysis
        if request.symbol:
            stock_data = market_service.get_stock_price(request.symbol)
            if 'error' not in stock_data:
                context['stock_data'] = stock_data
        
        # Get market context
        indices = market_service.get_market_indices()
        context['market_indices'] = indices
        
        # Get recent news and sentiment with enhanced service
        news_data = await enhanced_news_service.get_enhanced_financial_news(page_size=3)
        context['recent_news'] = news_data
        
        # Add market sentiment analysis
        sentiment_data = await enhanced_news_service.get_market_sentiment_analysis()
        context['market_sentiment'] = sentiment_data
        
        # Generate AI analysis
        analysis_prompt = f"""
        Provide a comprehensive financial analysis for: {request.query}
        
        Consider:
        1. Current market conditions
        2. Technical indicators (if applicable)
        3. Market sentiment from recent news
        4. Risk factors
        5. Investment recommendations (with disclaimers)
        
        Keep the analysis professional and include appropriate risk disclaimers.
        """
        
        analysis_text = await groq_service.generate_response(
            analysis_prompt, 
            context=context,
            system_prompt="You are a professional financial analyst. Provide detailed but responsible financial analysis."
        )
        
        # Extract recommendations from analysis
        recommendations = ["Consider broader market conditions before making decisions",
                           "Diversify your portfolio to manage risk",
                           "Always consult with a financial advisor for personalized advice"]
        
        return AnalysisResponse(
            analysis=analysis_text,
            data=context,
            recommendations=recommendations,
            confidence=0.8
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chart Data Models
class ChartDataRequest(BaseModel):
    symbol: str
    timeframe: str = "1D"  # 1D, 1W, 1M, 3M, 6M, 1Y

class CandlestickData(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float

class VolumeData(BaseModel):
    time: int
    value: int
    color: str

class ChartDataResponse(BaseModel):
    symbol: str
    timeframe: str
    candlestick_data: list[CandlestickData]
    volume_data: list[VolumeData]
    current_price: float
    change: float
    change_percent: float

@router.post("/chart-data", response_model=ChartDataResponse)
async def get_chart_data(request: ChartDataRequest):
    """Get real-time chart data for a specific symbol and timeframe"""
    try:
        symbol = request.symbol.upper()
        timeframe = request.timeframe
        
        # Convert timeframe to period format for APIs
        timeframe_mapping = {
            '1D': '1d',
            '1W': '5d', 
            '1M': '1mo',
            '3M': '3mo',
            '6M': '6mo',
            '1Y': '1y'
        }
        period = timeframe_mapping.get(timeframe, '1mo')
        
        # Get historical data from real APIs
        historical_data = await real_time_stock_service.get_historical_data(symbol, period)
        
        # Get current real-time data
        real_time_data = await real_time_stock_service.get_real_time_data(symbol)
        
        if historical_data and historical_data.data:
            candlestick_data = []
            volume_data = []
            
            for data_point in historical_data.data:
                timestamp = int(data_point.timestamp.timestamp())
                
                candlestick_data.append(CandlestickData(
                    time=timestamp,
                    open=round(data_point.open, 2),
                    high=round(data_point.high, 2),
                    low=round(data_point.low, 2),
                    close=round(data_point.close, 2)
                ))
                
                # Color volume bars based on price movement
                volume_color = "#4ade80" if data_point.close > data_point.open else "#ef4444"
                volume_data.append(VolumeData(
                    time=timestamp,
                    value=data_point.volume,
                    color=volume_color
                ))
            
            # Use real-time data if available, otherwise use last historical point
            if real_time_data:
                current_price = real_time_data.close
                change = real_time_data.change
                change_percent = real_time_data.change_percent
            else:
                latest_data = historical_data.data[-1]
                current_price = latest_data.close
                change = latest_data.change
                change_percent = latest_data.change_percent
            
            return ChartDataResponse(
                symbol=symbol,
                timeframe=timeframe,
                candlestick_data=candlestick_data,
                volume_data=volume_data,
                current_price=round(current_price, 2),
                change=round(change, 2),
                change_percent=round(change_percent, 2)
            )
        
        else:
            # Our service will now return sample data if all APIs fail
            # Get sample data from the _generate_sample_historical_data method
            logger.warning(f"No historical data found for {symbol}, trying to generate sample data")
            
            # Generate sample data
            sample_data = await real_time_stock_service._generate_sample_historical_data(symbol, period)
            
            if sample_data and sample_data.data:
                logger.info(f"Successfully generated sample data for {symbol}")
                
                candlestick_data = []
                volume_data = []
                
                for data_point in sample_data.data:
                    timestamp = int(data_point.timestamp.timestamp())
                    
                    candlestick_data.append(CandlestickData(
                        time=timestamp,
                        open=round(data_point.open, 2),
                        high=round(data_point.high, 2),
                        low=round(data_point.low, 2),
                        close=round(data_point.close, 2)
                    ))
                    
                    # Color volume bars based on price movement
                    volume_color = "#4ade80" if data_point.close > data_point.open else "#ef4444"
                    volume_data.append(VolumeData(
                        time=timestamp,
                        value=data_point.volume,
                        color=volume_color
                    ))
                
                # Use the last data point for current price/change
                latest_data = sample_data.data[-1]
                
                return ChartDataResponse(
                    symbol=symbol,
                    timeframe=timeframe,
                    candlestick_data=candlestick_data,
                    volume_data=volume_data,
                    current_price=round(latest_data.close, 2),
                    change=round(latest_data.change, 2),
                    change_percent=round(latest_data.change_percent, 2)
                )
            
            # If even sample generation fails, use minimal fallback
            logger.error(f"All data sources failed for {symbol}, using minimal fallback")
            
            # Base prices for different stocks (realistic current values)
            base_prices = {
                # Market Indices
                'NIFTY50': 22500.0,
                'SENSEX': 74000.0,
                'BANKNIFTY': 48000.0,
                # Individual Stocks
                'RELIANCE': 2500.0,
                'TCS': 3200.0,
                'INFY': 1400.0,
                'HDFC': 2700.0,
                'ITC': 450.0,
                'SBIN': 550.0,
                'BAJFINANCE': 6500.0,
                'LT': 2800.0,
                'WIPRO': 780.0,
                'HCLTECH': 1250.0,
            }
            
            base_price = base_prices.get(symbol, 1000.0)
            current_price = base_price * random.uniform(0.98, 1.02)  # ±2% variation
            change = current_price - base_price
            change_percent = (change / base_price) * 100
            
            # Generate minimal data points
            candlestick_data = []
            volume_data = []
            
            # Create just a few data points for the chart
            now = datetime.now()
            for i in range(5):
                timestamp = int((now - timedelta(hours=i)).timestamp())
                price_var = random.uniform(0.99, 1.01)
                open_price = current_price * price_var
                close_price = current_price * random.uniform(0.99, 1.01)
                high_price = max(open_price, close_price) * random.uniform(1.0, 1.005)
                low_price = min(open_price, close_price) * random.uniform(0.995, 1.0)
                volume = random.randint(100000, 1000000)
                
                candlestick_data.append(CandlestickData(
                    time=timestamp,
                    open=round(open_price, 2),
                    high=round(high_price, 2),
                    low=round(low_price, 2),
                    close=round(close_price, 2)
                ))
                
                volume_color = "#4ade80" if close_price > open_price else "#ef4444"
                volume_data.append(VolumeData(
                    time=timestamp,
                    value=volume,
                    color=volume_color
                ))
            
            return ChartDataResponse(
                symbol=symbol,
                timeframe=timeframe,
                candlestick_data=list(reversed(candlestick_data)),  # Reverse to show chronological order
                volume_data=list(reversed(volume_data)),
                current_price=round(current_price, 2),
                change=round(change, 2),
                change_percent=round(change_percent, 2)
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data: {str(e)}")
