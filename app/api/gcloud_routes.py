"""
Google Cloud Services API Routes

This module defines the API routes for accessing the
integrated Google Cloud services.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import base64
from pydantic import BaseModel

from app.models.schemas import ChatRequest, ChatResponse, StockRequest, AnalysisRequest
from app.services.financial_chatbot_integration import financial_chatbot

router = APIRouter(prefix="/google-cloud", tags=["Google Cloud Services"])

# ===== API Models =====

class MarketInsightsRequest(BaseModel):
    """Request model for market insights"""
    symbols: Optional[List[str]] = None


class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis"""
    base64_image: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


# ===== API Routes =====

@router.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    """
    Process a chat message using Google Cloud Services

    Args:
        request: The chat request from the user
    
    Returns:
        Processed chatbot response
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="Message text is required")
    
    response = await financial_chatbot.process_user_message(
        message=request.message,
        session_id=request.session_id,
        user_data=request.context if request.context else None
    )
    
    return response


@router.post("/financial-chat", response_model=ChatResponse)
async def process_financial_chat(request: ChatRequest):
    """
    Process a financial chat message using Google Cloud Services
    (Alternative endpoint for frontend compatibility)

    Args:
        request: The chat request from the user
    
    Returns:
        Processed chatbot response
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="Message text is required")
    
    response = await financial_chatbot.process_user_message(
        message=request.message,
        session_id=request.session_id,
        user_data=request.context if request.context else None
    )
    
    return response
@router.post("/image-analysis")
async def analyze_image(request: ImageAnalysisRequest):
    """
    Analyze an image containing financial charts or documents
    
    Args:
        request: The image analysis request containing base64-encoded image data
    
    Returns:
        Image analysis results
    """
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.base64_image)
        
        # Process the image
        response = await financial_chatbot.process_image_message(
            image_data=image_data,
            session_id=request.session_id if request.session_id else "anonymous",
            user_data=request.context or {}
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@router.post("/upload-image")
async def upload_image_for_analysis(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Upload and analyze an image containing financial charts or documents
    
    Args:
        file: The uploaded image file
        session_id: Optional session identifier
    
    Returns:
        Image analysis results
    """
    try:
        # Read the file content
        image_data = await file.read()
        
        # Process the image
        response = await financial_chatbot.process_image_message(
            image_data=image_data,
            session_id=session_id if session_id else "anonymous",
            user_data={}
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@router.post("/market-insights")
async def get_market_insights(request: MarketInsightsRequest):
    """
    Get comprehensive market insights using all available services
    
    Args:
        request: The market insights request containing optional stock symbols
    
    Returns:
        Comprehensive market insights
    """
    response = await financial_chatbot.get_market_insights(symbols=request.symbols or [])
    
    return response


# Add a specialized stock analysis endpoint
@router.post("/stock-analysis", response_model=ChatResponse)
async def analyze_stock(request: AnalysisRequest):
    """
    Get specialized stock analysis using AI services
    
    Args:
        request: The analysis request containing stock symbol and query
    
    Returns:
        AI-powered stock analysis
    """
    # Format the message for our AI processor
    message = f"Analyze {request.symbol or 'the market'}: {request.query}"
    
    response = await financial_chatbot.process_user_message(
        message=message,
        session_id="stock-analysis",
        user_data={
            "analysis_type": "stock",
            "symbol": request.symbol,
            "time_period": request.time_period
        }
    )
    
    return response


# Add a technical indicator endpoint
@router.post("/technical-indicators")
async def get_technical_indicators(request: StockRequest):
    """
    Get technical indicators for a stock
    
    Args:
        request: The stock request containing the symbol
    
    Returns:
        Technical indicators for the stock
    """
    if not request.symbol:
        raise HTTPException(status_code=400, detail="Stock symbol is required")
    
    # Get mock stock history for now
    history = financial_chatbot._get_mock_stock_history(request.symbol, 60)
    
    # Generate technical indicators
    indicators = await financial_chatbot.vertex_ai.generate_technical_indicators(
        request.symbol, {"history": history}
    )
    
    return indicators


# Export the router
gcloud_router = router
