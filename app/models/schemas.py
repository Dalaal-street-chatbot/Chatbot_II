from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    session_id: Optional[str] = Field(None, description="Session identifier")

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Bot response")
    intent: Optional[str] = Field(None, description="Detected intent")
    entities: Optional[List[str]] = Field(None, description="Extracted entities")
    confidence: Optional[float] = Field(None, description="Confidence score")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class StockRequest(BaseModel):
    """Stock data request model"""
    symbol: str = Field(..., description="Stock symbol")
    exchange: Optional[str] = Field("NSE", description="Exchange (NSE/BSE)")

class StockResponse(BaseModel):
    """Stock data response model"""
    symbol: str
    price: float
    currency: str = "INR"
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[int] = None
    source: Optional[str] = None

class NewsRequest(BaseModel):
    """News request model"""
    query: Optional[str] = Field(None, description="Search query")
    company: Optional[str] = Field(None, description="Company name")
    page_size: Optional[int] = Field(10, description="Number of articles")

class NewsArticle(BaseModel):
    """News article model"""
    title: str
    description: Optional[str] = None
    url: str
    source: str
    published_at: str
    image_url: Optional[str] = None

class NewsResponse(BaseModel):
    """News response model"""
    status: str
    total_results: int
    articles: List[NewsArticle]

class MarketIndicesResponse(BaseModel):
    """Market indices response model"""
    indices: Dict[str, Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.now)

class AnalysisRequest(BaseModel):
    """Financial analysis request model"""
    symbol: Optional[str] = None
    query: str = Field(..., description="Analysis query")
    time_period: Optional[str] = Field("1mo", description="Analysis time period")

class AnalysisResponse(BaseModel):
    """Financial analysis response model"""
    analysis: str = Field(..., description="Analysis result")
    data: Optional[Dict[str, Any]] = Field(None, description="Supporting data")
    recommendations: Optional[List[str]] = Field(None, description="Recommendations")
    confidence: Optional[float] = Field(None, description="Analysis confidence")
