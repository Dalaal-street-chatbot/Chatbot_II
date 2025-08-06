from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import config
from app.api.routes import router as api_router
from app.api.gcloud_routes import router as gcloud_api_router

# Validate configuration
if not config.validate_config():
    print("ERROR: Missing required configuration. Please check your .env file.")
    sys.exit(1)

app = FastAPI(
    title="Dalaal Street Chatbot API",
    description="AI-powered Indian financial markets chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["api"])

# Include Google Cloud Services routes
app.include_router(gcloud_api_router, prefix="/api/v1", tags=["google-cloud"])

# Serve static files (for React frontend)
if os.path.exists("build"):
    app.mount("/static", StaticFiles(directory="build/static"), name="static")

@app.get("/")
async def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to Dalaal Street Bot API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "chat": "/api/v1/chat",
            "stock": "/api/v1/stock",
            "news": "/api/v1/news",
            "indices": "/api/v1/indices",
            "analysis": "/api/v1/analysis"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "groq_configured": bool(config.GROQ_API_KEY),
        "news_api_configured": bool(config.NEWS_API),
        "indian_stock_api_configured": bool(config.INDIAN_STOCK_API_KEY)
    }

# Serve React app for any other routes (if build directory exists)
@app.get("/{path:path}")
async def serve_spa(path: str):
    """Serve React SPA"""
    if os.path.exists("build"):
        return FileResponse("build/index.html")
    else:
        raise HTTPException(status_code=404, detail="Frontend not built")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )