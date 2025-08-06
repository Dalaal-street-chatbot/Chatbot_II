import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Basic Configuration
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("PORT", 8000))
    
    # Groq AI Configuration (Main NLP AI)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Google APIs
    GOOGLE_FINANCE_API = os.getenv("GOOGLE_FINANCE_API")
    GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
    GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    CLOUD_VISION_API = os.getenv("CLOUD_VISION_API")
    
    # Vertex AI Configuration
    VERTEX_AI_API_CLIENT_ID = os.getenv("VERTEX_AI_API_CLIENT_ID")
    VERTEX_AI_API_CLIENT_SECRET = os.getenv("VERTEX_AI_API_CLIENT_SECRET")
    VERTEX_AI_API_REFRESH_TOKEN = os.getenv("VERTEX_AI_API_REFRESH_TOKEN")
    VERTEX_AI_API_PROJECT_ID = os.getenv("VERTEX_AI_API_PROJECT_ID", GOOGLE_CLOUD_PROJECT_ID)
    VERTEX_AI_API_LOCATION = os.getenv("VERTEX_AI_API_LOCATION", GOOGLE_CLOUD_LOCATION)
    
    # News API
    NEWS_API = os.getenv("NEWS_API")
    
    # Codestral API
    CODESTRAL_API_KEY = os.getenv("CODESTRAL_API_KEY")
    
    # Indian Stock Market API
    INDIAN_STOCK_API_KEY = os.getenv("INDIAN_STOCK_API_KEY")
    INDIAN_STOCK_API_BASE_URL = os.getenv("INDIAN_STOCK_API_BASE_URL")
    
    # Upstox API
    UPSTOX_API_KEY = os.getenv("UPSTOX_API_KEY")
    UPSTOX_API_SECRET = os.getenv("UPSTOX_API_SECRET")
    UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI")
    UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
    
    # AI/ML APIs
    DEEPSEEK_AI_R1_API = os.getenv("DEEPSEEK_AI_R1_API")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    RASA_PRO_LICENSE = os.getenv("RASA_PRO_LICENSE")
    
    # DialogFlow
    DIALOGFLOW_API = os.getenv("DIALOGFLOW_API")
    DIALOGFLOW_PROJECT_ID = os.getenv("DIALOGFLOW_PROJECT_ID")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # AlloyDB
    ALLOYDB_API_CLIENT_ID = os.getenv("ALLOYDB_API_CLIENT_ID")
    
    # Azure Configuration
    AZURE_ENV_NAME = os.getenv("AZURE_ENV_NAME")
    AZURE_LOCATION = os.getenv("AZURE_LOCATION")
    AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_EXISTING_AIPROJECT_ENDPOINT = os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration values are present"""
        required_keys = [
            "GROQ_API_KEY",
            "NEWS_API",
            "INDIAN_STOCK_API_KEY"
        ]
        
        missing_keys = []
        for key in required_keys:
            if not getattr(cls, key):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"Missing required environment variables: {', '.join(missing_keys)}")
            return False
        
        return True

# Create config instance
config = Config()
