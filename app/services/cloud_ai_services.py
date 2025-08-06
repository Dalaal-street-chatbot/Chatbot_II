import requests
from typing import Dict, List, Optional, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import config

class AzureOpenAIService:
    """Azure OpenAI service for enterprise-grade AI"""
    
    def __init__(self):
        if not config.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
        
        self.api_key = config.AZURE_OPENAI_API_KEY
        self.endpoint = config.AZURE_EXISTING_AIPROJECT_ENDPOINT
        self.api_version = "2024-02-15-preview"
    
    async def generate_financial_insights(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate financial insights using Azure OpenAI"""
        try:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an enterprise-grade financial AI assistant specializing in Indian markets. 
                    Provide professional, accurate, and actionable financial insights while maintaining compliance with financial regulations."""
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\nContext: {context or 'No additional context'}"
                }
            ]
            
            payload = {
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.3,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
            url = f"{self.endpoint}/openai/deployments/gpt-4/chat/completions?api-version={self.api_version}"
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "insights": data["choices"][0]["message"]["content"],
                    "model": "azure-gpt-4",
                    "usage": data.get("usage", {})
                }
            else:
                return {
                    "status": "error",
                    "response": f"Azure API error: {response.status_code}",
                    "intent": "error",
                    "entities": []
                }
                
        except Exception as e:
            print(f"Error with Azure OpenAI: {e}")
            return {
                "status": "error",
                "response": "Failed to generate Azure insights",
                "intent": "error",
                "entities": []
            }

class AzureTextAnalyticsService:
    """Azure Text Analytics for sentiment and entity analysis"""
    
    def __init__(self):
        self.endpoint = config.AZURE_EXISTING_AIPROJECT_ENDPOINT
        self.api_key = config.AZURE_OPENAI_API_KEY
    
    async def analyze_market_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of market news/comments"""
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            documents = [
                {"id": str(i), "text": text} 
                for i, text in enumerate(texts)
            ]
            
            payload = {"documents": documents}
            
            # Note: This is a simplified endpoint - actual Azure Text Analytics has different structure
            response = requests.post(
                f"{self.endpoint}/text/analytics/v3.1/sentiment",
                headers=headers,
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "sentiment_analysis": response.json(),
                    "service": "azure-text-analytics"
                }
            else:
                return {
                    "status": "error",
                    "response": f"Sentiment analysis error: {response.status_code}",
                    "intent": "error",
                    "entities": []
                }
                
        except Exception as e:
            print(f"Error with Azure Text Analytics: {e}")
            return {
                "status": "error",
                "response": "Sentiment analysis failed",
                "intent": "error",
                "entities": []
            }

class GoogleCloudAIService:
    """Google Cloud AI services integration"""
    
    def __init__(self):
        self.api_key = config.GOOGLE_AI_API_KEY
        self.project_id = config.GOOGLE_CLOUD_PROJECT_ID
    
    async def generate_market_predictions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Use Google AI for market predictions"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Based on the following market data, provide predictions and analysis:
            {data}
            
            Generate:
            1. Short-term outlook (1-7 days)
            2. Medium-term outlook (1-3 months)
            3. Key factors to watch
            4. Risk assessment
            """
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,
                    "topK": 32,
                    "topP": 1,
                    "maxOutputTokens": 1000
                }
            }
            
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "predictions": data["candidates"][0]["content"]["parts"][0]["text"],
                    "model": "gemini-pro"
                }
            else:
                return {
                    "status": "error",
                    "response": f"Google AI error: {response.status_code}",
                    "intent": "error",
                    "entities": []
                }
                
        except Exception as e:
            print(f"Error with Google AI: {e}")
            return {
                "status": "error",
                "response": "Prediction generation failed",
                "intent": "error",
                "entities": []
            }

# Create global instances
azure_openai_service = AzureOpenAIService()
azure_text_service = AzureTextAnalyticsService()
google_ai_service = GoogleCloudAIService()
