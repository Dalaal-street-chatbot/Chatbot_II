import os
import json
from typing import Dict, List, Optional, Any
from groq import Groq
from config.settings import config

class GroqAIService:
    """Main NLP AI service using Groq API"""
    
    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = "llama3-8b-8192"  # Updated to supported model
        
    async def generate_response(
        self, 
        user_message: str, 
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate AI response using Groq"""
        
        default_system_prompt = """
        You are Dalaal Street Bot, an expert Indian financial markets assistant. 
        You help users with:
        - Stock market analysis and data
        - Investment advice and strategies
        - Market news and trends
        - Financial education
        - Trading insights
        
        Always provide accurate, helpful, and responsible financial information.
        When discussing investments, remind users to do their own research and consult financial advisors.
        """
        
        messages = [
            {
                "role": "system",
                "content": system_prompt or default_system_prompt
            }
        ]
        
        # Add context if provided
        if context:
            context_message = f"Additional context: {json.dumps(context, indent=2)}"
            messages.append({
                "role": "system", 
                "content": context_message
            })
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating Groq response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again later."
    
    async def analyze_financial_query(self, query: str) -> Dict[str, Any]:
        """Analyze financial query and extract intent and entities"""
        
        analysis_prompt = f"""
        Analyze this financial query and extract:
        1. Intent (stock_price, market_analysis, news, investment_advice, general_info)
        2. Entities (stock symbols, company names, market sectors)
        3. Sentiment (positive, negative, neutral)
        4. Urgency (high, medium, low)
        
        Query: "{query}"
        
        Respond in JSON format:
        {{
            "intent": "intent_type",
            "entities": ["entity1", "entity2"],
            "sentiment": "sentiment_type",
            "urgency": "urgency_level",
            "confidence": 0.95
        }}
        """
        
        try:
            response = await self.generate_response(
                analysis_prompt,
                system_prompt="You are a financial query analyzer. Respond only in valid JSON format."
            )
            
            # Parse JSON response
            analysis = json.loads(response)
            return analysis
            
        except json.JSONDecodeError:
            return {
                "intent": "general_info",
                "entities": [],
                "sentiment": "neutral",
                "urgency": "low",
                "confidence": 0.5
            }
        except Exception as e:
            print(f"Error analyzing query: {e}")
            return {
                "intent": "general_info",
                "entities": [],
                "sentiment": "neutral",
                "urgency": "low",
                "confidence": 0.0
            }

# Create global instance
groq_service = GroqAIService()
