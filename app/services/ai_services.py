import requests
from typing import Dict, List, Optional, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import config

class CodestralService:
    """Codestral API service for advanced financial intelligence"""
    
    def __init__(self):
        if not config.CODESTRAL_API_KEY:
            raise ValueError("CODESTRAL_API_KEY environment variable is required")
        
        self.api_key = config.CODESTRAL_API_KEY
        self.base_url = "https://codestral.mistral.ai/v1"
    
    async def generate_financial_code(self, prompt: str) -> Dict[str, Any]:
        """Generate financial analysis code or algorithms"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "codestral-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial programming assistant. Generate Python code for financial analysis, trading algorithms, and market calculations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "code": data["choices"][0]["message"]["content"],
                    "model": "codestral"
                }
            else:
                return {
                    "status": "error",
                    "message": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            print(f"Error with Codestral API: {e}")
            return {
                "status": "error",
                "message": "Failed to generate code"
            }
    
    async def analyze_trading_strategy(self, strategy_description: str) -> Dict[str, Any]:
        """Analyze and optimize trading strategies"""
        prompt = f"""
        Analyze this trading strategy and provide:
        1. Code implementation in Python
        2. Risk assessment
        3. Backtesting approach
        4. Optimization suggestions
        
        Strategy: {strategy_description}
        
        Use pandas, numpy, and common financial libraries.
        """
        
        return await self.generate_financial_code(prompt)

class DeepSeekService:
    """DeepSeek AI service for advanced reasoning"""
    
    def __init__(self):
        if not config.DEEPSEEK_AI_R1_API:
            raise ValueError("DEEPSEEK_AI_R1_API environment variable is required")
        
        self.api_key = config.DEEPSEEK_AI_R1_API
        self.base_url = "https://api.deepseek.com/v1"
    
    async def deep_financial_analysis(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep financial reasoning and analysis"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Perform a deep financial analysis with step-by-step reasoning:
            
            Query: {query}
            Data: {data}
            
            Provide:
            1. Detailed reasoning process
            2. Multiple perspectives analysis
            3. Risk-reward assessment
            4. Market context consideration
            5. Actionable insights
            
            Use logical reasoning and consider market dynamics.
            """
            
            payload = {
                "model": "deepseek-reasoner",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst with deep reasoning capabilities. Provide thorough, step-by-step analysis with clear reasoning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "analysis": data["choices"][0]["message"]["content"],
                    "reasoning_steps": self._extract_reasoning_steps(data["choices"][0]["message"]["content"]),
                    "model": "deepseek-reasoner"
                }
            else:
                return {
                    "status": "error",
                    "message": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            print(f"Error with DeepSeek API: {e}")
            return {
                "status": "error",
                "message": "Failed to perform deep analysis"
            }
    
    def _extract_reasoning_steps(self, content: str) -> List[str]:
        """Extract reasoning steps from the analysis"""
        # Simple extraction - can be enhanced with NLP
        lines = content.split('\n')
        steps = []
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                steps.append(line.strip())
        return steps

class OllamaService:
    """Local Ollama service for offline AI capabilities"""
    
    def __init__(self):
        self.host = config.OLLAMA_HOST
        self.model = config.OLLAMA_MODEL
    
    async def local_analysis(self, prompt: str) -> Dict[str, Any]:
        """Perform local AI analysis using Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "response": data["response"],
                    "model": self.model,
                    "local": True
                }
            else:
                return {
                    "status": "error",
                    "message": f"Ollama error: {response.status_code}"
                }
                
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return {
                "status": "error",
                "message": "Local AI service unavailable"
            }

# Create global instances
codestral_service = CodestralService()
deepseek_service = DeepSeekService()
ollama_service = OllamaService()
