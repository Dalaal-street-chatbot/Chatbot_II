#!/usr/bin/env python3
"""
Comprehensive API Testing Script for Dalaal Street Chatbot
Tests all configured APIs and provides detailed status reports
"""

import asyncio
import aiohttp
import requests
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import config

class APITester:
    """Comprehensive API testing class"""
    
    def __init__(self):
        self.results = {}
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_azure_openai(self) -> Dict[str, Any]:
        """Test Azure OpenAI API"""
        print("🔵 Testing Azure OpenAI API...")
        
        if not config.AZURE_OPENAI_API_KEY or not config.AZURE_EXISTING_AIPROJECT_ENDPOINT:
            return {
                "status": "error",
                "message": "Azure OpenAI credentials missing",
                "details": "AZURE_OPENAI_API_KEY or AZURE_EXISTING_AIPROJECT_ENDPOINT not configured"
            }
        
        try:
            headers = {
                "api-key": config.AZURE_OPENAI_API_KEY,
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial AI assistant. Keep responses brief for testing."
                    },
                    {
                        "role": "user",
                        "content": "What is the current market sentiment for NIFTY 50?"
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3
            }
            
            # Try different deployment names
            deployment_names = ["gpt-4", "gpt-35-turbo", "gpt-4o", "gpt-4-turbo"]
            
            for deployment in deployment_names:
                try:
                    url = f"{config.AZURE_EXISTING_AIPROJECT_ENDPOINT}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview"
                    
                    async with self.session.post(url, headers=headers, json=payload, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                "status": "success",
                                "message": "Azure OpenAI API working",
                                "deployment": deployment,
                                "response_preview": data["choices"][0]["message"]["content"][:100] + "...",
                                "usage": data.get("usage", {}),
                                "model": deployment
                            }
                        elif response.status == 404:
                            continue  # Try next deployment
                        else:
                            error_text = await response.text()
                            return {
                                "status": "error",
                                "message": f"Azure OpenAI API error: {response.status}",
                                "details": error_text[:200]
                            }
                except Exception as e:
                    continue  # Try next deployment
            
            return {
                "status": "error",
                "message": "No working Azure OpenAI deployment found",
                "details": f"Tried deployments: {', '.join(deployment_names)}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Azure OpenAI test failed: {str(e)}",
                "details": str(e)
            }

    async def test_groq_api(self) -> Dict[str, Any]:
        """Test Groq AI API"""
        print("🟢 Testing Groq AI API...")
        
        if not config.GROQ_API_KEY:
            return {
                "status": "error",
                "message": "Groq API key missing",
                "details": "GROQ_API_KEY not configured"
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is RELIANCE stock current status? Keep it brief."
                    }
                ],
                "model": "llama-3.1-70b-versatile",
                "max_tokens": 100,
                "temperature": 0.3
            }
            
            async with self.session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "message": "Groq AI API working",
                        "response_preview": data["choices"][0]["message"]["content"][:100] + "...",
                        "usage": data.get("usage", {}),
                        "model": data.get("model", "llama-3.1-70b-versatile")
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "error",
                        "message": f"Groq API error: {response.status}",
                        "details": error_text[:200]
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Groq API test failed: {str(e)}",
                "details": str(e)
            }

    async def test_upstox_api(self) -> Dict[str, Any]:
        """Test Upstox API"""
        print("📈 Testing Upstox API...")
        
        if not config.UPSTOX_ACCESS_TOKEN:
            return {
                "status": "error",
                "message": "Upstox access token missing",
                "details": "UPSTOX_ACCESS_TOKEN not configured"
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {config.UPSTOX_ACCESS_TOKEN}",
                "Accept": "application/json"
            }
            
            # Test market data endpoint
            async with self.session.get(
                "https://api.upstox.com/v2/market-quote/ltp?instrument_key=NSE_EQ%7CINE002A01018",
                headers=headers,
                timeout=15
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "message": "Upstox API working",
                        "sample_data": {
                            "instrument": "RELIANCE",
                            "ltp": data.get("data", {}).get("NSE_EQ:INE002A01018", {}).get("last_price", "N/A")
                        }
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "error",
                        "message": f"Upstox API error: {response.status}",
                        "details": error_text[:200]
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Upstox API test failed: {str(e)}",
                "details": str(e)
            }

    async def test_news_api(self) -> Dict[str, Any]:
        """Test News API"""
        print("📰 Testing News API...")
        
        if not config.NEWS_API:
            return {
                "status": "error",
                "message": "News API key missing",
                "details": "NEWS_API not configured"
            }
        
        try:
            params = {
                "q": "NIFTY OR stock market OR BSE",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 5,
                "apiKey": config.NEWS_API
            }
            
            async with self.session.get(
                "https://newsapi.org/v2/everything",
                params=params,
                timeout=15
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "message": "News API working",
                        "total_results": data.get("totalResults", 0),
                        "sample_headline": data.get("articles", [{}])[0].get("title", "No articles found")[:100]
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "error",
                        "message": f"News API error: {response.status}",
                        "details": error_text[:200]
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"News API test failed: {str(e)}",
                "details": str(e)
            }

    async def test_google_ai_api(self) -> Dict[str, Any]:
        """Test Google AI API"""
        print("🔴 Testing Google AI API...")
        
        if not config.GOOGLE_AI_API_KEY:
            return {
                "status": "error",
                "message": "Google AI API key missing",
                "details": "GOOGLE_AI_API_KEY not configured"
            }
        
        try:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "What are the key factors affecting Indian stock market today? Keep it brief."}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,
                    "topK": 32,
                    "topP": 1,
                    "maxOutputTokens": 100
                }
            }
            
            async with self.session.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={config.GOOGLE_AI_API_KEY}",
                json=payload,
                timeout=20
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "message": "Google AI API working",
                        "response_preview": data["candidates"][0]["content"]["parts"][0]["text"][:100] + "...",
                        "model": "gemini-pro"
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "error",
                        "message": f"Google AI API error: {response.status}",
                        "details": error_text[:200]
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Google AI API test failed: {str(e)}",
                "details": str(e)
            }

    async def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all API tests"""
        print("🚀 Starting Comprehensive API Testing for Dalaal Street Chatbot")
        print("=" * 70)
        
        start_time = time.time()
        
        # Test all APIs
        self.results = {
            "azure_openai": await self.test_azure_openai(),
            "groq_ai": await self.test_groq_api(),
            "upstox": await self.test_upstox_api(),
            "news_api": await self.test_news_api(),
            "google_ai": await self.test_google_ai_api(),
        }
        
        # Add summary
        self.results["summary"] = self._generate_summary()
        self.results["test_duration"] = time.time() - start_time
        self.results["timestamp"] = datetime.now().isoformat()
        
        return self.results

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        working_apis = []
        failed_apis = []
        
        for api_name, result in self.results.items():
            if api_name == "summary":
                continue
                
            if result.get("status") == "success":
                working_apis.append(api_name)
            else:
                failed_apis.append(api_name)
        
        return {
            "total_apis_tested": len(self.results) - 1,  # Exclude summary itself
            "working_apis": working_apis,
            "failed_apis": failed_apis,
            "success_rate": f"{len(working_apis)}/{len(self.results) - 1}",
            "primary_nlp_recommendation": self._recommend_primary_nlp(working_apis)
        }

    def _recommend_primary_nlp(self, working_apis: List[str]) -> str:
        """Recommend primary NLP service based on working APIs"""
        if "azure_openai" in working_apis:
            return "azure_openai (Enterprise-grade, recommended for production)"
        elif "groq_ai" in working_apis:
            return "groq_ai (Fast and reliable, good fallback)"
        elif "google_ai" in working_apis:
            return "google_ai (Good alternative with Gemini Pro)"
        else:
            return "No working NLP APIs found"

    def print_results(self):
        """Print formatted test results"""
        print("\n" + "=" * 70)
        print("🎯 API TEST RESULTS")
        print("=" * 70)
        
        for api_name, result in self.results.items():
            if api_name in ["summary", "test_duration", "timestamp"]:
                continue
                
            status_emoji = "✅" if result.get("status") == "success" else "❌"
            print(f"\n{status_emoji} {api_name.upper().replace('_', ' ')}")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Message: {result.get('message', 'No message')}")
            
            if result.get("status") == "success":
                if "response_preview" in result:
                    print(f"   Response: {result['response_preview']}")
                if "model" in result:
                    print(f"   Model: {result['model']}")
                if "usage" in result:
                    print(f"   Usage: {result['usage']}")
            else:
                if "details" in result:
                    print(f"   Details: {result['details']}")
        
        # Print summary
        summary = self.results.get("summary", {})
        print(f"\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"Total APIs Tested: {summary.get('total_apis_tested', 0)}")
        print(f"Success Rate: {summary.get('success_rate', '0/0')}")
        print(f"Working APIs: {', '.join(summary.get('working_apis', []))}")
        print(f"Failed APIs: {', '.join(summary.get('failed_apis', []))}")
        print(f"Primary NLP Recommendation: {summary.get('primary_nlp_recommendation', 'None')}")
        print(f"Test Duration: {self.results.get('test_duration', 0):.2f} seconds")

async def main():
    """Main function to run API tests"""
    async with APITester() as tester:
        results = await tester.run_all_tests()
        tester.print_results()
        
        # Provide specific recommendations
        print("\n" + "=" * 70)
        print("🔧 CONFIGURATION RECOMMENDATIONS")
        print("=" * 70)
        
        summary = results.get("summary", {})
        working_apis = summary.get("working_apis", [])
        
        if "azure_openai" in working_apis:
            print("✅ Azure OpenAI is working and ready to be your primary NLP service!")
            print("   - Update comprehensive_chat.py to use Azure OpenAI as primary")
            print("   - Configure fallback chain: Azure OpenAI → Groq → Google AI")
        else:
            print("❌ Azure OpenAI not working. Check your configuration:")
            print("   - Verify AZURE_OPENAI_API_KEY is correct")
            print("   - Verify AZURE_EXISTING_AIPROJECT_ENDPOINT is correct")
            print("   - Check if you have the correct deployment name")
        
        if len(working_apis) > 0:
            print(f"\n✅ You have {len(working_apis)} working AI services!")
            print("   This provides good redundancy for your chatbot.")
        else:
            print("\n❌ No AI services are working. Please check your API keys.")

if __name__ == "__main__":
    asyncio.run(main())
