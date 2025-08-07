#!/usr/bin/env python3
"""
Simple API Testing Script for Dalaal Street Chatbot
Tests configured APIs without dependencies
"""

import requests
import json
import os
from datetime import datetime

# Environment variables (replace with your actual values for testing)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT")
# GROQ REMOVED - No longer using Groq service
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

def test_azure_openai():
    """Test Azure OpenAI API"""
    print("🔵 Testing Azure OpenAI API...")
    
    if not AZURE_OPENAI_API_KEY or not AZURE_ENDPOINT:
        return {
            "status": "error",
            "message": "Azure OpenAI credentials missing",
            "details": "Set AZURE_OPENAI_API_KEY and AZURE_EXISTING_AIPROJECT_ENDPOINT environment variables"
        }
    
    try:
        headers = {
            "api-key": AZURE_OPENAI_API_KEY,
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
                url = f"{AZURE_ENDPOINT}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview"
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "status": "success",
                        "message": "Azure OpenAI API working",
                        "deployment": deployment,
                        "response_preview": data["choices"][0]["message"]["content"][:100] + "...",
                        "usage": data.get("usage", {}),
                        "model": deployment
                    }
                elif response.status_code == 404:
                    continue  # Try next deployment
                else:
                    return {
                        "status": "error",
                        "message": f"Azure OpenAI API error: {response.status_code}",
                        "details": response.text[:200]
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

def test_google_ai_api():
    """Test Google AI API"""
    print("� Testing Google AI API...")

def test_upstox_api():
    """Test Upstox API"""
    print("📈 Testing Upstox API...")
    
    if not UPSTOX_ACCESS_TOKEN:
        return {
            "status": "error",
            "message": "Upstox access token missing",
            "details": "Set UPSTOX_ACCESS_TOKEN environment variable"
        }
    
    try:
        headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
        
        # Test market data endpoint
        response = requests.get(
            "https://api.upstox.com/v2/market-quote/ltp?instrument_key=NSE_EQ%7CINE002A01018",
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "success",
                "message": "Upstox API working",
                "sample_data": {
                    "instrument": "RELIANCE",
                    "ltp": data.get("data", {}).get("NSE_EQ:INE002A01018", {}).get("last_price", "N/A")
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Upstox API error: {response.status_code}",
                "details": response.text[:200]
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Upstox API test failed: {str(e)}",
            "details": str(e)
        }

def test_news_api():
    """Test News API"""
    print("📰 Testing News API...")
    
    if not NEWS_API_KEY:
        return {
            "status": "error",
            "message": "News API key missing",
            "details": "Set NEWS_API environment variable"
        }
    
    try:
        params = {
            "q": "NIFTY OR stock market OR BSE",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5,
            "apiKey": NEWS_API_KEY
        }
        
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "success",
                "message": "News API working",
                "total_results": data.get("totalResults", 0),
                "sample_headline": data.get("articles", [{}])[0].get("title", "No articles found")[:100]
            }
        else:
            return {
                "status": "error",
                "message": f"News API error: {response.status_code}",
                "details": response.text[:200]
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"News API test failed: {str(e)}",
            "details": str(e)
        }

def test_google_ai_api():
    """Test Google AI API"""
    print("🔴 Testing Google AI API...")
    
    if not GOOGLE_AI_API_KEY:
        return {
            "status": "error",
            "message": "Google AI API key missing",
            "details": "Set GOOGLE_AI_API_KEY environment variable"
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
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_AI_API_KEY}",
            json=payload,
            timeout=20
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "success",
                "message": "Google AI API working",
                "response_preview": data["candidates"][0]["content"]["parts"][0]["text"][:100] + "...",
                "model": "gemini-pro"
            }
        else:
            return {
                "status": "error",
                "message": f"Google AI API error: {response.status_code}",
                "details": response.text[:200]
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Google AI API test failed: {str(e)}",
            "details": str(e)
        }

def main():
    """Main function to run API tests"""
    print("🚀 Starting Comprehensive API Testing for Dalaal Street Chatbot")
    print("=" * 70)
    
    # Run all tests (Groq removed)
    results = {
        "azure_openai": test_azure_openai(),
        "upstox": test_upstox_api(),
        "news_api": test_news_api(),
        "google_ai": test_google_ai_api(),
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("🎯 API TEST RESULTS")
    print("=" * 70)
    
    working_apis = []
    failed_apis = []
    
    for api_name, result in results.items():
        status_emoji = "✅" if result.get("status") == "success" else "❌"
        print(f"\n{status_emoji} {api_name.upper().replace('_', ' ')}")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Message: {result.get('message', 'No message')}")
        
        if result.get("status") == "success":
            working_apis.append(api_name)
            if "response_preview" in result:
                print(f"   Response: {result['response_preview']}")
            if "model" in result:
                print(f"   Model: {result['model']}")
            if "usage" in result:
                print(f"   Usage: {result['usage']}")
        else:
            failed_apis.append(api_name)
            if "details" in result:
                print(f"   Details: {result['details']}")
    
    # Print summary
    print(f"\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total APIs Tested: {len(results)}")
    print(f"Success Rate: {len(working_apis)}/{len(results)}")
    print(f"Working APIs: {', '.join(working_apis) if working_apis else 'None'}")
    print(f"Failed APIs: {', '.join(failed_apis) if failed_apis else 'None'}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("🔧 CONFIGURATION RECOMMENDATIONS")
    print("=" * 70)
    
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
        
        # Recommend primary service (Groq removed)
        if "azure_openai" in working_apis:
            primary = "Azure OpenAI (Enterprise-grade, recommended for production)"
        elif "google_ai" in working_apis:
            primary = "Google AI (Good alternative with Gemini Pro)"
        else:
            primary = "None available"
        
        print(f"   Recommended Primary NLP: {primary}")
    else:
        print("\n❌ No AI services are working. Please check your API keys.")

if __name__ == "__main__":
    main()
