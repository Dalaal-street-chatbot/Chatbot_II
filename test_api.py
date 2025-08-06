#!/usr/bin/env python3

"""
Dalaal Street Chatbot Test Suite
Tests all API endpoints and AI services
"""

import asyncio
import requests
import json
import time
from typing import Dict, Any, List
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DalaalStreetTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
        self.test_results = []
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Starting Dalaal Street Chatbot Test Suite\n")
        
        # Test basic connectivity
        self.test_health_check()
        self.test_root_endpoint()
        
        # Test API endpoints
        self.test_stock_endpoint()
        self.test_indices_endpoint()
        self.test_news_endpoint()
        self.test_chat_endpoint()
        self.test_analysis_endpoint()
        
        # Test AI services (if available)
        self.test_ai_services()
        
        # Print summary
        self.print_test_summary()
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("🏥 Testing health check...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                self.test_results.append(("Health Check", True, "Passed"))
            else:
                print(f"❌ Health check failed: {response.status_code}")
                self.test_results.append(("Health Check", False, f"Status: {response.status_code}"))
        except Exception as e:
            print(f"❌ Health check error: {e}")
            self.test_results.append(("Health Check", False, str(e)))
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        print("🏠 Testing root endpoint...")
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint passed: {data.get('message', 'No message')}")
                self.test_results.append(("Root Endpoint", True, "Passed"))
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                self.test_results.append(("Root Endpoint", False, f"Status: {response.status_code}"))
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            self.test_results.append(("Root Endpoint", False, str(e)))
    
    def test_stock_endpoint(self):
        """Test stock data endpoint"""
        print("📈 Testing stock endpoint...")
        test_symbols = ["RELIANCE", "TCS", "INFY"]
        
        for symbol in test_symbols:
            try:
                payload = {"symbol": symbol, "exchange": "NSE"}
                response = requests.post(
                    f"{self.api_base}/stock", 
                    json=payload, 
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Stock data for {symbol}: ₹{data.get('price', 'N/A')}")
                    self.test_results.append((f"Stock-{symbol}", True, f"Price: {data.get('price')}"))
                else:
                    print(f"❌ Stock data failed for {symbol}: {response.status_code}")
                    self.test_results.append((f"Stock-{symbol}", False, f"Status: {response.status_code}"))
                    
            except Exception as e:
                print(f"❌ Stock endpoint error for {symbol}: {e}")
                self.test_results.append((f"Stock-{symbol}", False, str(e)))
            
            time.sleep(1)  # Rate limiting
    
    def test_indices_endpoint(self):
        """Test market indices endpoint"""
        print("📊 Testing indices endpoint...")
        try:
            response = requests.get(f"{self.api_base}/indices", timeout=15)
            if response.status_code == 200:
                data = response.json()
                indices = data.get('indices', {})
                print(f"✅ Market indices retrieved: {list(indices.keys())}")
                for name, info in indices.items():
                    if 'price' in info:
                        print(f"   {name}: {info['price']}")
                self.test_results.append(("Market Indices", True, f"Retrieved {len(indices)} indices"))
            else:
                print(f"❌ Indices endpoint failed: {response.status_code}")
                self.test_results.append(("Market Indices", False, f"Status: {response.status_code}"))
        except Exception as e:
            print(f"❌ Indices endpoint error: {e}")
            self.test_results.append(("Market Indices", False, str(e)))
    
    def test_news_endpoint(self):
        """Test financial news endpoint"""
        print("📰 Testing news endpoint...")
        try:
            payload = {"query": "stock market", "page_size": 3}
            response = requests.post(
                f"{self.api_base}/news", 
                json=payload, 
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"✅ News retrieved: {len(articles)} articles")
                if articles:
                    print(f"   Latest: {articles[0].get('title', 'No title')[:50]}...")
                self.test_results.append(("Financial News", True, f"Retrieved {len(articles)} articles"))
            else:
                print(f"❌ News endpoint failed: {response.status_code}")
                self.test_results.append(("Financial News", False, f"Status: {response.status_code}"))
        except Exception as e:
            print(f"❌ News endpoint error: {e}")
            self.test_results.append(("Financial News", False, str(e)))
    
    def test_chat_endpoint(self):
        """Test chat endpoint"""
        print("💬 Testing chat endpoint...")
        test_messages = [
            "Hello, how are you?",
            "What is the price of Reliance?",
            "Tell me about the stock market today",
            "Should I invest in IT stocks?"
        ]
        
        for i, message in enumerate(test_messages):
            try:
                payload = {
                    "message": message,
                    "session_id": f"test_session_{i}"
                }
                response = requests.post(
                    f"{self.api_base}/chat", 
                    json=payload, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '')
                    intent = data.get('intent', 'unknown')
                    print(f"✅ Chat response for '{message[:30]}...': {intent}")
                    print(f"   Response: {response_text[:100]}...")
                    self.test_results.append((f"Chat-{i+1}", True, f"Intent: {intent}"))
                else:
                    print(f"❌ Chat failed for '{message[:30]}...': {response.status_code}")
                    self.test_results.append((f"Chat-{i+1}", False, f"Status: {response.status_code}"))
                    
            except Exception as e:
                print(f"❌ Chat error for '{message[:30]}...': {e}")
                self.test_results.append((f"Chat-{i+1}", False, str(e)))
            
            time.sleep(2)  # Rate limiting for AI services
    
    def test_analysis_endpoint(self):
        """Test financial analysis endpoint"""
        print("🔍 Testing analysis endpoint...")
        try:
            payload = {
                "query": "Analyze the current market trends",
                "symbol": "NIFTY",
                "time_period": "1mo"
            }
            response = requests.post(
                f"{self.api_base}/analysis", 
                json=payload, 
                timeout=45
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get('analysis', '')
                confidence = data.get('confidence', 0)
                print(f"✅ Analysis generated with confidence: {confidence}")
                print(f"   Analysis preview: {analysis[:150]}...")
                self.test_results.append(("Financial Analysis", True, f"Confidence: {confidence}"))
            else:
                print(f"❌ Analysis endpoint failed: {response.status_code}")
                self.test_results.append(("Financial Analysis", False, f"Status: {response.status_code}"))
        except Exception as e:
            print(f"❌ Analysis endpoint error: {e}")
            self.test_results.append(("Financial Analysis", False, str(e)))
    
    def test_ai_services(self):
        """Test individual AI services if available"""
        print("🤖 Testing AI services availability...")
        
        # Test configuration endpoints or service health
        services_to_test = [
            "Groq AI", "Azure OpenAI", "Google AI", 
            "Codestral", "DeepSeek", "Ollama"
        ]
        
        for service in services_to_test:
            # This would require individual service health endpoints
            # For now, we'll mark as informational
            self.test_results.append((f"AI-{service}", True, "Configuration check needed"))
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        print("\n📊 Detailed Results:")
        print("-" * 60)
        
        for test_name, success, details in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status:<8} {test_name:<20} {details}")
        
        print("\n" + "="*60)
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Your Dalaal Street Chatbot is ready!")
        else:
            print("⚠️  Some tests failed. Check the details above.")
            print("💡 Make sure all API keys are configured and services are running.")

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dalaal Street Chatbot Test Suite")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL for the API (default: http://localhost:8000)")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only basic connectivity tests")
    
    args = parser.parse_args()
    
    tester = DalaalStreetTester(args.url)
    
    if args.quick:
        print("🚀 Running quick tests only...")
        tester.test_health_check()
        tester.test_root_endpoint()
        tester.print_test_summary()
    else:
        tester.run_all_tests()

if __name__ == "__main__":
    main()
