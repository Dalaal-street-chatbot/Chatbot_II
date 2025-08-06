import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.groq_service import groq_service
from config.settings import config

class GroqTrainingDataCollector:
    """Collect and prepare training data specifically for Groq AI fine-tuning"""
    
    def __init__(self):
        self.groq_service = groq_service
        self.training_conversations = []
        self.financial_patterns = []
        
    def collect_financial_conversation_data(self) -> List[Dict[str, Any]]:
        """Collect conversation data for financial chatbot training"""
        
        # Financial conversation templates
        conversation_templates = [
            # Stock price queries
            {
                "category": "stock_price",
                "user_patterns": [
                    "What's the price of {stock}?",
                    "Tell me the current price of {stock}",
                    "How much is {stock} trading at?",
                    "{stock} price today",
                    "What is {stock} share price?"
                ],
                "context_patterns": [
                    "market_data", "technical_analysis", "volume_analysis"
                ]
            },
            
            # Market analysis queries
            {
                "category": "market_analysis",
                "user_patterns": [
                    "How is the market performing today?",
                    "What's the market sentiment?",
                    "Should I buy {stock} now?",
                    "Is this a good time to invest in {sector}?",
                    "Market outlook for next week"
                ],
                "context_patterns": [
                    "market_indices", "news_sentiment", "technical_indicators"
                ]
            },
            
            # News and sentiment queries
            {
                "category": "news_analysis",
                "user_patterns": [
                    "What's the latest news on {stock}?",
                    "Any recent developments in {sector}?",
                    "Why is {stock} moving up/down?",
                    "Market news today",
                    "What's affecting {stock} price?"
                ],
                "context_patterns": [
                    "recent_news", "company_announcements", "sector_news"
                ]
            },
            
            # Trading strategy queries
            {
                "category": "trading_strategy",
                "user_patterns": [
                    "What's a good trading strategy for {stock}?",
                    "How to trade {stock} options?",
                    "When should I exit my {stock} position?",
                    "Risk management for {stock}",
                    "Stop loss strategy for {stock}"
                ],
                "context_patterns": [
                    "technical_analysis", "risk_metrics", "position_sizing"
                ]
            }
        ]
        
        # Popular Indian stocks and sectors
        stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
            "ICICIBANK", "KOTAKBANK", "SBIN", "BHARTIARTL", "ASIANPAINT",
            "MARUTI", "AXISBANK", "LT", "WIPRO", "NESTLEIND"
        ]
        
        sectors = [
            "IT", "Banking", "Pharma", "Auto", "FMCG", 
            "Energy", "Metals", "Realty", "Telecom"
        ]
        
        training_data = []
        
        for template in conversation_templates:
            for pattern in template["user_patterns"]:
                # Generate variations with different stocks/sectors
                entities = stocks if "{stock}" in pattern else sectors if "{sector}" in pattern else [""]
                
                for entity in entities[:5]:  # Limit to avoid too much data
                    user_query = pattern.format(stock=entity, sector=entity)
                    
                    # Create training example
                    training_example = {
                        "user_query": user_query,
                        "category": template["category"],
                        "entity": entity,
                        "context_needed": template["context_patterns"],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    training_data.append(training_example)
        
        return training_data
    
    async def generate_groq_responses(
        self, 
        training_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate high-quality responses using Groq for training data"""
        
        enhanced_training_data = []
        
        for i, example in enumerate(training_data):
            try:
                # Create context for the query
                context = await self._create_mock_context(example)
                
                # Generate response using Groq
                system_prompt = f"""
                You are an expert Indian stock market assistant. 
                Respond to the following query with accurate, helpful information.
                Category: {example['category']}
                Entity: {example.get('entity', 'N/A')}
                
                Provide a professional, informative response that:
                1. Directly answers the question
                2. Includes relevant data points
                3. Provides context and analysis
                4. Includes appropriate disclaimers
                5. Maintains a helpful, professional tone
                """
                
                response = await self.groq_service.generate_response(
                    example["user_query"],
                    context=context,
                    system_prompt=system_prompt
                )
                
                # Enhanced training example
                enhanced_example = {
                    **example,
                    "assistant_response": response,
                    "context_used": context,
                    "quality_score": await self._assess_response_quality(
                        example["user_query"], 
                        response
                    )
                }
                
                enhanced_training_data.append(enhanced_example)
                
                # Progress tracking
                if i % 10 == 0:
                    print(f"Generated {i}/{len(training_data)} responses")
                
                # Rate limiting
                import time
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating response for example {i}: {e}")
                continue
        
        return enhanced_training_data
    
    async def _create_mock_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Create realistic mock context for training"""
        
        context = {}
        entity = example.get("entity", "")
        category = example["category"]
        
        # Mock stock data
        if category in ["stock_price", "market_analysis"] and entity:
            context["stock_data"] = {
                "symbol": entity,
                "price": np.random.uniform(100, 3000),
                "change": np.random.uniform(-50, 50),
                "change_percent": np.random.uniform(-5, 5),
                "volume": np.random.randint(100000, 10000000),
                "market_cap": np.random.randint(10000, 500000) * 1000000
            }
        
        # Mock market indices
        if category == "market_analysis":
            context["market_indices"] = {
                "NIFTY50": {
                    "price": np.random.uniform(18000, 25000),
                    "change": np.random.uniform(-200, 200)
                },
                "SENSEX": {
                    "price": np.random.uniform(60000, 85000),
                    "change": np.random.uniform(-500, 500)
                }
            }
        
        # Mock news data
        if category == "news_analysis":
            context["recent_news"] = {
                "articles": [
                    {
                        "title": f"{entity} reports strong quarterly results" if entity else "Market shows positive momentum",
                        "sentiment": np.random.choice(["positive", "negative", "neutral"]),
                        "source": "Economic Times"
                    }
                ]
            }
        
        # Mock technical indicators
        if category == "trading_strategy":
            context["technical_indicators"] = {
                "rsi": np.random.uniform(20, 80),
                "macd": np.random.uniform(-10, 10),
                "bollinger_position": np.random.uniform(-1, 1)
            }
        
        return context
    
    async def _assess_response_quality(
        self, 
        query: str, 
        response: str
    ) -> float:
        """Assess the quality of generated responses"""
        
        quality_criteria = [
            len(response) > 50,  # Sufficient length
            "disclaimer" in response.lower() or "advice" in response.lower(),  # Includes disclaimers
            any(word in response.lower() for word in ["stock", "market", "price", "trading"]),  # Financial relevance
            "?" not in response or response.count("?") < 3,  # Not too many questions back
            len(response.split()) > 20  # Detailed enough
        ]
        
        quality_score = sum(quality_criteria) / len(quality_criteria)
        return quality_score
    
    def prepare_groq_fine_tuning_data(
        self, 
        enhanced_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare data in format suitable for Groq fine-tuning"""
        
        fine_tuning_data = []
        
        for example in enhanced_data:
            # Only include high-quality responses
            if example.get("quality_score", 0) >= 0.7:
                
                # Format for Groq fine-tuning
                formatted_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are an expert Indian financial markets assistant specializing in {example['category']} queries."
                        },
                        {
                            "role": "user", 
                            "content": example["user_query"]
                        },
                        {
                            "role": "assistant",
                            "content": example["assistant_response"]
                        }
                    ],
                    "metadata": {
                        "category": example["category"],
                        "entity": example.get("entity", ""),
                        "quality_score": example["quality_score"],
                        "context_types": example["context_needed"]
                    }
                }
                
                fine_tuning_data.append(formatted_example)
        
        return fine_tuning_data
    
    def save_training_data(
        self, 
        data: List[Dict[str, Any]], 
        filename: str = "groq_training_data.jsonl"
    ):
        """Save training data in JSONL format for Groq"""
        
        filepath = f"/home/codespace/Dalaal-street-chatbot/ml_training/data/{filename}"
        
        with open(filepath, 'w') as f:
            for example in data:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved {len(data)} training examples to {filepath}")
    
    def create_evaluation_dataset(
        self, 
        training_data: List[Dict[str, Any]], 
        eval_split: float = 0.2
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into training and evaluation sets"""
        
        np.random.shuffle(training_data)
        split_index = int(len(training_data) * (1 - eval_split))
        
        train_data = training_data[:split_index]
        eval_data = training_data[split_index:]
        
        return train_data, eval_data
    
    async def collect_real_conversation_data(self) -> List[Dict[str, Any]]:
        """Collect real conversation patterns for analysis"""
        
        # This would collect real user interactions
        # For now, we'll create synthetic but realistic conversations
        
        realistic_conversations = []
        
        conversation_flows = [
            [
                "Hi, what's the current price of Reliance?",
                "Can you tell me why Reliance is moving up today?",
                "Should I buy more Reliance shares?",
                "What's your price target for Reliance?"
            ],
            [
                "How is the market performing today?",
                "Which sectors are doing well?",
                "Any specific stock recommendations in IT sector?",
                "What about TCS vs Infosys?"
            ],
            [
                "I want to start investing in stocks",
                "What are some beginner-friendly stocks?",
                "How much should I invest initially?",
                "What's the risk in HDFC Bank?"
            ]
        ]
        
        for conversation in conversation_flows:
            for i, message in enumerate(conversation):
                realistic_conversations.append({
                    "user_message": message,
                    "conversation_id": len(realistic_conversations) // len(conversation),
                    "message_index": i,
                    "timestamp": datetime.now().isoformat(),
                    "context": "multi_turn_conversation"
                })
        
        return realistic_conversations

# Create global instance
groq_collector = GroqTrainingDataCollector()
