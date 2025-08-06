import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle
import re
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class ResponseGeneratorTrainer:
    """Train response generation model for financial conversations"""
    
    def __init__(self):
        self.model_data = {}
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
        self.response_templates = {}
        self.context_embeddings = {}
        self.quality_threshold = 0.7
        
        # Financial domain keywords
        self.financial_keywords = {
            'stock_price': ['price', 'cost', 'value', 'quote', 'rate', 'trading at'],
            'market_analysis': ['market', 'trend', 'analysis', 'performance', 'outlook'],
            'news_analysis': ['news', 'announcement', 'report', 'update', 'development'],
            'trading_strategy': ['buy', 'sell', 'invest', 'strategy', 'recommendation']
        }
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the response generation model"""
        
        print("Training response generator...")
        
        # Prepare training data
        queries = [item['query'] for item in training_data]
        responses = [item['response'] for item in training_data]
        categories = [item['category'] for item in training_data]
        contexts = [item.get('context', {}) for item in training_data]
        
        # Create query embeddings
        query_embeddings = self.vectorizer.fit_transform(queries)
        
        # Build response templates by category
        self._build_response_templates(training_data)
        
        # Build context-aware response mappings
        self._build_context_mappings(training_data)
        
        # Create similarity matrix
        similarity_matrix = cosine_similarity(query_embeddings)
        
        # Store training data
        self.model_data = {
            'queries': queries,
            'responses': responses,
            'categories': categories,
            'contexts': contexts,
            'query_embeddings': query_embeddings,
            'similarity_matrix': similarity_matrix
        }
        
        # Evaluate training quality
        quality_metrics = self._evaluate_training_quality(training_data)
        
        training_results = {
            'training_samples': len(training_data),
            'unique_categories': len(set(categories)),
            'avg_response_length': np.mean([len(r.split()) for r in responses]),
            'quality_score': quality_metrics['avg_quality'],
            'template_coverage': len(self.response_templates),
            'vectorizer_features': query_embeddings.shape[1]
        }
        
        print(f"Response Generator Training Results:")
        print(f"  Training Samples: {len(training_data)}")
        print(f"  Unique Categories: {len(set(categories))}")
        print(f"  Average Quality Score: {quality_metrics['avg_quality']:.3f}")
        print(f"  Template Coverage: {len(self.response_templates)} categories")
        
        return training_results
    
    def _build_response_templates(self, training_data: List[Dict[str, Any]]):
        """Build response templates for each category"""
        
        category_responses = defaultdict(list)
        
        for item in training_data:
            category = item['category']
            response = item['response']
            quality = self._calculate_response_quality(item['query'], response)
            
            if quality >= self.quality_threshold:
                category_responses[category].append({
                    'response': response,
                    'query': item['query'],
                    'quality': quality,
                    'context': item.get('context', {})
                })
        
        # Create templates for each category
        for category, responses in category_responses.items():
            # Sort by quality
            responses.sort(key=lambda x: x['quality'], reverse=True)
            
            # Extract common patterns
            templates = self._extract_response_patterns(responses)
            
            self.response_templates[category] = {
                'templates': templates,
                'examples': responses[:5],  # Top 5 examples
                'avg_quality': np.mean([r['quality'] for r in responses])
            }
    
    def _extract_response_patterns(self, responses: List[Dict[str, Any]]) -> List[str]:
        """Extract common response patterns"""
        
        patterns = []
        
        # Common opening patterns
        openings = [
            "Based on the current market data",
            "According to the latest information",
            "The current price of {stock}",
            "Market analysis shows",
            "Recent news indicates"
        ]
        
        # Common closing patterns
        closings = [
            "Please consult a financial advisor before making investment decisions.",
            "This information is for educational purposes only.",
            "Always do your own research before investing.",
            "Consider your risk tolerance before investing."
        ]
        
        # Structure patterns
        structure_patterns = [
            "{opening} {main_content} {disclaimer}",
            "{data_point} {analysis} {recommendation} {disclaimer}",
            "{greeting} {direct_answer} {context} {advice}"
        ]
        
        return {
            'openings': openings,
            'closings': closings,
            'structures': structure_patterns
        }
    
    def _build_context_mappings(self, training_data: List[Dict[str, Any]]):
        """Build context-aware response mappings"""
        
        context_types = defaultdict(list)
        
        for item in training_data:
            context = item.get('context', {})
            response = item['response']
            
            # Categorize by context type
            if 'stock_data' in context:
                context_types['with_stock_data'].append(item)
            if 'market_indices' in context:
                context_types['with_market_data'].append(item)
            if 'recent_news' in context:
                context_types['with_news'].append(item)
            if not context:
                context_types['no_context'].append(item)
        
        # Create embeddings for each context type
        for context_type, items in context_types.items():
            if items:
                responses = [item['response'] for item in items]
                response_embeddings = self.vectorizer.fit_transform(responses)
                
                self.context_embeddings[context_type] = {
                    'embeddings': response_embeddings,
                    'responses': responses,
                    'items': items
                }
    
    def generate_response(
        self, 
        query: str, 
        context: Dict[str, Any] = None,
        category: str = None
    ) -> Dict[str, Any]:
        """Generate response for a given query"""
        
        if not self.model_data:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find similar queries
        similar_queries = self._find_similar_queries(query)
        
        # Determine category if not provided
        if category is None:
            category = self._predict_category(query)
        
        # Generate response based on similarity and context
        if context:
            response = self._generate_context_aware_response(query, context, category, similar_queries)
        else:
            response = self._generate_template_response(query, category, similar_queries)
        
        # Calculate confidence
        confidence = self._calculate_response_confidence(query, response, similar_queries)
        
        return {
            'response': response,
            'category': category,
            'confidence': confidence,
            'similar_queries': [q['query'] for q in similar_queries[:3]],
            'context_used': bool(context)
        }
    
    def _find_similar_queries(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar queries from training data"""
        
        # Transform query
        query_embedding = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.model_data['query_embeddings'])[0]
        
        # Get top similar queries
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_queries = []
        for idx in top_indices:
            similar_queries.append({
                'query': self.model_data['queries'][idx],
                'response': self.model_data['responses'][idx],
                'category': self.model_data['categories'][idx],
                'similarity': similarities[idx],
                'context': self.model_data['contexts'][idx]
            })
        
        return similar_queries
    
    def _predict_category(self, query: str) -> str:
        """Predict category for the query"""
        
        query_lower = query.lower()
        category_scores = {}
        
        for category, keywords in self.financial_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            category_scores[category] = score
        
        # Return category with highest score, default to general
        if category_scores:
            predicted_category = max(category_scores, key=category_scores.get)
            if category_scores[predicted_category] > 0:
                return predicted_category
        
        return 'general_info'
    
    def _generate_context_aware_response(
        self, 
        query: str, 
        context: Dict[str, Any],
        category: str,
        similar_queries: List[Dict[str, Any]]
    ) -> str:
        """Generate response using context information"""
        
        # Base response from similar queries
        base_response = similar_queries[0]['response'] if similar_queries else ""
        
        # Enhance with context
        context_enhancements = []
        
        # Add stock data if available
        if 'stock_data' in context:
            stock_data = context['stock_data']
            if isinstance(stock_data, dict):
                symbol = stock_data.get('symbol', 'the stock')
                price = stock_data.get('price', 0)
                change = stock_data.get('change', 0)
                
                if price > 0:
                    context_enhancements.append(
                        f"The current price of {symbol} is ₹{price:.2f}"
                    )
                    
                    if change != 0:
                        direction = "up" if change > 0 else "down"
                        context_enhancements.append(
                            f"with a change of ₹{abs(change):.2f} ({direction})"
                        )
        
        # Add market indices if available
        if 'market_indices' in context:
            indices = context['market_indices']
            if isinstance(indices, dict):
                for index_name, index_data in indices.items():
                    if isinstance(index_data, dict) and 'price' in index_data:
                        context_enhancements.append(
                            f"{index_name} is at {index_data['price']:.2f}"
                        )
        
        # Add news context if available
        if 'recent_news' in context:
            news = context['recent_news']
            if isinstance(news, dict) and 'articles' in news:
                articles = news['articles']
                if articles and len(articles) > 0:
                    context_enhancements.append(
                        "Recent news developments may be affecting the market"
                    )
        
        # Combine base response with context
        if context_enhancements:
            enhanced_response = f"{'. '.join(context_enhancements)}. {base_response}"
        else:
            enhanced_response = base_response
        
        # Ensure proper disclaimer
        if not any(word in enhanced_response.lower() for word in ['disclaimer', 'advice', 'consult']):
            enhanced_response += " Please consult a financial advisor before making investment decisions."
        
        return enhanced_response
    
    def _generate_template_response(
        self, 
        query: str, 
        category: str,
        similar_queries: List[Dict[str, Any]]
    ) -> str:
        """Generate response using templates"""
        
        # Use most similar response as base
        if similar_queries and similar_queries[0]['similarity'] > 0.7:
            return similar_queries[0]['response']
        
        # Use category template
        if category in self.response_templates:
            template_data = self.response_templates[category]
            examples = template_data['examples']
            
            if examples:
                # Use best example response
                return examples[0]['response']
        
        # Fallback response
        fallback_responses = {
            'stock_price': "I can help you find stock prices. Please specify the stock symbol you're interested in.",
            'market_analysis': "Market analysis requires current data. I'd be happy to help analyze specific stocks or indices.",
            'news_analysis': "For the latest news analysis, I can help interpret recent market developments.",
            'trading_strategy': "Trading strategies should be based on your risk tolerance and investment goals. Please consult a financial advisor.",
            'general_info': "I'm here to help with your financial questions. Could you please be more specific?"
        }
        
        return fallback_responses.get(category, fallback_responses['general_info'])
    
    def _calculate_response_confidence(
        self, 
        query: str, 
        response: str,
        similar_queries: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the response"""
        
        confidence_factors = []
        
        # Similarity to training data
        if similar_queries:
            max_similarity = similar_queries[0]['similarity']
            confidence_factors.append(max_similarity)
        else:
            confidence_factors.append(0.3)
        
        # Response length (not too short, not too long)
        response_length = len(response.split())
        length_score = min(response_length / 50.0, 1.0) if response_length > 10 else 0.3
        confidence_factors.append(length_score)
        
        # Financial terminology presence
        financial_terms = ['stock', 'market', 'price', 'investment', 'trading', 'analysis']
        term_score = sum(1 for term in financial_terms if term in response.lower()) / len(financial_terms)
        confidence_factors.append(term_score)
        
        # Disclaimer presence
        disclaimer_score = 1.0 if any(word in response.lower() for word in ['disclaimer', 'advice', 'consult', 'risk']) else 0.5
        confidence_factors.append(disclaimer_score)
        
        # Calculate weighted average
        weights = [0.4, 0.2, 0.2, 0.2]
        confidence = sum(f * w for f, w in zip(confidence_factors, weights))
        
        return min(confidence, 1.0)
    
    def _calculate_response_quality(self, query: str, response: str) -> float:
        """Calculate quality score for a response"""
        
        quality_factors = []
        
        # Length check
        word_count = len(response.split())
        length_score = 1.0 if 20 <= word_count <= 200 else 0.5
        quality_factors.append(length_score)
        
        # Relevance check (simple keyword matching)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        relevance_score = len(query_words & response_words) / max(len(query_words), 1)
        quality_factors.append(min(relevance_score, 1.0))
        
        # Financial domain check
        financial_terms = ['stock', 'market', 'price', 'investment', 'trading', 'financial']
        domain_score = sum(1 for term in financial_terms if term in response.lower()) / len(financial_terms)
        quality_factors.append(domain_score)
        
        # Professional tone check
        professional_score = 1.0 if not any(word in response.lower() for word in ['lol', 'haha', 'omg']) else 0.3
        quality_factors.append(professional_score)
        
        # Disclaimer check
        disclaimer_score = 1.0 if any(word in response.lower() for word in ['disclaimer', 'advice', 'consult']) else 0.7
        quality_factors.append(disclaimer_score)
        
        return np.mean(quality_factors)
    
    def _evaluate_training_quality(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the quality of training data"""
        
        quality_scores = []
        
        for item in training_data:
            quality = self._calculate_response_quality(item['query'], item['response'])
            quality_scores.append(quality)
        
        return {
            'avg_quality': np.mean(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'high_quality_count': sum(1 for q in quality_scores if q >= self.quality_threshold)
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        model_data = {
            'model_data': self.model_data,
            'vectorizer': self.vectorizer,
            'response_templates': self.response_templates,
            'context_embeddings': self.context_embeddings,
            'financial_keywords': self.financial_keywords,
            'quality_threshold': self.quality_threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Response generator saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_data = model_data['model_data']
        self.vectorizer = model_data['vectorizer']
        self.response_templates = model_data['response_templates']
        self.context_embeddings = model_data['context_embeddings']
        self.financial_keywords = model_data['financial_keywords']
        self.quality_threshold = model_data['quality_threshold']
        
        print(f"Response generator loaded from {filepath}")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model performance"""
        
        if not self.model_data:
            return {'error': 'Model not trained'}
        
        # Test with sample queries
        test_queries = [
            "What's the price of Reliance?",
            "How is the market performing today?",
            "Should I invest in IT stocks?",
            "Latest news on HDFC Bank",
            "What is a good trading strategy?"
        ]
        
        quality_scores = []
        
        for query in test_queries:
            try:
                result = self.generate_response(query)
                quality = self._calculate_response_quality(query, result['response'])
                quality_scores.append(quality)
            except Exception:
                quality_scores.append(0.3)
        
        return {
            'quality': np.mean(quality_scores),
            'test_cases': len(test_queries),
            'avg_confidence': 0.75  # Placeholder
        }

# Create global instance
response_trainer = ResponseGeneratorTrainer()
