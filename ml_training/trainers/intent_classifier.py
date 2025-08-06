import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import re
from typing import Dict, List, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class IntentClassifierTrainer:
    """Train intent classification model for financial queries"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.stemmer = PorterStemmer()
        self.financial_entities = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
            'ICICIBANK', 'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ASIANPAINT',
            'NIFTY', 'SENSEX', 'BANKNIFTY', 'NSE', 'BSE'
        ]
        
        # Intent categories
        self.intent_categories = [
            'stock_price',
            'market_analysis', 
            'news_analysis',
            'trading_strategy',
            'general_info',
            'portfolio_advice'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better classification"""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Replace financial entities with standardized tokens
        for entity in self.financial_entities:
            text = re.sub(f'\\b{entity.lower()}\\b', f'STOCK_ENTITY_{entity}', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract features from text data"""
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
        
        # Additional features
        additional_features = []
        for text in texts:
            features = self._extract_additional_features(text)
            additional_features.append(features)
        
        additional_features = np.array(additional_features)
        
        # Combine features
        if additional_features.shape[0] > 0:
            from scipy.sparse import hstack
            combined_features = hstack([tfidf_features, additional_features])
        else:
            combined_features = tfidf_features
        
        return combined_features
    
    def _extract_additional_features(self, text: str) -> List[float]:
        """Extract additional hand-crafted features"""
        
        text_lower = text.lower()
        
        features = [
            # Price-related keywords
            1.0 if any(word in text_lower for word in ['price', 'cost', 'value', 'rate']) else 0.0,
            
            # Question indicators
            1.0 if any(word in text_lower for word in ['what', 'how', 'when', 'why', 'which']) else 0.0,
            
            # Time indicators
            1.0 if any(word in text_lower for word in ['today', 'now', 'current', 'latest']) else 0.0,
            
            # Investment keywords
            1.0 if any(word in text_lower for word in ['buy', 'sell', 'invest', 'trade']) else 0.0,
            
            # Analysis keywords
            1.0 if any(word in text_lower for word in ['analysis', 'trend', 'forecast', 'prediction']) else 0.0,
            
            # News keywords
            1.0 if any(word in text_lower for word in ['news', 'announcement', 'report', 'update']) else 0.0,
            
            # Market keywords
            1.0 if any(word in text_lower for word in ['market', 'index', 'nifty', 'sensex']) else 0.0,
            
            # Text length (normalized)
            min(len(text) / 100.0, 1.0),
            
            # Number of words
            min(len(text.split()) / 20.0, 1.0),
            
            # Has stock entity
            1.0 if any(entity.lower() in text_lower for entity in self.financial_entities) else 0.0
        ]
        
        return features
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the intent classification model"""
        
        # Prepare training data
        texts = [item['text'] for item in training_data]
        intents = [item['intent'] for item in training_data]
        
        # Extract features
        X = self.extract_features(texts)
        y = np.array(intents)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train pipeline
        self.model = Pipeline([
            ('classifier', self.classifier)
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        training_results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'training_samples': len(training_data),
            'feature_count': X.shape[1]
        }
        
        print(f"Intent Classifier Training Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  Training Samples: {len(training_data)}")
        
        return training_results
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent for a given text"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess and extract features
        processed_text = self.preprocess_text(text)
        features = self.vectorizer.transform([processed_text])
        
        # Add additional features
        additional_features = np.array([self._extract_additional_features(text)])
        
        if additional_features.shape[0] > 0:
            from scipy.sparse import hstack
            features = hstack([features, additional_features])
        
        # Predict
        intent = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get confidence
        confidence = max(probabilities)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'probabilities': dict(zip(self.model.classes_, probabilities))
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract financial entities from text"""
        
        entities = []
        text_upper = text.upper()
        
        for entity in self.financial_entities:
            if entity in text_upper:
                entities.append(entity)
        
        # Extract potential stock symbols (3-4 uppercase letters)
        import re
        potential_symbols = re.findall(r'\\b[A-Z]{3,4}\\b', text)
        for symbol in potential_symbols:
            if symbol not in entities and symbol not in ['THE', 'AND', 'FOR']:
                entities.append(symbol)
        
        return list(set(entities))
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'financial_entities': self.financial_entities,
            'intent_categories': self.intent_categories
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Intent classifier saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.financial_entities = model_data['financial_entities']
        self.intent_categories = model_data['intent_categories']
        
        print(f"Intent classifier loaded from {filepath}")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model performance"""
        
        if self.model is None:
            return {'error': 'Model not trained'}
        
        # Create test cases
        test_cases = [
            ("What's the price of Reliance?", "stock_price"),
            ("How is the market performing today?", "market_analysis"),
            ("Latest news on TCS", "news_analysis"),
            ("Should I buy HDFC Bank?", "trading_strategy"),
            ("What is a mutual fund?", "general_info")
        ]
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for text, expected_intent in test_cases:
            prediction = self.predict(text)
            if prediction['intent'] == expected_intent:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        
        return {
            'accuracy': accuracy,
            'test_cases': total_predictions,
            'correct_predictions': correct_predictions
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        
        if self.model is None:
            return {}
        
        # Get feature names from vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Add additional feature names
        additional_feature_names = [
            'has_price_keywords', 'has_question_words', 'has_time_indicators',
            'has_investment_keywords', 'has_analysis_keywords', 'has_news_keywords',
            'has_market_keywords', 'text_length_normalized', 'word_count_normalized',
            'has_stock_entity'
        ]
        
        all_feature_names = list(feature_names) + additional_feature_names
        
        # Get importance scores
        importance_scores = self.model.named_steps['classifier'].feature_importances_
        
        # Create feature importance dictionary
        feature_importance = dict(zip(all_feature_names, importance_scores))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:20])  # Top 20 features

# Create global instance
intent_trainer = IntentClassifierTrainer()
