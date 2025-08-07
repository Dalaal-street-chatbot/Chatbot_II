#!/usr/bin/env python3
"""
NSE Symbol Recognition Trainer for Dalaal Street Chatbot
This script trains a TensorFlow model to recognize and comprehend all NSE symbols
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import re
import csv
from datetime import datetime
from typing import List, Dict, Any, Tuple

class NSESymbolTrainer:
    """Train TensorFlow model to recognize and understand NSE symbols"""
    
    def __init__(self):
        # Set paths
        self.base_path = os.path.join('ml_training')
        self.models_path = os.path.join(self.base_path, 'models', 'nse_symbols')
        self.data_path = os.path.join(self.base_path, 'data', 'nse_symbols')
        
        # Create directories
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        # Model parameters
        self.max_length = 256
        self.embedding_dim = 128
        self.vocab_size = 10000
        self.batch_size = 32
        self.epochs = 15
        
        # Initialize components
        self.tokenizer = None
        self.label_encoder = None
        self.symbol_recognition_model = None
        self.symbol_classification_model = None
        
        # Data storage
        self.nse_symbols = []
        self.nifty_indices = []
        self.symbol_data = {}
        
        print(f"NSE Symbol Trainer initialized")
        print(f"Models will be saved to: {self.models_path}")
        print(f"Data will be processed from: {self.data_path}")

    def extract_symbols_from_csv(self, csv_file_path: str) -> Dict[str, Any]:
        """Extract NSE symbols and indices from the provided CSV file"""
        
        symbols_data = {
            'indices': [],
            'stocks': [],
            'raw_data': []
        }
        
        try:
            print(f"Extracting symbols from {csv_file_path}...")
            
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            
            # Extract indices (lines that start with index names)
            for index, row in df.iterrows():
                # Check if first column contains index names (starts with 'Nifty' or 'NIFTY')
                first_col = str(row.iloc[0]) if len(row) > 0 else ""
                
                if any(term in first_col.upper() for term in ['NIFTY', 'SENSEX', 'BANK', 'IT', 'AUTO', 'PHARMA', 'METAL', 'ENERGY']):
                    index_info = {
                        'name': first_col.strip(),
                        'type': 'index',
                        'data': row.to_dict()
                    }
                    symbols_data['indices'].append(index_info)
                    
                # Extract individual stock symbols from the "Top Securities" section
                elif 'SYMBOL' in str(row.iloc[0]).upper():
                    # This indicates start of stock symbols section
                    continue
                    
            # Extract stock symbols from the top securities section
            # Look for patterns like stock symbols (typically 3-15 characters, mostly uppercase)
            stock_pattern = re.compile(r'^[A-Z0-9]{2,15}$')
            
            for index, row in df.iterrows():
                for col_val in row:
                    col_str = str(col_val).strip()
                    if stock_pattern.match(col_str) and len(col_str) >= 3:
                        if col_str not in ['SYMBOL', 'SERIES', 'EQ', 'NC', 'GS', 'TB']:
                            stock_info = {
                                'symbol': col_str,
                                'type': 'stock',
                                'data': row.to_dict()
                            }
                            symbols_data['stocks'].append(stock_info)
            
            # Remove duplicates
            seen_indices = set()
            unique_indices = []
            for idx in symbols_data['indices']:
                if idx['name'] not in seen_indices:
                    unique_indices.append(idx)
                    seen_indices.add(idx['name'])
            symbols_data['indices'] = unique_indices
            
            seen_stocks = set()
            unique_stocks = []
            for stock in symbols_data['stocks']:
                if stock['symbol'] not in seen_stocks:
                    unique_stocks.append(stock)
                    seen_stocks.add(stock['symbol'])
            symbols_data['stocks'] = unique_stocks
            
            print(f"Extracted {len(symbols_data['indices'])} indices")
            print(f"Extracted {len(symbols_data['stocks'])} stock symbols")
            
            return symbols_data
            
        except Exception as e:
            print(f"Error extracting symbols: {e}")
            return symbols_data

    def load_comprehensive_nse_symbols(self) -> List[str]:
        """Load comprehensive list of NSE symbols including the most traded ones"""
        
        # Popular NSE stocks that should definitely be recognized
        popular_nse_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 
            'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ASIANPAINT', 'MARUTI', 'AXISBANK',
            'LT', 'WIPRO', 'NESTLEIND', 'HCLTECH', 'BAJFINANCE', 'TITAN', 'ULTRACEMCO',
            'POWERGRID', 'NTPC', 'TECHM', 'TATAMOTORS', 'TATASTEEL', 'ONGC', 'COALINDIA',
            'INDUSINDBK', 'GRASIM', 'ADANIPORTS', 'JSWSTEEL', 'HINDALCO', 'BAJAJFINSV',
            'CIPLA', 'HEROMOTOCO', 'DRREDDY', 'SUNPHARMA', 'EICHERMOT', 'BRITANNIA',
            'BPCL', 'DIVISLAB', 'APOLLOHOSP', 'SHREECEM', 'IOC', 'ADANIENSOL',
            'BAJAJ-AUTO', 'HDFCLIFE', 'SBILIFE', 'GODREJCP', 'M&M', 'DABUR',
            'VEDL', 'GAIL', 'TATACONSUM', 'PIDILITIND', 'BANDHANBNK', 'MCDOWELL-N',
            'LICHSGFIN', 'HAVELLS', 'JUBLFOOD', 'PAGEIND', 'COLPAL', 'MARICO',
            'BIOCON', 'CADILAHC', 'ZEEL', 'SAIL', 'NMDC', 'NATIONALUM', 'HINDZINC',
            'CANBK', 'PNB', 'UNIONBANK', 'BANKBARODA', 'YESBANK', 'FEDERALBNK',
            'IDEA', 'RCOM', 'SUZLON', 'JETAIRWAYS', 'SPICEJET', 'INDIGO'
        ]
        
        # Banking sector
        banking_stocks = [
            'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK',
            'BANKBARODA', 'CANBK', 'PNB', 'UNIONBANK', 'YESBANK', 'FEDERALBNK',
            'BANDHANBNK', 'RBLBANK', 'SOUTHBANK', 'CENTRALBK', 'INDIANB',
            'MAHABANK', 'SYNDIBANK', 'ALLAHABAD', 'VIJAYABANK', 'DENABANK'
        ]
        
        # IT sector
        it_stocks = [
            'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'MINDTREE', 'MPHASIS',
            'LTI', 'OFSS', 'NIIT', 'PERSISTENT', 'ZENSAR', 'COFORGE', 'LTTS',
            'CYIENT', 'RPOWER', 'KPIT', 'SONATSOFTW', 'HEXAWARE', 'POLYCAB'
        ]
        
        # Pharmaceutical sector
        pharma_stocks = [
            'SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'BIOCON', 'CADILAHC',
            'LUPIN', 'TORNTPHARM', 'AUROPHARMA', 'GLENMARK', 'ALKEM', 'IPCALAB',
            'PFIZER', 'ABBOTT', 'GLAXO', 'NOVARTIS', 'ROCHE', 'MERCK'
        ]
        
        # Auto sector
        auto_stocks = [
            'MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT',
            'ASHOKLEY', 'BHARATFORG', 'EXIDEIND', 'MOTHERSUMI', 'BALKRISIND',
            'APOLLOTYRE', 'MRF', 'CEAT', 'JKTYRE', 'TVSMOTORS', 'FORCEMOT'
        ]
        
        # FMCG sector
        fmcg_stocks = [
            'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP', 'MARICO',
            'COLPAL', 'PIDILITIND', 'TATACONSUM', 'MCDOWELL-N', 'UBL', 'RADICO',
            'VBL', 'CCL', 'EMAMILTD', 'JYOTHYLAB', 'CHOLAFIN', 'GILLETTE'
        ]
        
        # Energy and Oil
        energy_stocks = [
            'RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HPCL', 'GAIL', 'OIL', 'PETRONET',
            'GSPL', 'IGL', 'MGL', 'ATGL', 'ADANIGAS', 'ADANITRANS', 'ADANIPOWER'
        ]
        
        # Metals and Mining
        metals_stocks = [
            'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'SAIL', 'NMDC',
            'NATIONALUM', 'HINDZINC', 'COALINDIA', 'MOIL', 'JINDALSTEL',
            'RATNAMANI', 'WELCORP', 'WELSPUNIND', 'KALYANKJIL', 'MANAPPURAM'
        ]
        
        # Combine all symbols
        all_symbols = set()
        for symbol_list in [popular_nse_stocks, banking_stocks, it_stocks, pharma_stocks, 
                           auto_stocks, fmcg_stocks, energy_stocks, metals_stocks]:
            all_symbols.update(symbol_list)
        
        return list(all_symbols)

    def generate_training_data(self, symbols_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Generate training data for symbol recognition"""
        
        training_texts = []
        training_labels = []
        
        # Get comprehensive symbol list
        comprehensive_symbols = self.load_comprehensive_nse_symbols()
        
        # Add symbols from CSV
        csv_symbols = [stock['symbol'] for stock in symbols_data['stocks']]
        csv_indices = [idx['name'] for idx in symbols_data['indices']]
        
        # Combine all symbols
        all_symbols = set(comprehensive_symbols + csv_symbols)
        all_indices = set(csv_indices)
        
        print(f"Generating training data for {len(all_symbols)} symbols and {len(all_indices)} indices...")
        
        # Generate symbol recognition patterns
        question_patterns = [
            "What is the price of {}?",
            "Tell me about {} stock",
            "How is {} performing?",
            "Show me {} stock details",
            "What's the latest on {}?",
            "Give me {} analysis",
            "Is {} a good investment?",
            "What's happening with {}?",
            "{} stock price today",
            "Should I buy {}?",
            "What's the target price for {}?",
            "{} quarterly results",
            "Latest news on {}",
            "{} technical analysis",
            "What's the PE ratio of {}?",
            "Show {} chart",
            "Compare {} with other stocks",
            "{} dividend yield",
            "What's the 52-week high of {}?",
            "Is {} overvalued?"
        ]
        
        index_patterns = [
            "What's the {} level?",
            "How is {} moving?",
            "Show me {} chart",
            "What's the {} trend?",
            "{} performance today",
            "Where is {} heading?",
            "What's driving {} movement?",
            "{} technical levels",
            "Support and resistance for {}",
            "What's the outlook for {}?"
        ]
        
        # Generate training data for stocks
        for symbol in all_symbols:
            for pattern in question_patterns:
                text = pattern.format(symbol)
                training_texts.append(text)
                training_labels.append(f"STOCK:{symbol}")
                
                # Add variations with different cases
                training_texts.append(pattern.format(symbol.lower()))
                training_labels.append(f"STOCK:{symbol}")
                
                training_texts.append(pattern.format(symbol.title()))
                training_labels.append(f"STOCK:{symbol}")
        
        # Generate training data for indices
        for index in all_indices:
            for pattern in index_patterns:
                text = pattern.format(index)
                training_texts.append(text)
                training_labels.append(f"INDEX:{index}")
                
                # Add shortened versions
                short_index = index.replace('Nifty ', '').replace('NIFTY ', '')
                if short_index != index:
                    training_texts.append(pattern.format(short_index))
                    training_labels.append(f"INDEX:{index}")
        
        # Add general market queries
        general_patterns = [
            "How is the market today?",
            "What's the market sentiment?",
            "Show me market overview",
            "What are the top gainers?",
            "What are the top losers?",
            "Market news today",
            "What's happening in the markets?",
            "How are the indices performing?",
            "Market analysis",
            "Investment recommendations"
        ]
        
        for pattern in general_patterns:
            training_texts.append(pattern)
            training_labels.append("GENERAL:MARKET")
        
        print(f"Generated {len(training_texts)} training samples")
        return training_texts, training_labels

    def preprocess_data(self, texts: List[str], labels: List[str]):
        """Preprocess the training data"""
        
        print("Preprocessing training data...")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.X = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(labels)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")

    def build_symbol_recognition_model(self):
        """Build the TensorFlow model for symbol recognition"""
        
        print("Building symbol recognition model...")
        
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.GlobalAveragePooling1D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.symbol_recognition_model = model
        
        print("Model architecture:")
        model.summary()

    def build_advanced_model(self):
        """Build an advanced LSTM-based model for better symbol understanding"""
        
        print("Building advanced LSTM model...")
        
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.symbol_classification_model = model
        
        print("Advanced model architecture:")
        model.summary()

    def train_models(self):
        """Train both models"""
        
        print("Training symbol recognition model...")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3, restore_best_weights=True
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            os.path.join(self.models_path, 'best_symbol_model.h5'),
            monitor='val_accuracy', save_best_only=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001
        )
        
        # Train simple model
        history1 = self.symbol_recognition_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        print("\nTraining advanced LSTM model...")
        
        # Train advanced model
        history2 = self.symbol_classification_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history1, history2

    def save_models_and_data(self):
        """Save trained models and preprocessing data"""
        
        print("Saving models and preprocessing data...")
        
        # Save models
        self.symbol_recognition_model.save(
            os.path.join(self.models_path, 'nse_symbol_recognition_model.h5')
        )
        
        self.symbol_classification_model.save(
            os.path.join(self.models_path, 'nse_symbol_lstm_model.h5')
        )
        
        # Save tokenizer
        with open(os.path.join(self.models_path, 'tokenizer.pickle'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save label encoder
        with open(os.path.join(self.models_path, 'label_encoder.pickle'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save model config
        config = {
            'max_length': self.max_length,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'created_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.models_path, 'model_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models saved to {self.models_path}")

    def test_model_predictions(self):
        """Test the trained model with sample predictions"""
        
        print("\nTesting model predictions...")
        
        test_queries = [
            "What is the price of RELIANCE?",
            "Tell me about TCS stock",
            "How is Nifty 50 performing?",
            "Show me HDFC bank details",
            "What's the Nifty Bank level?",
            "INFY technical analysis",
            "Market sentiment today"
        ]
        
        for query in test_queries:
            # Preprocess query
            sequence = self.tokenizer.texts_to_sequences([query])
            padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
            
            # Predict with both models
            pred1 = self.symbol_recognition_model.predict(padded, verbose=0)
            pred2 = self.symbol_classification_model.predict(padded, verbose=0)
            
            # Get predictions
            class1 = self.label_encoder.inverse_transform([np.argmax(pred1)])[0]
            class2 = self.label_encoder.inverse_transform([np.argmax(pred2)])[0]
            
            conf1 = np.max(pred1)
            conf2 = np.max(pred2)
            
            print(f"Query: '{query}'")
            print(f"  Simple Model: {class1} (confidence: {conf1:.3f})")
            print(f"  LSTM Model: {class2} (confidence: {conf2:.3f})")
            print()

    def create_integration_guide(self):
        """Create integration guide for the chatbot"""
        
        integration_code = '''
# NSE Symbol Recognition Integration Guide

## Loading the Trained Models

```python
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class NSESymbolRecognizer:
    def __init__(self, models_path):
        # Load models
        self.simple_model = tf.keras.models.load_model(
            f"{models_path}/nse_symbol_recognition_model.h5"
        )
        self.lstm_model = tf.keras.models.load_model(
            f"{models_path}/nse_symbol_lstm_model.h5"
        )
        
        # Load tokenizer
        with open(f"{models_path}/tokenizer.pickle", 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Load label encoder
        with open(f"{models_path}/label_encoder.pickle", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load config
        with open(f"{models_path}/model_config.json", 'r') as f:
            self.config = json.load(f)
        
        self.max_length = self.config['max_length']
    
    def recognize_symbol(self, query: str, use_lstm: bool = True):
        """Recognize NSE symbol from user query"""
        
        # Preprocess query
        sequence = self.tokenizer.texts_to_sequences([query])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # Choose model
        model = self.lstm_model if use_lstm else self.simple_model
        
        # Predict
        prediction = model.predict(padded, verbose=0)
        predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)
        
        # Parse prediction
        if ':' in predicted_class:
            symbol_type, symbol = predicted_class.split(':', 1)
            return {
                'type': symbol_type.lower(),
                'symbol': symbol,
                'confidence': float(confidence),
                'query': query
            }
        else:
            return {
                'type': 'general',
                'symbol': predicted_class,
                'confidence': float(confidence),
                'query': query
            }

# Usage example:
recognizer = NSESymbolRecognizer('ml_training/models/nse_symbols')
result = recognizer.recognize_symbol("What is RELIANCE stock price?")
print(result)
# Output: {'type': 'stock', 'symbol': 'RELIANCE', 'confidence': 0.95, 'query': '...'}
```

## Integration with Chatbot

```python
# In your main chatbot service
async def process_user_query(self, user_message: str):
    # First, recognize any symbols
    symbol_recognition = self.nse_recognizer.recognize_symbol(user_message)
    
    if symbol_recognition['confidence'] > 0.7:
        if symbol_recognition['type'] == 'stock':
            # Handle stock query
            symbol = symbol_recognition['symbol']
            return await self.handle_stock_query(symbol, user_message)
        
        elif symbol_recognition['type'] == 'index':
            # Handle index query
            index = symbol_recognition['symbol']
            return await self.handle_index_query(index, user_message)
    
    # Fallback to general processing
    return await self.handle_general_query(user_message)
```
'''
        
        with open(os.path.join(self.models_path, 'integration_guide.md'), 'w') as f:
            f.write(integration_code)
        
        print(f"Integration guide saved to {self.models_path}/integration_guide.md")

    def run_complete_training_pipeline(self, csv_file_path: str):
        """Run the complete training pipeline"""
        
        print("🚀 Starting NSE Symbol Training Pipeline")
        print("="*50)
        
        # Step 1: Extract symbols from CSV
        print("\n📊 Step 1: Extracting symbols from CSV...")
        symbols_data = self.extract_symbols_from_csv(csv_file_path)
        
        if not symbols_data['stocks'] and not symbols_data['indices']:
            print("❌ No symbols found in CSV file. Using default symbol list.")
            symbols_data = {'stocks': [], 'indices': []}
        
        # Step 2: Generate training data
        print("\n🔧 Step 2: Generating training data...")
        texts, labels = self.generate_training_data(symbols_data)
        
        # Step 3: Preprocess data
        print("\n🛠️ Step 3: Preprocessing data...")
        self.preprocess_data(texts, labels)
        
        # Step 4: Build models
        print("\n🏗️ Step 4: Building models...")
        self.build_symbol_recognition_model()
        self.build_advanced_model()
        
        # Step 5: Train models
        print("\n🎯 Step 5: Training models...")
        try:
            history1, history2 = self.train_models()
            print("✅ Model training completed successfully!")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
        
        # Step 6: Save everything
        print("\n💾 Step 6: Saving models and data...")
        self.save_models_and_data()
        
        # Step 7: Test predictions
        print("\n🧪 Step 7: Testing predictions...")
        self.test_model_predictions()
        
        # Step 8: Create integration guide
        print("\n📚 Step 8: Creating integration guide...")
        self.create_integration_guide()
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"📁 Models saved to: {self.models_path}")
        print(f"📖 Integration guide: {self.models_path}/integration_guide.md")
        
        return True


def main():
    """Main function to run the training"""
    
    # Initialize trainer
    trainer = NSESymbolTrainer()
    
    # Check if CSV file exists
    csv_file_path = "/home/azureuser/Downloads/MA060825.csv"
    
    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        print("Please provide the correct path to your NSE data CSV file.")
        return
    
    print(f"📁 Using CSV file: {csv_file_path}")
    
    # Run training pipeline
    success = trainer.run_complete_training_pipeline(csv_file_path)
    
    if success:
        print("\n✅ NSE Symbol Recognition training completed successfully!")
        print("🤖 Your chatbot can now recognize and understand all NSE symbols!")
    else:
        print("\n❌ Training failed. Please check the errors above.")


if __name__ == "__main__":
    main()
