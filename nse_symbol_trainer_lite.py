#!/usr/bin/env python3
"""
NSE Symbol Recognition Trainer for Dalaal Street Chatbot (Lightweight Version)
This script trains the chatbot to recognize and comprehend all NSE symbols without TensorFlow
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pickle
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import Counter
import csv

class NSESymbolTrainerLite:
    """Lightweight NSE Symbol Recognition Trainer using traditional ML approaches"""
    
    def __init__(self):
        # Set paths
        self.base_path = os.path.join('ml_training')
        self.models_path = os.path.join(self.base_path, 'models', 'nse_symbols')
        self.data_path = os.path.join(self.base_path, 'data', 'nse_symbols')
        
        # Create directories
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        # Data storage
        self.nse_symbols = []
        self.nifty_indices = []
        self.symbol_data = {}
        self.symbol_patterns = {}
        
        print(f"NSE Symbol Trainer (Lite) initialized")
        print(f"Models will be saved to: {self.models_path}")
        print(f"Data will be processed from: {self.data_path}")

    def extract_symbols_from_csv(self, csv_file_path: str) -> Dict[str, Any]:
        """Extract NSE symbols and indices from the provided CSV file"""
        
        symbols_data = {
            'indices': [],
            'stocks': [],
            'price_data': {},
            'raw_data': []
        }
        
        try:
            print(f"Extracting symbols from {csv_file_path}...")
            
            # Read the CSV file
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Process each line
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma and clean data
                parts = [part.strip().strip('"').strip() for part in line.split(',')]
                symbols_data['raw_data'].append(parts)
                
                # Look for indices (lines containing Nifty, NIFTY, etc.)
                for i, part in enumerate(parts):
                    if any(term in part.upper() for term in ['NIFTY', 'SENSEX', 'BANK', 'IT', 'AUTO', 'PHARMA', 'METAL', 'ENERGY']):
                        if len(part) > 3 and part not in [p['name'] for p in symbols_data['indices']]:
                            symbols_data['indices'].append({
                                'name': part,
                                'type': 'index',
                                'line_number': line_num,
                                'data': parts
                            })
                
                # Look for stock symbols (typically 3-15 characters, mostly uppercase)
                for i, part in enumerate(parts):
                    if re.match(r'^[A-Z0-9]{3,15}$', part) and part not in ['SYMBOL', 'SERIES', 'EQ', 'NC', 'GS', 'TB']:
                        if part not in [s['symbol'] for s in symbols_data['stocks']]:
                            # Try to extract price data if available
                            price_info = {}
                            if i + 1 < len(parts) and parts[i + 1] == 'EQ' and i + 2 < len(parts):
                                try:
                                    price_info['close'] = float(parts[i + 2].replace(',', ''))
                                except:
                                    pass
                            
                            symbols_data['stocks'].append({
                                'symbol': part,
                                'type': 'stock',
                                'line_number': line_num,
                                'price_info': price_info,
                                'data': parts
                            })
            
            print(f"Extracted {len(symbols_data['indices'])} indices")
            print(f"Extracted {len(symbols_data['stocks'])} stock symbols")
            
            return symbols_data
            
        except Exception as e:
            print(f"Error extracting symbols: {e}")
            return symbols_data

    def load_comprehensive_nse_symbols(self) -> List[Dict[str, Any]]:
        """Load comprehensive list of NSE symbols with metadata"""
        
        # Most traded NSE stocks with sector information
        comprehensive_symbols = [
            # Banking
            {'symbol': 'HDFCBANK', 'sector': 'Banking', 'market_cap': 'Large'},
            {'symbol': 'ICICIBANK', 'sector': 'Banking', 'market_cap': 'Large'},
            {'symbol': 'SBIN', 'sector': 'Banking', 'market_cap': 'Large'},
            {'symbol': 'KOTAKBANK', 'sector': 'Banking', 'market_cap': 'Large'},
            {'symbol': 'AXISBANK', 'sector': 'Banking', 'market_cap': 'Large'},
            {'symbol': 'INDUSINDBK', 'sector': 'Banking', 'market_cap': 'Large'},
            {'symbol': 'BANKBARODA', 'sector': 'Banking', 'market_cap': 'Large'},
            {'symbol': 'PNB', 'sector': 'Banking', 'market_cap': 'Large'},
            {'symbol': 'CANBK', 'sector': 'Banking', 'market_cap': 'Mid'},
            {'symbol': 'UNIONBANK', 'sector': 'Banking', 'market_cap': 'Mid'},
            
            # IT
            {'symbol': 'TCS', 'sector': 'IT', 'market_cap': 'Large'},
            {'symbol': 'INFY', 'sector': 'IT', 'market_cap': 'Large'},
            {'symbol': 'WIPRO', 'sector': 'IT', 'market_cap': 'Large'},
            {'symbol': 'HCLTECH', 'sector': 'IT', 'market_cap': 'Large'},
            {'symbol': 'TECHM', 'sector': 'IT', 'market_cap': 'Large'},
            {'symbol': 'MINDTREE', 'sector': 'IT', 'market_cap': 'Mid'},
            {'symbol': 'MPHASIS', 'sector': 'IT', 'market_cap': 'Mid'},
            {'symbol': 'LTI', 'sector': 'IT', 'market_cap': 'Mid'},
            
            # Oil & Gas
            {'symbol': 'RELIANCE', 'sector': 'Oil & Gas', 'market_cap': 'Large'},
            {'symbol': 'ONGC', 'sector': 'Oil & Gas', 'market_cap': 'Large'},
            {'symbol': 'IOC', 'sector': 'Oil & Gas', 'market_cap': 'Large'},
            {'symbol': 'BPCL', 'sector': 'Oil & Gas', 'market_cap': 'Large'},
            {'symbol': 'HPCL', 'sector': 'Oil & Gas', 'market_cap': 'Large'},
            {'symbol': 'GAIL', 'sector': 'Oil & Gas', 'market_cap': 'Large'},
            
            # FMCG
            {'symbol': 'HINDUNILVR', 'sector': 'FMCG', 'market_cap': 'Large'},
            {'symbol': 'NESTLEIND', 'sector': 'FMCG', 'market_cap': 'Large'},
            {'symbol': 'BRITANNIA', 'sector': 'FMCG', 'market_cap': 'Large'},
            {'symbol': 'DABUR', 'sector': 'FMCG', 'market_cap': 'Large'},
            {'symbol': 'GODREJCP', 'sector': 'FMCG', 'market_cap': 'Mid'},
            {'symbol': 'MARICO', 'sector': 'FMCG', 'market_cap': 'Mid'},
            {'symbol': 'COLPAL', 'sector': 'FMCG', 'market_cap': 'Mid'},
            
            # Auto
            {'symbol': 'MARUTI', 'sector': 'Auto', 'market_cap': 'Large'},
            {'symbol': 'TATAMOTORS', 'sector': 'Auto', 'market_cap': 'Large'},
            {'symbol': 'M&M', 'sector': 'Auto', 'market_cap': 'Large'},
            {'symbol': 'BAJAJ-AUTO', 'sector': 'Auto', 'market_cap': 'Large'},
            {'symbol': 'HEROMOTOCO', 'sector': 'Auto', 'market_cap': 'Large'},
            {'symbol': 'EICHERMOT', 'sector': 'Auto', 'market_cap': 'Large'},
            
            # Pharma
            {'symbol': 'SUNPHARMA', 'sector': 'Pharma', 'market_cap': 'Large'},
            {'symbol': 'CIPLA', 'sector': 'Pharma', 'market_cap': 'Large'},
            {'symbol': 'DRREDDY', 'sector': 'Pharma', 'market_cap': 'Large'},
            {'symbol': 'DIVISLAB', 'sector': 'Pharma', 'market_cap': 'Large'},
            {'symbol': 'BIOCON', 'sector': 'Pharma', 'market_cap': 'Mid'},
            {'symbol': 'LUPIN', 'sector': 'Pharma', 'market_cap': 'Mid'},
            
            # Metals
            {'symbol': 'TATASTEEL', 'sector': 'Metals', 'market_cap': 'Large'},
            {'symbol': 'JSWSTEEL', 'sector': 'Metals', 'market_cap': 'Large'},
            {'symbol': 'HINDALCO', 'sector': 'Metals', 'market_cap': 'Large'},
            {'symbol': 'VEDL', 'sector': 'Metals', 'market_cap': 'Large'},
            {'symbol': 'SAIL', 'sector': 'Metals', 'market_cap': 'Large'},
            {'symbol': 'NMDC', 'sector': 'Metals', 'market_cap': 'Large'},
            {'symbol': 'NATIONALUM', 'sector': 'Metals', 'market_cap': 'Mid'},
            {'symbol': 'HINDZINC', 'sector': 'Metals', 'market_cap': 'Large'},
            
            # Others
            {'symbol': 'LT', 'sector': 'Engineering', 'market_cap': 'Large'},
            {'symbol': 'ULTRACEMCO', 'sector': 'Cement', 'market_cap': 'Large'},
            {'symbol': 'ASIANPAINT', 'sector': 'Paints', 'market_cap': 'Large'},
            {'symbol': 'BHARTIARTL', 'sector': 'Telecom', 'market_cap': 'Large'},
            {'symbol': 'POWERGRID', 'sector': 'Power', 'market_cap': 'Large'},
            {'symbol': 'NTPC', 'sector': 'Power', 'market_cap': 'Large'},
            {'symbol': 'COALINDIA', 'sector': 'Coal', 'market_cap': 'Large'},
            {'symbol': 'TITAN', 'sector': 'Jewelry', 'market_cap': 'Large'},
            {'symbol': 'BAJFINANCE', 'sector': 'NBFC', 'market_cap': 'Large'},
            {'symbol': 'BAJAJFINSV', 'sector': 'NBFC', 'market_cap': 'Large'},
        ]
        
        return comprehensive_symbols

    def generate_symbol_recognition_data(self, symbols_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate symbol recognition patterns and data"""
        
        recognition_data = {
            'symbol_patterns': {},
            'query_templates': {},
            'symbol_metadata': {},
            'symbol_mapping': {}
        }
        
        # Get comprehensive symbol list
        comprehensive_symbols = self.load_comprehensive_nse_symbols()
        csv_symbols = symbols_data['stocks']
        csv_indices = symbols_data['indices']
        
        print(f"Generating recognition data for symbols...")
        
        # Process stock symbols
        all_symbols = {}
        
        # Add comprehensive symbols
        for symbol_info in comprehensive_symbols:
            symbol = symbol_info['symbol']
            all_symbols[symbol] = {
                'type': 'stock',
                'sector': symbol_info.get('sector', 'Unknown'),
                'market_cap': symbol_info.get('market_cap', 'Unknown'),
                'source': 'comprehensive'
            }
        
        # Add CSV symbols
        for symbol_info in csv_symbols:
            symbol = symbol_info['symbol']
            if symbol not in all_symbols:
                all_symbols[symbol] = {
                    'type': 'stock',
                    'sector': 'Unknown',
                    'market_cap': 'Unknown',
                    'source': 'csv'
                }
            
            # Add price data if available
            if symbol_info.get('price_info'):
                all_symbols[symbol]['price_data'] = symbol_info['price_info']
        
        # Process indices
        all_indices = {}
        for index_info in csv_indices:
            index_name = index_info['name']
            all_indices[index_name] = {
                'type': 'index',
                'full_name': index_name,
                'source': 'csv'
            }
        
        # Generate query patterns for stocks
        stock_query_patterns = [
            "{symbol} price", "{symbol} stock price", "What is {symbol} price?",
            "Tell me about {symbol}", "Show me {symbol} details", "{symbol} analysis",
            "How is {symbol} performing?", "{symbol} stock", "Buy {symbol}",
            "Should I invest in {symbol}?", "{symbol} target price", "{symbol} news",
            "{symbol} chart", "{symbol} technical analysis", "{symbol} fundamentals",
            "What's the latest on {symbol}?", "{symbol} quarterly results",
            "{symbol} dividend", "{symbol} PE ratio", "{symbol} market cap"
        ]
        
        # Generate query patterns for indices  
        index_query_patterns = [
            "{index} level", "{index} today", "How is {index}?",
            "Show me {index}", "{index} chart", "{index} movement",
            "What's {index} doing?", "{index} performance", "{index} analysis",
            "Where is {index} heading?", "{index} trend", "{index} support resistance"
        ]
        
        # Build recognition patterns
        for symbol, info in all_symbols.items():
            patterns = []
            for template in stock_query_patterns:
                patterns.append(template.format(symbol=symbol))
                patterns.append(template.format(symbol=symbol.lower()))
                patterns.append(template.format(symbol=symbol.title()))
            
            recognition_data['symbol_patterns'][symbol] = patterns
            recognition_data['symbol_metadata'][symbol] = info
        
        for index, info in all_indices.items():
            patterns = []
            for template in index_query_patterns:
                patterns.append(template.format(index=index))
                
                # Add short versions
                short_index = index.replace('Nifty ', '').replace('NIFTY ', '')
                if short_index != index:
                    patterns.append(template.format(index=short_index))
            
            recognition_data['symbol_patterns'][index] = patterns
            recognition_data['symbol_metadata'][index] = info
        
        # Create symbol mapping for quick lookup
        for symbol in all_symbols.keys():
            recognition_data['symbol_mapping'][symbol.upper()] = symbol
            recognition_data['symbol_mapping'][symbol.lower()] = symbol
        
        for index in all_indices.keys():
            recognition_data['symbol_mapping'][index.upper()] = index
            recognition_data['symbol_mapping'][index.lower()] = index
        
        print(f"Generated patterns for {len(all_symbols)} stocks and {len(all_indices)} indices")
        
        return recognition_data

    def create_simple_recognition_model(self, recognition_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple rule-based recognition model"""
        
        model_data = {
            'symbol_keywords': {},
            'sector_keywords': {},
            'intent_keywords': {},
            'symbol_variants': {},
            'recognition_rules': []
        }
        
        print("Creating simple recognition model...")
        
        # Extract symbol keywords
        for symbol, metadata in recognition_data['symbol_metadata'].items():
            keywords = [symbol, symbol.lower(), symbol.upper()]
            
            # Add sector-based keywords
            sector = metadata.get('sector', '')
            if sector:
                keywords.extend([f"{sector} stock", f"{sector} shares"])
            
            model_data['symbol_keywords'][symbol] = keywords
            
            # Group by sector
            if sector not in model_data['sector_keywords']:
                model_data['sector_keywords'][sector] = []
            model_data['sector_keywords'][sector].append(symbol)
        
        # Define intent keywords
        model_data['intent_keywords'] = {
            'price_query': ['price', 'cost', 'value', 'quote', 'rate'],
            'analysis_query': ['analysis', 'review', 'research', 'study', 'fundamental', 'technical'],
            'news_query': ['news', 'updates', 'latest', 'announcement', 'report'],
            'performance_query': ['performance', 'return', 'gain', 'loss', 'movement'],
            'investment_query': ['buy', 'sell', 'invest', 'trade', 'hold', 'recommendation'],
            'comparison_query': ['compare', 'vs', 'versus', 'against', 'better'],
            'general_query': ['what', 'how', 'when', 'why', 'tell', 'show', 'explain']
        }
        
        # Create recognition rules
        model_data['recognition_rules'] = [
            {
                'type': 'exact_match',
                'description': 'Direct symbol mention',
                'priority': 1
            },
            {
                'type': 'keyword_match',
                'description': 'Symbol with keywords',
                'priority': 2
            },
            {
                'type': 'sector_match',
                'description': 'Sector-based matching',
                'priority': 3
            },
            {
                'type': 'fuzzy_match',
                'description': 'Similar symbol names',
                'priority': 4
            }
        ]
        
        return model_data

    def save_recognition_system(self, symbols_data: Dict[str, Any], recognition_data: Dict[str, Any], model_data: Dict[str, Any]):
        """Save the complete recognition system"""
        
        print("Saving recognition system...")
        
        # Save symbol data
        with open(os.path.join(self.models_path, 'nse_symbols_data.json'), 'w') as f:
            json.dump(symbols_data, f, indent=2, default=str)
        
        # Save recognition data
        with open(os.path.join(self.models_path, 'recognition_data.json'), 'w') as f:
            json.dump(recognition_data, f, indent=2, default=str)
        
        # Save model data
        with open(os.path.join(self.models_path, 'model_data.json'), 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        # Create a combined lookup file for fast access
        lookup_data = {
            'symbols': list(recognition_data['symbol_metadata'].keys()),
            'symbol_mapping': recognition_data['symbol_mapping'],
            'sectors': list(set([meta.get('sector', 'Unknown') for meta in recognition_data['symbol_metadata'].values()])),
            'created_at': datetime.now().isoformat(),
            'total_symbols': len(recognition_data['symbol_metadata']),
            'model_version': '1.0'
        }
        
        with open(os.path.join(self.models_path, 'symbol_lookup.json'), 'w') as f:
            json.dump(lookup_data, f, indent=2)
        
        print(f"Recognition system saved to {self.models_path}")

    def test_recognition_system(self, recognition_data: Dict[str, Any]):
        """Test the recognition system with sample queries"""
        
        print("\nTesting recognition system...")
        
        test_queries = [
            "What is the price of RELIANCE?",
            "Tell me about TCS stock",
            "How is Nifty 50 performing?", 
            "Show me HDFC bank details",
            "What's the Nifty Bank level?",
            "INFY technical analysis",
            "reliance news",
            "tcs vs infosys",
            "banking sector stocks",
            "IT companies performance"
        ]
        
        def simple_symbol_detection(query: str) -> List[str]:
            """Simple symbol detection logic"""
            detected = []
            query_upper = query.upper()
            
            # Check for exact matches
            for symbol in recognition_data['symbol_mapping']:
                if symbol.upper() in query_upper:
                    mapped_symbol = recognition_data['symbol_mapping'][symbol]
                    if mapped_symbol not in detected:
                        detected.append(mapped_symbol)
            
            return detected
        
        for query in test_queries:
            detected_symbols = simple_symbol_detection(query)
            print(f"Query: '{query}'")
            print(f"  Detected symbols: {detected_symbols}")
            
            if detected_symbols:
                for symbol in detected_symbols:
                    metadata = recognition_data['symbol_metadata'].get(symbol, {})
                    print(f"    {symbol}: {metadata.get('type', 'Unknown')} - {metadata.get('sector', 'Unknown')}")
            print()

    def create_integration_guide(self):
        """Create integration guide for the chatbot"""
        
        integration_code = '''# NSE Symbol Recognition Integration Guide (Lightweight)

## Loading the Recognition System

```python
import json
import re
from typing import List, Dict, Any

class NSESymbolRecognizer:
    def __init__(self, models_path: str):
        # Load recognition data
        with open(f"{models_path}/recognition_data.json", 'r') as f:
            self.recognition_data = json.load(f)
        
        # Load model data
        with open(f"{models_path}/model_data.json", 'r') as f:
            self.model_data = json.load(f)
        
        # Load lookup data
        with open(f"{models_path}/symbol_lookup.json", 'r') as f:
            self.lookup_data = json.load(f)
        
        print(f"Loaded {self.lookup_data['total_symbols']} symbols")
    
    def detect_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Detect NSE symbols in user query"""
        detected = []
        query_upper = query.upper()
        
        # Rule 1: Exact symbol match
        for symbol in self.lookup_data['symbol_mapping']:
            if symbol.upper() in query_upper:
                mapped_symbol = self.lookup_data['symbol_mapping'][symbol]
                metadata = self.recognition_data['symbol_metadata'].get(mapped_symbol, {})
                
                if mapped_symbol not in [d['symbol'] for d in detected]:
                    detected.append({
                        'symbol': mapped_symbol,
                        'type': metadata.get('type', 'unknown'),
                        'sector': metadata.get('sector', 'Unknown'),
                        'confidence': 0.9,
                        'match_type': 'exact'
                    })
        
        # Rule 2: Keyword-based detection
        for intent, keywords in self.model_data['intent_keywords'].items():
            for keyword in keywords:
                if keyword.lower() in query.lower():
                    # This indicates the type of query
                    for result in detected:
                        result['intent'] = intent
                    break
        
        return detected
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a symbol"""
        return self.recognition_data['symbol_metadata'].get(symbol, {})
    
    def get_symbols_by_sector(self, sector: str) -> List[str]:
        """Get all symbols in a specific sector"""
        symbols = []
        for symbol, metadata in self.recognition_data['symbol_metadata'].items():
            if metadata.get('sector', '').lower() == sector.lower():
                symbols.append(symbol)
        return symbols

# Usage example:
recognizer = NSESymbolRecognizer('ml_training/models/nse_symbols')
result = recognizer.detect_symbols("What is RELIANCE stock price?")
print(result)
# Output: [{'symbol': 'RELIANCE', 'type': 'stock', 'sector': 'Oil & Gas', 'confidence': 0.9, 'match_type': 'exact', 'intent': 'price_query'}]
```

## Integration with Chatbot

```python
# In your main chatbot service
async def process_user_query(self, user_message: str):
    # Detect symbols and intent
    symbol_results = self.nse_recognizer.detect_symbols(user_message)
    
    if symbol_results:
        for result in symbol_results:
            symbol = result['symbol']
            symbol_type = result['type']
            intent = result.get('intent', 'general_query')
            
            if symbol_type == 'stock':
                if intent == 'price_query':
                    return await self.handle_stock_price_query(symbol, user_message)
                elif intent == 'analysis_query':
                    return await self.handle_stock_analysis_query(symbol, user_message)
                elif intent == 'news_query':
                    return await self.handle_stock_news_query(symbol, user_message)
                else:
                    return await self.handle_general_stock_query(symbol, user_message)
            
            elif symbol_type == 'index':
                return await self.handle_index_query(symbol, user_message)
    
    # If no symbols detected, handle as general query
    return await self.handle_general_query(user_message)

async def handle_stock_price_query(self, symbol: str, query: str):
    # Get stock price using your existing services
    try:
        stock_data = await self.stock_service.get_real_time_data(symbol)
        symbol_info = self.nse_recognizer.get_symbol_info(symbol)
        
        response = f"**{symbol}** ({symbol_info.get('sector', 'Unknown')} sector)\\n"
        response += f"Current Price: ₹{stock_data.close:.2f}\\n"
        response += f"Change: {stock_data.change:+.2f} ({stock_data.change_percent:+.2f}%)\\n"
        
        return {
            "response": response,
            "symbol": symbol,
            "type": "stock_price",
            "data": stock_data
        }
    except Exception as e:
        return {
            "response": f"Sorry, I couldn't fetch the current price for {symbol}. {str(e)}",
            "error": True
        }
```

## Advanced Features

```python
# Symbol validation
def validate_symbol(self, symbol: str) -> bool:
    return symbol in self.lookup_data['symbol_mapping'].values()

# Sector-based queries
def handle_sector_query(self, sector: str) -> List[str]:
    return self.recognizer.get_symbols_by_sector(sector)

# Multiple symbol detection
def detect_multiple_symbols(self, query: str) -> List[str]:
    results = self.recognizer.detect_symbols(query)
    return [r['symbol'] for r in results]
```
'''
        
        with open(os.path.join(self.models_path, 'integration_guide.md'), 'w') as f:
            f.write(integration_code)
        
        print(f"Integration guide saved to {self.models_path}/integration_guide.md")

    def run_complete_training_pipeline(self, csv_file_path: str):
        """Run the complete training pipeline"""
        
        print("🚀 Starting NSE Symbol Training Pipeline (Lightweight)")
        print("="*60)
        
        # Step 1: Extract symbols from CSV
        print("\n📊 Step 1: Extracting symbols from CSV...")
        symbols_data = self.extract_symbols_from_csv(csv_file_path)
        
        if not symbols_data['stocks'] and not symbols_data['indices']:
            print("❌ No symbols found in CSV file. Using default symbol list.")
            symbols_data = {'stocks': [], 'indices': [], 'price_data': {}, 'raw_data': []}
        
        # Step 2: Generate recognition data
        print("\n🔧 Step 2: Generating recognition data...")
        recognition_data = self.generate_symbol_recognition_data(symbols_data)
        
        # Step 3: Create recognition model
        print("\n🏗️ Step 3: Creating recognition model...")
        model_data = self.create_simple_recognition_model(recognition_data)
        
        # Step 4: Save everything
        print("\n💾 Step 4: Saving recognition system...")
        self.save_recognition_system(symbols_data, recognition_data, model_data)
        
        # Step 5: Test the system
        print("\n🧪 Step 5: Testing recognition system...")
        self.test_recognition_system(recognition_data)
        
        # Step 6: Create integration guide
        print("\n📚 Step 6: Creating integration guide...")
        self.create_integration_guide()
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"📁 Recognition system saved to: {self.models_path}")
        print(f"📖 Integration guide: {self.models_path}/integration_guide.md")
        print(f"🔍 Total symbols trained: {len(recognition_data['symbol_metadata'])}")
        
        return True


def main():
    """Main function to run the training"""
    
    # Initialize trainer
    trainer = NSESymbolTrainerLite()
    
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
        print("\n🔗 Next steps:")
        print("1. Integrate the recognition system with your chatbot")
        print("2. Test with various stock queries")
        print("3. Update symbol data periodically")
    else:
        print("\n❌ Training failed. Please check the errors above.")


if __name__ == "__main__":
    main()
