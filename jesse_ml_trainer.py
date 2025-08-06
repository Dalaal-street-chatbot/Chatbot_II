#!/usr/bin/env python3
"""
Jesse ML Trainer for Dalaal Street Chatbot
This script trains a deep learning model on Jesse test files
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, Input, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Add necessary paths
JESSE_PATH = r'c:\Users\hatao\Downloads\jesse-master\jesse-master'
DESKTOP_PATH = r'c:\Users\hatao\OneDrive\Desktop'

# Add paths to system path
sys.path.append(JESSE_PATH)
sys.path.append(DESKTOP_PATH)

class JesseMLTrainer:
    """Train ML models on Jesse test data for the chatbot"""
    
    def __init__(self):
        self.jesse_path = JESSE_PATH
        self.data_path = os.path.join(DESKTOP_PATH, 'ml_training', 'data', 'jesse')
        self.models_path = os.path.join(DESKTOP_PATH, 'ml_training', 'models', 'jesse')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.models_path, exist_ok=True)
        
        # Model parameters
        self.max_words = 10000
        self.max_sequence_length = 100
        self.embedding_dim = 100
        self.lstm_units = 128
        self.dense_units = 64
        self.num_epochs = 10
        self.batch_size = 32
        self.dropout_rate = 0.2
        
        # Data containers
        self.conversation_data = None
        self.tokenizer = None
        self.label_encoder = None
        self.test_category_classifier = None
        self.response_generator = None
    
    def load_conversation_data(self):
        """Load processed conversation data"""
        print(f"📂 Loading conversation data from {self.data_path}")
        
        # Check if conversation data exists
        conversation_file = os.path.join(self.data_path, 'jesse_conversation_data.json')
        if not os.path.exists(conversation_file):
            print(f"❌ Conversation data not found at {conversation_file}")
            print("   Please run jesse_test_integrator.py first to generate the data")
            return False
        
        # Load conversation data
        with open(conversation_file, 'r') as f:
            self.conversation_data = json.load(f)
        
        print(f"[SUCCESS] Loaded {len(self.conversation_data)} conversation examples")
        return True
    
    def prepare_data_for_test_classifier(self):
        """Prepare data for test category classification model"""
        print("🔄 Preparing data for test category classifier")
        
        # Extract questions and categories
        questions = []
        categories = []
        
        for item in self.conversation_data:
            questions.append(item['user_query'])
            categories.append(item['context_used']['test_file'].replace('test_', '').replace('.py', ''))
        
        # Tokenize the questions
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(questions)
        sequences = self.tokenizer.texts_to_sequences(questions)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Encode the categories
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(categories)
        y = to_categorical(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"[SUCCESS] Prepared data with {len(X_train)} training samples and {len(X_test)} test samples")
        print(f"   Number of unique categories: {len(self.label_encoder.classes_)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_test_category_classifier(self):
        """Build and compile the test category classification model"""
        print("🔧 Building test category classifier model")
        
        # Get number of classes
        num_classes = len(self.label_encoder.classes_)
        
        # Build model
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_sequence_length),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            Bidirectional(LSTM(self.lstm_units)),
            Dropout(self.dropout_rate),
            Dense(self.dense_units, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        self.test_category_classifier = model
        print("[SUCCESS] Test category classifier model built successfully")
        
        return model
    
    def train_test_category_classifier(self, X_train, X_test, y_train, y_test):
        """Train the test category classification model"""
        print("🏋️ Training test category classifier model")
        
        # Train the model
        history = self.test_category_classifier.fit(
            X_train, y_train,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate the model
        loss, accuracy = self.test_category_classifier.evaluate(X_test, y_test)
        print(f"[SUCCESS] Model training completed with test accuracy: {accuracy:.4f}")
        
        # Save the model
        model_path = os.path.join(self.models_path, 'jesse_test_category_classifier.keras')
        self.test_category_classifier.save(model_path)
        
        # Save tokenizer and label encoder
        with open(os.path.join(self.models_path, 'tokenizer.json'), 'w') as f:
            json.dump(self.tokenizer.to_json(), f)
        
        with open(os.path.join(self.models_path, 'label_encoder.json'), 'w') as f:
            json.dump({
                'classes': self.label_encoder.classes_.tolist()
            }, f)
        
        print(f"[SUCCESS] Model saved to {model_path}")
        
        # Plot training history
        self.plot_training_history(history, 'test_category_classifier')
        
        return history
    
    def prepare_data_for_response_generator(self):
        """Prepare data for the response generator model"""
        print("🔄 Preparing data for response generator")
        
        # Extract questions and responses
        questions = []
        responses = []
        test_files = []
        
        for item in self.conversation_data:
            questions.append(item['user_query'])
            responses.append(item['assistant_response'])
            test_files.append(item['context_used']['test_file'].replace('test_', '').replace('.py', ''))
        
        # Tokenize the questions
        question_tokenizer = Tokenizer(num_words=self.max_words)
        question_tokenizer.fit_on_texts(questions)
        question_sequences = question_tokenizer.texts_to_sequences(questions)
        X_questions = pad_sequences(question_sequences, maxlen=self.max_sequence_length)
        
        # Tokenize the responses (this will be a placeholder - real generation will use Groq)
        response_tokenizer = Tokenizer(num_words=self.max_words*2)
        response_tokenizer.fit_on_texts(responses)
        response_sequences = response_tokenizer.texts_to_sequences(responses)
        max_response_length = max([len(seq) for seq in response_sequences])
        X_responses = pad_sequences(response_sequences, maxlen=max_response_length)
        
        # One-hot encode the test files
        file_encoder = LabelEncoder()
        file_encoded = file_encoder.fit_transform(test_files)
        file_onehot = to_categorical(file_encoded)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            [X_questions, file_onehot], X_responses, test_size=0.2, random_state=42
        )
        
        print(f"[SUCCESS] Prepared data with {len(X_train[0])} training samples and {len(X_test[0])} test samples")
        
        # Save tokenizers for later use
        with open(os.path.join(self.models_path, 'question_tokenizer.json'), 'w') as f:
            json.dump(question_tokenizer.to_json(), f)
        
        with open(os.path.join(self.models_path, 'response_tokenizer.json'), 'w') as f:
            json.dump(response_tokenizer.to_json(), f)
        
        with open(os.path.join(self.models_path, 'file_encoder.json'), 'w') as f:
            json.dump({
                'classes': file_encoder.classes_.tolist()
            }, f)
        
        return (X_train, X_test, y_train, y_test, 
                question_tokenizer, response_tokenizer, 
                max_response_length, file_encoder)
    
    def build_response_generator(self, max_response_length, num_test_categories):
        """Build and compile the response generator model"""
        print("🔧 Building response generator model")
        
        # Input layers
        question_input = Input(shape=(self.max_sequence_length,), name='question_input')
        category_input = Input(shape=(num_test_categories,), name='category_input')
        
        # Question encoding branch
        question_embedding = Embedding(self.max_words, self.embedding_dim)(question_input)
        question_lstm = Bidirectional(LSTM(self.lstm_units, return_sequences=False))(question_embedding)
        question_dense = Dense(self.dense_units, activation='relu')(question_lstm)
        
        # Category branch
        category_dense = Dense(self.dense_units // 2, activation='relu')(category_input)
        
        # Merge branches
        merged = concatenate([question_dense, category_dense])
        common = Dense(self.dense_units*2, activation='relu')(merged)
        common = Dropout(self.dropout_rate)(common)
        
        # This is just a placeholder for response generation 
        # In practice, we'll use Groq for actual generation
        output = Dense(self.max_words*2, activation='softmax')(common)
        
        # Build model
        model = Model(inputs=[question_input, category_input], outputs=output)
        
        # Compile model
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        self.response_generator = model
        print("[SUCCESS] Response generator model built successfully")
        
        return model
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{model_name} Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{model_name} Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_path, f'{model_name}_history.png'))
        plt.close()
    
    def run_sample_prediction(self):
        """Run a sample prediction to test the model"""
        print("🔄 Testing model with sample predictions")
        
        # Sample questions
        sample_questions = [
            "How does Jesse test backtest functionality?",
            "What assertions are used in the indicators test?",
            "Can you explain how position testing works in Jesse?",
            "Show me how to test the router in Jesse",
            "What does the metrics test verify in Jesse?"
        ]
        
        # Tokenize and pad the questions
        sequences = self.tokenizer.texts_to_sequences(sample_questions)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Predict categories
        predictions = self.test_category_classifier.predict(padded_sequences)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_categories = self.label_encoder.inverse_transform(predicted_classes)
        
        # Print results
        print("\nSample Predictions:")
        print("------------------")
        for i, question in enumerate(sample_questions):
            print(f"Question: {question}")
            print(f"Predicted Category: {predicted_categories[i]}")
            print(f"Confidence: {np.max(predictions[i]):.4f}")
            print()
    
    def create_integration_guide(self):
        """Create a guide for integrating the model with the chatbot"""
        guide = """# Integrating Jesse ML Models with Dalaal Street Chatbot

## Overview
This guide explains how to integrate the trained Jesse test category classifier and response generator models with your existing Dalaal Street chatbot.

## Model Files
- `jesse_test_category_classifier.keras`: TensorFlow model for test category classification
- `tokenizer.json`: Tokenizer used for preprocessing text input
- `label_encoder.json`: Label encoder used for category encoding
- `question_tokenizer.json`: Tokenizer used for preprocessing questions
- `response_tokenizer.json`: Tokenizer used for preprocessing responses
- `file_encoder.json`: Encoder for test file categories

## Integration Steps

### 1. Load the Models
```python
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the test category classifier
model_path = "ml_training/models/jesse/jesse_test_category_classifier.keras"
test_category_classifier = load_model(model_path)

# Load tokenizer and label encoder
with open("ml_training/models/jesse/tokenizer.json", "r") as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

with open("ml_training/models/jesse/label_encoder.json", "r") as f:
    label_encoder_data = json.load(f)
    label_classes = label_encoder_data["classes"]
```

### 2. Process User Input
```python
def classify_jesse_test_query(query):
    # Tokenize and pad the query
    sequence = tokenizer.texts_to_sequences([query])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    
    # Predict category
    prediction = test_category_classifier.predict(padded_sequence)
    predicted_class = np.argmax(prediction[0])
    predicted_category = label_classes[predicted_class]
    confidence = np.max(prediction[0])
    
    return {
        "category": predicted_category,
        "confidence": float(confidence)
    }
```

### 3. Integrate with Intent Classifier
Update your existing intent classifier to recognize Jesse-related queries:

```python
def classify_intent(query):
    # Check if query is about Jesse tests
    jesse_keywords = ["jesse", "test", "trading", "backtest", "strategy", "indicator"]
    if any(keyword in query.lower() for keyword in jesse_keywords):
        # Classify using Jesse model
        jesse_classification = classify_jesse_test_query(query)
        if jesse_classification["confidence"] > 0.6:
            return {
                "intent": "jesse_test",
                "category": jesse_classification["category"],
                "confidence": jesse_classification["confidence"]
            }
    
    # Fall back to your regular intent classifier
    return original_intent_classifier(query)
```

### 4. Generate Responses
For actual response generation, use your existing integration with Groq:

```python
async def generate_response(query, intent_data):
    if intent_data["intent"] == "jesse_test":
        # Use Groq for response generation with proper context
        prompt = f"The user is asking about Jesse's {intent_data['category']} tests. Query: {query}"
        response = await groq_collector.generate_response(prompt)
        return response
    else:
        # Use your regular response generation
        return await generate_regular_response(query, intent_data)
```

### 5. Update Training Pipeline
Add Jesse data to your regular training pipeline:

```python
# In ml_orchestrator.py, update the run_complete_training_pipeline method
async def run_complete_training_pipeline(self):
    # ... existing code ...
    
    # Add Jesse training data to the conversation data
    try:
        with open("ml_training/data/jesse/jesse_conversation_data.json", "r") as f:
            jesse_data = json.load(f)
        
        with open("ml_training/data/conversation_data.json", "r") as f:
            existing_data = json.load(f)
        
        # Combine data
        combined_data = existing_data + jesse_data
        
        # Save combined data
        with open("ml_training/data/conversation_data.json", "w") as f:
            json.dump(combined_data, f, indent=2)
            
        print(f"[SUCCESS] Added {len(jesse_data)} Jesse examples to training data")
    except Exception as e:
        print(f"[WARNING] Could not add Jesse training data: {e}")
    
    # ... rest of existing code ...
```

## Testing the Integration
1. Start your chatbot
2. Ask a Jesse-related question like "How does Jesse test indicators?"
3. Verify that the response contains relevant information about Jesse's indicator tests

## Maintenance
- Retrain models periodically with new test data
- Monitor performance and adjust confidence thresholds as needed
- Update conversation templates if Jesse's test suite changes
"""
        
        guide_path = os.path.join(self.models_path, 'jesse_integration_guide.md')
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide)
        
        print(f"[SUCCESS] Integration guide saved to {guide_path}")
    
    async def run_full_training(self):
        """Run the full model training pipeline"""
        print("🚀 Starting Jesse ML Training Process")
        print("=" * 60)
        
        # Step 1: Load conversation data
        if not self.load_conversation_data():
            return
        
        # Step 2: Train test category classifier
        X_train, X_test, y_train, y_test = self.prepare_data_for_test_classifier()
        self.build_test_category_classifier()
        self.train_test_category_classifier(X_train, X_test, y_train, y_test)
        
        # Step 3: Test model with sample predictions
        self.run_sample_prediction()
        
        # Step 4: Create integration guide
        self.create_integration_guide()
        
        # Step 5: Prepare data for response generator
        # Note: In practice, we'll use Groq for generation
        # This is just to demonstrate the full pipeline
        (X_train, X_test, y_train, y_test, 
         question_tokenizer, response_tokenizer, 
         max_response_length, file_encoder) = self.prepare_data_for_response_generator()
        
        print("\n🎉 Jesse ML Training Completed Successfully!")
        print("\nNote: For production use, we recommend using the Groq integration")
        print("      for response generation as demonstrated in jesse_test_integrator.py")

async def main():
    """Main execution function"""
    
    print("🚀 Jesse ML Trainer for Dalaal Street Chatbot")
    print("=" * 60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check TensorFlow availability
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print()
    
    trainer = JesseMLTrainer()
    await trainer.run_full_training()
    
    print()
    print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())
