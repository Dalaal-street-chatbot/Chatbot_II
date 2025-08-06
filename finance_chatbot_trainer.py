"""
Finance Chatbot Trainer
This script trains a model on the finance-alpaca dataset for financial Q&A
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from transformers import AutoTokenizer
import datetime

class FinanceTrainer:
    def __init__(self):
        # Set paths
        self.base_path = os.path.join('ml_training')
        self.data_path = os.path.join(self.base_path, 'data', 'finance')
        self.models_path = os.path.join(self.base_path, 'models', 'finance')
        
        # Create directories if they don't exist
        os.makedirs(self.models_path, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Model parameters
        self.max_length = 128
        self.embedding_dim = 384
        self.batch_size = 32
        self.epochs = 10
        
        print(f"Finance trainer initialized with data path: {self.data_path}")
        print(f"Models will be saved to: {self.models_path}")
    
    def load_data(self):
        """Load the finance conversation data"""
        try:
            data_file = os.path.join(self.data_path, 'finance_conversation_data.json')
            print(f"Loading data from {data_file}")
            
            with open(data_file, 'r', encoding='utf-8') as f:
                self.conversation_data = json.load(f)
            
            print(f"Loaded {len(self.conversation_data)} conversations")
            
            # Extract questions and responses
            self.questions = []
            self.responses = []
            
            for conv in self.conversation_data:
                if len(conv['conversation']) >= 2:
                    user_message = conv['conversation'][0]['content']
                    assistant_message = conv['conversation'][1]['content']
                    self.questions.append(user_message)
                    self.responses.append(assistant_message)
            
            print(f"Extracted {len(self.questions)} question-response pairs")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data for training"""
        try:
            print("Preprocessing data...")
            
            # Generate intent categories (simplified for demo)
            # In a real system, you might want to cluster or classify intents
            self.intents = []
            for q in self.questions:
                # Simple heuristic: first few words as category
                words = q.split()[:3]
                intent = ' '.join(words).lower()
                self.intents.append(intent)
            
            # Encode intents
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.intents)
            
            # Tokenize questions
            self.X = []
            for question in self.questions:
                tokens = self.tokenizer(
                    question,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='tf'
                )
                self.X.append(tokens['input_ids'][0].numpy())
            
            self.X = np.array(self.X)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            print(f"Preprocessing complete. Training on {len(self.X_train)} samples, testing on {len(self.X_test)}")
            return True
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return False
    
    def build_model(self):
        """Build the model architecture"""
        try:
            print("Building model...")
            
            vocab_size = self.tokenizer.vocab_size
            num_classes = len(self.label_encoder.classes_)
            
            model = models.Sequential([
                layers.Input(shape=(self.max_length,)),
                layers.Embedding(vocab_size, self.embedding_dim),
                layers.GlobalAveragePooling1D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=optimizers.Adam(1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            print(f"Model built with {num_classes} output classes")
            return True
            
        except Exception as e:
            print(f"Error building model: {e}")
            return False
    
    def train_model(self):
        """Train the model"""
        try:
            print("Training model...")
            
            # Create TensorBoard callback
            log_dir = os.path.join(self.models_path, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            
            # Train the model
            history = self.model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_test, self.y_test),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[tensorboard_callback]
            )
            
            # Save the model
            model_path = os.path.join(self.models_path, 'finance_intent_classifier.keras')
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Save the label encoder
            encoder_path = os.path.join(self.models_path, 'label_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"Label encoder saved to {encoder_path}")
            
            # Save the response data for retrieval
            response_data = {
                'questions': self.questions,
                'responses': self.responses,
                'intents': self.intents
            }
            response_path = os.path.join(self.models_path, 'response_data.json')
            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(response_data, f)
            print(f"Response data saved to {response_path}")
            
            # Save tokenizer config
            tokenizer_path = os.path.join(self.models_path, 'tokenizer_config.json')
            with open(tokenizer_path, 'w', encoding='utf-8') as f:
                json.dump({'max_length': self.max_length}, f)
            print(f"Tokenizer config saved to {tokenizer_path}")
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def generate_integration_guide(self):
        """Generate a guide for integrating the model"""
        try:
            guide_text = """# Finance Chatbot Integration Guide

## Overview

This guide explains how to integrate the finance chatbot model with your application.

## Files Generated

- `finance_intent_classifier.keras`: The trained TensorFlow model
- `label_encoder.pkl`: The scikit-learn LabelEncoder for intent labels
- `response_data.json`: Questions, responses, and intent mappings
- `tokenizer_config.json`: Configuration for the tokenizer

## Integration Steps

1. **Load the model and supporting files**
   ```python
   import tensorflow as tf
   import pickle
   import json
   from transformers import AutoTokenizer
   
   # Load model
   model = tf.keras.models.load_model('ml_training/models/finance/finance_intent_classifier.keras')
   
   # Load label encoder
   with open('ml_training/models/finance/label_encoder.pkl', 'rb') as f:
       label_encoder = pickle.load(f)
   
   # Load response data
   with open('ml_training/models/finance/response_data.json', 'r', encoding='utf-8') as f:
       response_data = json.load(f)
   
   # Load tokenizer
   tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   with open('ml_training/models/finance/tokenizer_config.json', 'r', encoding='utf-8') as f:
       tokenizer_config = json.load(f)
   max_length = tokenizer_config['max_length']
   ```

2. **Create a prediction function**
   ```python
   def get_finance_response(user_query):
       # Tokenize the input
       tokens = tokenizer(
           user_query,
           padding='max_length',
           truncation=True,
           max_length=max_length,
           return_tensors='tf'
       )
       
       # Make prediction
       predictions = model.predict(tokens['input_ids'])
       predicted_class = predictions.argmax(axis=1)[0]
       
       # Get the predicted intent
       predicted_intent = label_encoder.inverse_transform([predicted_class])[0]
       
       # Find matching responses
       responses = []
       for i, intent in enumerate(response_data['intents']):
           if intent == predicted_intent:
               responses.append(response_data['responses'][i])
       
       # Return the first matching response or a default
       if responses:
           return responses[0]
       else:
           return "I'm not sure how to answer that financial question."
   ```

3. **Integrate with your chatbot**
   ```python
   # Example integration with your existing chatbot handler
   async def handle_chat_message(message):
       # Check if it's a finance-related question
       if is_finance_question(message):
           return get_finance_response(message)
       else:
           # Use your existing chatbot logic
           return await your_existing_chatbot(message)
   ```

## Additional Tips

- You can fine-tune the model with your own data by adding examples
- The model works best for finance-related questions similar to the training data
- Consider implementing a confidence threshold for fallback to more general responses
"""
            
            guide_path = os.path.join(self.models_path, 'finance_integration_guide.md')
            with open(guide_path, 'w', encoding='utf-8') as f:
                f.write(guide_text)
            print(f"Integration guide saved to {guide_path}")
            return True
            
        except Exception as e:
            print(f"Error generating integration guide: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("\n===== FINANCE CHATBOT TRAINING PIPELINE =====\n")
        
        success = self.load_data()
        if not success:
            print("Failed to load data. Aborting.")
            return False
        
        success = self.preprocess_data()
        if not success:
            print("Failed to preprocess data. Aborting.")
            return False
        
        success = self.build_model()
        if not success:
            print("Failed to build model. Aborting.")
            return False
        
        success = self.train_model()
        if not success:
            print("Failed to train model. Aborting.")
            return False
        
        success = self.generate_integration_guide()
        if not success:
            print("Failed to generate integration guide. Aborting.")
            return False
        
        print("\n===== TRAINING PIPELINE COMPLETED SUCCESSFULLY =====\n")
        return True

if __name__ == "__main__":
    trainer = FinanceTrainer()
    trainer.run_complete_pipeline()
