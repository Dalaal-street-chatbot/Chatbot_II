#!/usr/bin/env python3
"""
Lean ML Trainer for Dalaal Street Chatbot
This script builds deep learning models based on Lean codebase for the chatbot
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.layers import Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# Add necessary paths - update these to your actual paths
LEAN_PATH = r'c:\path\to\Lean-Master'  # Update this to your Lean path
DESKTOP_PATH = r'c:\Users\hatao\OneDrive\Desktop'
DATA_PATH = os.path.join(DESKTOP_PATH, 'ml_training', 'data', 'lean')
MODEL_PATH = os.path.join(DESKTOP_PATH, 'ml_training', 'models', 'lean')

# Add paths to system path
sys.path.append(LEAN_PATH)
sys.path.append(DESKTOP_PATH)

class LeanMLTrainer:
    """Class to build and train ML models based on Lean code data"""
    
    def __init__(self):
        # Paths
        self.data_path = DATA_PATH
        self.model_path = MODEL_PATH
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # ML parameters
        self.max_words = 10000  # Max vocabulary size
        self.max_sequence_length = 300  # Max sequence length for padding
        self.embedding_dim = 300  # Embedding dimension
        self.validation_split = 0.2  # Validation split ratio
        
        # Model artifacts
        self.tokenizer = None
        self.label_encoder = None
        self.intent_model = None
        self.response_classifier = None
        self.component_classifier = None
        
        # Initialize counters
        self.num_training_examples = 0
        self.num_categories = 0
        self.num_components = 0
        
    def load_conversation_data(self):
        """Load previously processed conversation data"""
        print(f"📂 Loading conversation data from {self.data_path}")
        
        data_file = os.path.join(self.data_path, 'lean_conversation_data.json')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}. Run lean_code_integrator.py first.")
        
        with open(data_file, 'r') as f:
            conversation_data = json.load(f)
        
        self.num_training_examples = len(conversation_data)
        print(f"✅ Loaded {self.num_training_examples} conversation examples")
        
        return conversation_data
    
    def preprocess_conversation_data(self, conversation_data):
        """Preprocess conversation data for training"""
        print("🔄 Preprocessing conversation data")
        
        # Extract features and targets
        user_queries = []
        categories = []
        component_names = []
        responses = []
        
        for item in conversation_data:
            user_queries.append(item['user_query'])
            categories.append(item['category'])
            component_names.append(item['component_name'])
            responses.append(item['assistant_response'])
        
        # Create tokenizer for user queries
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(user_queries)
        
        # Convert text to sequences
        user_sequences = self.tokenizer.texts_to_sequences(user_queries)
        X = pad_sequences(user_sequences, maxlen=self.max_sequence_length)
        
        # Encode categories
        self.category_encoder = LabelEncoder()
        y_category = self.category_encoder.fit_transform(categories)
        
        # Encode component names
        self.component_encoder = LabelEncoder()
        y_component = self.component_encoder.fit_transform(component_names)
        
        self.num_categories = len(self.category_encoder.classes_)
        self.num_components = len(self.component_encoder.classes_)
        
        print(f"✅ Preprocessed {len(user_queries)} examples")
        print(f"  • Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"  • Categories: {self.num_categories}")
        print(f"  • Components: {self.num_components}")
        
        # Create train-test splits
        X_train, X_test, y_cat_train, y_cat_test = train_test_split(
            X, y_category, test_size=self.validation_split, random_state=42)
        
        _, _, y_comp_train, y_comp_test = train_test_split(
            X, y_component, test_size=self.validation_split, random_state=42)
        
        return (X_train, X_test, y_cat_train, y_cat_test, y_comp_train, y_comp_test)
    
    def build_intent_classifier(self, num_categories):
        """Build an intent classifier model"""
        print("🔄 Building intent classifier model")
        
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_sequence_length),
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(64)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_categories, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        print(f"✅ Built intent classifier model")
        model.summary()
        
        return model
    
    def build_component_classifier(self, num_components):
        """Build a component classifier model"""
        print("🔄 Building component classifier model")
        
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_sequence_length),
            Bidirectional(LSTM(256, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_components, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        print(f"✅ Built component classifier model with {num_components} components")
        model.summary()
        
        return model
    
    def train_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """Train a model with early stopping"""
        print(f"🏋️ Training {model_name}...")
        
        # Define callbacks
        checkpoint_path = os.path.join(self.model_path, f'{model_name}_best.h5')
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, save_best_only=True)
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=32,
            callbacks=callbacks
        )
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"✅ {model_name} - Test accuracy: {accuracy:.4f}")
        
        # Save the final model
        final_model_path = os.path.join(self.model_path, f'{model_name}.h5')
        model.save(final_model_path)
        print(f"✅ Saved {model_name} to {final_model_path}")
        
        # Plot training history
        self.plot_training_history(history, model_name)
        
        return model, history
    
    def plot_training_history(self, history, model_name):
        """Plot and save training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{model_name} - Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{model_name} - Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Save the plot
        plot_path = os.path.join(self.model_path, f'{model_name}_history.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    
    def save_model_artifacts(self):
        """Save model artifacts for inference"""
        print("💾 Saving model artifacts")
        
        # Save tokenizer
        tokenizer_path = os.path.join(self.model_path, 'tokenizer.pickle')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save category encoder
        category_encoder_path = os.path.join(self.model_path, 'category_encoder.pickle')
        with open(category_encoder_path, 'wb') as f:
            pickle.dump(self.category_encoder, f)
        
        # Save component encoder
        component_encoder_path = os.path.join(self.model_path, 'component_encoder.pickle')
        with open(component_encoder_path, 'wb') as f:
            pickle.dump(self.component_encoder, f)
        
        # Save configuration
        config = {
            'max_words': self.max_words,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim,
            'num_categories': self.num_categories,
            'num_components': self.num_components,
            'trained_at': datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.model_path, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Saved model artifacts to {self.model_path}")
    
    def create_inference_pipeline(self):
        """Create and save an inference pipeline class"""
        print("🔄 Creating inference pipeline")
        
        inference_code = """
import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class LeanModelInference:
    \"\"\"Class for making predictions using trained Lean models\"\"\"
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._load_artifacts()
    
    def _load_artifacts(self):
        \"\"\"Load all required model artifacts\"\"\"
        # Load tokenizer
        with open(os.path.join(self.model_dir, 'tokenizer.pickle'), 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Load encoders
        with open(os.path.join(self.model_dir, 'category_encoder.pickle'), 'rb') as f:
            self.category_encoder = pickle.load(f)
        
        with open(os.path.join(self.model_dir, 'component_encoder.pickle'), 'rb') as f:
            self.component_encoder = pickle.load(f)
        
        # Load config
        with open(os.path.join(self.model_dir, 'model_config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Load models
        self.intent_model = load_model(os.path.join(self.model_dir, 'intent_classifier.h5'))
        self.component_model = load_model(os.path.join(self.model_dir, 'component_classifier.h5'))
        
        print(f"Loaded models and artifacts from {self.model_dir}")
    
    def preprocess_query(self, query):
        \"\"\"Preprocess a user query\"\"\"
        sequence = self.tokenizer.texts_to_sequences([query])
        padded = pad_sequences(sequence, maxlen=self.config['max_sequence_length'])
        return padded
    
    def predict_category(self, query):
        \"\"\"Predict the category of a query\"\"\"
        padded = self.preprocess_query(query)
        predictions = self.intent_model.predict(padded)
        category_idx = np.argmax(predictions, axis=1)[0]
        category = self.category_encoder.inverse_transform([category_idx])[0]
        confidence = predictions[0][category_idx]
        return category, confidence
    
    def predict_component(self, query):
        \"\"\"Predict the component a query is about\"\"\"
        padded = self.preprocess_query(query)
        predictions = self.component_model.predict(padded)
        component_idx = np.argmax(predictions, axis=1)[0]
        component = self.component_encoder.inverse_transform([component_idx])[0]
        confidence = predictions[0][component_idx]
        return component, confidence
    
    def get_top_components(self, query, top_k=3):
        \"\"\"Get top k component predictions\"\"\"
        padded = self.preprocess_query(query)
        predictions = self.component_model.predict(padded)[0]
        top_indices = predictions.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            component = self.component_encoder.inverse_transform([idx])[0]
            confidence = predictions[idx]
            results.append({
                'component': component,
                'confidence': float(confidence)
            })
        
        return results
    
    def analyze_query(self, query):
        \"\"\"Analyze a query and return all predictions\"\"\"
        category, category_confidence = self.predict_category(query)
        component, component_confidence = self.predict_component(query)
        top_components = self.get_top_components(query)
        
        return {
            'query': query,
            'category': {
                'name': category,
                'confidence': float(category_confidence)
            },
            'component': {
                'name': component,
                'confidence': float(component_confidence)
            },
            'top_components': top_components
        }
"""
        
        # Save the inference code
        inference_path = os.path.join(self.model_path, 'lean_model_inference.py')
        with open(inference_path, 'w') as f:
            f.write(inference_code.strip())
        
        print(f"✅ Created inference pipeline at {inference_path}")
    
    def run_full_training(self):
        """Run the full model training pipeline"""
        print("🚀 Starting Lean ML Training Process")
        print("=" * 60)
        
        # Step 1: Load conversation data
        conversation_data = self.load_conversation_data()
        
        # Step 2: Preprocess conversation data
        preprocessed = self.preprocess_conversation_data(conversation_data)
        X_train, X_test, y_cat_train, y_cat_test, y_comp_train, y_comp_test = preprocessed
        
        # Step 3: Build and train intent classifier
        intent_model = self.build_intent_classifier(self.num_categories)
        self.intent_model, _ = self.train_model(
            intent_model, X_train, y_cat_train, X_test, y_cat_test, 'intent_classifier')
        
        # Step 4: Build and train component classifier
        component_model = self.build_component_classifier(self.num_components)
        self.component_classifier, _ = self.train_model(
            component_model, X_train, y_comp_train, X_test, y_comp_test, 'component_classifier')
        
        # Step 5: Save model artifacts
        self.save_model_artifacts()
        
        # Step 6: Create inference pipeline
        self.create_inference_pipeline()
        
        # Step 7: Create training report
        self.create_training_report()
        
        print("\n🎉 Lean ML Training Completed Successfully!")
    
    def create_training_report(self):
        """Create a training report"""
        report = f"""
# Lean ML Training Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Summary
- **Training Examples**: {self.num_training_examples}
- **Categories**: {self.num_categories}
- **Components**: {self.num_components}
- **Vocabulary Size**: {len(self.tokenizer.word_index)}

## Model Architecture
### Intent Classifier
- Bidirectional LSTM network
- Embedding dimension: {self.embedding_dim}
- LSTM units: 128 (first layer), 64 (second layer)
- Output categories: {self.num_categories}

### Component Classifier
- Bidirectional LSTM network
- Embedding dimension: {self.embedding_dim}
- LSTM units: 256
- Output components: {self.num_components}

## Model Files
- `intent_classifier.h5`: Intent classification model
- `component_classifier.h5`: Component prediction model
- `tokenizer.pickle`: Text tokenizer
- `category_encoder.pickle`: Category label encoder
- `component_encoder.pickle`: Component label encoder
- `model_config.json`: Model configuration
- `lean_model_inference.py`: Inference pipeline

## Integration with Dalaal Street Chatbot
The models are ready for integration with your existing Dalaal Street chatbot.
Use the LeanModelInference class for making predictions in your application.

## Next Steps
1. Integrate the models with your chatbot's inference pipeline
2. Add the Lean conversation data to your fine-tuning dataset
3. Evaluate and improve the models with real user feedback
"""
        
        report_file = os.path.join(self.model_path, 'training_report.md')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Training report saved to {report_file}")

def main():
    """Main execution function"""
    
    print("🚀 Lean ML Trainer for Dalaal Street Chatbot")
    print("=" * 60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    trainer = LeanMLTrainer()
    trainer.run_full_training()
    
    print()
    print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
