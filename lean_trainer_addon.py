#!/usr/bin/env python3
"""
Lean Trainer Addon for Dalaal Street Chatbot
This script integrates Lean's codebase ML models with the existing chatbot
"""

import os
import sys
import json
from datetime import datetime
import importlib
import argparse
import shutil

# Add necessary paths - update these to your actual paths
LEAN_PATH = r'c:\path\to\Lean-Master'  # Update this to your Lean path
DESKTOP_PATH = r'c:\Users\hatao\OneDrive\Desktop'
DATA_PATH = os.path.join(DESKTOP_PATH, 'ml_training', 'data')
MODEL_PATH = os.path.join(DESKTOP_PATH, 'ml_training', 'models')

# Add paths to system path
sys.path.append(LEAN_PATH)
sys.path.append(DESKTOP_PATH)

# Import ml_orchestrator from existing project
try:
    from ml_training.ml_orchestrator import MLOrchestrator
except ImportError:
    print("Warning: Could not import MLOrchestrator, some functionality may be limited")
    class MLOrchestrator:
        """Mock class for MLOrchestrator"""
        def __init__(self):
            self.trainers = {}
        
        def register_trainer(self, name, trainer):
            """Register a trainer"""
            self.trainers[name] = trainer
        
        def train(self, trainer_name, *args, **kwargs):
            """Train a model"""
            print(f"Would train {trainer_name} if implemented")

class LeanTrainer:
    """Integration class for Lean ML training with Dalaal Street chatbot"""
    
    def __init__(self):
        self.data_path = os.path.join(DATA_PATH, 'lean')
        self.model_path = os.path.join(MODEL_PATH, 'lean')
        
        # Create paths if they don't exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
    
    def check_data_availability(self):
        """Check if Lean data is available"""
        conversation_data = os.path.join(self.data_path, 'lean_conversation_data.json')
        groq_data = os.path.join(self.data_path, 'lean_groq_data.jsonl')
        
        if not os.path.exists(conversation_data):
            print("⚠️ Lean conversation data not found")
            return False
        
        if not os.path.exists(groq_data):
            print("⚠️ Lean Groq training data not found")
            return False
        
        return True
    
    def preprocess(self):
        """Preprocess Lean data - called by orchestrator"""
        print("🔄 Preprocessing Lean data")
        
        if not self.check_data_availability():
            print("❌ Required data files not found. Run lean_code_integrator.py first.")
            return False
        
        # Count conversation examples
        conversation_data_file = os.path.join(self.data_path, 'lean_conversation_data.json')
        with open(conversation_data_file, 'r') as f:
            conversation_data = json.load(f)
        
        # Create preprocessing summary
        summary = {
            'num_examples': len(conversation_data),
            'categories': set([item['category'] for item in conversation_data]),
            'preprocessed_at': datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = os.path.join(self.data_path, 'preprocessing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Preprocessed {len(conversation_data)} Lean conversation examples")
        return True
    
    def train(self, *args, **kwargs):
        """Train Lean models - called by orchestrator"""
        print("🚀 Training Lean models")
        
        # Call the lean_ml_trainer.py script
        lean_ml_trainer_path = os.path.join(DESKTOP_PATH, 'lean_ml_trainer.py')
        if not os.path.exists(lean_ml_trainer_path):
            print(f"❌ Lean ML trainer not found at {lean_ml_trainer_path}")
            return False
        
        # Execute the trainer
        print("📣 Executing Lean ML trainer")
        print("=" * 60)
        
        try:
            # Use importlib to avoid subprocess
            spec = importlib.util.spec_from_file_location("lean_ml_trainer", lean_ml_trainer_path)
            lean_ml_trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lean_ml_trainer)
            
            # Run the trainer
            lean_ml_trainer.main()
            print("=" * 60)
            print("✅ Lean ML training completed")
            return True
            
        except Exception as e:
            print(f"❌ Error running Lean ML trainer: {e}")
            return False
    
    def evaluate(self, *args, **kwargs):
        """Evaluate Lean models - called by orchestrator"""
        print("🔍 Evaluating Lean models")
        
        # Check if models exist
        intent_model_path = os.path.join(self.model_path, 'intent_classifier.h5')
        component_model_path = os.path.join(self.model_path, 'component_classifier.h5')
        
        if not os.path.exists(intent_model_path) or not os.path.exists(component_model_path):
            print("❌ Trained models not found. Run training first.")
            return False
        
        # For demonstration, we'll just report model file sizes
        intent_size = os.path.getsize(intent_model_path) / (1024 * 1024)  # MB
        component_size = os.path.getsize(component_model_path) / (1024 * 1024)  # MB
        
        print(f"✅ Lean models evaluation:")
        print(f"  • Intent classifier: {intent_size:.2f} MB")
        print(f"  • Component classifier: {component_size:.2f} MB")
        
        # Actual evaluation would be done by loading the models and running test data
        return True
    
    def register_with_orchestrator(self, orchestrator):
        """Register this trainer with the ML orchestrator"""
        print("🔄 Registering Lean trainer with ML orchestrator")
        
        orchestrator.register_trainer('lean', self)
        print("✅ Lean trainer registered")
    
    def integrate_with_chatbot(self):
        """Integrate Lean models with the Dalaal Street chatbot"""
        print("🔄 Integrating Lean models with Dalaal Street chatbot")
        
        # 1. Copy inference file to the chatbot's services directory
        inference_file = os.path.join(self.model_path, 'lean_model_inference.py')
        if os.path.exists(inference_file):
            dest_path = os.path.join(DESKTOP_PATH, 'app', 'services', 'lean_model_inference.py')
            shutil.copy(inference_file, dest_path)
            print(f"✅ Copied inference pipeline to {dest_path}")
        
        # 2. Add integration code
        integration_code = """
# Add this to your chatbot's chat processor

from app.services.lean_model_inference import LeanModelInference

# Initialize Lean model inference
lean_model_path = os.path.join(BASE_DIR, 'ml_training', 'models', 'lean')
lean_inference = LeanModelInference(lean_model_path)

# In your chat processing function, add:
def process_query(query, context=None):
    # Analyze with Lean models
    lean_analysis = lean_inference.analyze_query(query)
    
    # If Lean-related query is detected with high confidence (e.g., >0.7)
    if lean_analysis['category']['name'].startswith('lean_') and lean_analysis['category']['confidence'] > 0.7:
        # Use Lean models for response
        component = lean_analysis['component']['name']
        # Lookup response based on component or generate with LLM
        response = generate_lean_response(query, component)
        return response
    
    # Otherwise, use existing chatbot logic
    # ...

def generate_lean_response(query, component):
    # Use the component information to enhance response
    # This could reference documentation, code examples, etc.
    # Or you could use a pre-trained response model
    pass
"""
        
        # Save integration example
        example_path = os.path.join(DESKTOP_PATH, 'lean_integration_example.py')
        with open(example_path, 'w') as f:
            f.write(integration_code.strip())
        
        print(f"✅ Created integration example at {example_path}")
        print("📣 Add this code to your chatbot's processing pipeline")
        
        return True

def install_lean_integration():
    """Install Lean integration with the Dalaal Street chatbot"""
    print("🚀 Lean Integration for Dalaal Street Chatbot")
    print("=" * 60)
    
    # Create Lean trainer
    lean_trainer = LeanTrainer()
    
    # Try to import the ML orchestrator
    try:
        from ml_training.ml_orchestrator import ml_orchestrator
        
        # Register with orchestrator
        lean_trainer.register_with_orchestrator(ml_orchestrator)
        
        print("📣 Lean trainer registered with ML orchestrator")
        print("You can now run the following command to train:")
        print("python train_bot.py lean")
        
    except ImportError:
        print("⚠️ ML orchestrator not found")
        print("You need to register the Lean trainer manually")
        print("See lean_integration_example.py for integration code")
    
    # Create integration guide
    create_integration_guide()
    
    print("\n🎉 Lean Integration Setup Completed!")

def create_integration_guide():
    """Create a comprehensive integration guide"""
    guide = """# Lean Chatbot Integration Guide

## Overview
This guide explains how to integrate the Lean algorithmic trading framework with your existing Dalaal Street chatbot, providing users with the ability to ask questions about Lean's codebase, architecture, and functionality.

## Prerequisites
1. A working Dalaal Street chatbot
2. The Lean codebase (Lean-Master directory)
3. Python 3.7+
4. TensorFlow 2.x
5. Required Python packages: numpy, pandas, scikit-learn, matplotlib

## Integration Process

### Step 1: Clone the Lean Repository
If you haven't already, clone the Lean repository:
```bash
git clone https://github.com/QuantConnect/Lean.git Lean-Master
```

### Step 2: Set Up Directory Structure
Ensure you have the following directory structure:
```
Desktop/
  ├── ml_training/
  │   ├── data/
  │   │   └── lean/  # Lean training data will be stored here
  │   └── models/
  │       └── lean/  # Lean models will be stored here
  ├── app/
  │   └── services/  # Inference pipeline will be copied here
  ├── lean_code_integrator.py
  ├── lean_ml_trainer.py
  └── lean_trainer_addon.py
```

### Step 3: Configure Paths
Update the paths in each script to match your environment:
1. Open `lean_code_integrator.py`, `lean_ml_trainer.py`, and `lean_trainer_addon.py`
2. Update `LEAN_PATH` to point to your Lean-Master directory
3. Update `DESKTOP_PATH` if necessary

### Step 4: Extract Lean Code Data
Run the code integrator to extract and process Lean's codebase:
```bash
python lean_code_integrator.py
```
This will:
- Scan Lean's codebase for relevant files
- Extract classes, methods, and their documentation
- Generate conversation templates for training
- Create data files in the `ml_training/data/lean/` directory

### Step 5: Train Machine Learning Models
Run the ML trainer to build models based on the extracted data:
```bash
python lean_ml_trainer.py
```
This will:
- Load and preprocess the conversation data
- Train intent and component classifier models
- Save the trained models and artifacts
- Create an inference pipeline

### Step 6: Integrate with Dalaal Street Chatbot
Register the Lean trainer with your ML orchestrator:
```bash
python lean_trainer_addon.py
```
This will:
- Register the Lean trainer with your ML orchestrator
- Copy the inference pipeline to your services directory
- Create an integration example

### Step 7: Run Combined Training
Train your chatbot with both Dalaal Street and Lean data:
```bash
python train_bot.py all
```

### Step 8: Update Your Chat Processing Pipeline
Modify your chat processing logic to include Lean model inference, as shown in the integration example.

## Usage Examples

### Asking About Lean Components
User can ask questions like:
- "How does the QCAlgorithm class work in Lean?"
- "What methods are available in Lean's Portfolio class?"
- "Can you explain the purpose of Universe Selection in Lean?"
- "How do I implement a custom alpha model in Lean?"

### Getting Code Examples
User can ask for examples like:
- "Show me an example of a simple Lean algorithm"
- "How do I create a custom indicator in Lean?"
- "What's the structure of a backtest in Lean?"

## Troubleshooting

### Model Training Issues
If you encounter issues during model training:
1. Check that TensorFlow is installed correctly
2. Ensure enough training data was generated
3. Try adjusting the model parameters in `lean_ml_trainer.py`

### Integration Issues
If the chatbot isn't using Lean models correctly:
1. Check that the models are loaded properly
2. Verify that the inference pipeline is accessible
3. Check confidence thresholds for Lean-related queries

### Data Generation Issues
If not enough data is generated:
1. Check that the Lean path is correct
2. Ensure the code integrator can access Lean files
3. Try expanding the directories scanned in `lean_code_integrator.py`

## Extending the Integration

### Adding More Training Data
You can enhance the training data by:
1. Adding more conversation templates to `lean_conversation_data.json`
2. Including more specific examples of Lean usage
3. Generating templates for additional Lean components

### Improving Model Performance
To improve the models:
1. Adjust model architecture in `lean_ml_trainer.py`
2. Add more training examples with varied question formats
3. Fine-tune the embedding dimensions and LSTM parameters

### Enhancing Response Generation
To provide better responses:
1. Add a retrieval-augmented generation system
2. Index Lean documentation for more accurate answers
3. Add code example generation based on templates

## Maintenance

### Updating with New Lean Versions
When Lean is updated:
1. Run the code integrator again to extract new components
2. Retrain the models with the updated data
3. Update the inference pipeline if necessary

### Monitoring Performance
Monitor the performance of Lean-related queries by:
1. Tracking user satisfaction with responses
2. Logging queries that received low confidence scores
3. Periodically evaluating model accuracy

## Resources
- [Lean Documentation](https://www.lean.io/docs/)
- [QuantConnect](https://www.quantconnect.com/)
- [Lean GitHub Repository](https://github.com/QuantConnect/Lean)
"""
    
    guide_path = os.path.join(DESKTOP_PATH, 'LEAN_CHATBOT_INTEGRATION.md')
    with open(guide_path, 'w') as f:
        f.write(guide)
    
    print(f"✅ Created integration guide at {guide_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lean Trainer Addon for Dalaal Street Chatbot')
    parser.add_argument('--install', action='store_true', help='Install Lean integration')
    parser.add_argument('--train', action='store_true', help='Train Lean models')
    parser.add_argument('--integrate', action='store_true', help='Integrate with chatbot')
    
    args = parser.parse_args()
    
    if args.install:
        install_lean_integration()
    elif args.train:
        trainer = LeanTrainer()
        trainer.train()
    elif args.integrate:
        trainer = LeanTrainer()
        trainer.integrate_with_chatbot()
    else:
        install_lean_integration()  # Default action
