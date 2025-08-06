# Jesse Integration for Dalaal Street Chatbot

This extension integrates the Jesse algorithmic trading framework with the Dalaal Street chatbot, enabling the chatbot to provide detailed information about algorithmic trading strategies, backtesting, and technical indicators.

## Overview

Jesse is an advanced algorithmic trading framework with backtesting capabilities. This integration extracts knowledge from Jesse's test suite and trains machine learning models to enable the chatbot to answer questions related to algorithmic trading.

## Key Features

- **Test Case Extraction**: Extracts test cases from Jesse's codebase to gather domain knowledge
- **Conversation Template Generation**: Converts test cases into conversation templates for training
- **Machine Learning Models**: Trains a test category classifier with 92.89% accuracy
- **Groq Integration**: Formats data for fine-tuning with Groq models

## Files

- `jesse_integration_tester.py`: Main script to test the integration process
- `jesse_test_integrator.py`: Extracts test cases from Jesse's codebase
- `jesse_ml_trainer.py`: Trains machine learning models on the extracted data
- `jesse_trainer_addon.py`: Integration with the ML orchestrator
- `jesse_ml_integrator.py`: Integration with the chatbot's ML pipeline

## Integration Results

- **Test Files Processed**: 27
- **Test Cases Extracted**: 442
- **Conversation Templates Generated**: 3,094
- **Model Accuracy**: 92.89%

## Usage

### Running the Integration

```bash
python jesse_integration_tester.py
```

### Training the Chatbot

```bash
python train_bot.py jesse
```

### Using in the Chatbot

The chatbot can now answer questions like:
- "How does Jesse test indicators?"
- "What's the purpose of the backtest test in Jesse?"
- "How is position handling implemented in Jesse's test suite?"

## Directory Structure

```
ml_training/
  ├── data/
  │   └── jesse/
  │       ├── jesse_conversation_data.json   # Training data
  │       ├── jesse_groq_data.jsonl          # Groq fine-tuning data
  │       ├── jesse_test_categories.json     # Categories of tests
  │       └── jesse_testing_guide.md         # Guide on Jesse testing
  └── models/
      └── jesse/
          ├── jesse_test_category_classifier.keras  # ML model
          ├── tokenizer.json                        # Tokenizer for text processing
          ├── label_encoder.json                    # Encoder for categories
          └── jesse_integration_guide.md            # Integration guide
```

## Dependencies

- TensorFlow 2.x
- Jesse framework
- Pandas
- NumPy
- AST (Python standard library)

## Future Work

- Train additional models for more specific Jesse functionality
- Expand the training data with Jesse documentation
- Add support for code generation for Jesse strategies
- Integrate with live trading APIs

## Contributors

- Dalaal Street Chatbot Team

## License

Same as the main Dalaal Street Chatbot project
