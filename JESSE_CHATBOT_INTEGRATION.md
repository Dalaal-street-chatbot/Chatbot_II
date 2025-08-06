# Integrating Jesse Test Files with Dalaal Street Chatbot

This guide explains how to train your Dalaal Street chatbot using Jesse's test files with TensorFlow-based deep learning.

## Overview

The integration involves:
1. Extracting test cases from Jesse's test files
2. Converting them to chatbot conversation templates
3. Training deep learning models to understand and respond to Jesse-related queries
4. Fine-tuning your existing chatbot model with this data

## Setup Instructions

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Jesse codebase (from https://github.com/jesse-ai/jesse)
- Dalaal Street chatbot codebase
- Groq API key (if using Groq for fine-tuning)

### Installation

1. Copy the following files to your Desktop:
   - `jesse_test_integrator.py`
   - `jesse_ml_trainer.py`
   - `jesse_trainer_addon.py`

2. Ensure paths are correct:
   ```python
   JESSE_PATH = r'c:\Users\hatao\Downloads\jesse-master\jesse-master'
   DESKTOP_PATH = r'c:\Users\hatao\OneDrive\Desktop'
   ```
   Update these paths if necessary to match your environment.

3. Create directories:
   ```
   mkdir -p ml_training/data/jesse
   mkdir -p ml_training/models/jesse
   ```

### Step 1: Extract and Process Jesse Test Files

Run the test integrator to extract and process Jesse's test files:

```bash
python jesse_test_integrator.py
```

This will:
- Scan Jesse's test directory
- Extract test cases and assertions
- Generate chatbot conversation templates
- Save the data in `ml_training/data/jesse/`

### Step 2: Train Deep Learning Models

Train deep learning models on the extracted test data:

```bash
python jesse_ml_trainer.py
```

This will:
- Train a test category classifier using TensorFlow
- Prepare data for response generation
- Save models and tokenizers in `ml_training/models/jesse/`
- Generate an integration guide

### Step 3: Integrate with Dalaal Street Chatbot

#### Option 1: Use the provided addon

Follow the instructions in `jesse_trainer_addon.py` to integrate with your existing ML orchestrator:

1. Add the `JesseTrainer` class to your `ml_orchestrator.py` file
2. Update the `MLTrainingOrchestrator.__init__` method
3. Add the `_train_jesse` method
4. Update `run_complete_training_pipeline` to include Jesse training
5. Update `train_specific_component` to include "jesse" option
6. Update `get_training_status` to include Jesse status
7. Update usage instructions in `train_bot.py`

#### Option 2: Add a new training command

Add a new command to your `train_bot.py`:

```python
# In train_bot.py, update the main function
async def main():
    # ... existing code ...
    
    if len(sys.argv) > 1:
        component = sys.argv[1].lower()
        if component == "jesse":
            print("🎯 Training Jesse component...")
            from jesse_trainer_addon import JesseTrainer
            jesse_trainer = JesseTrainer()
            await jesse_trainer.run_complete_pipeline()
        # ... existing code ...
```

### Step 4: Run the Combined Training Pipeline

Train your chatbot with both Dalaal Street and Jesse data:

```bash
python train_bot.py all
```

Or train only the Jesse component:

```bash
python train_bot.py jesse
```

## Technical Implementation Details

### Data Processing

The Jesse test files are processed in several steps:
1. Test files are scanned using Python's AST (Abstract Syntax Tree)
2. Test functions and assertions are extracted
3. Conversation templates are generated based on the test cases
4. Templates are formatted for TensorFlow training and Groq fine-tuning

### Model Architecture

The test category classifier uses:
- Embedding layer for text representation
- Bidirectional LSTM layers for sequence processing
- Dense layers with dropout for classification

### Integration with Groq

For Groq integration, the system:
1. Formats conversation templates as JSONL files
2. Creates training and evaluation datasets
3. Prepares fine-tuning instructions

## Usage Example

Once integrated, you can ask Jesse-related questions:

```
User: How does Jesse test indicators?

Bot: In Jesse's testing framework, the `test_indicators` test (from `test_indicators.py`) verifies the functionality of the indicators component.

Here's what this test checks:
- It tests various technical indicators including MACD, RSI, and Bollinger Bands
- It verifies both single value and sequential output modes
- It ensures indicator calculations are accurate by comparing with expected values

This test is important because it ensures that the indicators functionality works correctly, which is crucial for Jesse's algorithmic trading operations.

To run this specific test, you could use:
```python
pytest tests/test_indicators.py::test_indicators
```

Would you like me to explain any specific indicator test in more detail?
```

## Troubleshooting

### Common Issues

1. **Path Issues**: Ensure the Jesse path and Desktop path are correctly set in all files.

2. **Import Errors**: If you get import errors, check that the system paths are correctly added:
   ```python
   sys.path.append(JESSE_PATH)
   sys.path.append(DESKTOP_PATH)
   ```

3. **TensorFlow Errors**: If you encounter TensorFlow errors, ensure you have the correct version installed:
   ```bash
   pip install tensorflow==2.9.0
   ```

4. **Memory Issues**: If you run out of memory during training, try reducing batch size or model complexity:
   ```python
   self.batch_size = 16  # Lower from 32
   self.lstm_units = 64  # Lower from 128
   ```

### Getting Help

If you encounter issues with the integration:
1. Check the generated integration guide at `ml_training/models/jesse/jesse_integration_guide.md`
2. Consult the integration report at `ml_training/data/jesse/jesse_integration_report.md`
3. Review the training history plots in `ml_training/models/jesse/`

## Maintenance and Updates

### Keeping the Model Updated

When Jesse's test files change:
1. Re-run the test integrator: `python jesse_test_integrator.py`
2. Re-train the models: `python jesse_ml_trainer.py`
3. Update your chatbot: `python train_bot.py jesse`

### Performance Monitoring

Monitor the performance of Jesse-related queries:
1. Track accuracy of category classification
2. Collect user feedback on response quality
3. Periodically review conversation logs

## Resources

- [Jesse Documentation](https://docs.jesse.trade)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Groq Fine-tuning Guide](https://console.groq.com/docs/fine-tuning)

## License

This integration is provided under the same license as your Dalaal Street chatbot and Jesse projects.
