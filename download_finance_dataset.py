"""
Dataset downloader for finance-alpaca dataset from Hugging Face
This script downloads the gbharti/finance-alpaca dataset and prepares it for training
"""

import os
from datasets import load_dataset
import json
from tqdm import tqdm

# Create directories if they don't exist
os.makedirs('ml_training/data/finance', exist_ok=True)

print("Downloading gbharti/finance-alpaca dataset from Hugging Face...")
try:
    # Download the dataset
    # Note: This requires being logged in with `huggingface-cli login`
    ds = load_dataset("gbharti/finance-alpaca")
    print(f"Dataset downloaded successfully. Contains {len(ds['train'])} training examples.")

    # Save the raw dataset
    raw_data_path = 'ml_training/data/finance/finance_alpaca_raw.json'
    with open(raw_data_path, 'w', encoding='utf-8') as f:
        json.dump(ds['train'], f, indent=2)
    print(f"Raw dataset saved to {raw_data_path}")

    # Process into a format suitable for our chatbot
    processed_data = []
    for item in tqdm(ds['train']):
        # Format each example as a conversation
        conversation = {
            "id": f"finance_{item.get('id', len(processed_data))}",
            "conversation": [
                {
                    "role": "user",
                    "content": item["instruction"]
                },
                {
                    "role": "assistant",
                    "content": item["output"]
                }
            ]
        }
        processed_data.append(conversation)

    # Save the processed data
    processed_data_path = 'ml_training/data/finance/finance_conversation_data.json'
    with open(processed_data_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    print(f"Processed dataset saved to {processed_data_path}")

    # Create a jsonl file for possible fine-tuning
    jsonl_path = 'ml_training/data/finance/finance_groq_data.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    print(f"JSONL dataset saved to {jsonl_path}")

except Exception as e:
    print(f"Error downloading or processing dataset: {e}")
    print("Make sure you're logged in with `huggingface-cli login` and have access to the dataset.")
