#!/usr/bin/env python
"""
Main script to run the finance dataset download and training process
"""

import os
import asyncio
import sys
from tqdm import tqdm
import time

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    banner = """
    ┌────────────────────────────────────────────────┐
    │                                                │
    │   Finance-Alpaca Dataset Training Pipeline     │
    │                                                │
    │   Dalaal Street Chatbot - Finance Edition      │
    │                                                │
    └────────────────────────────────────────────────┘
    """
    print(banner)

async def main():
    print_banner()
    
    # Step 1: Download the Finance-Alpaca dataset
    print("\n[Step 1/3] Downloading Finance-Alpaca dataset...")
    
    # Run the dataset downloader script
    try:
        print("Running download_finance_dataset.py...")
        from download_finance_dataset import main as download_main
        download_main()
        print("[SUCCESS] Dataset downloaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print("Please make sure you're logged in with `huggingface-cli login`")
        return
    
    # Step 2: Train the finance model
    print("\n[Step 2/3] Training finance chatbot model...")
    
    try:
        from finance_chatbot_trainer import FinanceTrainer
        
        trainer = FinanceTrainer()
        result = trainer.run_complete_pipeline()
        
        if not result:
            print("[ERROR] Failed to train finance model")
            return
        
        print("[SUCCESS] Finance model trained successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to train model: {e}")
        return
    
    # Step 3: Integrate with ML orchestrator
    print("\n[Step 3/3] Integrating with ML orchestrator...")
    
    try:
        # Create necessary directories
        os.makedirs("ml_training/models/finance", exist_ok=True)
        
        print("Testing the finance model integration...")
        from finance_trainer_addon import FinanceTrainerAddon
        
        trainer_addon = FinanceTrainerAddon()
        result = await trainer_addon.verify_integration()
        
        if result:
            print("[SUCCESS] Finance trainer integration verified!")
        else:
            print("[ERROR] Finance trainer integration verification failed")
            return
    except Exception as e:
        print(f"[ERROR] Failed to integrate with ML orchestrator: {e}")
        return
    
    print("\n=== All steps completed successfully! ===")
    print("\nThe Finance-Alpaca dataset has been downloaded, processed, and a model has been trained.")
    print("You can now use this model with your Dalaal Street chatbot.")
    print("\nTo run the complete training pipeline including this model, use:")
    print("    python ml_training/run_ml_training.py")
    print("\nTo train just the finance component, use:")
    print("    python ml_training/run_ml_training.py --component finance")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
