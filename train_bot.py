#!/usr/bin/env python3
"""
Dalaal Street Chatbot ML Training Script
This script trains the chatbot on Upstox and Groq API functionalities
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_training.ml_orchestrator import ml_orchestrator

async def main():
    """Main training function"""
    
    print("🚀 Dalaal Street Chatbot ML Training")
    print("=" * 50)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check current training status
    print("📋 Checking current training status...")
    status = ml_orchestrator.get_training_status()
    
    for component, is_complete in status.items():
        status_icon = "✅" if is_complete else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}")
    
    print()
    
    # Ask user what to train
    if len(sys.argv) > 1:
        component = sys.argv[1].lower()
        if component == "all":
            print("🎯 Running complete training pipeline...")
            await ml_orchestrator.run_complete_training_pipeline()
        else:
            print(f"🎯 Training {component} component...")
            await ml_orchestrator.train_specific_component(component)
    else:
        print("🎯 Running complete training pipeline...")
        await ml_orchestrator.run_complete_training_pipeline()
    
    print()
    print("✨ Training completed successfully!")
    print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show final status
    print("\n📋 Final training status:")
    final_status = ml_orchestrator.get_training_status()
    for component, is_complete in final_status.items():
        status_icon = "✅" if is_complete else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}")

def show_usage():
    """Show usage instructions"""
    print("""
Usage: python train_bot.py [component]

Components:
  all       - Run complete training pipeline (default)
  data      - Collect training data only
  intent    - Train intent classifier only
  market    - Train market predictor only  
  response  - Train response generator only
  groq      - Prepare Groq fine-tuning data only

Examples:
  python train_bot.py           # Train everything
  python train_bot.py all       # Train everything
  python train_bot.py data      # Collect data only
  python train_bot.py groq      # Prepare Groq data only
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_usage()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
