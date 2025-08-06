#!/usr/bin/env python3
"""
Jesse Integration Tester
Runs a simple test to verify Jesse integration with Dalaal Street Chatbot
"""

import os
import sys
import asyncio

# Add necessary paths
JESSE_PATH = r'c:\Users\hatao\Downloads\jesse-master\jesse-master'
DESKTOP_PATH = r'c:\Users\hatao\OneDrive\Desktop'

# Add paths to system path
sys.path.append(JESSE_PATH)
sys.path.append(DESKTOP_PATH)

async def run_jesse_test_integrator():
    """Run Jesse test integrator"""
    print("\n[INFO] Running Jesse Test Integrator")
    print("=" * 60)
    
    # Import the module and run
    try:
        from jesse_test_integrator import JesseTestIntegrator
        integrator = JesseTestIntegrator()
        await integrator.run_full_integration()
        print("[SUCCESS] Jesse Test Integrator completed successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Jesse Test Integrator failed: {str(e)}")
        return False

async def run_jesse_ml_trainer():
    """Run Jesse ML trainer"""
    print("\n[INFO] Running Jesse ML Trainer")
    print("=" * 60)
    
    # Import the module and run
    try:
        from jesse_ml_trainer import JesseMLTrainer
        trainer = JesseMLTrainer()
        await trainer.run_full_training()
        print("[SUCCESS] Jesse ML Trainer completed successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Jesse ML Trainer failed: {str(e)}")
        return False

async def main():
    """Main execution function"""
    print("[INFO] Jesse Integration Tester")
    print("=" * 60)
    
    # Step 1: Run Jesse Test Integrator
    test_integrator_success = await run_jesse_test_integrator()
    
    if not test_integrator_success:
        print("\n[ERROR] Integration test failed at Test Integrator stage")
        return
    
    # Step 2: Run Jesse ML Trainer
    ml_trainer_success = await run_jesse_ml_trainer()
    
    if not ml_trainer_success:
        print("\n[ERROR] Integration test failed at ML Trainer stage")
        return
    
    # Step 3: Check if integration files exist
    jesse_groq_data = os.path.join(DESKTOP_PATH, 'ml_training', 'data', 'jesse', 'jesse_groq_data.jsonl')
    jesse_model = os.path.join(DESKTOP_PATH, 'ml_training', 'models', 'jesse', 'jesse_test_category_classifier')
    
    if os.path.exists(jesse_groq_data) and os.path.exists(jesse_model):
        print("\n[SUCCESS] Jesse integration test completed successfully!")
        print("\nThe following files were created:")
        print(f"- Groq training data: {jesse_groq_data}")
        print(f"- ML model: {jesse_model}")
        
        print("\nTo use these in your chatbot, add the following code to your ml_orchestrator.py:")
        print('''
from jesse_trainer_addon import JesseTrainer

# In __init__:
self.jesse_trainer = JesseTrainer()

# Add this method:
async def _train_jesse(self):
    """Train the Jesse component"""
    
    print("  • Training Jesse component...")
    success = await self.jesse_trainer.run_complete_pipeline()
    
    if success:
        print("    [SUCCESS] Jesse component trained successfully")
    else:
        print("    [ERROR] Jesse component training failed")
    
    return success

# In run_complete_training_pipeline:
# Step: Train Jesse Component
print("\\n[INFO] Training Jesse Component")
await self._train_jesse()

# In train_specific_component:
elif component == "jesse":
    await self._train_jesse()
        ''')
    else:
        print("\n[ERROR] Integration test failed: Expected files not created")

if __name__ == "__main__":
    asyncio.run(main())
