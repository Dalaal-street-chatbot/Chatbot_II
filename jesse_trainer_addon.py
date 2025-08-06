# Add Jesse training component to ml_orchestrator.py

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Jesse paths
JESSE_PATH = r'c:\Users\hatao\Downloads\jesse-master\jesse-master'
DESKTOP_PATH = r'c:\Users\hatao\OneDrive\Desktop'

class JesseTrainer:
    """Component for handling Jesse test training"""
    
    def __init__(self):
        self.jesse_path = JESSE_PATH
        self.data_path = os.path.join(DESKTOP_PATH, 'ml_training', 'data', 'jesse')
        self.models_path = os.path.join(DESKTOP_PATH, 'ml_training', 'models', 'jesse')
        
        # Create directories if they don't exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
    
    async def collect_training_data(self) -> bool:
        """Collect training data from Jesse test files"""
        print("📊 Collecting Jesse test training data")
        
        try:
            # Run the Jesse test integrator script
            integrator_path = os.path.join(DESKTOP_PATH, 'jesse_test_integrator.py')
            process = await asyncio.create_subprocess_exec(
                sys.executable, integrator_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                print(f"❌ Error collecting Jesse test data: {stderr.decode()}")
                return False
            
            print("✅ Jesse test data collected successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error collecting Jesse test data: {e}")
            return False
    
    async def train_models(self) -> bool:
        """Train models on Jesse test data"""
        print("🏋️ Training models on Jesse test data")
        
        try:
            # Run the Jesse ML trainer script
            trainer_path = os.path.join(DESKTOP_PATH, 'jesse_ml_trainer.py')
            process = await asyncio.create_subprocess_exec(
                sys.executable, trainer_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                print(f"❌ Error training Jesse models: {stderr.decode()}")
                return False
            
            print("✅ Jesse models trained successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error training Jesse models: {e}")
            return False
    
    async def prepare_groq_data(self) -> bool:
        """Prepare Jesse data for Groq fine-tuning"""
        print("🔄 Preparing Jesse data for Groq fine-tuning")
        
        try:
            # Check if Jesse conversation data exists
            conversation_file = os.path.join(self.data_path, 'jesse_conversation_data.json')
            if not os.path.exists(conversation_file):
                print(f"❌ Jesse conversation data not found at {conversation_file}")
                return False
            
            # Check if Jesse Groq data exists
            groq_file = os.path.join(self.data_path, 'jesse_groq_data.jsonl')
            if not os.path.exists(groq_file):
                print(f"❌ Jesse Groq data not found at {groq_file}")
                return False
            
            # Copy the Jesse Groq data to the main Groq directory
            main_groq_dir = os.path.join(DESKTOP_PATH, 'ml_training', 'data')
            target_file = os.path.join(main_groq_dir, 'jesse_groq_data.jsonl')
            
            # Read Jesse Groq data
            with open(groq_file, 'r') as f:
                jesse_data = [json.loads(line) for line in f.readlines()]
            
            # Write to target file
            with open(target_file, 'w') as f:
                for item in jesse_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"✅ Jesse Groq data prepared: {len(jesse_data)} examples")
            return True
            
        except Exception as e:
            print(f"❌ Error preparing Jesse Groq data: {e}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get Jesse training status"""
        return {
            "jesse_data_collected": os.path.exists(os.path.join(self.data_path, 'jesse_conversation_data.json')),
            "jesse_models_trained": os.path.exists(os.path.join(self.models_path, 'jesse_test_category_classifier')),
            "jesse_groq_data_prepared": os.path.exists(os.path.join(self.data_path, 'jesse_groq_data.jsonl')),
            "jesse_integration_guide_created": os.path.exists(os.path.join(self.models_path, 'jesse_integration_guide.md'))
        }
    
    async def run_complete_pipeline(self) -> bool:
        """Run the complete Jesse training pipeline"""
        
        # Step 1: Collect training data
        if not await self.collect_training_data():
            return False
        
        # Step 2: Train models
        if not await self.train_models():
            return False
        
        # Step 3: Prepare Groq data
        if not await self.prepare_groq_data():
            return False
        
        return True

# Instructions for adding this component to ml_orchestrator.py:
"""
1. Add the JesseTrainer class to your ml_orchestrator.py file

2. Update the MLTrainingOrchestrator.__init__ method to include:
    self.jesse_trainer = JesseTrainer()

3. Add a method to train Jesse component:
    async def _train_jesse(self):
        """Train the Jesse component"""
        
        print("  • Training Jesse component...")
        success = await self.jesse_trainer.run_complete_pipeline()
        
        if success:
            print("    ✓ Jesse component trained successfully")
        else:
            print("    ✗ Jesse component training failed")
        
        return success

4. Update run_complete_training_pipeline to include Jesse training:
    async def run_complete_training_pipeline(self):
        # ... existing code ...
        
        # Step 8: Train Jesse Component
        print("\n🛠️ Step 8: Training Jesse Component")
        await self._train_jesse()
        
        # ... existing code ...

5. Update train_specific_component to include "jesse" option:
    async def train_specific_component(self, component: str):
        # ... existing code ...
        
        elif component == "jesse":
            await self._train_jesse()
        
        # ... existing code ...

6. Update get_training_status to include Jesse status:
    def get_training_status(self) -> Dict[str, Any]:
        status = {
            # ... existing status items ...
        }
        
        # Add Jesse status
        jesse_status = self.jesse_trainer.get_training_status()
        status.update(jesse_status)
        
        return status

7. Update the usage instructions in train_bot.py:
    def show_usage():
        print('''
        Usage: python train_bot.py [component]

        Components:
          # ... existing components ...
          jesse     - Train Jesse test component only
        ''')
"""
