#!/usr/bin/env python3
"""
Jesse ML Integrator for Dalaal Street Chatbot
This script adds Jesse training integration to the existing ML orchestrator
"""

import os
import sys
import json
from typing import Dict, Any
import asyncio
from datetime import datetime

# Add necessary paths
JESSE_PATH = r'c:\Users\hatao\Downloads\jesse-master\jesse-master'
DESKTOP_PATH = r'c:\Users\hatao\OneDrive\Desktop'

# Add paths to system path
sys.path.append(JESSE_PATH)
sys.path.append(DESKTOP_PATH)

# Try to import ml_orchestrator from existing project
try:
    from ml_training.ml_orchestrator import ml_orchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    print("Warning: Could not import ml_orchestrator, some functionality may be limited")
    ORCHESTRATOR_AVAILABLE = False
    ml_orchestrator = None

# Import Jesse trainer addon
from jesse_trainer_addon import JesseTrainer

class JesseMLIntegrator:
    """Class to integrate Jesse training into the ML orchestrator"""
    
    def __init__(self):
        self.jesse_trainer = JesseTrainer()
        self.ml_orchestrator = ml_orchestrator if ORCHESTRATOR_AVAILABLE else None
        self.backup_path = os.path.join(DESKTOP_PATH, 'ml_training', 'backup')
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_path, exist_ok=True)
    
    async def backup_ml_orchestrator(self):
        """Create a backup of the current ml_orchestrator.py file"""
        print("📂 Creating backup of ml_orchestrator.py")
        
        ml_orchestrator_path = os.path.join(DESKTOP_PATH, 'ml_training', 'ml_orchestrator.py')
        if not os.path.exists(ml_orchestrator_path):
            print("❌ ml_orchestrator.py not found. Skipping backup.")
            return False
        
        # Create timestamped backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.backup_path, f'ml_orchestrator_backup_{timestamp}.py')
        
        # Copy the file
        try:
            with open(ml_orchestrator_path, 'r') as src_file:
                content = src_file.read()
            
            with open(backup_file, 'w') as dest_file:
                dest_file.write(content)
                
            print(f"✅ Backup created at {backup_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to create backup: {str(e)}")
            return False
    
    def update_ml_orchestrator(self):
        """Update ml_orchestrator.py to include Jesse training"""
        print("🔄 Updating ml_orchestrator.py to include Jesse training")
        
        ml_orchestrator_path = os.path.join(DESKTOP_PATH, 'ml_training', 'ml_orchestrator.py')
        if not os.path.exists(ml_orchestrator_path):
            print("❌ ml_orchestrator.py not found. Cannot update.")
            return False
        
        # Read the current ml_orchestrator.py file
        with open(ml_orchestrator_path, 'r') as f:
            content = f.read()
        
        # Check if Jesse integration is already added
        if 'JesseTrainer' in content:
            print("ℹ️ Jesse integration is already added to ml_orchestrator.py")
            return True
        
        # Check for key patterns to make additions
        imports_pattern = 'import asyncio'
        imports_addition = """import asyncio
from jesse_trainer_addon import JesseTrainer"""
        
        init_pattern = '    def __init__(self'
        init_addition = """        # Jesse trainer
        self.jesse_trainer = JesseTrainer()"""
        
        train_method_pattern = '    async def run_complete_training_pipeline'
        train_method_addition = """        # Step: Train Jesse Component
        print("\\n🛠️ Training Jesse Component")
        await self._train_jesse()"""
        
        specific_component_pattern = '    async def train_specific_component(self, component: str'
        specific_component_addition = """        elif component == "jesse":
            await self._train_jesse()"""
        
        status_pattern = '    def get_training_status'
        status_addition = """        # Add Jesse status
        jesse_status = self.jesse_trainer.get_training_status()
        status.update(jesse_status)"""
        
        # Add _train_jesse method after a specific pattern
        train_pattern = '    async def train_specific_component'
        train_jesse_method = """
    async def _train_jesse(self):
        """Train the Jesse component"""
        
        print("  • Training Jesse component...")
        success = await self.jesse_trainer.run_complete_pipeline()
        
        if success:
            print("    ✓ Jesse component trained successfully")
        else:
            print("    ✗ Jesse component training failed")
        
        return success
"""
        
        # Make the updates
        content = content.replace(imports_pattern, imports_addition)
        
        # Add init code
        init_lines = content.split('\n')
        for i, line in enumerate(init_lines):
            if init_pattern in line:
                # Find the end of __init__ method to add our code
                for j in range(i+1, len(init_lines)):
                    if init_lines[j].strip() and not init_lines[j].startswith(' ' * 8):
                        # Insert our addition before this line
                        init_lines.insert(j, init_addition)
                        break
                break
        
        # Reconstruct content with updated init
        content = '\n'.join(init_lines)
        
        # Add training step to run_complete_training_pipeline
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if train_method_pattern in line:
                # Find end of method to add our code before return statement
                for j in range(i+1, len(lines)):
                    if 'return' in lines[j] and lines[j].strip().startswith('return'):
                        # Insert our addition before return
                        lines.insert(j, train_method_addition)
                        break
                break
        
        # Reconstruct content with updated training pipeline
        content = '\n'.join(lines)
        
        # Add jesse to train_specific_component
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if specific_component_pattern in line:
                # Find an elif statement to insert after
                for j in range(i+1, len(lines)):
                    if 'elif component ==' in lines[j]:
                        # Insert our addition after an existing elif
                        lines.insert(j+1, specific_component_addition)
                        break
                break
        
        # Reconstruct content with updated specific component
        content = '\n'.join(lines)
        
        # Add jesse status to get_training_status
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if status_pattern in line:
                # Find return statement to insert before
                for j in range(i+1, len(lines)):
                    if 'return status' in lines[j]:
                        # Insert our addition before return
                        lines.insert(j, status_addition)
                        break
                break
        
        # Reconstruct content with updated status
        content = '\n'.join(lines)
        
        # Add _train_jesse method
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if train_pattern in line:
                # Find the end of train_specific_component method
                for j in range(i+1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith(' ' * 4):
                        # Insert our method before this line
                        lines.insert(j, train_jesse_method)
                        break
                break
        
        # Reconstruct content with _train_jesse method
        content = '\n'.join(lines)
        
        # Write the updated content back to ml_orchestrator.py
        with open(ml_orchestrator_path, 'w') as f:
            f.write(content)
        
        print("✅ ml_orchestrator.py updated with Jesse training integration")
        return True
    
    def update_train_bot(self):
        """Update train_bot.py to include Jesse option"""
        print("🔄 Updating train_bot.py to include Jesse option")
        
        train_bot_path = os.path.join(DESKTOP_PATH, 'train_bot.py')
        if not os.path.exists(train_bot_path):
            print("❌ train_bot.py not found. Cannot update.")
            return False
        
        # Read the current train_bot.py file
        with open(train_bot_path, 'r') as f:
            content = f.read()
        
        # Check if Jesse option is already added
        if 'component == "jesse"' in content:
            print("ℹ️ Jesse option is already added to train_bot.py")
            return True
        
        # Add jesse to usage message
        usage_pattern = 'def show_usage'
        usage_addition = """          jesse     - Train Jesse test component only"""
        
        # Make the update
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if usage_pattern in line:
                # Find the end of usage components list
                for j in range(i+1, len(lines)):
                    if ']' in lines[j] or '```' in lines[j]:
                        # Insert our addition before this line
                        lines.insert(j, usage_addition)
                        break
                break
        
        # Reconstruct content with updated usage
        content = '\n'.join(lines)
        
        # Write the updated content back to train_bot.py
        with open(train_bot_path, 'w') as f:
            f.write(content)
        
        print("✅ train_bot.py updated with Jesse training option")
        return True
    
    def verify_jesse_training(self):
        """Verify Jesse training is working"""
        print("🔍 Verifying Jesse training integration")
        
        # Check for required files
        files_to_check = [
            (os.path.join(DESKTOP_PATH, 'jesse_test_integrator.py'), "Jesse test integrator"),
            (os.path.join(DESKTOP_PATH, 'jesse_ml_trainer.py'), "Jesse ML trainer"),
            (os.path.join(DESKTOP_PATH, 'jesse_trainer_addon.py'), "Jesse trainer addon"),
            (os.path.join(DESKTOP_PATH, 'ml_training', 'ml_orchestrator.py'), "ML orchestrator")
        ]
        
        all_files_present = True
        for file_path, file_desc in files_to_check:
            if not os.path.exists(file_path):
                print(f"❌ {file_desc} not found at {file_path}")
                all_files_present = False
        
        if not all_files_present:
            print("❌ Some required files are missing")
            return False
        
        # Check ml_orchestrator.py for Jesse integration
        with open(os.path.join(DESKTOP_PATH, 'ml_training', 'ml_orchestrator.py'), 'r') as f:
            content = f.read()
            
        if 'JesseTrainer' not in content:
            print("❌ Jesse integration not found in ml_orchestrator.py")
            return False
        
        if 'async def _train_jesse' not in content:
            print("❌ _train_jesse method not found in ml_orchestrator.py")
            return False
        
        # All checks passed
        print("✅ Jesse training integration verified successfully")
        return True
    
    async def run_test_integration(self):
        """Test Jesse integration by running a small test"""
        print("🧪 Testing Jesse integration")
        
        if not ORCHESTRATOR_AVAILABLE:
            print("⚠️ ML orchestrator not available, skipping integration test")
            return False
        
        try:
            # Initialize the orchestrator
            print("  • Initializing ML orchestrator")
            ml_orchestrator.init()
            
            # Run Jesse training
            print("  • Running Jesse training")
            result = await ml_orchestrator.train_specific_component("jesse")
            
            if result:
                print("✅ Jesse integration test passed")
                return True
            else:
                print("❌ Jesse integration test failed")
                return False
        except Exception as e:
            print(f"❌ Error testing Jesse integration: {str(e)}")
            return False
    
    async def run_integration(self):
        """Run the full integration process"""
        print("🚀 Starting Jesse ML Integration")
        print("=" * 60)
        
        # Step 1: Backup ml_orchestrator.py
        await self.backup_ml_orchestrator()
        
        # Step 2: Update ml_orchestrator.py
        self.update_ml_orchestrator()
        
        # Step 3: Update train_bot.py
        self.update_train_bot()
        
        # Step 4: Verify Jesse training integration
        integration_verified = self.verify_jesse_training()
        
        if integration_verified:
            print("\n🎉 Jesse ML Integration Completed Successfully!")
            print("\nYou can now run Jesse training with:")
            print("  python train_bot.py jesse")
        else:
            print("\n❌ Jesse ML Integration Failed")
            print("\nPlease check the error messages above and try again")

async def main():
    """Main execution function"""
    
    print("🚀 Jesse ML Integrator for Dalaal Street Chatbot")
    print("=" * 60)
    print(f"Integration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    integrator = JesseMLIntegrator()
    await integrator.run_integration()
    
    print()
    print(f"Integration finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())
