"""
Finance Trainer Addon for ML Orchestrator
This module integrates the Finance dataset trainer with the main ML orchestrator
"""

import os
import asyncio
from finance_chatbot_trainer import FinanceTrainer

class FinanceTrainerAddon:
    def __init__(self):
        self.trainer = FinanceTrainer()
        print("Finance trainer addon initialized")
    
    async def run_complete_pipeline(self):
        """Run the complete finance training pipeline asynchronously"""
        # Run the trainer in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.trainer.run_complete_pipeline)
        return result
    
    async def verify_integration(self):
        """Verify that the integration is working properly"""
        # Check for essential files
        try:
            model_path = os.path.join(self.trainer.models_path, 'finance_intent_classifier.keras')
            guide_path = os.path.join(self.trainer.models_path, 'finance_integration_guide.md')
            encoder_path = os.path.join(self.trainer.models_path, 'label_encoder.pkl')
            response_path = os.path.join(self.trainer.models_path, 'response_data.json')
            
            files_exist = (
                os.path.exists(model_path) and
                os.path.exists(guide_path) and
                os.path.exists(encoder_path) and
                os.path.exists(response_path)
            )
            
            if files_exist:
                print("[SUCCESS] Finance chatbot integration verified successfully")
                return True
            else:
                print("[ERROR] Finance chatbot integration verification failed - missing files")
                return False
                
        except Exception as e:
            print(f"[ERROR] Verification error: {e}")
            return False
