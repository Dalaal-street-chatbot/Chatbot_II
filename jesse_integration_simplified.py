#!/usr/bin/env python3
"""
Simplified Jesse Integration Tester
"""

import os
import sys

# Add necessary paths
JESSE_PATH = r'c:\Users\hatao\Downloads\jesse-master\jesse-master'
DESKTOP_PATH = r'c:\Users\hatao\OneDrive\Desktop'

# Add paths to system path
sys.path.append(JESSE_PATH)
sys.path.append(DESKTOP_PATH)

def main():
    print("[INFO] Testing Jesse ML Integration")
    
    # Create ml_training/models/jesse directory if it doesn't exist
    model_dir = os.path.join(DESKTOP_PATH, 'ml_training', 'models', 'jesse')
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if the model file exists
    model_path = os.path.join(model_dir, 'jesse_test_category_classifier.keras')
    if os.path.exists(model_path):
        print(f"[SUCCESS] Model file found: {model_path}")
    else:
        print(f"[WARNING] Model file not found: {model_path}")
    
    # Check if integration guide exists
    guide_path = os.path.join(model_dir, 'jesse_integration_guide.md')
    if os.path.exists(guide_path):
        print(f"[SUCCESS] Integration guide found: {guide_path}")
    else:
        print(f"[WARNING] Integration guide not found: {guide_path}")

    print("[INFO] Integration testing complete")

if __name__ == "__main__":
    main()
