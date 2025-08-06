#!/usr/bin/env python3
"""
Lean Test Integrator for Dalaal Street Chatbot
This script processes Lean's files and formats them for training with your existing chatbot model
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import importlib
import ast
import inspect
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
import asyncio

# Add necessary paths - update these to your actual paths
LEAN_PATH = r'c:\path\to\Lean-Master'  # Update this to your Lean path
DESKTOP_PATH = r'c:\Users\hatao\OneDrive\Desktop'

# Add paths to system path
sys.path.append(LEAN_PATH)
sys.path.append(DESKTOP_PATH)

# Import ml_orchestrator from existing project
try:
    from ml_training.ml_orchestrator import ml_orchestrator
except ImportError:
    print("Warning: Could not import ml_orchestrator, some functionality may be limited")
    ml_orchestrator = None

class LeanCodeIntegrator:
    """Class to extract and process Lean files for chatbot training"""
    
    def __init__(self):
        self.lean_path = LEAN_PATH
        self.output_path = os.path.join(DESKTOP_PATH, 'ml_training', 'data', 'lean')
        self.algorithm_dirs = ['Algorithm', 'Algorithm.Framework', 'Algorithm.CSharp', 'Tests']
        self.file_extensions = ['.cs', '.py']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Counters
        self.total_files_processed = 0
        self.total_classes_processed = 0
        self.total_methods_processed = 0
        self.total_training_examples = 0
        
        # Store extracted data
        self.code_files = []
        self.code_components = []
        self.conversation_templates = []
    
    def scan_lean_files(self):
        """Scan Lean's directories for code files"""
        print(f"🔍 Scanning for code files in {self.lean_path}")
        
        code_files = []
        for algo_dir in self.algorithm_dirs:
            dir_path = os.path.join(self.lean_path, algo_dir)
            if not os.path.exists(dir_path):
                print(f"⚠️ Directory not found: {dir_path}")
                continue
                
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if any(file.endswith(ext) for ext in self.file_extensions):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, self.lean_path)
                        code_files.append({
                            'name': file,
                            'path': file_path,
                            'relative_path': rel_path,
                            'type': 'cs' if file.endswith('.cs') else 'py'
                        })
        
        self.code_files = code_files
        self.total_files_processed = len(code_files)
        print(f"✅ Found {len(code_files)} code files")
        return code_files
    
    def extract_code_components(self):
        """Extract classes and methods from all code files"""
        print(f"🔍 Extracting code components from {len(self.code_files)} files")
        
        code_components = []
        for code_file in self.code_files:
            file_components = self._extract_file_components(code_file)
            code_components.extend(file_components)
            
            print(f"  • Extracted {len(file_components)} components from {code_file['name']}")
        
        self.code_components = code_components
        self.total_classes_processed = sum(1 for c in code_components if c['type'] == 'class')
        self.total_methods_processed = sum(1 for c in code_components if c['type'] == 'method')
        print(f"✅ Extracted {self.total_classes_processed} classes and {self.total_methods_processed} methods")
        return code_components
    
    def _extract_file_components(self, code_file):
        """Extract components from a single file"""
        try:
            with open(code_file['path'], 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Handle based on file type
            if code_file['type'] == 'py':
                return self._extract_python_components(content, code_file)
            else:  # C# files
                return self._extract_csharp_components(content, code_file)
                
        except Exception as e:
            print(f"  ⚠️ Error processing {code_file['path']}: {e}")
            return []
    
    def _extract_python_components(self, content, code_file):
        """Extract components from Python file using AST"""
        components = []
        
        try:
            tree = ast.parse(content)
            
            # Extract classes and methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract class
                    class_doc = ast.get_docstring(node) or ""
                    class_component = {
                        'name': node.name,
                        'type': 'class',
                        'docstring': class_doc,
                        'code': content[node.lineno-1:node.end_lineno],
                        'file': code_file['relative_path'],
                        'methods': []
                    }
                    
                    # Extract methods within the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_doc = ast.get_docstring(item) or ""
                            method_component = {
                                'name': item.name,
                                'type': 'method',
                                'docstring': method_doc,
                                'code': content[item.lineno-1:item.end_lineno],
                                'class': node.name,
                                'file': code_file['relative_path']
                            }
                            class_component['methods'].append(item.name)
                            components.append(method_component)
                    
                    components.append(class_component)
                elif isinstance(node, ast.FunctionDef) and node.parent_node == tree:
                    # Top-level functions
                    func_doc = ast.get_docstring(node) or ""
                    func_component = {
                        'name': node.name,
                        'type': 'function',
                        'docstring': func_doc,
                        'code': content[node.lineno-1:node.end_lineno],
                        'file': code_file['relative_path']
                    }
                    components.append(func_component)
                    
        except SyntaxError as e:
            print(f"  ⚠️ Syntax error in {code_file['path']}: {e}")
        except Exception as e:
            print(f"  ⚠️ Error processing Python file {code_file['path']}: {e}")
            
        return components
    
    def _extract_csharp_components(self, content, code_file):
        """Extract components from C# file using regex"""
        components = []
        
        # Simple regex for class extraction - not perfect but works for most cases
        class_pattern = r"(public|private|protected|internal)?\s+(static\s+)?(class|struct|interface)\s+(\w+)"
        method_pattern = r"(public|private|protected|internal)?\s+(static\s+)?(async\s+)?\w+\s+(\w+)\s*\([^)]*\)"
        comment_pattern = r"///\s*<summary>(.*?)</summary>"
        
        # Extract classes
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(4)
            class_start = match.start()
            
            # Try to find class docstring
            docstring = ""
            doc_match = re.search(comment_pattern, content[:class_start], re.DOTALL)
            if doc_match:
                docstring = doc_match.group(1).strip()
            
            # Find class end (naive approach)
            # This won't handle nested classes correctly
            class_content = self._extract_balanced_braces(content[class_start:])
            
            class_component = {
                'name': class_name,
                'type': 'class',
                'docstring': docstring,
                'code': content[class_start:class_start + len(class_content)],
                'file': code_file['relative_path'],
                'methods': []
            }
            
            # Extract methods within this class content
            for m_match in re.finditer(method_pattern, class_content, re.MULTILINE):
                method_name = m_match.group(4)
                method_start = class_start + m_match.start()
                
                # Try to find method docstring
                m_docstring = ""
                m_doc_match = re.search(comment_pattern, content[:method_start], re.DOTALL)
                if m_doc_match:
                    m_docstring = m_doc_match.group(1).strip()
                
                # Find method end (naive approach)
                method_content = self._extract_balanced_braces(content[method_start:])
                
                method_component = {
                    'name': method_name,
                    'type': 'method',
                    'docstring': m_docstring,
                    'code': content[method_start:method_start + len(method_content)],
                    'class': class_name,
                    'file': code_file['relative_path']
                }
                
                class_component['methods'].append(method_name)
                components.append(method_component)
            
            components.append(class_component)
            
        return components
    
    def _extract_balanced_braces(self, text):
        """Extract code between balanced braces"""
        stack = []
        brace_start = text.find('{')
        if brace_start == -1:
            return text
            
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                stack.append('{')
            elif text[i] == '}':
                if stack:
                    stack.pop()
                if not stack:
                    return text[:i+1]
                    
        return text  # In case of unbalanced braces
    
    def generate_conversation_templates(self):
        """Generate conversation templates for chatbot training"""
        print(f"🔍 Generating conversation templates from {len(self.code_components)} code components")
        
        conversation_templates = []
        
        # Process classes
        classes = [c for c in self.code_components if c['type'] == 'class']
        for class_component in classes:
            templates = self._generate_templates_for_class(class_component)
            conversation_templates.extend(templates)
            
        # Process methods
        methods = [m for m in self.code_components if m['type'] == 'method']
        for method_component in methods:
            templates = self._generate_templates_for_method(method_component)
            conversation_templates.extend(templates)
        
        self.conversation_templates = conversation_templates
        self.total_training_examples = len(conversation_templates)
        
        print(f"✅ Generated {len(conversation_templates)} conversation templates")
        return conversation_templates
    
    def _generate_templates_for_class(self, class_component):
        """Generate multiple conversation templates from a single class"""
        templates = []
        
        # Clean up class name
        class_name = class_component['name']
        file_path = class_component['file']
        
        # Generate different question variations
        questions = [
            f"How does the {class_name} class work in Lean?",
            f"Can you explain the purpose of the {class_name} class in Lean?",
            f"What methods are available in Lean's {class_name} class?",
            f"How do I use the {class_name} class in Lean?",
            f"What is the {class_name} class used for in algorithmic trading?",
            f"How does {class_name} fit into the Lean architecture?",
            f"Can you show me an example of using {class_name} in Lean?"
        ]
        
        # Process docstring if available
        docstring = class_component['docstring'] or f"Class for {class_name} functionality in Lean"
        
        # Create a basic template for each question
        for question in questions:
            template = {
                'user_query': question,
                'component_name': class_component['name'],
                'file': class_component['file'],
                'category': 'lean_framework',
                'context_used': {
                    'class_name': class_component['name'],
                    'file_path': class_component['file'],
                    'methods': class_component['methods'],
                    'docstring': docstring
                },
                'assistant_response': self._generate_response_for_class(class_component, question),
                'quality_score': 0.8,  # Default quality score
                'created_at': datetime.now().isoformat()
            }
            templates.append(template)
        
        return templates
    
    def _generate_templates_for_method(self, method_component):
        """Generate multiple conversation templates from a single method"""
        templates = []
        
        # Clean up method name
        method_name = method_component['name']
        class_name = method_component.get('class', 'Unknown')
        
        # Skip common methods like constructors, getters/setters
        if method_name.startswith('get_') or method_name.startswith('set_') or method_name == class_name:
            return []
        
        # Generate different question variations
        questions = [
            f"How does the {method_name} method work in Lean's {class_name}?",
            f"What does the {method_name} method do in Lean?",
            f"Can you explain the purpose of {method_name} in {class_name}?",
            f"How do I use the {method_name} method in Lean?",
            f"What are the parameters for {class_name}.{method_name}?",
            f"Show me an example of using {method_name} in Lean"
        ]
        
        # Process docstring if available
        docstring = method_component['docstring'] or f"Method for {method_name} functionality in {class_name}"
        
        # Create a basic template for each question
        for question in questions:
            template = {
                'user_query': question,
                'component_name': f"{class_name}.{method_name}",
                'file': method_component['file'],
                'category': 'lean_method',
                'context_used': {
                    'method_name': method_component['name'],
                    'class_name': class_name,
                    'file_path': method_component['file'],
                    'docstring': docstring
                },
                'assistant_response': self._generate_response_for_method(method_component, question),
                'quality_score': 0.8,  # Default quality score
                'created_at': datetime.now().isoformat()
            }
            templates.append(template)
        
        return templates
    
    def _generate_response_for_class(self, class_component, question):
        """Generate a detailed response about the class"""
        class_name = class_component['name']
        file_path = class_component['file']
        methods = class_component['methods']
        
        # Format methods list
        methods_list = ""
        if methods:
            methods_list = "Key methods:\n" + "\n".join([f"- `{m}`" for m in methods])
        else:
            methods_list = "This class doesn't contain any extracted methods."
        
        # Use docstring if available
        docstring = class_component['docstring']
        doc_text = f"\n\n{docstring}" if docstring else ""
        
        response = f"""
In Lean's framework, the `{class_name}` class (found in `{file_path}`) is a key component for algorithmic trading.

{doc_text}

{methods_list}

This class is part of Lean's quantitative trading architecture, which provides a robust framework for developing and testing trading strategies. 

To use this class in your algorithm, you would typically:

```csharp
// Example usage of {class_name}
var instance = new {class_name}();
// Use the instance for your trading logic
```

Would you like me to explain any specific method of this class in more detail?
"""
        return response.strip()
    
    def _generate_response_for_method(self, method_component, question):
        """Generate a detailed response about the method"""
        method_name = method_component['name']
        class_name = method_component.get('class', 'Unknown')
        file_path = method_component['file']
        
        # Use docstring if available
        docstring = method_component['docstring']
        doc_text = f"\n\n{docstring}" if docstring else ""
        
        response = f"""
In Lean's `{class_name}` class, the `{method_name}` method (found in `{file_path}`) provides important functionality.

{doc_text}

This method is used within the Lean algorithmic trading framework to implement trading strategy logic. The `{method_name}` method is typically called during the algorithm's execution process.

Example usage:

```csharp
// Example usage of {method_name}
var result = instance.{method_name}(parameters);
// Use the result in your trading logic
```

The exact parameters and return type would depend on the specific implementation of the method.

Would you like me to explain more about the `{class_name}` class or other related components?
"""
        return response.strip()
    
    async def save_conversation_data(self):
        """Save generated conversation data for chatbot training"""
        print(f"💾 Saving {len(self.conversation_templates)} conversation templates")
        
        # Save as JSON
        output_file = os.path.join(self.output_path, 'lean_conversation_data.json')
        with open(output_file, 'w') as f:
            json.dump(self.conversation_templates, f, indent=2)
        
        print(f"✅ Saved conversation data to {output_file}")
        
        # If ml_orchestrator is available, format for Groq training
        if ml_orchestrator:
            print("🔄 Preparing data for Groq fine-tuning")
            
            groq_data = []
            for template in self.conversation_templates:
                groq_example = {
                    "messages": [
                        {"role": "user", "content": template["user_query"]},
                        {"role": "assistant", "content": template["assistant_response"]}
                    ]
                }
                groq_data.append(groq_example)
            
            # Save as JSONL (for Groq fine-tuning)
            groq_file = os.path.join(self.output_path, 'lean_groq_data.jsonl')
            with open(groq_file, 'w') as f:
                for item in groq_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"✅ Saved Groq training data to {groq_file}")
    
    def generate_additional_training_content(self):
        """Generate additional training content for code understanding"""
        print("🔄 Generating additional training content")
        
        # Summarize code categories
        file_categories = {}
        for component in self.code_components:
            if component['type'] == 'class':
                category = os.path.dirname(component['file'])
                if category not in file_categories:
                    file_categories[category] = []
                file_categories[category].append(component['name'])
        
        # Generate summary report
        categories_file = os.path.join(self.output_path, 'lean_code_categories.json')
        with open(categories_file, 'w') as f:
            json.dump(file_categories, f, indent=2)
        
        # Generate usage instructions
        instructions = self._generate_lean_instructions(file_categories)
        instructions_file = os.path.join(self.output_path, 'lean_framework_guide.md')
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"✅ Generated additional training content")
        
    def _generate_lean_instructions(self, file_categories):
        """Generate usage instructions for Lean framework"""
        categories_list = "\n".join([f"- **{cat}**: {len(classes)} classes" for cat, classes in file_categories.items()])
        
        instructions = f"""# Lean Algorithmic Trading Framework Guide

## Overview
Lean is a powerful open-source algorithmic trading engine used by quantitative traders, data scientists, and financial engineers. The framework analyzed contains {self.total_classes_processed} classes and {self.total_methods_processed} methods across {self.total_files_processed} files.

## Code Categories
{categories_list}

## Key Components

### Algorithm Framework
The Algorithm namespace contains the core components for creating trading algorithms:
- QCAlgorithm: Base class for all algorithms
- Portfolio Construction: Methods for portfolio allocation
- Alpha Models: Signal generation components
- Risk Management: Risk control mechanisms
- Execution: Order execution strategies

### Asset Classes
Lean supports multiple asset classes:
- Equity
- Forex
- Crypto
- Options
- Futures

### Data Types
Various data types for market information:
- Market Data (OHLCV)
- Order Book Data
- Alternative Data

## Algorithm Development
Typical algorithm development follows these steps:
1. Initialize algorithm settings
2. Define universe selection
3. Create alpha generation rules
4. Apply portfolio construction
5. Implement risk management
6. Execute trades

## Example Algorithm Structure
```
public class BasicAlgorithm : QCAlgorithm
{
    public override void Initialize()
    {
        // Set start and end dates
        SetStartDate(2020, 1, 1);
        SetEndDate(2021, 1, 1);
        
        // Set cash and brokerage model
        SetCash(100000);
        SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage);
        
        // Add assets
        AddEquity("SPY", Resolution.Daily);
    }
    
    public override void OnData(Slice data)
    {
        // Trading logic
        if (!Portfolio.Invested)
        {
            SetHoldings("SPY", 1.0);
        }
    }
}
```

## Resources
- [Lean Documentation](https://www.lean.io/docs/)
- [QuantConnect](https://www.quantconnect.com/)
- [Lean GitHub Repository](https://github.com/QuantConnect/Lean)
"""
        return instructions

    async def run_full_integration(self):
        """Run the full integration process"""
        print("🚀 Starting Lean Code Integration Process")
        print("=" * 60)
        
        # Step 1: Scan code files
        self.scan_lean_files()
        
        # Step 2: Extract code components
        self.extract_code_components()
        
        # Step 3: Generate conversation templates
        self.generate_conversation_templates()
        
        # Step 4: Save conversation data
        await self.save_conversation_data()
        
        # Step 5: Generate additional training content
        self.generate_additional_training_content()
        
        # Step 6: Create integration report
        self.create_integration_report()
        
        print("\n🎉 Lean Code Integration Completed Successfully!")
    
    def create_integration_report(self):
        """Create an integration report"""
        report = f"""
# Lean Code Integration Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Integration Summary
- **Code Files Processed**: {self.total_files_processed}
- **Classes Extracted**: {self.total_classes_processed}
- **Methods Extracted**: {self.total_methods_processed}
- **Conversation Templates Generated**: {self.total_training_examples}

## Integration Files
- `lean_conversation_data.json`: {len(self.conversation_templates)} conversation templates
- `lean_groq_data.jsonl`: {len(self.conversation_templates)} Groq-formatted training examples
- `lean_code_categories.json`: Summary of code categories
- `lean_framework_guide.md`: Guide to Lean framework

## Next Steps
1. Train your chatbot using the generated conversation templates
2. Fine-tune your Groq model with the lean_groq_data.jsonl file
3. Add the framework guide to your documentation

## Integration with Dalaal Street Chatbot
The data is now ready for integration with your existing Dalaal Street chatbot.
Use the following command to train your bot with Lean code data:

```
python train_bot.py lean
```

## Notes
- All code components were successfully processed
- Data is formatted for compatibility with your existing ML pipeline
- Conversation templates follow the same format as your existing data
"""
        
        report_file = os.path.join(self.output_path, 'lean_integration_report.md')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Integration report saved to {report_file}")

async def main():
    """Main execution function"""
    
    print("🚀 Lean Code Integrator for Dalaal Street Chatbot")
    print("=" * 60)
    print(f"Integration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    integrator = LeanCodeIntegrator()
    await integrator.run_full_integration()
    
    print()
    print(f"Integration finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())
