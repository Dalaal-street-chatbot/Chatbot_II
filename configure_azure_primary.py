#!/usr/bin/env python3
"""
Configuration Update Script - Make Azure OpenAI Primary NLP Service
Updates comprehensive_chat.py to prioritize Azure OpenAI
"""

import os
import sys
from datetime import datetime

def create_backup(file_path):
    """Create a backup of the original file"""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        with open(file_path, 'r') as original:
            content = original.read()
        with open(backup_path, 'w') as backup:
            backup.write(content)
        print(f"✅ Backup created: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return False

def update_comprehensive_chat():
    """Update comprehensive_chat.py to use Azure OpenAI as primary"""
    
    file_path = "vscode-vfs://github/Dalaal-street-chatbot/Chatbot_II/app/services/comprehensive_chat.py"
    
    print("🔧 Updating comprehensive_chat.py to use Azure OpenAI as primary...")
    
    # Create backup first
    if not create_backup(file_path):
        return False
    
    try:
        # Read current file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Define the new comprehensive chat service with Azure OpenAI priority
        new_comprehensive_chat = '''async def get_comprehensive_financial_response(
    query: str,
    user_context: Optional[Dict] = None,
    use_azure_priority: bool = True
) -> Dict[str, Any]:
    """
    Enhanced financial assistant with Azure OpenAI priority
    
    Args:
        query: User's financial query
        user_context: Optional user context and preferences
        use_azure_priority: Whether to prioritize Azure OpenAI (default: True)
    
    Returns:
        Comprehensive response with financial insights
    """
    
    try:
        # Initialize response structure
        response = {
            "status": "success",
            "query": query,
            "response": "",
            "service_used": "",
            "timestamp": datetime.now().isoformat(),
            "context": user_context or {},
            "confidence": 0.0,
            "sources": []
        }
        
        # Service priority chain (Azure OpenAI first, no Groq)
        service_chain = [
            ("azure_openai", "Azure OpenAI - Enterprise Grade"),
            ("google_ai", "Google AI - Gemini Pro"),
            ("deepseek", "DeepSeek - Alternative"),
            ("local_ollama", "Local Ollama - Fallback")
        ] if use_azure_priority else [
            ("google_ai", "Google AI - Gemini Pro"),
            ("azure_openai", "Azure OpenAI - Enterprise Grade"),
            ("deepseek", "DeepSeek - Alternative"),
            ("local_ollama", "Local Ollama - Fallback")
        ]
        
        # Enhance query with financial context
        enhanced_query = f"""
        Financial Query: {query}
        
        Context: You are a specialized Indian stock market AI assistant. Provide accurate, 
        actionable financial insights with focus on NSE/BSE stocks, market indices (NIFTY 50, SENSEX, BANK NIFTY), 
        and current market conditions.
        
        Guidelines:
        - Provide specific stock recommendations when appropriate
        - Include risk disclaimers for investment advice
        - Reference current market trends and news when relevant
        - Use technical and fundamental analysis perspectives
        - Keep responses practical and actionable
        
        Query: {query}
        """
        
        # Try each service in priority order
        for service_name, service_description in service_chain:
            try:
                print(f"🔄 Trying {service_description}...")
                
                if service_name == "azure_openai":
                    # Use Azure OpenAI Cloud Service
                    from app.services.cloud_ai_services import AzureOpenAIService
                    azure_service = AzureOpenAIService()
                    
                    result = await azure_service.get_financial_insights(
                        query=enhanced_query,
                        context=user_context
                    )
                    
                    if result and result.get("status") == "success":
                        response.update({
                            "response": result.get("insights", ""),
                            "service_used": f"Azure OpenAI ({result.get('model', 'gpt-4')})",
                            "confidence": 0.95,  # High confidence for Azure OpenAI
                            "sources": ["Azure OpenAI API", "Financial Analysis Engine"],
                            "analysis": result.get("analysis", {}),
                            "recommendations": result.get("recommendations", [])
                        })
                        print(f"✅ Success with {service_description}")
                        return response
                        
                elif service_name == "google_ai":
                    # Use Google AI Service
                    google_response = await get_google_ai_response(enhanced_query)
                    
                    if google_response and google_response.get("status") == "success":
                        response.update({
                            "response": google_response.get("response", ""),
                            "service_used": "Google AI (Gemini Pro)",
                            "confidence": 0.80,
                            "sources": ["Google AI API", "Gemini Pro Model"]
                        })
                        print(f"✅ Success with {service_description}")
                        return response
                        
                elif service_name == "deepseek":
                    # Use DeepSeek Service (if available)
                    deepseek_response = await get_deepseek_response(enhanced_query)
                    
                    if deepseek_response and deepseek_response.get("status") == "success":
                        response.update({
                            "response": deepseek_response.get("response", ""),
                            "service_used": "DeepSeek AI",
                            "confidence": 0.75,
                            "sources": ["DeepSeek API"]
                        })
                        print(f"✅ Success with {service_description}")
                        return response
                        
                elif service_name == "local_ollama":
                    # Use Local Ollama Service
                    ollama_response = await get_ollama_response(enhanced_query)
                    
                    if ollama_response and ollama_response.get("status") == "success":
                        response.update({
                            "response": ollama_response.get("response", ""),
                            "service_used": "Local Ollama",
                            "confidence": 0.70,
                            "sources": ["Local Ollama Instance"]
                        })
                        print(f"✅ Success with {service_description}")
                        return response
                
            except Exception as e:
                print(f"⚠️ {service_description} failed: {str(e)}")
                continue
        
        # If all services fail, return fallback response
        response.update({
            "status": "partial_failure",
            "response": """I apologize, but I'm currently experiencing technical difficulties with my AI services. 
            
However, I can still help you with basic financial information:
            
📈 **Market Status**: Please check NSE/BSE websites for current market status
🔍 **Stock Research**: Use reliable sources like MoneyControl, ET Markets
📊 **Technical Analysis**: Consider consulting financial advisors for investment decisions
⚠️ **Risk Warning**: All investments carry risk. Please do your own research.

Please try again in a few moments, or contact support if this issue persists.""",
            "service_used": "Fallback Response",
            "confidence": 0.30,
            "sources": ["System Fallback"]
        })
        
        return response
        
    except Exception as e:
        # Error response
        return {
            "status": "error",
            "query": query,
            "response": f"I encountered an error while processing your request: {str(e)}. Please try again.",
            "service_used": "Error Handler",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }'''
        
        # Update the file content
        # Find and replace the existing function
        import re
        
        # Pattern to match the existing function
        pattern = r'async def get_comprehensive_financial_response\([\s\S]*?(?=\n\nasync def|\n\ndef|\Z)'
        
        # Replace with new function
        updated_content = re.sub(pattern, new_comprehensive_chat, content, flags=re.MULTILINE)
        
        # If pattern not found, append the function
        if updated_content == content:
            updated_content = content + "\n\n" + new_comprehensive_chat
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print("✅ comprehensive_chat.py updated successfully!")
        print("🔵 Azure OpenAI is now the primary NLP service")
        print("📋 Service priority order (Groq removed):")
        print("   1. Azure OpenAI (Enterprise Grade)")
        print("   2. Google AI (Gemini Pro)")
        print("   3. DeepSeek (Alternative)")
        print("   4. Local Ollama (Fallback)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update comprehensive_chat.py: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Azure OpenAI Primary Configuration Update")
    print("=" * 50)
    
    success = update_comprehensive_chat()
    
    if success:
        print("\n✅ Configuration update completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Run test_all_apis.py to verify all services work")
        print("2. Test the chatbot with financial queries")
        print("3. Monitor Azure OpenAI usage and performance")
        print("4. Configure rate limiting if needed")
        
    else:
        print("\n❌ Configuration update failed!")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
