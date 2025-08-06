from typing import Dict, List, Optional, Any, Tuple
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.groq_service import groq_service
from app.services.ai_services import codestral_service, deepseek_service, ollama_service
from app.services.cloud_ai_services import azure_openai_service, google_ai_service
from app.services.enhanced_news_service import enhanced_news_service
from app.services.google_cloud_services import google_cloud_services
from app.services.financial_chatbot_integration import financial_chatbot
from market_data import market_service

class ComprehensiveChatService:
    """Orchestrates multiple AI services for comprehensive financial assistance"""
    
    def __init__(self):
        self.primary_ai = groq_service  # Groq as main NLP AI
        self.services = {
            'groq': groq_service,
            'codestral': codestral_service,
            'deepseek': deepseek_service,
            'azure': azure_openai_service,
            'google': google_ai_service,
            'ollama': ollama_service
        }
    
    async def process_comprehensive_query(
        self, 
        user_message: str, 
        session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process query using multiple AI services for comprehensive response"""
        
        try:
            # Step 1: Analyze query intent and complexity
            analysis = await self.primary_ai.analyze_financial_query(user_message)
            
            # Step 2: Gather relevant data based on intent
            context_data = await self._gather_context_data(analysis, user_message)
            
            # Step 3: Determine which AI services to use
            services_to_use = self._select_ai_services(analysis, user_message)
            
            # Step 4: Generate responses from multiple services
            responses = await self._generate_multi_service_responses(
                user_message, 
                context_data, 
                services_to_use,
                analysis
            )
            
            # Step 5: Synthesize final response
            final_response = await self._synthesize_final_response(
                user_message,
                responses,
                context_data,
                analysis
            )
            
            return {
                'response': final_response,
                'intent': analysis['intent'],
                'entities': analysis.get('entities', []),
                'confidence': analysis.get('confidence', 0.0),
                'services_used': list(services_to_use.keys()),
                'context_data': context_data,
                'analysis': analysis
            }
            
        except Exception as e:
            print(f"Error in comprehensive chat processing: {e}")
            # Fallback to primary AI only
            fallback_response = await self.primary_ai.generate_response(user_message)
            return {
                'response': fallback_response,
                'intent': 'general_info',
                'entities': [],
                'confidence': 0.5,
                'services_used': ['groq'],
                'error': str(e)
            }
    
    async def _gather_context_data(
        self, 
        analysis: Dict[str, Any], 
        user_message: str
    ) -> Dict[str, Any]:
        """Gather relevant context data based on query analysis"""
        
        context = {}
        intent = analysis.get('intent', 'general_info')
        entities = analysis.get('entities', [])
        
        try:
            # Store entities in context for services
            context['entities'] = entities
            
            # Gather stock data if relevant
            if intent in ['stock_price', 'market_analysis', 'stock_forecast', 'technical_analysis'] or any(
                entity.upper() in ['NIFTY', 'SENSEX', 'RELIANCE', 'TCS', 'HDFC'] 
                for entity in entities
            ):
                if entities:
                    for entity in entities[:3]:  # Limit to 3 stocks
                        stock_data = market_service.get_stock_price(entity.upper())
                        if 'error' not in stock_data:
                            context[f'stock_{entity}'] = stock_data
                
                # Always include major indices
                context['market_indices'] = market_service.get_market_indices()
                
                # If technical analysis intent, get historical data from Google Cloud BigQuery
                if intent == 'technical_analysis' and entities:
                    for entity in entities[:1]:  # Just the first stock for technical analysis
                        try:
                            history_data = await google_cloud_services.get_market_history(entity.upper(), 30)
                            if history_data.get('status') == 'success':
                                context[f'history_{entity}'] = history_data.get('history', [])
                        except Exception as e:
                            print(f"Error getting historical data: {e}")
            
            # Gather news if relevant
            if intent in ['news', 'market_analysis', 'sentiment_analysis']:
                if entities:
                    # Get company-specific news
                    for entity in entities[:2]:
                        news_data = await enhanced_news_service.get_enhanced_company_news(entity, page_size=3)
                        context[f'news_{entity}'] = news_data
                else:
                    # Get general market news
                    context['market_news'] = await enhanced_news_service.get_enhanced_financial_news(page_size=5)
                
                # Add market sentiment for analysis
                if intent == 'sentiment_analysis':
                    try:
                        sentiment_data = await enhanced_news_service.get_market_sentiment_analysis()
                        context['market_sentiment'] = sentiment_data
                    except AttributeError:
                        # Fallback to a basic sentiment implementation
                        context['market_sentiment'] = {'sentiment': 'neutral', 'score': 0.5}
                    
                    # Add expert opinions for deeper analysis
                    try:
                        expert_opinions = await enhanced_news_service.get_expert_opinions(limit=3)
                        context['expert_opinions'] = expert_opinions
                    except AttributeError:
                        # Skip expert opinions if not available
                        pass
                        
            # Add forecast data for prediction intents
            if intent in ['forecast', 'prediction', 'market_prediction'] and entities:
                symbol = entities[0].upper()
                try:
                    forecast_data = await google_cloud_services.generate_market_forecast(symbol)
                    if forecast_data.get('status') == 'success':
                        context[f'forecast_{symbol}'] = forecast_data.get('forecast', {})
                except Exception as e:
                    print(f"Error getting forecast data: {e}")
            
            # Add time-sensitive context
            context['query_timestamp'] = str(asyncio.get_event_loop().time())
            
        except Exception as e:
            print(f"Error gathering context data: {e}")
            context['error'] = 'Some context data unavailable'
        
        return context
    
    def _select_ai_services(
        self, 
        analysis: Dict[str, Any], 
        user_message: str
    ) -> Dict[str, Any]:
        """Select appropriate AI services based on query characteristics"""
        
        services = {'groq': self.services['groq']}  # Always include primary
        intent = analysis.get('intent', 'general_info')
        urgency = analysis.get('urgency', 'low')
        
        # Add services based on intent and complexity
        if 'code' in user_message.lower() or 'algorithm' in user_message.lower():
            services['codestral'] = self.services['codestral']
        
        if 'analysis' in user_message.lower() or 'reasoning' in user_message.lower():
            services['deepseek'] = self.services['deepseek']
        
        if urgency == 'high' or 'enterprise' in user_message.lower():
            services['azure'] = self.services['azure']
        
        if 'prediction' in user_message.lower() or 'forecast' in user_message.lower():
            services['google'] = self.services['google']
            
        # Add Google Cloud services for specific use cases
        lower_msg = user_message.lower()
        if ('chart' in lower_msg or 'image' in lower_msg or 'document' in lower_msg):
            services['gcloud_vision'] = google_cloud_services
        
        if ('dialogflow' in lower_msg or 'conversation' in lower_msg or 'finance chat' in lower_msg):
            services['gcloud_dialogflow'] = google_cloud_services
            
        if ('market trend' in lower_msg or 'trend analysis' in lower_msg or 'technical analysis' in lower_msg):
            services['gcloud_vertexai'] = google_cloud_services
            
        if ('big data' in lower_msg or 'data warehouse' in lower_msg or 'historical data' in lower_msg):
            services['gcloud_bigquery'] = google_cloud_services
            
        # For comprehensive financial analysis, use our integrated financial chatbot
        if ('financial analysis' in lower_msg or 'comprehensive' in lower_msg or 
            'complete analysis' in lower_msg or 'full market view' in lower_msg):
            services['financial_chatbot'] = financial_chatbot
        
        # Use local AI for privacy-sensitive queries
        if 'confidential' in lower_msg or 'private' in lower_msg:
            services['ollama'] = self.services['ollama']
        
        return services
    
    async def _generate_multi_service_responses(
        self,
        user_message: str,
        context_data: Dict[str, Any],
        services: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate responses from multiple AI services"""
        
        responses = {}
        
        # Create tasks for parallel execution
        tasks = []
        
        for service_name, service in services.items():
            if service_name == 'groq':
                task = self._call_groq_service(user_message, context_data)
            elif service_name == 'codestral':
                task = self._call_codestral_service(user_message, analysis)
            elif service_name == 'deepseek':
                task = self._call_deepseek_service(user_message, context_data)
            elif service_name == 'azure':
                task = self._call_azure_service(user_message, context_data)
            elif service_name == 'google':
                task = self._call_google_service(user_message, context_data)
            elif service_name == 'ollama':
                task = self._call_ollama_service(user_message)
            # Add Google Cloud services
            elif service_name == 'gcloud_vision':
                task = self._call_gcloud_vision_service(user_message, context_data)
            elif service_name == 'gcloud_dialogflow':
                task = self._call_gcloud_dialogflow_service(user_message, context_data)
            elif service_name == 'gcloud_vertexai':
                task = self._call_gcloud_vertexai_service(user_message, context_data)
            elif service_name == 'gcloud_bigquery':
                task = self._call_gcloud_bigquery_service(user_message, context_data)
            elif service_name == 'financial_chatbot':
                task = self._call_financial_chatbot_service(user_message, context_data)
            else:
                continue
            
            tasks.append((service_name, task))
        
        # Execute tasks with timeout
        for service_name, task in tasks:
            try:
                response = await asyncio.wait_for(task, timeout=15.0)
                responses[service_name] = response
            except asyncio.TimeoutError:
                responses[service_name] = f"{service_name} service timeout"
            except Exception as e:
                responses[service_name] = f"{service_name} service error: {str(e)}"
        
        return responses
    
    async def _call_groq_service(self, message: str, context: Dict[str, Any]) -> str:
        """Call Groq service"""
        return await groq_service.generate_response(message, context)
    
    async def _call_codestral_service(self, message: str, analysis: Dict[str, Any]) -> str:
        """Call Codestral service if code-related"""
        if 'code' in message.lower() or 'algorithm' in message.lower():
            result = await codestral_service.generate_financial_code(message)
            return result.get('code', 'Code generation failed')
        return "Not applicable for this query"
    
    async def _call_deepseek_service(self, message: str, context: Dict[str, Any]) -> str:
        """Call DeepSeek service for deep analysis"""
        result = await deepseek_service.deep_financial_analysis(message, context)
        return result.get('analysis', 'Deep analysis failed')
    
    async def _call_azure_service(self, message: str, context: Dict[str, Any]) -> str:
        """Call Azure OpenAI service"""
        result = await azure_openai_service.generate_financial_insights(message, context)
        return result.get('insights', 'Azure insights failed')
    
    async def _call_google_service(self, message: str, context: Dict[str, Any]) -> str:
        """Call Google AI service"""
        if 'prediction' in message.lower() or 'forecast' in message.lower():
            result = await google_ai_service.generate_market_predictions(context)
            return result.get('predictions', 'Prediction generation failed')
        return "Not applicable for this query"
    
    async def _call_ollama_service(self, message: str) -> str:
        """Call local Ollama service"""
        result = await ollama_service.local_analysis(message)
        return result.get('response', 'Local analysis failed')
        
    async def _call_gcloud_vision_service(self, message: str, context: Dict[str, Any]) -> str:
        """Call Google Cloud Vision service for image/chart analysis"""
        # Check if we have image data in the context
        image_data = context.get('image_data')
        if image_data:
            result = await google_cloud_services.analyze_chart_image(image_data)
            if result.get('status') == 'success':
                return f"Chart Analysis: {result.get('chart_data', {}).get('text_content', 'No analysis available')}"
            else:
                return f"Chart analysis error: {result.get('message', 'Unknown error')}"
        else:
            return "Chart analysis requires image data"
    
    async def _call_gcloud_dialogflow_service(self, message: str, context: Dict[str, Any]) -> str:
        """Call Google Cloud Dialogflow service for conversational finance"""
        # Generate session ID from user context if available
        session_id = context.get('session_id', f"session_{str(asyncio.get_event_loop().time())}")
        result = await google_cloud_services.handle_finance_query(session_id, message)
        if result.get('status') == 'success':
            return result.get('message', 'No response from Dialogflow')
        else:
            return f"Dialogflow error: {result.get('message', 'Unknown error')}"
    
    async def _call_gcloud_vertexai_service(self, message: str, context: Dict[str, Any]) -> str:
        """Call Google Cloud Vertex AI service for market trends and forecasts"""
        # Check if we have a specific stock symbol in context
        symbols = context.get('entities', [])
        if symbols and len(symbols) > 0:
            symbol = symbols[0]
            result = await google_cloud_services.generate_market_forecast(symbol)
            if result.get('status') == 'success':
                forecast = result.get('forecast', {})
                return f"Market Forecast for {symbol}: {forecast}"
            else:
                return f"Market forecast error: {result.get('message', 'Unknown error')}"
        else:
            # General market trend analysis
            market_data = context.get('market_indices', {})
            if market_data:
                # Convert to list format for analysis
                market_list = [{'symbol': k, 'price': v.get('price', 0), 'volume': v.get('volume', 0)} 
                              for k, v in market_data.items()]
                result = await google_cloud_services.analyze_market_trend(market_list)
                if result.get('status') == 'success':
                    analysis = result.get('analysis', {})
                    return f"Market Trend Analysis: {analysis}"
                else:
                    return f"Market trend analysis error: {result.get('message', 'Unknown error')}"
            else:
                return "Market analysis requires market data"
    
    async def _call_gcloud_bigquery_service(self, message: str, context: Dict[str, Any]) -> str:
        """Call Google Cloud BigQuery service for historical data analysis"""
        # Check if we have specific symbols to analyze
        symbols = context.get('entities', [])
        if symbols and len(symbols) > 0:
            symbol = symbols[0]
            days = 30  # Default to 30 days of history
            
            # Check if the message specifies a time period
            if "year" in message.lower() or "365" in message:
                days = 365
            elif "quarter" in message.lower() or "90" in message:
                days = 90
            elif "month" in message.lower() or "30" in message:
                days = 30
            elif "week" in message.lower() or "7" in message:
                days = 7
            
            result = await google_cloud_services.get_market_history(symbol, days)
            if result.get('status') == 'success':
                history = result.get('history', [])
                count = result.get('count', 0)
                return f"Retrieved {count} days of historical data for {symbol}"
            else:
                return f"Historical data retrieval error: {result.get('message', 'Unknown error')}"
        else:
            # Try to run a general market query
            result = await google_cloud_services.query_market_data(
                "SELECT AVG(close_price) as avg_price, MAX(volume) as max_volume " +
                "FROM market_data.daily_prices " +
                "WHERE date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY) " +
                "GROUP BY symbol LIMIT 5"
            )
            if result.get('status') == 'success':
                return f"Market query returned {result.get('row_count', 0)} results"
            else:
                return f"Market query error: {result.get('message', 'Unknown error')}"
    
    async def _call_financial_chatbot_service(self, message: str, context: Dict[str, Any]) -> str:
        """Call the integrated financial chatbot service"""
        session_id = context.get('session_id', f"session_{str(asyncio.get_event_loop().time())}")
        result = await financial_chatbot.process_user_message(
            message=message,
            session_id=session_id,
            user_data=context
        )
        
        if result.get('status') == 'success':
            return result.get('response', 'No response from financial chatbot')
        else:
            return f"Financial chatbot error: {result.get('message', 'Unknown error')}"
    
    async def _synthesize_final_response(
        self,
        user_message: str,
        responses: Dict[str, Any],
        context_data: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """Synthesize final response from multiple AI outputs"""
        
        # Use Groq as the primary synthesizer
        synthesis_prompt = f"""
        Synthesize a comprehensive response to: "{user_message}"
        
        Available AI responses:
        {self._format_responses_for_synthesis(responses)}
        
        Context data available:
        {list(context_data.keys())}
        
        Provide a unified, coherent response that:
        1. Directly answers the user's question
        2. Incorporates the best insights from multiple AI services
        3. Includes relevant data and context
        4. Maintains professional financial advisory tone
        5. Includes appropriate disclaimers
        
        Keep the response comprehensive but not overwhelming.
        """
        
        try:
            final_response = await groq_service.generate_response(
                synthesis_prompt,
                context=context_data,
                system_prompt="You are a master financial synthesizer. Create unified responses from multiple AI insights."
            )
            return final_response
        
        except Exception as e:
            # Fallback to primary response
            return responses.get('groq', 'I apologize, but I encountered an issue processing your request.')
    
    def _format_responses_for_synthesis(self, responses: Dict[str, Any]) -> str:
        """Format responses for synthesis prompt"""
        formatted = []
        for service, response in responses.items():
            if isinstance(response, str) and len(response) > 10:
                formatted.append(f"{service.upper()}: {response[:500]}...")
        return '\n\n'.join(formatted)

# Create global instance
comprehensive_chat = ComprehensiveChatService()
