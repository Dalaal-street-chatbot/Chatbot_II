# Azure Functions Adaptation Guide for FastAPI Backend

## ⚠️ Current Challenges

Your FastAPI application has:
- **7 different endpoints** (chat, stock, indices, news, analysis, chart-data, google-cloud-chat)
- **Complex service dependencies** (GROQ, Google Cloud, UPSTOX)
- **Session management** for chat functionality
- **CORS middleware** for frontend integration

## 🔄 Required Changes for Azure Functions

### 1. Split Each Endpoint into Separate Functions

**Current structure:**
```
app/api/routes.py (7 endpoints in one file)
```

**Functions structure needed:**
```
functions/
├── chat/
│   ├── __init__.py
│   └── function.json
├── stock/
│   ├── __init__.py  
│   └── function.json
├── chart_data/
│   ├── __init__.py
│   └── function.json
├── news/
│   ├── __init__.py
│   └── function.json
├── analysis/
│   ├── __init__.py
│   └── function.json
├── indices/
│   ├── __init__.py
│   └── function.json
├── google_cloud_chat/
│   ├── __init__.py
│   └── function.json
├── requirements.txt
└── host.json
```

### 2. Example Function Implementation

Here's how your chat endpoint would look as an Azure Function:

**functions/chat/__init__.py:**
```python
import azure.functions as func
import json
import logging
from app.services.groq_service import groq_service

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Chat function processed a request.')
    
    try:
        # Get request data
        req_body = req.get_json()
        message = req_body.get('message')
        session_id = req_body.get('session_id')
        
        # Process with your existing service
        response = groq_service.get_response(message, session_id)
        
        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            headers={
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
    except Exception as e:
        logging.error(f"Error in chat function: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error"}),
            status_code=500
        )
```

**functions/chat/function.json:**
```json
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "authLevel": "anonymous",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": ["post", "options"],
      "route": "chat"
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
```

### 3. Update Frontend API Configuration

You'd need to update your `api.ts` to point to individual function URLs:

```typescript
// Base URL for Azure Functions
const FUNCTIONS_BASE_URL = 'https://your-function-app.azurewebsites.net/api';

export const API_ENDPOINTS = {
  CHAT: `${FUNCTIONS_BASE_URL}/chat`,
  GOOGLE_CLOUD_CHAT: `${FUNCTIONS_BASE_URL}/google-cloud-chat`,
  STOCK: `${FUNCTIONS_BASE_URL}/stock`,
  INDICES: `${FUNCTIONS_BASE_URL}/indices`,
  CHART_DATA: `${FUNCTIONS_BASE_URL}/chart-data`,
  NEWS: `${FUNCTIONS_BASE_URL}/news`,
  ANALYSIS: `${FUNCTIONS_BASE_URL}/analysis`,
} as const;
```

## ⏱️ **Time Investment:**

- **Azure Functions refactoring**: 2-3 days of work
- **Azure Container Apps deployment**: 2-3 hours
- **Azure App Service deployment**: 1-2 hours

## 🎯 **My Recommendation:**

**Skip Azure Functions for now.** Here's why:

1. **Your current FastAPI app works perfectly as-is**
2. **Container Apps gives you serverless benefits without refactoring**
3. **You can always migrate to Functions later if needed**

## 🚀 **Quick Container Apps Deployment:**

```bash
# This will deploy your current FastAPI app as-is
az containerapp up \
  --name dalaal-street-api \
  --resource-group dalaal-street-bot \
  --source . \
  --target-port 8000 \
  --ingress external
```

## 🤔 **Final Decision Guide:**

**Choose Azure Functions if:**
- ✅ You want to learn Azure Functions
- ✅ You have time for refactoring
- ✅ You plan to add more event-driven features

**Choose Container Apps if:**
- ✅ You want to deploy quickly (recommended)
- ✅ You want serverless scaling
- ✅ You don't want to change your code

**What would you prefer to do?**
