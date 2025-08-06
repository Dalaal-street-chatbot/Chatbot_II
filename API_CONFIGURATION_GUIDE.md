# 🎯 API Configuration Guide for Dalaal Street Chatbot

## ✅ What's Already Configured

Your frontend is now properly configured with centralized API management:

### 📁 New Files Created:
- `src/config/api.ts` - Centralized API configuration
- `.env.example.frontend` - Environment variables template

### 🔧 Updated Components:
- `TradingChart.tsx` - Now uses API_ENDPOINTS.CHART_DATA
- `ChatWindow.tsx` - Now uses API_ENDPOINTS.CHAT and API_ENDPOINTS.GOOGLE_CLOUD_CHAT

## 🚀 Next Steps - Backend Deployment

### Option 1: Deploy to Azure App Service (Recommended)
```bash
# 1. Create Azure App Service for Python
az webapp create --resource-group dalaal-street-bot --plan MyAppServicePlan --name dalaal-street-chatbot-api --runtime "PYTHON|3.11"

# 2. Deploy your backend code
az webapp deployment source config-zip --resource-group dalaal-street-bot --name dalaal-street-chatbot-api --src backend.zip
```

### Option 2: Deploy to Azure Container Apps
```bash
# Build and push Docker image
docker build -t dalaal-street-chatbot-api .
docker tag dalaal-street-chatbot-api youracr.azurecr.io/dalaal-street-chatbot-api
docker push youracr.azurecr.io/dalaal-street-chatbot-api

# Deploy to Container Apps
az containerapp create --resource-group dalaal-street-bot --name dalaal-street-chatbot-api --image youracr.azurecr.io/dalaal-street-chatbot-api
```

## 🔄 After Backend Deployment

### 1. Update Frontend Configuration
Create a `.env.local` file in your frontend:
```env
REACT_APP_API_BASE_URL=https://your-backend-app-name.azurewebsites.net
```

### 2. Update CORS in Backend
Update your `main.py` CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "https://gray-desert-02b53f100.1.azurestaticapps.net",  # Your deployed frontend
        "https://dalaalstreet.dev"  # Your custom domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

### 3. Rebuild and Redeploy Frontend
```bash
npm run build
swa deploy ./build --deployment-token="YOUR_DEPLOYMENT_TOKEN"
```

## 📋 Current API Endpoints to Deploy

Your backend has these endpoints that need to be accessible:

| Endpoint | Purpose | Current Status |
|----------|---------|----------------|
| `POST /api/v1/chat` | Main chat functionality | ✅ Configured |
| `POST /api/v1/google-cloud/financial-chat` | Google Cloud chat | ✅ Configured |
| `POST /api/v1/chart-data` | Trading chart data | ✅ Configured |
| `POST /api/v1/stock` | Stock data | ⚠️ Not used in frontend yet |
| `GET /api/v1/indices` | Market indices | ⚠️ Not used in frontend yet |
| `POST /api/v1/news` | News service | ⚠️ Not used in frontend yet |
| `POST /api/v1/analysis` | Financial analysis | ⚠️ Not used in frontend yet |

## 🔧 Environment Variables Needed for Backend

Make sure your backend deployment has these environment variables:
```env
# Required for your backend
GROQ_API_KEY=your_groq_key
GOOGLE_CLOUD_PROJECT_ID=your_project_id
UPSTOX_API_KEY=your_upstox_key
# ... other keys from your .env file
```

## 🧪 Testing Your Deployment

1. **Frontend Test**: Visit https://gray-desert-02b53f100-preview.eastasia.1.azurestaticapps.net
2. **Backend Test**: Visit https://your-backend-url/docs (FastAPI docs)
3. **Integration Test**: Try the chat functionality on your frontend

## 💡 Pro Tips

1. **Use Azure Key Vault** for storing sensitive API keys
2. **Set up Application Insights** for monitoring
3. **Configure custom domains** for both frontend and backend
4. **Set up CI/CD pipelines** with GitHub Actions

## 🆘 Quick Fix for Immediate Testing

If you want to test locally right now:
1. Start your backend: `python main.py`
2. Your frontend will automatically connect to `http://localhost:8000`
3. Everything should work as expected

---

**Next Action Required**: Deploy your Python backend to Azure, then update the REACT_APP_API_BASE_URL environment variable and redeploy your frontend.
