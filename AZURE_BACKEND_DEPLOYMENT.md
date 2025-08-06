# 🚀 Deploy FastAPI Backend to Azure

## Option 1: Azure Container Apps (RECOMMENDED)

### Step 1: Update your API configuration
Update your frontend API base URL to point to your Container App:

```typescript
// In src/config/api.ts - update line 9:
return process.env.REACT_APP_API_BASE_URL || 'https://dalaal-street-api.nicegrass-12345678.eastasia.azurecontainerapps.io';
```

### Step 2: Deploy using Azure CLI

```bash
# 1. Login to Azure
az login

# 2. Create resource group (if not exists)
az group create --name dalaal-street-bot --location eastasia

# 3. Create Container Apps environment
az containerapp env create \
  --name dalaal-street-env \
  --resource-group dalaal-street-bot \
  --location eastasia

# 4. Build and deploy your FastAPI app
az containerapp up \
  --name dalaal-street-api \
  --resource-group dalaal-street-bot \
  --environment dalaal-street-env \
  --source . \
  --target-port 8000 \
  --ingress external \
  --env-vars GROQ_API_KEY=secretref:groq-key \
           GOOGLE_CLOUD_PROJECT_ID=secretref:gcp-project \
           UPSTOX_API_KEY=secretref:upstox-key
```

### Step 3: Add your secrets
```bash
# Add your API keys as secrets
az containerapp secret set \
  --name dalaal-street-api \
  --resource-group dalaal-street-bot \
  --secrets groq-key="YOUR_GROQ_KEY" \
           gcp-project="YOUR_GCP_PROJECT" \
           upstox-key="YOUR_UPSTOX_KEY"
```

## Option 2: Azure App Service

### Step 1: Create App Service
```bash
# Create App Service plan
az appservice plan create \
  --name dalaal-street-plan \
  --resource-group dalaal-street-bot \
  --sku B1 \
  --is-linux

# Create Web App
az webapp create \
  --name dalaal-street-api \
  --resource-group dalaal-street-bot \
  --plan dalaal-street-plan \
  --runtime "PYTHON|3.11"
```

### Step 2: Deploy from GitHub
```bash
# Configure GitHub deployment
az webapp deployment source config \
  --name dalaal-street-api \
  --resource-group dalaal-street-bot \
  --repo-url https://github.com/Dalaal-street-chatbot/Dalaal-street-chatbot \
  --branch main \
  --manual-integration
```

## Option 3: Azure Functions (If you insist)

This requires significant code refactoring. Each endpoint becomes a separate function:

### Required Changes:
1. **Split each route into separate function files**
2. **Remove FastAPI dependencies**
3. **Use Azure Functions HTTP triggers**
4. **Restructure your project**

### Example Function Structure:
```
functions/
├── chat/
│   ├── __init__.py
│   └── function.json
├── stock/
│   ├── __init__.py
│   └── function.json
└── host.json
```

## 💡 Recommendation

**Go with Azure Container Apps** because:
- ✅ **Zero code changes** needed
- ✅ **Serverless pricing** - pay only for usage
- ✅ **Auto-scaling** from 0 to many instances
- ✅ **Full FastAPI compatibility**
- ✅ **Easy CI/CD** setup

## 🔄 After Backend Deployment

### Update Frontend Configuration:
```bash
# Create .env.local with your backend URL
echo "REACT_APP_API_BASE_URL=https://your-container-app-url" > .env.local

# Rebuild and redeploy frontend
npm run build
swa deploy ./build --deployment-token="YOUR_TOKEN"
```

### Update CORS in main.py:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://gray-desert-02b53f100.1.azurestaticapps.net",
        "https://dalaalstreet.dev"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

---

**Would you like me to help you deploy to Azure Container Apps? It's the fastest path to get your backend running!**
