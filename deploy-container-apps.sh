#!/bin/bash

# 🚀 Azure Container Apps Deployment Script for Dalaal Street Chatbot
# This script will deploy your FastAPI backend to Azure Container Apps

set -e  # Exit on any error

# Configuration
RESOURCE_GROUP="dalaal-street-bot"
CONTAINER_APP_NAME="dalaal-street-api"
CONTAINER_ENV_NAME="dalaal-street-env"
LOCATION="eastasia"
IMAGE_NAME="dalaal-street-chatbot"

echo "🚀 Starting Azure Container Apps deployment..."

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first."
    echo "   Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Login check
echo "🔐 Checking Azure login status..."
if ! az account show &> /dev/null; then
    echo "Please login to Azure first:"
    az login
fi

# Show current subscription
echo "📋 Current Azure subscription:"
az account show --query "{name:name, id:id}" -o table

# Create resource group if it doesn't exist
echo "📦 Creating/checking resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION --output table

# Create Container Apps environment
if ! az containerapp env show --name $CONTAINER_ENV_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    az containerapp env create \
        --name $CONTAINER_ENV_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --output table
    echo "✅ Container Apps environment created"
else
    echo "✅ Container Apps environment already exists"
fi

# Create Azure Container Registry
ACR_NAME="${RESOURCE_GROUP}acr"
echo "🧰 Creating/checking Azure Container Registry..."
if ! az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $ACR_NAME \
        --sku Basic \
        --location $LOCATION \
        --admin-enabled true \
        --output table
    echo "✅ ACR created"
else
    echo "✅ ACR already exists"
fi

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show \
    --name $ACR_NAME \
    --resource-group $RESOURCE_GROUP \
    --query loginServer \
    --output tsv)

# Build and push image to ACR
echo "🔨 Building and pushing image to ACR..."
az acr build \
    --registry $ACR_NAME \
    --image $IMAGE_NAME:latest \
    --file Dockerfile \
    .

# Deploy the container app with the image from ACR
echo "🐳 Deploying container app..."
az containerapp up \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_ENV_NAME \
    --image $ACR_LOGIN_SERVER/$IMAGE_NAME:latest \
    --target-port 8000 \
    --ingress external \
    --output table

# Get the app URL
echo "🌍 Getting application URL..."
APP_URL=$(az containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "properties.configuration.ingress.fqdn" \
    --output tsv)

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Your API is available at: https://$APP_URL"
echo "2. API docs available at: https://$APP_URL/docs"
echo "3. Update your frontend REACT_APP_API_BASE_URL to: https://$APP_URL"
echo ""
echo "🔧 To add environment variables (API keys), run:"
echo "   az containerapp update \\"
echo "     --name $CONTAINER_APP_NAME \\"
echo "     --resource-group $RESOURCE_GROUP \\"
echo "     --set-env-vars GROQ_API_KEY=your_key \\"
echo "                   GOOGLE_CLOUD_PROJECT_ID=your_project \\"
echo "                   UPSTOX_API_KEY=your_key"
echo ""
echo "🔄 To update your frontend, run:"
echo "   echo 'REACT_APP_API_BASE_URL=https://$APP_URL' > .env.local"
echo "   npm run build"
echo "   swa deploy ./build --deployment-token=\"YOUR_TOKEN\""
echo ""

# Test the deployment
echo "🧪 Testing deployment..."
if curl -f "https://$APP_URL/docs" &> /dev/null; then
    echo "✅ API is responding correctly!"
else
    echo "⚠️  API might still be starting up. Check logs with:"
    echo "   az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP"
fi
