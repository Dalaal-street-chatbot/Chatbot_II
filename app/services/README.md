# Google Cloud Services Integration

This directory contains modules for integrating with Google Cloud services for the Dalaal Street Chatbot application.

## Services Implemented

### 1. Vertex AI Financial Analysis
- Specialized financial analysis using Vertex AI
- Technical indicators calculation
- Stock price prediction
- Financial sentiment analysis
- Financial chart and document analysis

### 2. DialogFlow Financial Chat
- Intent recognition for financial conversations
- Specialized financial query processing
- Portfolio tracking conversation flows
- Financial planning conversation flows

### 3. Cloud Vision Integration
- Chart image analysis
- Financial document extraction
- Technical analysis chart recognition

### 4. BigQuery Analytics
- Financial data warehousing
- Advanced market analytics
- Historical performance analysis
- Custom reporting capabilities

## Integration

The `financial_chatbot_integration.py` file serves as the main integration point for all Google Cloud services in the chatbot application. It orchestrates the flow between different services based on user queries and intents.

## API Routes

The Google Cloud services are exposed through FastAPI routes in `app/api/gcloud_routes.py` with the following endpoints:

- `/api/v1/gcloud/chat` - Process chat messages using integrated Google Cloud services
- `/api/v1/gcloud/image-analysis` - Analyze financial charts or documents from base64-encoded image
- `/api/v1/gcloud/upload-image` - Analyze financial charts or documents from uploaded file
- `/api/v1/gcloud/market-insights` - Get comprehensive market insights using all available services
- `/api/v1/gcloud/stock-analysis` - Get specialized stock analysis using AI services
- `/api/v1/gcloud/technical-indicators` - Get technical indicators for a stock

## Usage

### Chat Example
```python
import requests
import json

url = "http://localhost:8000/api/v1/gcloud/chat"
payload = {
    "message": "What's the current market sentiment for RELIANCE?",
    "session_id": "user-123",
    "context": {"last_query": "stock_price"}
}

response = requests.post(url, json=payload)
result = response.json()
print(json.dumps(result, indent=2))
```

### Image Analysis Example
```python
import requests
import base64
import json

# Read image file
with open("financial_chart.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

url = "http://localhost:8000/api/v1/gcloud/image-analysis"
payload = {
    "base64_image": encoded_string,
    "session_id": "user-123",
    "context": {"analysis_type": "chart"}
}

response = requests.post(url, json=payload)
result = response.json()
print(json.dumps(result, indent=2))
```

### Market Insights Example
```python
import requests
import json

url = "http://localhost:8000/api/v1/gcloud/market-insights"
payload = {
    "symbols": ["RELIANCE", "HDFCBANK", "TCS"]
}

response = requests.post(url, json=payload)
result = response.json()
print(json.dumps(result, indent=2))
```

## Configuration

To use these Google Cloud services, you need to set up the following environment variables:

```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
VERTEXAI_PROJECT_ID=your-project-id
VERTEXAI_LOCATION=us-central1
DIALOGFLOW_PROJECT_ID=your-dialogflow-project-id
VISION_API_KEY=your-vision-api-key
```

## Dependencies

The integration requires the following Python packages:
- google-cloud-aiplatform
- google-cloud-dialogflow
- google-cloud-vision
- google-cloud-bigquery
- google-cloud-storage
- vertexai

All dependencies are specified in the main `requirements.txt` file.
