# Google Cloud Services for Financial Chatbot

This document provides instructions on how to integrate and use Google Cloud services in the Dalaal Street Chatbot application.

## Services Overview

The chatbot uses the following Google Cloud services:

1. **Vertex AI** - For financial analysis, stock prediction, and sentiment analysis
2. **Dialogflow** - For specialized financial conversations and intent recognition
3. **Cloud Vision API** - For chart image analysis and financial document extraction
4. **BigQuery** - For financial data warehousing and advanced analytics

## Setup Instructions

### 1. Prerequisites

- Google Cloud Platform (GCP) account
- Project created in GCP with billing enabled
- Service account with appropriate permissions

### 2. Enable Required APIs

Enable the following APIs in your GCP project:

- Vertex AI API
- Dialogflow API
- Cloud Vision API
- BigQuery API
- Cloud Storage API

### 3. Create Service Account

1. Go to IAM & Admin > Service Accounts in GCP Console
2. Create a new service account with the following roles:
   - Vertex AI User
   - Dialogflow API Admin
   - Cloud Vision API User
   - BigQuery User
   - Storage Object Admin
3. Create and download a JSON key for this service account

### 4. Environment Configuration

Set the following environment variables:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
export VERTEXAI_PROJECT_ID="your-project-id"
export VERTEXAI_LOCATION="us-central1"
export DIALOGFLOW_PROJECT_ID="your-dialogflow-project-id"
```

Or add them to your `.env` file:

```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
VERTEXAI_PROJECT_ID=your-project-id
VERTEXAI_LOCATION=us-central1
DIALOGFLOW_PROJECT_ID=your-dialogflow-project-id
```

### 5. Install Required Packages

```bash
pip install -r requirements.txt
```

## Testing the Integration

After setting up the environment, you can test the integration with the provided example script:

```bash
python gcloud_examples.py
```

This script demonstrates:
- Chat processing with Dialogflow and Vertex AI
- Image analysis with Cloud Vision API
- Market insights with multiple services
- Stock analysis with Vertex AI
- Technical indicators generation

## API Endpoints

The following endpoints are available:

- `/api/v1/gcloud/chat` - Process chat messages
- `/api/v1/gcloud/image-analysis` - Analyze financial charts/documents
- `/api/v1/gcloud/upload-image` - Upload and analyze images
- `/api/v1/gcloud/market-insights` - Get comprehensive market insights
- `/api/v1/gcloud/stock-analysis` - Get AI-powered stock analysis
- `/api/v1/gcloud/technical-indicators` - Get technical indicators

See the example script for detailed usage patterns.

## Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Dialogflow Documentation](https://cloud.google.com/dialogflow/docs)
- [Cloud Vision API Documentation](https://cloud.google.com/vision/docs)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)

## Support

For issues with the Google Cloud services integration, please check the logs or contact your system administrator.
