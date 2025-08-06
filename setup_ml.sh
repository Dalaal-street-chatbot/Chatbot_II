#!/bin/bash

echo "🚀 Setting up Dalaal Street Chatbot ML Training Environment"
echo "=========================================================="

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config

# Upgrade pip
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip

# Install Python ML dependencies
echo "🧠 Installing Python ML dependencies..."
pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.1 \
    scikit-learn==1.4.1 \
    matplotlib==3.8.3 \
    seaborn==0.13.2

# Install deep learning frameworks
echo "🤖 Installing deep learning frameworks..."
pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    tensorflow==2.15.0

# Install NLP libraries
echo "💬 Installing NLP libraries..."
pip install --no-cache-dir \
    transformers==4.38.1 \
    datasets==2.17.0 \
    huggingface-hub==0.21.0 \
    sentence-transformers==2.4.0 \
    nltk==3.8.1 \
    spacy==3.7.4 \
    textblob==0.17.1

# Install financial libraries
echo "📈 Installing financial libraries..."
pip install --no-cache-dir \
    yfinance==0.2.37 \
    ta==0.10.2 \
    quantlib==1.32

# Install API and web libraries
echo "🌐 Installing API and web libraries..."
pip install --no-cache-dir \
    groq==0.4.1 \
    openai==1.12.0 \
    newsapi-python==0.2.7 \
    requests==2.31.0 \
    httpx==0.27.0 \
    aiohttp==3.9.3

# Install FastAPI and related
echo "⚡ Installing FastAPI and related..."
pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    python-multipart==0.0.9

# Install Google Cloud libraries
echo "☁️  Installing Google Cloud libraries..."
pip install --no-cache-dir \
    google-cloud-aiplatform==1.42.1 \
    google-cloud-vision==3.7.0 \
    google-cloud-bigquery==3.17.2 \
    google-auth==2.27.0 \
    google-auth-oauthlib==1.2.0

# Install Azure libraries
echo "🔵 Installing Azure libraries..."
pip install --no-cache-dir \
    azure-cognitiveservices-language-textanalytics==5.3.0 \
    azure-identity==1.15.0

# Install ML experiment tracking
echo "📊 Installing ML experiment tracking..."
pip install --no-cache-dir \
    wandb==0.16.4 \
    mlflow==2.10.2 \
    optuna==3.5.0

# Install development tools
echo "🛠️  Installing development tools..."
pip install --no-cache-dir \
    jupyter==1.0.0 \
    notebook==7.1.0 \
    pytest==8.0.2 \
    pytest-asyncio==0.23.5 \
    black==24.2.0 \
    flake8==7.0.0

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
print('NLTK data downloaded successfully')
"

# Download spaCy model
echo "🌍 Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm

# Create necessary directories
echo "📁 Creating ML training directories..."
mkdir -p ml_training/data
mkdir -p ml_training/models
mkdir -p ml_training/logs
mkdir -p ml_training/experiments

# Make training script executable
echo "🔑 Making training script executable..."
chmod +x train_bot.py

# Test imports
echo "🧪 Testing critical imports..."
python3 -c "
import numpy as np
import pandas as pd
import sklearn
import torch
import tensorflow as tf
import transformers
import groq
import fastapi
print('✅ All critical imports successful')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Ensure your .env file contains all required API keys"
echo "2. Run: python train_bot.py"
echo "3. Monitor training progress in the terminal"
echo ""
echo "For help: python train_bot.py --help"
