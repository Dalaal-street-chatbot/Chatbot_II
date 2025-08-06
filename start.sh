#!/bin/bash

# Dalaal Street Chatbot Startup Script

echo "🚀 Starting Dalaal Street Chatbot..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with your API keys."
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Validate configuration
echo "🔧 Validating configuration..."
python -c "
from config.settings import config
if not config.validate_config():
    print('❌ Configuration validation failed!')
    exit(1)
print('✅ Configuration validated successfully!')
"

if [ $? -ne 0 ]; then
    echo "❌ Configuration validation failed!"
    exit 1
fi

# Start the application
echo "🎯 Starting the API server..."
echo "📊 Access the API at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔄 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"

python main.py
