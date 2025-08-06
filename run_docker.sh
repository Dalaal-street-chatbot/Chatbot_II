#!/bin/bash

# Run Dalaal Street Chatbot in Docker Environment
# This script runs the chatbot in a Docker environment

set -e  # Exit on error

echo "🚀 Setting up Dalaal Street Chatbot Docker Environment"
echo "======================================================"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Creating a template .env file from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file from example. Please edit it with your actual API keys."
        echo "ℹ️ Edit the file and then run this script again."
        exit 1
    else
        echo "❌ .env.example file not found either. Cannot continue."
        exit 1
    fi
fi

# Build and start the containers
echo "🏗️ Building Docker containers..."

# Use appropriate docker-compose command based on what's available
if command -v docker-compose &> /dev/null; then
    docker-compose build
    echo "🚀 Starting services..."
    docker-compose up -d
elif docker compose version &> /dev/null; then
    docker compose build
    echo "🚀 Starting services..."
    docker compose up -d
else
    echo "❌ Could not determine Docker Compose command."
    exit 1
fi

# Display service status
echo ""
echo "📊 Service Status:"

if command -v docker-compose &> /dev/null; then
    docker-compose ps
elif docker compose version &> /dev/null; then
    docker compose ps
fi

echo ""
echo "✅ Dalaal Street Chatbot is now running!"
echo "📊 Access the API at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔄 Health Check: http://localhost:8000/health"
echo ""
echo "💻 To view logs, run: docker-compose logs -f dalaal-street-bot"
echo "🛑 To stop the service, run: docker-compose down"
