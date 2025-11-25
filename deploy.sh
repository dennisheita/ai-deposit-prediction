#!/bin/bash

# AI Deposit Prediction - Docker Deployment Script
# This script builds and runs the application using Docker

set -e

echo "🚀 Starting AI Deposit Prediction deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    echo "❌ docker-compose is not available. Please install docker-compose."
    exit 1
fi

echo "📦 Building Docker image..."
$DOCKER_COMPOSE_CMD build

echo "🏃 Starting the application..."
$DOCKER_COMPOSE_CMD up -d

echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8501/healthz &> /dev/null; then
    echo "✅ Application is running successfully!"
    echo "🌐 Access the application at: http://localhost:8501"
    echo ""
    echo "📋 Useful commands:"
    echo "  • View logs: $DOCKER_COMPOSE_CMD logs -f"
    echo "  • Stop application: $DOCKER_COMPOSE_CMD down"
    echo "  • Restart application: $DOCKER_COMPOSE_CMD restart"
else
    echo "❌ Application failed to start. Check logs with: $DOCKER_COMPOSE_CMD logs"
    exit 1
fi