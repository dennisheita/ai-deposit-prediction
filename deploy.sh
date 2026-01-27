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

echo "📦 Building all Docker images (Frontend, Backend, Dashboard)..."
$DOCKER_COMPOSE_CMD build

echo "🏃 Starting all services..."
$DOCKER_COMPOSE_CMD up -d

echo "⏳ Waiting for services to initialize..."
sleep 15

# Check if the Main UI is running
if curl -f http://localhost:3000 &> /dev/null; then
    echo "✅ AI Deposit Prediction System is LIVE!"
    echo "------------------------------------------------"
    echo "📊 Main UI:        http://localhost:3000"
    echo "⚙️  Dashboard:      http://localhost:8501"
    echo "🔌 API Backend:    http://localhost:8000/health"
    echo "------------------------------------------------"
    echo ""
    echo "📋 Useful commands:"
    echo "  • View logs: $DOCKER_COMPOSE_CMD logs -f"
    echo "  • Stop app:  $DOCKER_COMPOSE_CMD down"
    echo "  • Restart:   $DOCKER_COMPOSE_CMD restart"
else
    echo "❌ Some services failed to start. Check logs with: $DOCKER_COMPOSE_CMD logs"
    exit 1
fi