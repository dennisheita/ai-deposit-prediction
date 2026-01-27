#!/bin/bash

# AI Deposit Prediction - Server Deployment Script
# This script deploys the application to a remote server using Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - MODIFY THESE VALUES
SERVER_USER="${SERVER_USER:-your_username}"      # Server username
SERVER_HOST="${SERVER_HOST:-your_server_ip}"     # Server IP or hostname
SERVER_DIR="${SERVER_DIR:-/opt/ai-deposit-prediction}"  # Directory on server
SSH_KEY="${SSH_KEY:-}"                            # Path to SSH key (optional)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -u, --user USERNAME     Server username (default: $SERVER_USER)"
    echo "  -h, --host HOST         Server hostname or IP (default: $SERVER_HOST)"
    echo "  -d, --dir DIRECTORY     Server deployment directory (default: $SERVER_DIR)"
    echo "  -k, --key SSH_KEY       Path to SSH private key"
    echo "  --help                  Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SERVER_USER             Server username"
    echo "  SERVER_HOST             Server hostname or IP"
    echo "  SERVER_DIR              Server deployment directory"
    echo "  SSH_KEY                 Path to SSH private key"
    echo ""
    echo "Example:"
    echo "  $0 -u admin -h 192.168.1.100 -d /var/www/app"
    echo "  SERVER_USER=admin SERVER_HOST=192.168.1.100 $0"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--user)
            SERVER_USER="$2"
            shift 2
            ;;
        -h|--host)
            SERVER_HOST="$2"
            shift 2
            ;;
        -d|--dir)
            SERVER_DIR="$2"
            shift 2
            ;;
        -k|--key)
            SSH_KEY="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ "$SERVER_USER" = "your_username" ] || [ -z "$SERVER_USER" ]; then
    print_error "Server username not set. Use -u option or set SERVER_USER environment variable."
    show_usage
    exit 1
fi

if [ "$SERVER_HOST" = "your_server_ip" ] || [ -z "$SERVER_HOST" ]; then
    print_error "Server host not set. Use -h option or set SERVER_HOST environment variable."
    show_usage
    exit 1
fi

# Build SSH command
SSH_CMD="ssh"
if [ -n "$SSH_KEY" ]; then
    SSH_CMD="ssh -i $SSH_KEY"
fi

SERVER="$SERVER_USER@$SERVER_HOST"

print_status "🚀 Starting deployment to $SERVER"
print_status "Target directory: $SERVER_DIR"

# Check if we can connect to the server
print_status "Checking server connectivity..."
if ! $SSH_CMD -o ConnectTimeout=10 -o BatchMode=yes "$SERVER" "echo 'Connection successful'" > /dev/null 2>&1; then
    print_error "Cannot connect to server $SERVER"
    print_error "Please check:"
    print_error "  1. Server is running and accessible"
    print_error "  2. SSH key is configured correctly"
    print_error "  3. Username and host are correct"
    exit 1
fi
print_success "Server connection verified"

# Create deployment directory on server
print_status "Creating deployment directory on server..."
$SSH_CMD "$SERVER" "sudo mkdir -p $SERVER_DIR && sudo chown \$USER:\$USER $SERVER_DIR" || {
    print_error "Failed to create directory on server. Trying without sudo..."
    $SSH_CMD "$SERVER" "mkdir -p $SERVER_DIR" || {
        print_error "Failed to create directory on server"
        exit 1
    }
}

# Check if Docker is installed on server
print_status "Checking Docker on server..."
if ! $SSH_CMD "$SERVER" "command -v docker" > /dev/null 2>&1; then
    print_warning "Docker not found on server. Installing Docker..."
    $SSH_CMD "$SERVER" "
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        sudo usermod -aG docker \$USER
        rm get-docker.sh
    " || {
        print_error "Failed to install Docker on server"
        exit 1
    }
    print_success "Docker installed successfully"
else
    print_success "Docker is already installed"
fi

# Check if docker-compose is available on server
print_status "Checking docker-compose on server..."
if ! $SSH_CMD "$SERVER" "docker compose version" > /dev/null 2>&1 && ! $SSH_CMD "$SERVER" "docker-compose --version" > /dev/null 2>&1; then
    print_warning "docker-compose not found. Installing..."
    $SSH_CMD "$SERVER" "
        sudo curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    " || {
        print_error "Failed to install docker-compose"
        exit 1
    }
fi
print_success "docker-compose is available"

# Determine docker-compose command on server
DOCKER_COMPOSE_CMD="docker compose"
if ! $SSH_CMD "$SERVER" "docker compose version" > /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
fi

# Sync files to server
print_status "Syncing files to server..."
RSYNC_OPTS="-avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='venv' --exclude='node_modules' --exclude='.next' --exclude='data/*.csv' --exclude='models/*.pkl'"

# Add SSH key to rsync if provided
if [ -n "$SSH_KEY" ]; then
    RSYNC_OPTS="$RSYNC_OPTS -e 'ssh -i $SSH_KEY'"
fi

eval rsync $RSYNC_OPTS ./ "$SERVER:$SERVER_DIR/" || {
    print_error "Failed to sync files to server"
    exit 1
}
print_success "Files synced successfully"

# Build and start Docker containers on server
print_status "Building and starting Docker containers on server..."
$SSH_CMD "$SERVER" "cd $SERVER_DIR && $DOCKER_COMPOSE_CMD -f docker-compose.prod.yml down" 2>/dev/null || true

$SSH_CMD "$SERVER" "cd $SERVER_DIR && $DOCKER_COMPOSE_CMD -f docker-compose.prod.yml build --no-cache" || {
    print_error "Failed to build Docker images on server"
    exit 1
}

$SSH_CMD "$SERVER" "cd $SERVER_DIR && $DOCKER_COMPOSE_CMD -f docker-compose.prod.yml up -d" || {
    print_error "Failed to start containers on server"
    exit 1
}

print_success "Docker containers started successfully"

# Wait for services to be ready
print_status "Waiting for services to initialize..."
sleep 15

# Check service health
print_status "Checking service health..."
HEALTHY=true

# Check backend
if $SSH_CMD "$SERVER" "curl -sf http://localhost:8000/health" > /dev/null 2>&1; then
    print_success "Backend API is healthy (port 8000)"
else
    print_error "Backend API is not responding"
    HEALTHY=false
fi

# Check dashboard
if $SSH_CMD "$SERVER" "curl -sf http://localhost:8501/healthz" > /dev/null 2>&1; then
    print_success "Dashboard is healthy (port 8501)"
else
    print_error "Dashboard is not responding"
    HEALTHY=false
fi

# Check frontend
if $SSH_CMD "$SERVER" "curl -sf http://localhost:3000" > /dev/null 2>&1; then
    print_success "Frontend is healthy (port 3000)"
else
    print_error "Frontend is not responding"
    HEALTHY=false
fi

echo ""
echo "================================================"
if [ "$HEALTHY" = true ]; then
    print_success "🎉 Deployment successful!"
    echo ""
    echo "Application URLs:"
    echo "  📊 Frontend:    http://$SERVER_HOST:3000"
    echo "  ⚙️  Dashboard:   http://$SERVER_HOST:8501"
    echo "  🔌 API:         http://$SERVER_HOST:8000"
    echo ""
    echo "Server Commands:"
    echo "  View logs:  $SSH_CMD $SERVER 'cd $SERVER_DIR && $DOCKER_COMPOSE_CMD -f docker-compose.prod.yml logs -f'"
    echo "  Stop:       $SSH_CMD $SERVER 'cd $SERVER_DIR && $DOCKER_COMPOSE_CMD -f docker-compose.prod.yml down'"
    echo "  Restart:    $SSH_CMD $SERVER 'cd $SERVER_DIR && $DOCKER_COMPOSE_CMD -f docker-compose.prod.yml restart'"
else
    print_warning "⚠️  Deployment completed but some services are not responding"
    echo "Check logs with: $SSH_CMD $SERVER 'cd $SERVER_DIR && $DOCKER_COMPOSE_CMD -f docker-compose.prod.yml logs'"
fi
echo "================================================"
