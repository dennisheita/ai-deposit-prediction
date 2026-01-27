# AI Deposit Prediction - Server Deployment Guide

This guide explains how to deploy the AI Deposit Prediction application to a remote server using Docker.

## Overview

The application consists of three services:
- **Backend API** (FastAPI) - Port 8000
- **Dashboard** (Streamlit) - Port 8501
- **Frontend** (Next.js) - Port 3000

## Prerequisites

### Local Machine
- Docker and Docker Compose installed
- SSH access to the target server
- rsync installed (for file synchronization)

### Server Requirements
- Ubuntu 20.04+ or similar Linux distribution
- SSH access enabled
- Ports 3000, 8000, and 8501 open in firewall (or configure as needed)
- Sudo privileges (for Docker installation if not already installed)

## Quick Deployment

### Option 1: Using Environment Variables

```bash
export SERVER_USER=your_username
export SERVER_HOST=your_server_ip
export SERVER_DIR=/opt/ai-deposit-prediction
export SSH_KEY=/path/to/your/ssh/key  # Optional

./deploy-to-server.sh
```

### Option 2: Using Command Line Arguments

```bash
./deploy-to-server.sh -u your_username -h your_server_ip -d /opt/ai-deposit-prediction -k /path/to/ssh/key
```

## Manual Deployment Steps

If you prefer to deploy manually or the script doesn't work for your setup:

### 1. Prepare the Server

SSH into your server and install Docker:

```bash
# SSH to server
ssh your_username@your_server_ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and back in for group changes to take effect
exit
```

### 2. Copy Files to Server

From your local machine:

```bash
# Create directory on server
ssh your_username@your_server_ip "mkdir -p /opt/ai-deposit-prediction"

# Copy files (excluding unnecessary files)
rsync -avz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='node_modules' \
  --exclude='.next' \
  ./ your_username@your_server_ip:/opt/ai-deposit-prediction/
```

### 3. Build and Run on Server

SSH into the server and run:

```bash
ssh your_username@your_server_ip
cd /opt/ai-deposit-prediction

# Build and start containers
docker compose -f docker-compose.prod.yml up -d --build

# Check status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f
```

## Docker Configuration Files

### docker-compose.prod.yml

This file configures the production deployment with:
- Health checks for all services
- Automatic restart policies
- Production-optimized settings
- 4 workers for the backend API

### Dockerfile

The main Dockerfile:
- Uses Python 3.9 slim base image
- Installs GDAL and geospatial dependencies
- Sets up the application environment
- Configures Streamlit for headless operation

### ui/Dockerfile

The frontend Dockerfile:
- Uses Node.js 20 Alpine
- Multi-stage build for smaller image size
- Standalone Next.js output

## Accessing the Application

After successful deployment:

- **Frontend UI**: http://your_server_ip:3000
- **Dashboard**: http://your_server_ip:8501
- **API**: http://your_server_ip:8000
- **Health Check**: http://your_server_ip:8000/health

## Managing the Deployment

### View Logs

```bash
ssh your_username@your_server_ip "cd /opt/ai-deposit-prediction && docker compose -f docker-compose.prod.yml logs -f"
```

### Stop Services

```bash
ssh your_username@your_server_ip "cd /opt/ai-deposit-prediction && docker compose -f docker-compose.prod.yml down"
```

### Restart Services

```bash
ssh your_username@your_server_ip "cd /opt/ai-deposit-prediction && docker compose -f docker-compose.prod.yml restart"
```

### Update Deployment

To update the code and redeploy:

```bash
# Run the deployment script again
./deploy-to-server.sh

# Or manually:
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.venv' --exclude='venv' --exclude='node_modules' --exclude='.next' \
  ./ your_username@your_server_ip:/opt/ai-deposit-prediction/

ssh your_username@your_server_ip "cd /opt/ai-deposit-prediction && docker compose -f docker-compose.prod.yml up -d --build"
```

## Troubleshooting

### Services Not Starting

Check logs for specific errors:
```bash
docker compose -f docker-compose.prod.yml logs
```

### Port Already in Use

If ports are already in use, modify the port mappings in `docker-compose.prod.yml`:
```yaml
ports:
  - "8080:8000"  # Use port 8080 instead of 8000
```

### Permission Denied

Ensure your user is in the docker group:
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Data Persistence

Data is persisted through Docker volumes:
- `./data:/app/data` - Data files
- `./models:/app/models` - Trained models
- `./logs:/app/logs` - Application logs

These directories on the server will retain data between container restarts.

## Security Considerations

1. **Firewall**: Configure your server's firewall to only allow necessary ports
2. **SSH Keys**: Use SSH key authentication instead of passwords
3. **Environment Variables**: For sensitive data, use environment files (`.env`) that are not committed to git
4. **HTTPS**: For production use, consider setting up a reverse proxy (nginx/traefik) with SSL certificates

## Advanced Configuration

### Using a Reverse Proxy (nginx)

For production deployments with HTTPS, you can add an nginx service:

```yaml
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend
```

### Environment Variables

Create a `.env` file on the server for sensitive configuration:

```bash
# On server
cd /opt/ai-deposit-prediction
cat > .env << EOF
ENVIRONMENT=production
SECRET_KEY=your_secret_key_here
DATABASE_URL=your_database_url
EOF
```

Update `docker-compose.prod.yml` to use the env file:
```yaml
env_file:
  - .env
```

## Support

For issues or questions:
1. Check the logs using `docker compose logs`
2. Verify all services are running with `docker compose ps`
3. Ensure all required ports are open on your server
