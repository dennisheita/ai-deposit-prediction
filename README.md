# AI Deposit Prediction System

A comprehensive machine learning application for mineral deposit prediction with multi-mineral categorization and complete data isolation.

## 🌟 Features

- **🏔️ Multi-Mineral Support**: Separate environments for Copper, Diamonds, Gold, Lead, REE, Tin, and Uranium
- **🔒 Complete Isolation**: Each mineral operates in its own isolated environment
- **🗺️ Interactive Maps**: Real-time prediction visualization with heatmaps
- **📊 Advanced Analytics**: Performance monitoring and feature importance analysis
- **🔄 Continuous Learning**: Automated model retraining with new data
- **📁 Data Management**: Comprehensive file upload, processing, and management
- **🎯 Batch Processing**: Hyperparameter tuning and model optimization

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB RAM recommended
- 10GB free disk space

### Deployment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dennisheita/ai-deposit-prediction.git
   cd ai-deposit-prediction
   ```

2. **Run the deployment script:**
   ```bash
   ./deploy.sh
   ```

   Or manually:
   ```bash
   # Build and run with Docker Compose
   docker-compose up -d
   ```

3. **Access the application:**
   - Open http://localhost:8501 in your browser
   - The application will be ready in about 30-60 seconds

## 🐳 Docker Deployment

### Files Created

- **`Dockerfile`**: Multi-stage build with geospatial libraries
- **`docker-compose.yml`**: Complete deployment configuration
- **`.dockerignore`**: Optimized build context
- **`deploy.sh`**: Automated deployment script

### Manual Docker Commands

```bash
# Build the image
docker build -t ai-deposit-prediction .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  ai-deposit-prediction
```

### Docker Compose (Recommended)

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Restart
docker-compose restart
```

## 📁 Project Structure

```
ai-deposit-prediction/
├── app.py                    # Main Streamlit application
├── train_model.py           # External training script
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── deploy.sh              # Deployment automation script
├── .dockerignore          # Docker build exclusions
├── src/
│   ├── __init__.py
│   ├── data_architecture.py # Database operations
│   ├── data_ingestion.py   # File processing
│   ├── feature_engineering.py # Feature computation
│   ├── training_pipeline.py # ML training
│   ├── prediction.py       # Prediction system
│   └── monitoring.py       # Performance tracking
├── data/                   # Application data (mounted volume)
├── models/                 # Trained models (mounted volume)
├── logs/                   # Application logs (mounted volume)
└── ml_data/               # Training datasets (optional mount)
```

## 🏔️ Mineral Categories

The system supports complete isolation for these minerals:

- **Copper**
- **Diamonds**
- **Gold**
- **Lead**
- **REE (Rare Earth Elements)**
- **Tin**
- **Uranium**

Each mineral has its own:
- Models and training runs
- Uploaded datasets
- Predictions and results
- Performance statistics

## 🔧 Configuration

### Environment Variables

```bash
# Streamlit Configuration
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Application will create these directories automatically:
# - data/features/
# - data/deposits/
# - data/predictions/
# - models/
# - logs/
```

### Data Persistence

The following directories are mounted as Docker volumes for data persistence:

- `./data` → `/app/data` (databases, uploaded files)
- `./models` → `/app/models` (trained models)
- `./logs` → `/app/logs` (application logs)
- `./ml_data` → `/app/ml_data` (training datasets)

## 📊 Usage Guide

### 1. Select Mineral
Choose your target mineral from the sidebar dropdown to enter its isolated environment.

### 2. Upload Data
- **Features**: Geological feature data (elevation, slope, aspect, etc.)
- **Deposits**: Known mineral deposit locations
- **Formats**: CSV, Shapefile (.shp), GeoJSON

### 3. Train Models
- **Quick Training**: Uses pre-split datasets
- **Custom Training**: Upload your own data
- **Batch Processing**: Hyperparameter optimization
- **Continuous Learning**: Automatic retraining

### 4. Run Predictions
- Upload prediction areas
- Select trained models
- View interactive maps with probability heatmaps
- Download results as shapefiles

### 5. Monitor Performance
- View training statistics
- Analyze feature importance
- Track model performance over time

## 🔍 Troubleshooting

### Application Won't Start

```bash
# Check container logs
docker-compose logs ai-deposit-prediction

# Check if port 8501 is available
netstat -tlnp | grep 8501

# Restart the application
docker-compose restart
```

### Database Issues

If you encounter database schema errors:

1. Go to **Data Management** → **Database Reset**
2. Click **"Reset Database (Fix Mineral Support)"**
3. This will recreate the database with proper mineral columns

### Memory Issues

The application requires significant memory for ML training:

```bash
# Increase Docker memory limit
docker-compose.yml:
services:
  ai-deposit-prediction:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Docker logs: `docker-compose logs -f`
3. Open an issue on GitHub

---

**Built with:** Python, Streamlit, GeoPandas, Scikit-learn, Docker
**Geospatial Support:** GDAL, PROJ, GEOS
**Visualization:** Folium, Matplotlib