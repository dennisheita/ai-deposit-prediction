from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import zipfile
import json
from typing import List, Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import shapely
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import io
import base64
import subprocess
import threading

from src.data_ingestion import ingest_files
from src.prediction import run_prediction_pipeline, run_prediction_pipeline_from_geojson
from src.data_architecture import get_models, get_models_by_mineral, get_files
from src.monitoring import get_active_alerts, generate_performance_report, plot_performance_trends, plot_feature_importance_evolution
from src.training_pipeline import run_training_pipeline
from src.advanced_training import run_advanced_training_pipeline

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "FastAPI backend is running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs('data/features', exist_ok=True)
os.makedirs('data/deposits', exist_ok=True)
os.makedirs('data/predictions', exist_ok=True)
os.makedirs('models', exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        file_path = tmp.name
    results = ingest_files([file_path])
    os.unlink(file_path)
    return {"results": results}

@app.post("/train")
async def train_model(
    features_file: str = Form(...),
    deposits_file: str = Form(...),
    mineral: str = Form(...),
    advanced: bool = Form(False),
    negatives: int = Form(1),
    folds: int = Form(10),
    trials: int = Form(50)
):
    try:
        if advanced:
            result = run_advanced_training_pipeline(
                features_file, 
                deposits_file, 
                mineral=mineral,
                n_negatives_per_positive=negatives,
                k=folds,
                n_trials=trials
            )
        else:
            result = run_training_pipeline(features_file, deposits_file, mineral=mineral)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_advanced")
async def train_advanced_model(
    features_file: str = Form(...),
    deposits_file: str = Form(...),
    mineral: str = Form(...),
    negatives: int = Form(1),
    folds: int = Form(10),
    trials: int = Form(50)
):
    try:
        result = run_advanced_training_pipeline(
            features_file, 
            deposits_file, 
            mineral=mineral,
            n_negatives_per_positive=negatives,
            k=folds,
            n_trials=trials
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mineral: str = Form(...),
    threshold: float = Form(0.5)
):
    # Process file
    if file.filename.endswith('.zip'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(await file.read())
            zip_path = tmp.name
        extract_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        shp_path = None
        for f in os.listdir(extract_dir):
            if f.endswith('.shp'):
                shp_path = os.path.join(extract_dir, f)
                break
        prediction_area_path = shp_path
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            tmp.write(await file.read())
            prediction_area_path = tmp.name

    # Get model
    if mineral == 'All Minerals':
        models = get_models()
    else:
        models = get_models_by_mineral(mineral)
    if not models:
        raise HTTPException(status_code=404, detail="No models available")
    model_version = models[-1][1]

    # Run prediction
    output_filename, pred_gdf = run_prediction_pipeline(prediction_area_path, model_version, threshold, mineral=mineral)

    # Create map
    # Reproject to a projected CRS for accurate centroid calculation
    # Use EPSG:3857 (Web Mercator) which is commonly used for mapping
    projected_gdf = pred_gdf.to_crs('EPSG:3857')
    center_lat = projected_gdf.geometry.centroid.y.mean()
    center_lon = projected_gdf.geometry.centroid.x.mean()
    # Convert back to geographic CRS for folium
    center = gpd.GeoSeries([Point(center_lon, center_lat)], crs='EPSG:3857').to_crs('EPSG:4326').iloc[0]
    center_lat, center_lon = center.y, center.x
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    prob_col = 'probabilit' if 'probabilit' in pred_gdf.columns else 'probability'
    pred_col = 'predictio' if 'predictio' in pred_gdf.columns else 'prediction'

    # Helper function to get coordinates from any geometry type
    def get_coords(geom):
        if geom.geom_type == 'Point':
            return geom.y, geom.x
        else:
            return geom.centroid.y, geom.centroid.x

    # Load KNOWN DEPOSITS for comparison
    def load_known_deposits(mineral_name):
        """Load known deposit locations from training data."""
        known_deposits = []
        try:
            if mineral_name in ['Gold', 'All Minerals']:
                df = pd.read_csv('data/gold grid_with_all_features.csv')
                gold_deps = df[df['gold_present'] == 1][['latitude', 'longitude']]
                for _, row in gold_deps.iterrows():
                    known_deposits.append({'lat': row['latitude'], 'lon': row['longitude'], 'mineral': 'Gold'})
            
            if mineral_name in ['Copper', 'All Minerals']:
                df = pd.read_csv('data/copper_grid_with_features.csv')
                copper_deps = df[df['copper_present'] == 1][['latitude', 'longitude']]
                for _, row in copper_deps.iterrows():
                    known_deposits.append({'lat': row['latitude'], 'lon': row['longitude'], 'mineral': 'Copper'})
            
            if mineral_name in ['Uranium', 'All Minerals']:
                df = pd.read_csv('data/uranium_grid_with_features.csv')
                uranium_deps = df[df['uranium_present'] == 1][['latitude', 'longitude']]
                for _, row in uranium_deps.iterrows():
                    known_deposits.append({'lat': row['latitude'], 'lon': row['longitude'], 'mineral': 'Uranium'})
        except Exception as e:
            print(f"Error loading known deposits: {e}")
        return known_deposits

    # Add KNOWN DEPOSITS to map (GREEN markers)
    known_deposits = load_known_deposits(mineral)
    known_group = folium.FeatureGroup(name='Known Deposits (Green)')
    for dep in known_deposits:
        folium.CircleMarker(
            location=[dep['lat'], dep['lon']],
            radius=6,
            color='green',
            fill=True,
            fill_color='green',
            popup=f"KNOWN DEPOSIT<br>Mineral: {dep['mineral']}",
            fill_opacity=0.9
        ).add_to(known_group)
    known_group.add_to(m)
    print(f"Added {len(known_deposits)} known {mineral} deposits to map (green)")

    # Filter to only show DEPOSIT predictions (prediction = 1)
    if pred_col in pred_gdf.columns:
        deposit_gdf = pred_gdf[pred_gdf[pred_col] == 1]
        no_deposit_count = len(pred_gdf) - len(deposit_gdf)
        print(f"Showing {len(deposit_gdf)} predicted deposits (filtered out {no_deposit_count} no-deposit points)")
    else:
        deposit_gdf = pred_gdf

    # Add PREDICTED DEPOSITS to map (RED markers)
    predicted_group = folium.FeatureGroup(name='Predicted Deposits (Red)')
    
    # Heatmap for deposit areas only
    if prob_col in deposit_gdf.columns and len(deposit_gdf) > 0:
        heat_data = []
        for idx, row in deposit_gdf.iterrows():
            lat, lon = get_coords(row.geometry)
            heat_data.append([lat, lon, row[prob_col]])
        HeatMap(heat_data, radius=15).add_to(predicted_group)

    # Show PREDICTED DEPOSIT markers (red)
    if pred_col in deposit_gdf.columns:
        for idx, row in deposit_gdf.iterrows():
            lat, lon = get_coords(row.geometry)
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color='red',
                fill=True,
                fill_color='red',
                popup=f"PREDICTED DEPOSIT<br>Probability: {row.get(prob_col, 'N/A'):.2f}",
                fill_opacity=0.8
            ).add_to(predicted_group)
    predicted_group.add_to(m)

    folium.LayerControl().add_to(m)
    map_html = m._repr_html_()

    # Clean up
    if 'zip_path' in locals():
        os.unlink(zip_path)
    if 'extract_dir' in locals():
        import shutil
        shutil.rmtree(extract_dir)
    if 'prediction_area_path' in locals():
        os.unlink(prediction_area_path)

    return {"map_html": map_html, "filename": output_filename}


@app.post("/predict-area")
async def predict_area(
    request: dict
):
    """
    Predict mineral deposits in a custom polygon area.
    
    Accepts GeoJSON polygon and returns prediction map.
    
    Request body should include:
    {
        "geometry": {
            "type": "Polygon",
            "coordinates": [...]
        },
        "mineral": "All Minerals",
        "threshold": 0.5
    }
    """
    try:
        # Parse request
        geometry = request.get("geometry")
        mineral = request.get("mineral", "All Minerals")
        threshold = request.get("threshold", 0.5)
        
        if not geometry:
            raise HTTPException(status_code=400, detail="Geometry is required")

        # Create GeoDataFrame from GeoJSON
        gdf = gpd.GeoDataFrame(
            [{"id": 1}],
            geometry=[shapely.geometry.shape(geometry)],
            crs="EPSG:4326"
        )

        # Get model
        if mineral == 'All Minerals':
            models = get_models()
        else:
            models = get_models_by_mineral(mineral)
        if not models:
            raise HTTPException(status_code=404, detail="No models available")
        model_version = models[-1][1]

        # Run prediction pipeline directly on the GeoDataFrame
        output_filename, pred_gdf = run_prediction_pipeline_from_geojson(gdf, model_version, threshold, mineral=mineral)

        # Create map
        # Calculate map center
        if pred_gdf.empty:
            center_lat, center_lon = -22.0, 15.0
        else:
            try:
                # Reproject to a projected CRS for accurate centroid calculation
                projected_gdf = pred_gdf.to_crs('EPSG:3857')
                center_lat = projected_gdf.geometry.centroid.y.mean()
                center_lon = projected_gdf.geometry.centroid.x.mean()
                center = gpd.GeoSeries([Point(center_lon, center_lat)], crs='EPSG:3857').to_crs('EPSG:4326').iloc[0]
                center_lat, center_lon = center.y, center.x
            except Exception as e:
                center_lat, center_lon = -22.0, 15.0
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        prob_col = 'probabilit' if 'probabilit' in pred_gdf.columns else 'probability'
        pred_col = 'predictio' if 'predictio' in pred_gdf.columns else 'prediction'

        if prob_col in pred_gdf.columns:
            heat_data = [[row.geometry.y, row.geometry.x, row[prob_col]] for idx, row in pred_gdf.iterrows()]
            HeatMap(heat_data, radius=15).add_to(m)

        if pred_col in pred_gdf.columns:
            for idx, row in pred_gdf.iterrows():
                color = 'red' if row[pred_col] == 1 else 'blue'
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"Probability: {row.get(prob_col, 'N/A'):.2f}<br>Prediction: {'Deposit' if row[pred_col] == 1 else 'No Deposit'}",
                    fill_opacity=0.7
                ).add_to(m)

        folium.LayerControl().add_to(m)
        map_html = m._repr_html_()

        return {"map_html": map_html, "filename": output_filename}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(mineral: str = "All Minerals"):
    if mineral == 'All Minerals':
        models = get_models()
    else:
        models = get_models_by_mineral(mineral)

    alerts = get_active_alerts()
    report = generate_performance_report()

    # Performance trends plot
    fig_trends = plot_performance_trends()
    trends_img = None
    if fig_trends:
        buf = io.BytesIO()
        fig_trends.savefig(buf, format='png')
        buf.seek(0)
        trends_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_trends)

    # Feature importance plot
    fig_fi = plot_feature_importance_evolution()
    fi_img = None
    if fig_fi:
        buf = io.BytesIO()
        fig_fi.savefig(buf, format='png')
        buf.seek(0)
        fi_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_fi)

    return {
        "models": models,
        "alerts": alerts,
        "report": report,
        "trends_img": trends_img,
        "fi_img": fi_img
    }

@app.get("/models")
async def get_models_list(mineral: str = "All Minerals"):
    if mineral == 'All Minerals':
        models = get_models()
    else:
        models = get_models_by_mineral(mineral)
    return {"models": models}

@app.get("/files")
async def get_files_list(mineral: str = "All Minerals"):
    if mineral == 'All Minerals':
        files = get_files()
    else:
        files = [f for f in get_files() if f[6] == mineral]
    return {"files": files}

@app.get("/map")
async def get_map(mineral: str = "All Minerals"):
    prediction_files = [f for f in get_files() if f[3] == 'prediction' and (mineral == 'All Minerals' or f[6] == mineral)]
    m = folium.Map(location=[0, 0], zoom_start=2)
    for f in prediction_files:
        shp_path = f"data/predictions/{f[1]}.shp"
        if os.path.exists(shp_path):
            pred_gdf = gpd.read_file(shp_path)
            prob_col = 'probabilit' if 'probabilit' in pred_gdf.columns else 'probability'
            pred_col = 'predictio' if 'predictio' in pred_gdf.columns else 'prediction'
            for idx, row in pred_gdf.iterrows():
                if row[pred_col] == 1:
                    # Use centroid for polygons
                    if row.geometry.geom_type == 'Polygon':
                        center = row.geometry.centroid
                        lat, lon = center.y, center.x
                    else:
                        lat, lon = row.geometry.y, row.geometry.x
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=5,
                        color='red',
                        fill=True,
                        fill_color='red',
                        popup=f"Mineral: {f[6] or 'Unknown'}<br>Probability: {row.get(prob_col, 'N/A'):.2f}",
                        fill_opacity=0.7
                    ).add_to(m)
    folium.LayerControl().add_to(m)
    map_html = m._repr_html_()
    return {"map_html": map_html}

@app.get("/download/{file_id}")
async def download_file(file_id: int):
    files = get_files()
    file_info = next((f for f in files if f[0] == file_id), None)
    if file_info and os.path.exists(file_info[5]):
        return FileResponse(file_info[5], media_type='application/octet-stream', filename=file_info[1])
    raise HTTPException(status_code=404, detail="File not found")

# For batch processing, training iterations
training_active = False
training_iterations = 0

@app.post("/start_batch_training")
async def start_batch_training(mineral: str = Form(...)):
    global training_active, training_iterations
    training_active = True
    training_iterations = 0
    # Start training in background
    threading.Thread(target=batch_train, args=(mineral,)).start()
    return {"message": "Batch training started"}

@app.post("/stop_batch_training")
async def stop_batch_training():
    global training_active
    training_active = False
    return {"message": "Batch training stopped"}

@app.get("/training_status")
async def get_training_status():
    return {"active": training_active, "iterations": training_iterations}

def batch_train(mineral: str):
    global training_active, training_iterations
    while training_active:
        training_iterations += 1
        # Run training
        try:
            result = subprocess.run(['python3', 'train_model.py', '100', mineral], capture_output=True, text=True, cwd='.')
            if result.returncode != 0:
                break
        except:
            break
        import time
        time.sleep(3)  # Pause between iterations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)