import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import os
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from src.data_architecture import get_models, save_prediction_data, load_geoparquet, get_files
import geopandas as gpd
from src.feature_engineering import FeatureEngineer
import json

def load_prediction_area(file_path, file_type='shp'):
    """Load prediction area from shapefile or GeoJSON."""
    if file_type == 'shp':
        gdf = gpd.read_file(file_path)
    elif file_type == 'geojson':
        gdf = gpd.read_file(file_path, driver='GeoJSON')
    else:
        raise ValueError("Unsupported file type")
    return gdf

def apply_feature_engineering(prediction_gdf, base_features_file='features.parquet'):
    """Apply feature engineering to prediction area using FeatureEngineer."""
    # Load base features to get context (e.g., for spatial features)
    try:
        base_gdf = load_geoparquet(base_features_file, 'data/features/')
    except FileNotFoundError:
        base_gdf = prediction_gdf  # fallback

    # Initialize FeatureEngineer
    fe = FeatureEngineer()
    # Load deposits from database
    deposit_files = [f for f in get_files() if f[3] == 'deposit']
    if deposit_files:
        deposits_path = deposit_files[0][5]
        fe.deposits_gdf = gpd.read_file(deposits_path)
    else:
        fe.deposits_gdf = None
    fe.base_gdf = base_gdf

    # Compute features on prediction_gdf
    spatial_feat = fe.compute_spatial_features(prediction_gdf)
    topo_feat = fe.compute_topographic_features(prediction_gdf)
    geo_feat = fe.compute_geological_features(prediction_gdf)
    hydro_feat = fe.compute_hydrological_features(prediction_gdf)
    clim_feat = fe.compute_climatic_features(prediction_gdf)
    temp_feat = fe.compute_temporal_features(prediction_gdf)

    # Combine
    all_features = pd.concat([spatial_feat.drop(columns='geometry'), topo_feat.drop(columns='geometry'),
                              geo_feat.drop(columns='geometry'), hydro_feat.drop(columns='geometry'),
                              clim_feat.drop(columns='geometry'), temp_feat.drop(columns='geometry')], axis=1)

    # Spatial lag and multi-scale
    lag_feat = fe.compute_spatial_lag_features(prediction_gdf, all_features[['elevation', 'distance_to_deposit']] if 'elevation' in all_features.columns else pd.DataFrame())
    multi_feat = fe.compute_multi_scale_features(prediction_gdf, all_features)

    final_features = pd.concat([all_features, lag_feat, multi_feat], axis=1)

    # Merge to prediction_gdf
    for col in final_features.columns:
        prediction_gdf[col] = final_features[col]

    return prediction_gdf

def load_model(model_version):
    """Load model from registry."""
    models = get_models()
    model_info = next((m for m in models if m[1] == model_version), None)
    if not model_info:
        raise ValueError(f"Model version {model_version} not found")
    model_path = json.loads(model_info[2])['model_path'] if model_info[2] else f"models/model_v{model_version}.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model = joblib.load(model_path)
    return model

def generate_predictions(model, features_gdf, feature_cols=None):
    """Generate probability predictions."""
    if feature_cols is None:
        # Assume all numeric columns except geometry
        feature_cols = features_gdf.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != 'geometry']

    X = features_gdf[feature_cols]
    # Handle NaNs
    X = X.fillna(X.mean())

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)  # if binary classifier without proba

    features_gdf['probability'] = probs
    return features_gdf

def threshold_predictions(predictions_gdf, threshold=0.5):
    """Threshold probabilities to binary predictions."""
    predictions_gdf['prediction'] = (predictions_gdf['probability'] >= threshold).astype(int)
    predictions_gdf['confidence'] = np.abs(predictions_gdf['probability'] - 0.5) * 2  # confidence score
    return predictions_gdf

def export_to_shapefile(predictions_gdf, output_path):
    """Export predictions to shapefile."""
    predictions_gdf.to_file(output_path, driver='ESRI Shapefile')

def run_prediction_pipeline(prediction_area_path, model_version, threshold=0.5, output_filename=None, base_features_file='features.parquet'):
    """Main prediction pipeline."""
    # Load prediction area
    prediction_gdf = load_prediction_area(prediction_area_path)

    # Apply feature engineering
    prediction_gdf = apply_feature_engineering(prediction_gdf, base_features_file)

    # Load model
    model = load_model(model_version)

    # Generate predictions
    prediction_gdf = generate_predictions(model, prediction_gdf)

    # Threshold
    prediction_gdf = threshold_predictions(prediction_gdf, threshold)

    # Export to shapefile
    if output_filename is None:
        output_filename = f"prediction_{model_version}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.shp"
    output_path = f"data/predictions/{output_filename}"
    export_to_shapefile(prediction_gdf, output_path)

    # Save to data architecture
    save_prediction_data(prediction_gdf, output_filename)

    return output_filename, prediction_gdf