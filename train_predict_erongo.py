#!/usr/bin/env python3
"""
Train and predict on Erongo area specifically for Gold, Uranium, and Copper.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import joblib
import datetime
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cluster import KMeans
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories
DATA_DIR = 'data'
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
DEPOTSITS_DIR = os.path.join(DATA_DIR, 'deposits')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
MODELS_DIR = 'models'
ERONGO_DIR = os.path.join(DATA_DIR, 'erongo')

# Ensure directories exist
for directory in [FEATURES_DIR, DEPOTSITS_DIR, PREDICTIONS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


def load_erongo_boundary():
    """Load Erongo boundary from GeoJSON."""
    erongo_path = os.path.join(ERONGO_DIR, 'Erongo.geojson')
    erongo_gdf = gpd.read_file(erongo_path)
    erongo_gdf = erongo_gdf.to_crs('EPSG:4326')  # Ensure WGS84
    logging.info(f"Loaded Erongo boundary with CRS: {erongo_gdf.crs}")
    logging.info(f"Erongo bounds: {erongo_gdf.total_bounds}")
    return erongo_gdf


def load_and_convert_feature_file(file_path):
    """Load parquet file and convert to GeoDataFrame."""
    df = pd.read_parquet(file_path)
    
    # Create geometry from lat/lon
    geometry = [Point(lon, lat) for lat, lon in zip(df['latitude'], df['longitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    return gdf


def filter_to_erongo(gdf, erongo_gdf):
    """Filter GeoDataFrame to Erongo region."""
    # Check if geometries are within Erongo boundary
    erongo_geom = erongo_gdf.geometry.iloc[0]
    filtered_gdf = gdf[gdf.geometry.within(erongo_geom)].copy()
    
    logging.info(f"Filtered from {len(gdf)} to {len(filtered_gdf)} points in Erongo")
    return filtered_gdf


def prepare_training_data(mineral, erongo_gdf):
    """Prepare training data for specific mineral."""
    logging.info(f"Preparing training data for {mineral}")
    
    # Find corresponding feature file
    mineral_files = {
        'Gold': 'tmpo5nqdddr_gold_chain_data.parquet',
        'Uranium': 'tmpxy4ga2f4_uranium_grid_with_features.parquet',
        'Copper': 'tmpj42yxw1e_copper_grid_with_features.parquet'
    }
    
    file_name = mineral_files.get(mineral)
    if not file_name:
        raise ValueError(f"Unknown mineral: {mineral}")
    
    file_path = os.path.join(FEATURES_DIR, file_name)
    
    # Load and convert to GeoDataFrame
    gdf = load_and_convert_feature_file(file_path)
    
    # Filter to Erongo region
    erongo_gdf = filter_to_erongo(gdf, erongo_gdf)
    
    # Check if we have any data
    if len(erongo_gdf) == 0:
        raise ValueError(f"No {mineral} data available in Erongo region")
    
    logging.info(f"Found {len(erongo_gdf)} training points for {mineral} in Erongo")
    
    return erongo_gdf


def spatial_kfold_cv(X, y, coords, k=10):
    """Create spatial k-fold cross-validation using spatial clustering for groups."""
    kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
    groups = kmeans.labels_
    return GroupKFold(n_splits=k).split(X, y, groups)


def train_model(X, y, coords, mineral):
    """Train Random Forest model with spatial CV."""
    logging.info(f"Training model for {mineral} with {len(X)} samples")
    
    # Spatial CV
    cv = list(spatial_kfold_cv(X, y, coords, k=5))
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [10, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Train
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    
    logging.info(f"Best {mineral} model: AUC = {best_score:.4f}, Params = {best_params}")
    
    return best_model, best_score, best_params


def save_model(model, mineral):
    """Save trained model with versioning."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    version = f"{mineral.lower()}_erongo_{timestamp}"
    file_path = os.path.join(MODELS_DIR, f"model_{version}.joblib")
    joblib.dump(model, file_path)
    
    logging.info(f"Model saved to: {file_path}")
    return version, file_path


def sample_erongo_to_points(erongo_gdf, spacing=0.0003):
    """Sample Erongo polygon into grid points for prediction."""
    min_x, min_y, max_x, max_y = erongo_gdf.total_bounds
    
    x_coords = np.arange(min_x, max_x, spacing)
    y_coords = np.arange(min_y, max_y, spacing)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            points.append(point)
    
    points_gdf = gpd.GeoDataFrame(
        geometry=points,
        crs=erongo_gdf.crs
    )
    
    within_points = points_gdf[points_gdf.within(erongo_gdf.geometry.iloc[0])]
    
    logging.info(f"Generated {len(within_points)} prediction points in Erongo")
    return within_points


def extract_features_from_points(points_gdf, all_features_gdf):
    """Extract features for each prediction point using nearest neighbor interpolation."""
    from scipy.spatial import cKDTree
    
    # Prepare spatial index for all features
    all_coords = np.array([(point.x, point.y) for point in all_features_gdf.geometry])
    tree = cKDTree(all_coords)
    
    # Query nearest neighbors for each prediction point
    points_coords = np.array([(point.x, point.y) for point in points_gdf.geometry])
    distances, indices = tree.query(points_coords, k=1)
    
    # Extract features from nearest neighbors
    points_gdf['nearest_idx'] = indices
    
    for col in all_features_gdf.columns:
        if col not in ['geometry', 'latitude', 'longitude']:
            points_gdf[col] = all_features_gdf.iloc[indices][col].values
    
    logging.info(f"Extracted features for {len(points_gdf)} points")
    return points_gdf


def predict_erongo_area(model, features_gdf, erongo_gdf):
    """Generate predictions for entire Erongo area."""
    logging.info("Generating prediction grid for Erongo")
    
    # Sample Erongo polygon
    points_gdf = sample_erongo_to_points(erongo_gdf)
    
    if points_gdf.empty:
        raise ValueError("Could not generate any prediction points")
    
    # Extract features
    points_gdf = extract_features_from_points(points_gdf, features_gdf)
    
    # Prepare features for prediction
    feature_cols = [c for c in points_gdf.columns if c not in 
                   ['geometry', 'nearest_idx', 'latitude', 'longitude', 
                    'copper_present', 'gold_present', 'uranium_present']]
    
    # Handle missing features
    X = points_gdf[feature_cols]
    X = X.fillna(X.mean())
    
    # Generate predictions
    points_gdf['probability'] = model.predict_proba(X)[:, 1]
    points_gdf['prediction'] = (points_gdf['probability'] >= 0.5).astype(int)
    points_gdf['confidence'] = np.abs(points_gdf['probability'] - 0.5) * 2
    
    return points_gdf


def export_to_shapefile(gdf, mineral, version):
    """Export predictions to shapefile."""
    output_filename = f"prediction_{mineral.lower()}_erongo_{version}.shp"
    output_path = os.path.join(PREDICTIONS_DIR, output_filename)
    gdf.to_file(output_path, driver='ESRI Shapefile')
    logging.info(f"Predictions exported to: {output_path}")
    return output_filename


def main():
    """Main pipeline."""
    # Load Erongo boundary
    erongo_gdf = load_erongo_boundary()
    
    # Process each mineral
    for mineral in ['Gold', 'Uranium', 'Copper']:
        try:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing {mineral}")
            logging.info('='*50)
            
            # Step 1: Prepare training data
            features_gdf = prepare_training_data(mineral, erongo_gdf)
            
            # Step 2: Separate features and labels
            label_col = f'{mineral.lower()}_present'
            if label_col not in features_gdf.columns:
                logging.warning(f"Label column '{label_col}' not found. Using default 'label' column.")
                label_col = 'label'
            
            # Check if label column exists
            if label_col not in features_gdf.columns:
                # Try to infer from other columns (e.g., copper_present for Copper)
                available_cols = list(features_gdf.columns)
                raise ValueError(f"Label column '{label_col}' not found. Available columns: {available_cols}")
            
            # Extract features and labels
            feature_cols = [c for c in features_gdf.columns if c not in 
                           ['geometry', 'latitude', 'longitude', label_col]]
            X = features_gdf[feature_cols]
            y = features_gdf[label_col].astype(int)
            
            # Extract coordinates for spatial CV
            coords = np.array([[point.x, point.y] for point in features_gdf.geometry])
            
            # Step 3: Train model
            model, best_score, best_params = train_model(X, y, coords, mineral)
            
            # Step 4: Save model
            version, model_path = save_model(model, mineral)
            
            # Step 5: Predict on Erongo area
            predictions_gdf = predict_erongo_area(model, features_gdf, erongo_gdf)
            
            # Step 6: Add mineral information
            predictions_gdf['mineral'] = mineral
            predictions_gdf['model_version'] = version
            
            # Step 7: Export to shapefile
            output_filename = export_to_shapefile(predictions_gdf, mineral, version)
            
            logging.info(f"Successfully completed processing for {mineral}")
            
        except Exception as e:
            logging.error(f"Error processing {mineral}: {str(e)}")
            import traceback
            logging.error(f"Stack trace: {traceback.format_exc()}")
            continue
    
    logging.info("\nProcessing complete!")
    logging.info(f"Check {PREDICTIONS_DIR} for shapefiles")
    logging.info(f"Check {MODELS_DIR} for trained models")


if __name__ == "__main__":
    main()
