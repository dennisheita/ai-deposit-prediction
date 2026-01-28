import os
import logging
import datetime
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cluster import KMeans
from shapely.geometry import Point
from .data_architecture import load_geoparquet, insert_model
import multiprocessing
from .monitoring import track_training_start, track_training_end, log_training_failure, calculate_data_quality, update_training_run

# Directories
MODELS_DIR = 'models/'
LOGS_DIR = 'logs/'

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(filename=os.path.join(LOGS_DIR, 'training.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def stratified_spatial_negative_sampling(positives_gdf, features_gdf, n_negatives_per_positive=1,
                                         strata_cols=['elevation'], min_distance=1000):
    """
    Perform stratified spatial negative sampling proportional to positives, distance-based, environmental stratification.
    """
    n_positives = len(positives_gdf)
    n_negatives = n_positives * n_negatives_per_positive

    # Ensure features_gdf has geometry
    if not hasattr(features_gdf, 'geometry') or features_gdf.geometry.isna().all():
        logging.warning("Features GeoDataFrame has no valid geometry. Creating from lat/lon if available.")
        if 'lon' in features_gdf.columns and 'lat' in features_gdf.columns:
            features_gdf = gpd.GeoDataFrame(
                features_gdf,
                geometry=gpd.points_from_xy(features_gdf['lon'], features_gdf['lat']),
                crs='EPSG:4326'
            )
        else:
            logging.error("Cannot perform spatial negative sampling: no geometry or lat/lon columns found.")
            # Return empty DataFrame with label column instead of GeoDataFrame
            empty_df = pd.DataFrame(columns=list(features_gdf.columns) + ['label'])
            return empty_df

    # Exclude areas within min_distance of positives
    try:
        positives_buffer = positives_gdf.buffer(min_distance)
        union_buffer = positives_buffer.unary_union
        candidates = features_gdf[~features_gdf.geometry.intersects(union_buffer)]
    except Exception as e:
        logging.warning(f"Error in spatial buffer operation: {e}. Using all features as candidates.")
        candidates = features_gdf.copy()

    if len(candidates) == 0:
        logging.warning("No candidates available for negative sampling after spatial filtering.")
        # Fallback: use features without spatial filtering
        candidates = features_gdf.copy()

    if len(candidates) == 0:
        logging.error("No candidates available for negative sampling at all.")
        # Return empty DataFrame with label column instead of GeoDataFrame
        empty_df = pd.DataFrame(columns=list(features_gdf.columns) + ['label'])
        return empty_df

    # Try stratified sampling if strata column exists and has valid data
    if strata_cols and strata_cols[0] in candidates.columns:
        try:
            # Check if strata column has numeric data
            strata_col = candidates[strata_cols[0]]
            if pd.api.types.is_numeric_dtype(strata_col) and not strata_col.isna().all():
                candidates = candidates.copy()
                candidates['strata'] = pd.qcut(candidates[strata_cols[0]], q=5, labels=False, duplicates='drop')
                strata_counts = candidates['strata'].value_counts()
                negatives = []
                for strata in strata_counts.index:
                    strata_candidates = candidates[candidates['strata'] == strata]
                    sample_size = int(n_negatives / len(strata_counts))
                    if sample_size > 0:
                        sampled = strata_candidates.sample(min(sample_size, len(strata_candidates)), random_state=42)
                        negatives.append(sampled)
                negatives_gdf = pd.concat(negatives) if negatives else gpd.GeoDataFrame()
            else:
                raise ValueError("Strata column is not numeric")
        except Exception as e:
            logging.warning(f"Stratified sampling failed: {e}. Using simple random sampling.")
            sample_size = min(n_negatives, len(candidates))
            negatives_gdf = candidates.sample(sample_size, random_state=42) if len(candidates) > 0 else gpd.GeoDataFrame()
    else:
        # Simple random sampling
        sample_size = min(n_negatives, len(candidates))
        negatives_gdf = candidates.sample(sample_size, random_state=42) if len(candidates) > 0 else gpd.GeoDataFrame()

    if len(negatives_gdf) == 0:
        logging.error("Failed to generate any negative samples.")
    else:
        logging.info(f"Generated {len(negatives_gdf)} negative samples.")

    negatives_gdf['label'] = 0
    return negatives_gdf

def spatial_kfold_cv(X, y, coords, k=10):
    """
    Create spatial k-fold cross-validation using spatial clustering for groups.
    """
    # Cluster coordinates into k groups
    kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
    groups = kmeans.labels_
    return GroupKFold(n_splits=k).split(X, y, groups)

def train_rf_parallel(X, y, param_grid, cv, n_jobs=-1):
    """
    Train Random Forest with hyperparameter grid search in parallel.
    """
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=n_jobs, verbose=1)
    grid_search.fit(X, y)
    return grid_search

def select_top_models(grid_search, top_percent=0.1):
    """
    Select top 10% models based on CV scores.
    """
    results = pd.DataFrame(grid_search.cv_results_)
    n_top = max(1, int(len(results) * top_percent))
    top_indices = results['mean_test_score'].nlargest(n_top).index
    top_params = [results.loc[i, 'params'] for i in top_indices]
    # For ensemble, we can return top params, but for simplicity, return best
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, top_params

def save_model(model, version, path=MODELS_DIR):
    """
    Save model with versioning.
    """
    filename = f'model_v{version}.joblib'
    filepath = os.path.join(path, filename)
    joblib.dump(model, filepath)
    return filepath

def log_metrics(metrics, log_path=os.path.join(LOGS_DIR, 'training.log')):
    """
    Log training metrics.
    """
    logging.info(f"Training metrics: {metrics}")

def run_training_pipeline(features_file, deposits_file, mineral=None, n_negatives_per_positive=1, param_grid=None, k=10, top_percent=0.1):
    """
    Main training pipeline with monitoring.
    """
    logging.info("Starting training pipeline.")

    # Start monitoring
    version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id, start_time = track_training_start(version)

    try:
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 500, 1000, 2000],
                'max_depth': [10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        # Load data
        features_gdf = load_geoparquet(features_file, 'data/features/')
        deposits_gdf = load_geoparquet(deposits_file, 'data/deposits/')

        # Calculate data quality
        completeness, spatial_coverage = calculate_data_quality(features_gdf)

        # Check if deposits_gdf already has labels (from CSV upload)
        if 'label' in deposits_gdf.columns:
            # If labels exist, we might need to merge with features if features are separate
            # But based on the user's data, it seems they might be uploading pre-labeled data
            # Let's handle the case where deposits_gdf is actually the full training set
            if len(deposits_gdf.columns) > 5: # Heuristic: if it has many columns, it might be the full dataset
                 train_data = deposits_gdf.copy()
                 if 'geometry' in train_data.columns:
                     train_data = train_data.drop(columns=['geometry'])
                 X = train_data.drop(columns=['label'])
                 y = train_data['label']
                 # Create dummy coords if geometry is missing, or use lat/lon if available
                 if 'lon' in deposits_gdf.columns and 'lat' in deposits_gdf.columns:
                     coords = np.array(list(zip(deposits_gdf['lon'], deposits_gdf['lat'])))
                 else:
                     # Fallback to random coords or 0 if no spatial info (not ideal for spatial CV but works for code)
                     coords = np.zeros((len(train_data), 2))
            else:
                 # Standard flow: deposits are just locations
                 positives = deposits_gdf.copy()
                 positives['label'] = 1
                 negatives = stratified_spatial_negative_sampling(positives, features_gdf, n_negatives_per_positive)
                 train_data = pd.concat([positives, negatives], ignore_index=True)
                 # Only drop geometry if it exists
                 cols_to_drop = ['label']
                 if 'geometry' in train_data.columns:
                     cols_to_drop.append('geometry')
                 X = train_data.drop(columns=cols_to_drop)
                 y = train_data['label']
                 if 'geometry' in train_data.columns and hasattr(train_data, 'geometry'):
                     coords = np.array([[geom.centroid.x, geom.centroid.y] if hasattr(geom, 'centroid') else [geom.x, geom.y] for geom in train_data.geometry])
                 else:
                     coords = np.zeros((len(train_data), 2))
        else:
            # Standard flow
            positives = deposits_gdf.copy()
            positives['label'] = 1
            negatives = stratified_spatial_negative_sampling(positives, features_gdf, n_negatives_per_positive)
            train_data = pd.concat([positives, negatives], ignore_index=True)
            
            # Check if we have valid training data
            if len(train_data) == 0:
                raise ValueError("No training data available. Cannot train model.")
            if len(negatives) == 0:
                raise ValueError("No negative samples generated. Cannot train model.")
            
            # Only drop geometry if it exists
            cols_to_drop = ['label']
            if 'geometry' in train_data.columns:
                cols_to_drop.append('geometry')
            X = train_data.drop(columns=cols_to_drop)
            y = train_data['label']
            
            # Handle case where geometry might be None or not exist
            coords_list = []
            if 'geometry' in train_data.columns and hasattr(train_data, 'geometry'):
                for geom in train_data.geometry:
                    if geom is not None and hasattr(geom, 'x'):
                        coords_list.append([geom.x, geom.y])
                    elif geom is not None and hasattr(geom, 'centroid'):
                        coords_list.append([geom.centroid.x, geom.centroid.y])
                    else:
                        # Fallback: use random coordinates or zeros
                        coords_list.append([0.0, 0.0])
                coords = np.array(coords_list)
            else:
                coords = np.zeros((len(train_data), 2))

        logging.info(f"Training data: {len(train_data)} samples, {sum(y)} positives, {len(y) - sum(y)} negatives.")

        # Check if we have both positive and negative samples
        if sum(y) == 0:
            raise ValueError("No positive samples in training data. Cannot train model.")
        if sum(y) == len(y):
            raise ValueError("No negative samples in training data. Cannot train model.")

        # Clean the feature data - handle 'No Data' and other string values
        # Drop non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found in training data.")
        
        X = X[numeric_columns]
        
        # Handle missing values
        X = X.replace(['No Data', 'ND', 'N/A', 'null', 'NULL', ''], np.nan)
        
        # Check for any remaining non-numeric values
        for col in X.columns:
            if X[col].dtype == object:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    logging.warning(f"Dropping column {col} - cannot convert to numeric")
                    X = X.drop(columns=[col])
        
        # Fill NaN values with median
        X = X.fillna(X.median())
        
        logging.info(f"After cleaning: X shape = {X.shape}, numeric columns = {len(X.columns)}")

        # Spatial CV
        cv = list(spatial_kfold_cv(X, y, coords, k))

        # Train
        grid_search = train_rf_parallel(X, y, param_grid, cv, n_jobs=multiprocessing.cpu_count())

        # Select top
        best_model, best_params, best_score, top_params = select_top_models(grid_search, top_percent)

        # Save best model
        model_path = save_model(best_model, version)

        # Metrics
        metrics = {
            'best_params': best_params,
            'best_score': best_score,
            'model_path': model_path,
            'n_top_models': len(top_params)
        }

        # End monitoring
        feature_names = list(X.columns)
        track_training_end(run_id, best_model, X, y, feature_names)

        # Update training run with data quality
        update_training_run(run_id, data_completeness=completeness, spatial_coverage=spatial_coverage)

        # Log
        log_metrics(metrics)

        # Update metadata
        insert_model(version, metrics, mineral)

        logging.info(f"Training completed. Best model saved at {model_path} with AUC {best_score}")
        return f"Training completed. Best model saved at {model_path} with AUC {best_score}"

    except Exception as e:
        log_training_failure(version, str(e))
        update_training_run(run_id, status='failed')
        logging.error(f"Training failed: {str(e)}")
        raise