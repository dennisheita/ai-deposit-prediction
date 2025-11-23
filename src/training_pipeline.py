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

    # Exclude areas within min_distance of positives
    positives_buffer = positives_gdf.buffer(min_distance)
    union_buffer = positives_buffer.unary_union
    candidates = features_gdf[~features_gdf.geometry.intersects(union_buffer)]

    if strata_cols and strata_cols[0] in candidates.columns:
        # Create strata based on quantiles
        candidates = candidates.copy()
        candidates['strata'] = pd.qcut(candidates[strata_cols[0]], q=5, labels=False, duplicates='drop')
        strata_counts = candidates['strata'].value_counts()
        negatives = []
        for strata in strata_counts.index:
            strata_candidates = candidates[candidates['strata'] == strata]
            # Proportional to positives in strata, but simplified: equal across strata
            sample_size = int(n_negatives / len(strata_counts))
            if sample_size > 0:
                sampled = strata_candidates.sample(min(sample_size, len(strata_candidates)), random_state=42)
                negatives.append(sampled)
        negatives_gdf = pd.concat(negatives) if negatives else gpd.GeoDataFrame()
    else:
        # Simple random sampling
        negatives_gdf = candidates.sample(min(n_negatives, len(candidates)), random_state=42)

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

def run_training_pipeline(features_file, deposits_file, n_negatives_per_positive=1, param_grid=None, k=10, top_percent=0.1):
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

        # Assume deposits_gdf has geometry and is positives (label=1), features_gdf has predictors
        positives = deposits_gdf.copy()
        positives['label'] = 1

        # Negative sampling from features not near positives
        negatives = stratified_spatial_negative_sampling(positives, features_gdf, n_negatives_per_positive)

        # Combine
        train_data = pd.concat([positives, negatives], ignore_index=True)
        X = train_data.drop(columns=['label', 'geometry'])
        y = train_data['label']
        coords = np.array([[geom.x, geom.y] for geom in train_data.geometry])

        logging.info(f"Training data: {len(train_data)} samples, {sum(y)} positives, {len(y) - sum(y)} negatives.")

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
        insert_model(version, metrics)

        logging.info(f"Training completed. Best model saved at {model_path} with AUC {best_score}")
        return f"Training completed. Best model saved at {model_path} with AUC {best_score}"

    except Exception as e:
        log_training_failure(version, str(e))
        update_training_run(run_id, status='failed')
        logging.error(f"Training failed: {str(e)}")
        raise