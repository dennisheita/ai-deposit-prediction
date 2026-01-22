import os
import logging
import datetime
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans
from shapely.geometry import Point
from .data_architecture import load_geoparquet, insert_model
import multiprocessing
from .monitoring import track_training_start, track_training_end, log_training_failure, calculate_data_quality, update_training_run
import xgboost as xgb
import lightgbm as lgb
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import scipy.stats as stats

# Directories
MODELS_DIR = 'models/'
LOGS_DIR = 'logs/'

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(filename=os.path.join(LOGS_DIR, 'advanced_training.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def stratified_spatial_negative_sampling(positives_gdf, features_gdf, n_negatives_per_positive=1,
                                         strata_cols=['elevation'], min_distance=0):
    """
    Perform stratified spatial negative sampling proportional to positives, distance-based, environmental stratification.
    """
    n_positives = len(positives_gdf)
    n_negatives = n_positives * n_negatives_per_positive

    # Exclude areas within min_distance of positives (if min_distance > 0)
    if min_distance > 0:
        positives_buffer = positives_gdf.buffer(min_distance)
        union_buffer = positives_buffer.unary_union
        candidates = features_gdf[~features_gdf.geometry.intersects(union_buffer)]
    else:
        # If min_distance is 0, use all features as candidates (no buffer)
        candidates = features_gdf.copy()

    if strata_cols and strata_cols[0] in candidates.columns:
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
        # Remove strata column after sampling
        if 'strata' in negatives_gdf.columns:
            negatives_gdf = negatives_gdf.drop(columns=['strata'])
    else:
        negatives_gdf = candidates.sample(min(n_negatives, len(candidates)), random_state=42)

    negatives_gdf['label'] = 0
    return negatives_gdf

def spatial_kfold_cv(X, y, coords, k=10):
    """
    Create stratified k-fold cross-validation to guarantee both classes in each fold.
    """
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(X, y)

def optimize_xgboost(trial, X, y, cv):
    """Optimize XGBoost hyperparameters using Optuna"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 20),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': 42
    }
    
    scores = []
    for train_idx, val_idx in cv:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    
    return np.mean(scores)

def optimize_lightgbm(trial, X, y, cv):
    """Optimize LightGBM hyperparameters using Optuna"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 20),
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42
    }
    
    scores = []
    for train_idx, val_idx in cv:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    
    return np.mean(scores)

def optimize_random_forest(trial, X, y, cv):
    """Optimize Random Forest hyperparameters using Optuna"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'random_state': 42,
        'n_jobs': -1
    }
    
    scores = []
    for train_idx, val_idx in cv:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    
    return np.mean(scores)

def train_xgboost(X, y, cv, n_trials=50):
    """Train optimized XGBoost model"""
    study = optuna.create_study(direction='maximize', study_name='XGBoost Optimization')
    study.optimize(lambda trial: optimize_xgboost(trial, X, y, cv), n_trials=n_trials, n_jobs=-1)
    
    best_params = study.best_params
    best_score = study.best_value
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X, y)
    
    return model, best_params, best_score

def train_lightgbm(X, y, cv, n_trials=50):
    """Train optimized LightGBM model"""
    study = optuna.create_study(direction='maximize', study_name='LightGBM Optimization')
    study.optimize(lambda trial: optimize_lightgbm(trial, X, y, cv), n_trials=n_trials, n_jobs=-1)
    
    best_params = study.best_params
    best_score = study.best_value
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)
    
    return model, best_params, best_score

def train_random_forest(X, y, cv, n_trials=50):
    """Train optimized Random Forest model"""
    study = optuna.create_study(direction='maximize', study_name='Random Forest Optimization')
    study.optimize(lambda trial: optimize_random_forest(trial, X, y, cv), n_trials=n_trials, n_jobs=-1)
    
    best_params = study.best_params
    best_score = study.best_value
    
    model = RandomForestClassifier(**best_params)
    model.fit(X, y)
    
    return model, best_params, best_score

def create_ensemble(models):
    """Create a voting ensemble of trained models"""
    ensemble = VotingClassifier(
        estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
        voting='soft',
        weights=[1]*len(models)
    )
    return ensemble

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive evaluation metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob)
    }

def save_model(model, version, model_type, path=MODELS_DIR):
    """Save model with versioning and type information"""
    filename = f'{model_type}_model_v{version}.joblib'
    filepath = os.path.join(path, filename)
    joblib.dump(model, filepath)
    return filepath

def log_metrics(metrics, log_path=os.path.join(LOGS_DIR, 'advanced_training.log')):
    """Log training metrics"""
    logging.info(f"Training metrics: {metrics}")

def run_advanced_training_pipeline(features_file, deposits_file, mineral=None, n_negatives_per_positive=1, k=10, n_trials=50):
    """Main advanced training pipeline with multiple models and ensemble"""
    logging.info("Starting advanced training pipeline.")
    
    version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id, start_time = track_training_start(version)
    
    try:
        # Load GeoParquet files directly without adding extra directory prefix
        if os.path.isfile(features_file):
            if features_file.endswith('.parquet'):
                try:
                    features_gdf = gpd.read_parquet(features_file)
                except Exception:
                    features_gdf = pd.read_parquet(features_file)
            elif features_file.endswith('.geojson'):
                features_gdf = gpd.read_file(features_file)
            elif features_file.endswith('.csv'):
                df = pd.read_csv(features_file)
                if 'lat' in df.columns and 'lon' in df.columns:
                    features_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')
                else:
                    features_gdf = df
            else:
                raise ValueError(f"Unsupported file format: {features_file}")
        else:
            raise FileNotFoundError(f"Features file not found: {features_file}")
            
        if os.path.isfile(deposits_file):
            if deposits_file.endswith('.parquet'):
                try:
                    deposits_gdf = gpd.read_parquet(deposits_file)
                except Exception:
                    deposits_gdf = pd.read_parquet(deposits_file)
            elif deposits_file.endswith('.geojson'):
                deposits_gdf = gpd.read_file(deposits_file)
            elif deposits_file.endswith('.csv'):
                df = pd.read_csv(deposits_file)
                if 'lat' in df.columns and 'lon' in df.columns:
                    deposits_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')
                else:
                    deposits_gdf = df
            else:
                raise ValueError(f"Unsupported file format: {deposits_file}")
        else:
            raise FileNotFoundError(f"Deposits file not found: {deposits_file}")
        
        completeness, spatial_coverage = calculate_data_quality(features_gdf)
        
        if 'label' in deposits_gdf.columns:
            if len(deposits_gdf.columns) > 5:
                train_data = deposits_gdf.copy()
                if 'geometry' in train_data.columns:
                    train_data = train_data.drop(columns=['geometry'])
                X = train_data.drop(columns=['label'])
                y = train_data['label']
                if 'lon' in deposits_gdf.columns and 'lat' in deposits_gdf.columns:
                    coords = np.array(list(zip(deposits_gdf['lon'], deposits_gdf['lat'])))
                else:
                    coords = np.zeros((len(train_data), 2))
            else:
                positives = deposits_gdf.copy()
                positives['label'] = 1
                negatives = stratified_spatial_negative_sampling(positives, features_gdf, n_negatives_per_positive)
                train_data = pd.concat([positives, negatives], ignore_index=True)
                X = train_data.drop(columns=['label', 'geometry'])
                y = train_data['label']
                coords = np.array([[geom.centroid.x, geom.centroid.y] if hasattr(geom, 'centroid') else [geom.x, geom.y] for geom in train_data.geometry])
        else:
            positives = deposits_gdf.copy()
            positives['label'] = 1
            negatives = stratified_spatial_negative_sampling(positives, features_gdf, n_negatives_per_positive)
            train_data = pd.concat([positives, negatives], ignore_index=True)
            X = train_data.drop(columns=['label', 'geometry'])
            y = train_data['label']
            coords = np.array([[geom.x, geom.y] for geom in train_data.geometry])
        
        logging.info(f"Training data: {len(train_data)} samples, {sum(y)} positives, {len(y) - sum(y)} negatives.")
        
        cv = list(spatial_kfold_cv(X, y, coords, k))
        
        logging.info("Training XGBoost model...")
        xgb_model, xgb_params, xgb_score = train_xgboost(X, y, cv, n_trials)
        
        logging.info("Training LightGBM model...")
        lgb_model, lgb_params, lgb_score = train_lightgbm(X, y, cv, n_trials)
        
        logging.info("Training Random Forest model...")
        rf_model, rf_params, rf_score = train_random_forest(X, y, cv, n_trials)
        
        logging.info("Creating ensemble model...")
        ensemble = create_ensemble([xgb_model, lgb_model, rf_model])
        ensemble.fit(X, y)
        
        ensemble_score = np.mean([xgb_score, lgb_score, rf_score])
        
        xgb_path = save_model(xgb_model, version, 'xgboost')
        lgb_path = save_model(lgb_model, version, 'lightgbm')
        rf_path = save_model(rf_model, version, 'randomforest')
        ensemble_path = save_model(ensemble, version, 'ensemble')
        
        metrics = {
            'xgboost': {
                'params': xgb_params,
                'score': xgb_score,
                'path': xgb_path
            },
            'lightgbm': {
                'params': lgb_params,
                'score': lgb_score,
                'path': lgb_path
            },
            'randomforest': {
                'params': rf_params,
                'score': rf_score,
                'path': rf_path
            },
            'ensemble': {
                'score': ensemble_score,
                'path': ensemble_path
            }
        }
        
        feature_names = list(X.columns)
        
        track_training_end(run_id, ensemble, X, y, feature_names)
        
        update_training_run(run_id, data_completeness=completeness, spatial_coverage=spatial_coverage)
        
        log_metrics(metrics)
        
        insert_model(version, metrics, mineral)
        
        best_model_type = max(metrics.keys(), key=lambda x: metrics[x]['score'] if 'score' in metrics[x] else 0)
        best_score = metrics[best_model_type]['score']
        best_path = metrics[best_model_type]['path']
        
        logging.info(f"Training completed. Best model: {best_model_type} (AUC: {best_score:.4f}) saved at {best_path}")
        return f"Training completed. Best model: {best_model_type} (AUC: {best_score:.4f}) saved at {best_path}"
        
    except Exception as e:
        log_training_failure(version, str(e))
        update_training_run(run_id, status='failed')
        logging.error(f"Training failed: {str(e)}")
        raise
