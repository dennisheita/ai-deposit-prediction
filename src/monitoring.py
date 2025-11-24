import os
import logging
import datetime
import json
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import geopandas as gpd
from shapely.geometry import Point
from .data_architecture import insert_alert, get_alerts, insert_training_run, update_training_run, get_training_runs, get_models

# Directories
LOGS_DIR = 'logs/'

# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(filename=os.path.join(LOGS_DIR, 'monitoring.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_system_resources():
    """Get current system resource usage."""
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    return cpu, memory, disk

def calculate_data_quality(gdf):
    """Calculate data quality metrics: completeness and spatial coverage."""
    # Completeness: percentage of non-null values
    completeness = gdf.drop(columns=['geometry']).notnull().mean().mean()

    # Spatial coverage: area in square degrees (approximate)
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    total_area = gdf.unary_union.convex_hull.area if len(gdf) > 0 else 0

    return completeness, total_area

def track_training_start(model_version):
    """Start tracking a training run."""
    start_time = datetime.datetime.now().isoformat()
    run_id = insert_training_run(model_version, start_time)
    logging.info(f"Training started for model {model_version}, run_id {run_id}")
    return run_id, start_time

def track_training_end(run_id, model, X, y, feature_names):
    """End tracking a training run with metrics."""
    end_time = datetime.datetime.now().isoformat()
    duration = (datetime.datetime.fromisoformat(end_time) - datetime.datetime.fromisoformat(get_training_runs()[-1][2])).total_seconds()  # approximate

    cpu, memory, disk = get_system_resources()

    # Predictions for metrics
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    f1 = f1_score(y, y_pred)

    feature_importances = dict(zip(feature_names, model.feature_importances_)) if hasattr(model, 'feature_importances_') else None

    update_training_run(run_id, end_time=end_time, duration=duration, cpu_usage=cpu, memory_usage=memory,
                        disk_usage=disk, accuracy=accuracy, auc=auc, f1=f1,
                        feature_importances=feature_importances, status='completed')

    logging.info(f"Training completed for run_id {run_id}: AUC={auc}, Accuracy={accuracy}, F1={f1}")

    # Check for alerts
    check_performance_degradation(run_id)
    check_storage_capacity(disk)

    return accuracy, auc, f1

def check_performance_degradation(current_run_id):
    """Check for performance degradation compared to previous runs."""
    runs = get_training_runs()
    if len(runs) < 2:
        return

    current = runs[-1]
    previous = runs[-2]

    if current[9] < previous[9] * 0.95:  # AUC dropped by 5%
        insert_alert('performance_degradation', f"AUC dropped from {previous[9]} to {current[9]}", 'warning')
        logging.warning("Performance degradation detected")

def check_storage_capacity(disk_usage):
    """Check if storage capacity is low."""
    if disk_usage > 90:
        insert_alert('storage_capacity', f"Disk usage at {disk_usage}%", 'critical')
        logging.critical("Storage capacity warning")

def log_training_failure(model_version, error_message):
    """Log a training failure."""
    insert_alert('training_failure', f"Training failed for {model_version}: {error_message}", 'error')
    logging.error(f"Training failure: {error_message}")

def generate_performance_report():
    """Generate a report on model performance over time."""
    runs = get_training_runs()
    if not runs:
        return "No training runs available."

    df = pd.DataFrame(runs, columns=['id', 'model_version', 'start_time', 'end_time', 'duration', 'cpu_usage',
                                     'memory_usage', 'disk_usage', 'accuracy', 'auc', 'f1', 'feature_importances',
                                     'data_completeness', 'spatial_coverage', 'status'])

    report = f"Total Training Runs: {len(df)}\n"
    report += f"Average AUC: {df['auc'].mean():.3f}\n"
    report += f"Average Accuracy: {df['accuracy'].mean():.3f}\n"
    report += f"Average F1: {df['f1'].mean():.3f}\n"
    report += f"Average Duration: {df['duration'].mean():.2f} seconds\n"

    return report

def plot_performance_trends():
    """Generate plots for performance trends."""
    runs = get_training_runs()
    if len(runs) < 2:
        return None

    df = pd.DataFrame(runs, columns=['id', 'model_version', 'start_time', 'end_time', 'duration', 'cpu_usage',
                                     'memory_usage', 'disk_usage', 'accuracy', 'auc', 'f1', 'feature_importances',
                                     'data_completeness', 'spatial_coverage', 'status'])

    df['start_time'] = pd.to_datetime(df['start_time'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].plot(df['start_time'], df['auc'], marker='o')
    axes[0,0].set_title('AUC Over Time')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('AUC')

    axes[0,1].plot(df['start_time'], df['accuracy'], marker='o')
    axes[0,1].set_title('Accuracy Over Time')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Accuracy')

    axes[1,0].plot(df['start_time'], df['f1'], marker='o')
    axes[1,0].set_title('F1 Score Over Time')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('F1')

    axes[1,1].plot(df['start_time'], df['duration'], marker='o')
    axes[1,1].set_title('Training Duration Over Time')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Duration (s)')

    plt.tight_layout()
    return fig

def plot_feature_importance_evolution():
    """Plot how feature importances have evolved."""
    runs = get_training_runs()
    if not runs:
        return None

    importances = []
    times = []
    for run in runs:
        if run[11]:  # feature_importances
            fi = json.loads(run[11])
            importances.append(fi)
            times.append(pd.to_datetime(run[2]))  # start_time

    if not importances:
        return None

    df_fi = pd.DataFrame(importances)
    df_fi['time'] = times

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in df_fi.columns[:-1]:  # exclude time
        ax.plot(df_fi['time'], df_fi[col], label=col, marker='o')

    ax.set_title('Feature Importance Evolution')
    ax.set_xlabel('Time')
    ax.set_ylabel('Importance')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def resolve_alert(alert_id):
    """Mark an alert as resolved."""
    from .data_architecture import get_alerts
    # Since get_alerts is already imported at the top, but to be safe
    # Actually, it's imported as get_alerts from data_architecture
    # But we need to update the alert in the database
    # The data_architecture has resolve_alert? No.
    # I need to add resolve_alert to data_architecture.py

    # For now, let's implement it here
    import sqlite3
    from .data_architecture import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE alerts SET resolved = 1 WHERE id = ?', (alert_id,))
    conn.commit()
    conn.close()

def get_active_alerts():
    """Get unresolved alerts for display."""
    alerts = get_alerts(resolved=False)
    return alerts