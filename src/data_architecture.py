import sqlite3
import os
import datetime
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import pyarrow as pa
import pyarrow.parquet as pq

# Database path
DB_PATH = 'data/metadata.db'

def initialize_database():
    """Initialize the SQLite database and create tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            type TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            status TEXT NOT NULL,
            path TEXT NOT NULL
        )
    ''')

    # Create models table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL,
            performance_metrics TEXT,
            created_date TEXT NOT NULL
        )
    ''')

    # Create alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT NOT NULL,
            created_date TEXT NOT NULL,
            resolved INTEGER DEFAULT 0
        )
    ''')

    # Create training_runs table for detailed metrics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration REAL,
            cpu_usage REAL,
            memory_usage REAL,
            disk_usage REAL,
            accuracy REAL,
            auc REAL,
            f1 REAL,
            feature_importances TEXT,
            data_completeness REAL,
            spatial_coverage REAL,
            status TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def insert_file(filename, file_type, status, path):
    """Insert a new file record into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    upload_date = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO files (filename, type, upload_date, status, path)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, file_type, upload_date, status, path))
    conn.commit()
    conn.close()

def get_files():
    """Retrieve all file records from the database."""
    initialize_database()  # Ensure tables exist
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM files')
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_file_status(file_id, status):
    """Update the status of a file."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE files SET status = ? WHERE id = ?', (status, file_id))
    conn.commit()
    conn.close()

def insert_model(version, performance_metrics):
    """Insert a new model record into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    created_date = datetime.datetime.now().isoformat()
    metrics_json = json.dumps(performance_metrics) if performance_metrics else None
    cursor.execute('''
        INSERT INTO models (version, performance_metrics, created_date)
        VALUES (?, ?, ?)
    ''', (version, metrics_json, created_date))
    conn.commit()
    conn.close()

def get_models():
    """Retrieve all model records from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM models')
    rows = cursor.fetchall()
    conn.close()
    return rows

def insert_alert(alert_type, message, severity):
    """Insert a new alert record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    created_date = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO alerts (type, message, severity, created_date)
        VALUES (?, ?, ?, ?)
    ''', (alert_type, message, severity, created_date))
    conn.commit()
    conn.close()

def get_alerts(resolved=False):
    """Retrieve alerts, optionally only unresolved."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if resolved:
        cursor.execute('SELECT * FROM alerts')
    else:
        cursor.execute('SELECT * FROM alerts WHERE resolved = 0')
    rows = cursor.fetchall()
    conn.close()
    return rows

def resolve_alert(alert_id):
    """Mark an alert as resolved."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE alerts SET resolved = 1 WHERE id = ?', (alert_id,))
    conn.commit()
    conn.close()

def insert_training_run(model_version, start_time, end_time=None, duration=None, cpu_usage=None,
                        memory_usage=None, disk_usage=None, accuracy=None, auc=None, f1=None,
                        feature_importances=None, data_completeness=None, spatial_coverage=None, status='running'):
    """Insert a new training run record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    fi_json = json.dumps(feature_importances) if feature_importances else None
    cursor.execute('''
        INSERT INTO training_runs (model_version, start_time, end_time, duration, cpu_usage, memory_usage,
                                   disk_usage, accuracy, auc, f1, feature_importances, data_completeness,
                                   spatial_coverage, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (model_version, start_time, end_time, duration, cpu_usage, memory_usage, disk_usage,
          accuracy, auc, f1, fi_json, data_completeness, spatial_coverage, status))
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def update_training_run(run_id, **kwargs):
    """Update a training run record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    set_clause = ', '.join([f'{k} = ?' for k in kwargs.keys()])
    values = list(kwargs.values())
    values.append(run_id)
    cursor.execute(f'UPDATE training_runs SET {set_clause} WHERE id = ?', values)
    conn.commit()
    conn.close()

def get_training_runs():
    """Retrieve all training run records."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM training_runs')
    rows = cursor.fetchall()
    conn.close()
    return rows

def save_geoparquet(gdf, filename, directory='data/features/'):
    """Save a GeoDataFrame to GeoParquet format in the specified directory."""
    path = os.path.join(directory, filename)
    gdf.to_parquet(path)
    # Insert into database
    insert_file(filename, 'geoparquet', 'uploaded', path)

def load_geoparquet(filename, directory='data/features/'):
    """Load a GeoDataFrame from GeoParquet format."""
    path = os.path.join(directory, filename)
    return gpd.read_parquet(path)

def validate_crs(gdf, expected_crs='EPSG:4326'):
    """Validate if the GeoDataFrame has the expected CRS."""
    if gdf.crs != expected_crs:
        raise ValueError(f"CRS mismatch: expected {expected_crs}, got {gdf.crs}")
    return True

def save_deposit_data(data, filename, directory='data/deposits/'):
    """Save deposit data (e.g., CSV or Parquet) to the deposits directory."""
    path = os.path.join(directory, filename)
    if isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False)
    elif isinstance(data, gpd.GeoDataFrame):
        data.to_file(path, driver='GeoJSON')  # or other format
    insert_file(filename, 'deposit', 'uploaded', path)

def save_prediction_data(data, filename, directory='data/predictions/'):
    """Save prediction data to the predictions directory."""
    path = os.path.join(directory, filename)
    if isinstance(data, gpd.GeoDataFrame):
        data.to_file(path, driver='ESRI Shapefile')
    elif isinstance(data, pd.DataFrame):
        data.to_parquet(path)
    insert_file(filename, 'prediction', 'uploaded', path)

# Initialize the database when the module is imported
initialize_database()