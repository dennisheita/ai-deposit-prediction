
import pandas as pd
import os
import sys
from src.training_pipeline import run_training_pipeline

def train_fast(mineral, file_path, label_col):
    print(f"Fast training for {mineral}...")
    
    if not os.path.exists(file_path):
        print(f"Skipping {mineral} - file not found")
        return

    df = pd.read_csv(file_path)
    
    # Rename label column
    if label_col in df.columns:
        df = df.rename(columns={label_col: 'label'})
    
    # Rename lat/lon
    if 'latitude' in df.columns:
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    # Save temp files
    os.makedirs('data/deposits', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    filename = f"{mineral.lower()}_fast.csv"
    df.to_csv(f"data/deposits/{filename}", index=False)
    df.to_csv(f"data/features/{filename}", index=False)

    # Minimal param grid for speed
    fast_grid = {
        'n_estimators': [10],
        'max_depth': [5],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    
    try:
        res = run_training_pipeline(filename, filename, mineral=mineral, param_grid=fast_grid, k=2)
        print(f"Success: {res}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Train Uranium, Gold, Copper quickly
    train_fast('Uranium', 'data/uranium_grid_with_features.csv', 'uranium_present')
    train_fast('Gold', 'data/gold grid_with_all_features.csv', 'gold_present')
    train_fast('Copper', 'data/copper_grid_with_features.csv', 'copper_present')
