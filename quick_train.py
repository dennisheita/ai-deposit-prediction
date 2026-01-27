#!/usr/bin/env python3
"""
Quick training script that uses CSV files directly without GeoParquet conversion.
Designed for speed and simplicity.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys
from src.training_pipeline import run_training_pipeline
import time

def main():
    # Define the data files and corresponding minerals
    data_files = [
        ('data/copper_complete_real.csv', 'Copper', 'copper_present'),
        ('data/gold_complete_real.csv', 'Gold', 'gold_present'),
        ('data/uranium_complete_real.csv', 'Uranium', 'uranium_present')
    ]
    
    # Create a temporary directory for GeoParquet files
    temp_dir = 'data/temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each mineral
    for csv_path, mineral, label_col in data_files:
        print(f"\n{'='*50}")
        print(f"Processing {mineral} data...")
        print('='*50)
        
        # Load CSV file
        df = pd.read_csv(csv_path)
        
        # Create GeoDataFrame from lat/lon
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
            crs='EPSG:4326'
        )
        
        # Save as GeoParquet
        filename = os.path.basename(csv_path).replace('.csv', '.geoparquet')
        features_path = os.path.join(temp_dir, f"{mineral.lower()}_features.geoparquet")
        deposits_path = os.path.join(temp_dir, f"{mineral.lower()}_deposits.geoparquet")
        
        gdf.to_parquet(features_path)
        gdf.to_parquet(deposits_path)
        
        print(f"Saved {mineral} features to: {features_path}")
        print(f"Saved {mineral} deposits to: {deposits_path}")
        
        # Train models
        num_runs = 15
        print(f"\nTraining {num_runs} models for {mineral}...")
        
        for i in range(num_runs):
            run_num = i + 1
            print(f"\nRun {run_num}/{num_runs} for {mineral}")
            
            try:
                result = run_training_pipeline(
                    os.path.basename(features_path),
                    os.path.basename(deposits_path),
                    mineral=mineral,
                    n_negatives_per_positive=2,
                    k=10
                )
                print(f"Success: {result}")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
    
    print(f"\n{'='*50}")
    print("Training completed!")
    print('='*50)
    
    # Cleanup temporary files if needed
    # import shutil
    # shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
