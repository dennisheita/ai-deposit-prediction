#!/usr/bin/env python3
"""
Script to prepare and train models using the new real data files.
Converts CSV files to GeoParquet and runs training pipeline multiple times.
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from src.training_pipeline import run_training_pipeline
import time

def csv_to_geoparquet(csv_path, output_dir):
    """Convert CSV file with lat/lon to GeoParquet format."""
    df = pd.read_csv(csv_path)
    
    # Create geometry from latitude and longitude
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as GeoParquet
    filename = os.path.basename(csv_path).replace('.csv', '.geoparquet')
    output_path = os.path.join(output_dir, filename)
    gdf.to_parquet(output_path)
    
    # Also copy to deposits directory since these files contain both features and labels
    deposits_dir = 'data/deposits'
    os.makedirs(deposits_dir, exist_ok=True)
    deposits_path = os.path.join(deposits_dir, filename)
    import shutil
    shutil.copy2(output_path, deposits_path)
    
    print(f"Converted {csv_path} to {output_path}")
    print(f"Copied to deposits directory: {deposits_path}")
    return output_path

def main():
    # Define the data files and corresponding minerals
    data_files = [
        ('data/copper_complete_real.csv', 'Copper', 'copper_present'),
        ('data/gold_complete_real.csv', 'Gold', 'gold_present'),
        ('data/uranium_complete_real.csv', 'Uranium', 'uranium_present')
    ]
    
    # Prepare data
    prepared_files = []
    for csv_path, mineral, label_col in data_files:
        # Convert CSV to GeoParquet
        geoparquet_path = csv_to_geoparquet(csv_path, 'data/features')
        
        # Extract just the filename, not full path
        filename = os.path.basename(geoparquet_path)
        
        # Since these files contain both features and labels, we'll use them as both deposits and features
        prepared_files.append((filename, filename, mineral))
    
    # Train models multiple times
    num_runs = 15
    total_runs = len(prepared_files) * num_runs
    print(f"Starting {total_runs} training runs...")
    
    run_count = 0
    for features_path, deposits_path, mineral in prepared_files:
        print(f"\n{'='*50}")
        print(f"Training {mineral} models ({num_runs} runs)...")
        print('='*50)
        
        for i in range(num_runs):
            run_count += 1
            print(f"\nRun {run_count}/{total_runs} for {mineral}...")
            
            try:
                result = run_training_pipeline(
                    features_path, 
                    deposits_path, 
                    mineral=mineral,
                    n_negatives_per_positive=2,
                    k=10
                )
                print(f"Success: {result}")
                
                # Add a small delay to prevent database issues
                time.sleep(1)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
    
    print(f"\n{'='*50}")
    print(f"Training completed! {run_count} runs executed.")
    print('='*50)

if __name__ == "__main__":
    main()
