#!/usr/bin/env python3
"""
Convert CSV data to GeoParquet format for training
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

def convert_csv_to_geoparquet(csv_path, output_path, latitude_col='latitude', longitude_col='longitude', label_col='gold_present'):
    """Convert CSV with lat/lon to GeoParquet"""
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create geometry column
    geometry = [Point(xy) for xy in zip(df[longitude_col], df[latitude_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GeoParquet
    gdf.to_parquet(output_path)
    print(f"Successfully converted {csv_path} to {output_path}")
    
    return gdf

if __name__ == "__main__":
    # Convert grid_with_all_features.csv to features.parquet
    features_csv = 'data/grid_with_all_features - grid_with_all_features.csv'
    features_output = 'data/features/grid_features.parquet'
    
    if os.path.exists(features_csv):
        features_gdf = convert_csv_to_geoparquet(features_csv, features_output)
        print(f"Features file created: {features_gdf.shape}")
    
    # Convert gold_chain_data.csv to deposits.parquet (only points with gold_present = 1)
    deposits_csv = 'data/gold_chain_data.csv'
    deposits_output = 'data/deposits/gold_deposits.parquet'
    
    if os.path.exists(deposits_csv):
        df = pd.read_csv(deposits_csv)
        # Filter for deposits with gold_present = 1
        deposits_df = df[df['gold_present'] == 1]
        if not deposits_df.empty:
            geometry = [Point(0, 0) for _ in range(len(deposits_df))]  # Dummy geometry - need to add real lat/lon
            # Add lat/lon columns if they don't exist
            if 'latitude' not in deposits_df.columns or 'longitude' not in deposits_df.columns:
                # We need to extract lat/lon from chain_id and sequence_id or use dummy values
                # For demonstration, use dummy coordinates
                deposits_df['latitude'] = -29.5 + (deposits_df['chain_id'] * 0.1)
                deposits_df['longitude'] = 12.5 + (deposits_df['sequence_id'] * 0.05)
            
            geometry = [Point(xy) for xy in zip(deposits_df['longitude'], deposits_df['latitude'])]
            deposits_gdf = gpd.GeoDataFrame(deposits_df, geometry=geometry, crs='EPSG:4326')
            deposits_gdf.to_parquet(deposits_output)
            print(f"Deposits file created: {deposits_gdf.shape}")
        else:
            print("No deposits with gold_present = 1 found in gold_chain_data.csv")
