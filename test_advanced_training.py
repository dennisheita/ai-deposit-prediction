#!/usr/bin/env python3
"""
Test advanced training pipeline with existing training data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from src.advanced_training import run_advanced_training_pipeline

def test_advanced_training():
    # Load existing training data
    try:
        X_train = pd.read_csv('ml_data copy/X_train.csv')
        y_train = pd.read_csv('ml_data copy/y_train.csv').squeeze()
        X_val = pd.read_csv('ml_data copy/X_val.csv')
        y_val = pd.read_csv('ml_data copy/y_val.csv').squeeze()
        X_test = pd.read_csv('ml_data copy/X_test.csv')
        y_test = pd.read_csv('ml_data copy/y_test.csv').squeeze()
        
        print('✅ Training data loaded:')
        print(f'  X_train shape: {X_train.shape}')
        print(f'  y_train shape: {y_train.shape}')
        print(f'  X_val shape: {X_val.shape}')
        print(f'  y_val shape: {y_val.shape}')
        print(f'  X_test shape: {X_test.shape}')
        print(f'  y_test shape: {y_test.shape}')
        
        # Check label distribution
        print(f'\nLabel distribution:')
        print(f'  Train: {y_train.value_counts()}')
        print(f'  Val: {y_val.value_counts()}')
        print(f'  Test: {y_test.value_counts()}')
        
    except Exception as e:
        print(f'❌ Error loading training data: {str(e)}')
        return False
    
    # Create GeoDataFrame from training data (dummy coordinates for spatial sampling)
    try:
        # Create dummy coordinates (since we don't have real lat/lon in training data)
        np.random.seed(42)
        train_coords = np.random.rand(len(X_train), 2) * 10  # 0-10 degrees
        val_coords = np.random.rand(len(X_val), 2) * 10
        test_coords = np.random.rand(len(X_test), 2) * 10
        
        # Combine into single training dataset
        X_all = pd.concat([X_train, X_val, X_test], ignore_index=True)
        y_all = pd.concat([y_train, y_val, y_test], ignore_index=True)
        coords_all = np.vstack([train_coords, val_coords, test_coords])
        
        # Create GeoDataFrame with dummy points
        geometry = [Point(x, y) for x, y in coords_all]
        train_gdf = gpd.GeoDataFrame(
            X_all.assign(label=y_all),
            geometry=geometry,
            crs='EPSG:4326'
        )
        
        # Save as GeoParquet files for testing
        train_gdf.to_parquet('data/features/test_train_features.parquet')
        
        # Create deposits GeoDataFrame (just positive samples)
        deposits_gdf = train_gdf[train_gdf['label'] == 1].drop(columns=['label'])
        deposits_gdf.to_parquet('data/deposits/test_train_deposits.parquet')
        
        print('\n✅ Test data created and saved')
        
    except Exception as e:
        print(f'\n❌ Error creating test data: {str(e)}')
        return False
    
    # Run advanced training
    try:
        print('\n🚀 Starting advanced training pipeline...')
        result = run_advanced_training_pipeline(
            features_file='data/features/test_train_features.parquet',
            deposits_file='data/deposits/test_train_deposits.parquet',
            mineral='gold',
            n_negatives_per_positive=1,
            k=3,
            n_trials=3  # Use small number of trials for testing
        )
        
        print(f'\n✅ Training completed successfully: {result}')
        
    except Exception as e:
        print(f'\n❌ Training failed: {str(e)}')
        return False
    
    # Clean up test files
    try:
        os.remove('data/features/test_train_features.parquet')
        os.remove('data/deposits/test_train_deposits.parquet')
    except:
        pass
        
    return True

if __name__ == "__main__":
    print('Testing advanced training pipeline...')
    test_advanced_training()
