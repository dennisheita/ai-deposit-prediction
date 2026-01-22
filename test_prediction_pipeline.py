#!/usr/bin/env python3
"""
Test prediction pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.prediction import predict_with_model
import pandas as pd
import geopandas as gpd

def test_prediction_pipeline():
    # Load test data
    try:
        features = gpd.read_parquet('data/features/grid_features.parquet')
        print('✅ Features loaded:', len(features))
    except Exception as e:
        print(f'❌ Error loading features: {str(e)}')
        return False
    
    # Test prediction with each model type
    model_types = ['rf', 'xgb', 'lgb', 'ensemble']
    
    for model_type in model_types:
        try:
            print(f'\nTesting {model_type.upper()} model...')
            results = predict_with_model(features, model_type=model_type)
            
            if results is not None:
                print(f'✅ Prediction results shape: {results.shape}')
                print(f'✅ Prediction results columns: {list(results.columns)}')
                
                if 'probability' in results.columns:
                    prob_range = (results['probability'].min(), results['probability'].max())
                    print(f'✅ Probability range: {prob_range[0]:.4f} - {prob_range[1]:.4f}')
                    
                    positive_preds = sum(results['probability'] > 0.5)
                    print(f'✅ Positive predictions (prob > 0.5): {positive_preds}')
            else:
                print(f'❌ No results returned for {model_type}')
                
        except Exception as e:
            print(f'❌ Error with {model_type} model: {str(e)}')
            print('Continuing to test other models...')
    
    print('\n✅ All model types tested!')
    return True

if __name__ == "__main__":
    test_prediction_pipeline()
