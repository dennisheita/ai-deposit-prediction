#!/usr/bin/env python3
"""
Check model features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib

# Load models
model_files = [
    ('RF', 'models/rf_deposit_model.joblib'),
    ('XGB', 'models/xgb_deposit_model.joblib'),
    ('LGB', 'models/lgb_deposit_model.joblib'),
    ('Ensemble', 'models/ensemble_deposit_model.joblib')
]

for model_name, model_path in model_files:
    if os.path.exists(model_path):
        print(f"\nChecking {model_name} model:")
        model = joblib.load(model_path)
        
        if hasattr(model, 'feature_names_in_'):
            print(f"Feature names in: {model.feature_names_in_}")
            print(f"Number of features: {len(model.feature_names_in_)}")
        
        elif hasattr(model, 'get_booster'):
            try:
                print(f"Feature names: {model.get_booster().feature_names}")
                print(f"Number of features: {len(model.get_booster().feature_names)}")
            except Exception as e:
                print(f"Error getting XGBoost feature names: {e}")
        
        elif hasattr(model, 'feature_importances_'):
            print(f"Number of features (from importances): {len(model.feature_importances_)}")
        
        else:
            print("Could not determine feature information from model")
    else:
        print(f"\nModel file not found: {model_path}")
