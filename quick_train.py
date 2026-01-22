#!/usr/bin/env python3
"""
Quick training script using existing CSV files
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.advanced_training import train_xgboost, train_lightgbm, train_random_forest, create_ensemble, calculate_metrics, save_model
import datetime
import joblib

def quick_train():
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
        
    except Exception as e:
        print(f'❌ Error loading training data: {str(e)}')
        return False
    
    # Combine train and validation data for training
    X_all = pd.concat([X_train, X_val], ignore_index=True)
    y_all = pd.concat([y_train, y_val], ignore_index=True)
    
    # Create dummy cross-validation folds (simple stratified split)
    from sklearn.model_selection import StratifiedKFold
    cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X_all, y_all))
    
    # Train models
    print('\n🚀 Training models...')
    
    try:
        version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print('  XGBoost...')
        xgb_model, xgb_params, xgb_score = train_xgboost(X_all, y_all, cv, n_trials=3)
        
        print('  LightGBM...')
        lgb_model, lgb_params, lgb_score = train_lightgbm(X_all, y_all, cv, n_trials=3)
        
        print('  Random Forest...')
        rf_model, rf_params, rf_score = train_random_forest(X_all, y_all, cv, n_trials=3)
        
        print('  Ensemble...')
        ensemble = create_ensemble([xgb_model, lgb_model, rf_model])
        ensemble.fit(X_all, y_all)
        
        print('\n✅ Training completed!')
        
        # Evaluate on test set
        print('\n📊 Test set evaluation:')
        models = {
            'XGBoost': xgb_model,
            'LightGBM': lgb_model,
            'Random Forest': rf_model,
            'Ensemble': ensemble
        }
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            print(f'\n{name}:')
            print(f'  AUC: {metrics["auc"]:.4f}')
            print(f'  Accuracy: {metrics["accuracy"]:.4f}')
            print(f'  Precision: {metrics["precision"]:.4f}')
            print(f'  Recall: {metrics["recall"]:.4f}')
            print(f'  F1: {metrics["f1"]:.4f}')
            
            # Save model
            joblib.dump(model, f'models/{name.lower().replace(" ", "")}_model_v{version}.joblib')
        
        # Save best model
        best_score = 0
        best_model = None
        best_name = ''
        
        for name, model in models.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = np.mean([calculate_metrics(y_test, model.predict(X_test), y_prob)['auc']])
            if auc > best_score:
                best_score = auc
                best_model = model
                best_name = name
        
        print(f'\n🏆 Best model: {best_name} (AUC: {best_score:.4f})')
        joblib.dump(best_model, f'models/best_model_v{version}.joblib')
        
        return True
        
    except Exception as e:
        print(f'\n❌ Training failed: {str(e)}')
        return False

if __name__ == "__main__":
    print('Quick training using existing CSV files...')
    quick_train()
