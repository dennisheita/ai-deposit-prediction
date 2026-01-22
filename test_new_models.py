#!/usr/bin/env python3
"""
Test newly trained models
"""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def test_new_models():
    # Load test data
    try:
        X_test = pd.read_csv('ml_data copy/X_test.csv')
        y_test = pd.read_csv('ml_data copy/y_test.csv').squeeze()
        
        print('✅ Test data loaded:')
        print(f'  X_test shape: {X_test.shape}')
        print(f'  y_test shape: {y_test.shape}')
        
    except Exception as e:
        print(f'❌ Error loading test data: {str(e)}')
        return False
    
    # Load newly trained models
    model_files = [
        ('XGBoost', 'models/xgboost_model_v20260122_024746.joblib'),
        ('LightGBM', 'models/lightgbm_model_v20260122_024746.joblib'),
        ('Random Forest', 'models/randomforest_model_v20260122_024746.joblib'),
        ('Ensemble', 'models/ensemble_model_v20260122_024746.joblib')
    ]
    
    print('\nTesting newly trained models:')
    print('-' * 50)
    
    for name, model_path in model_files:
        if joblib.os.path.exists(model_path):
            print(f'\n{name}:')
            try:
                model = joblib.load(model_path)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)
                
                print(f'  Accuracy: {accuracy:.4f}')
                print(f'  Precision: {precision:.4f}')
                print(f'  Recall: {recall:.4f}')
                print(f'  F1: {f1:.4f}')
                print(f'  AUC: {auc:.4f}')
                
            except Exception as e:
                print(f'  Error: {str(e)}')
        else:
            print(f'\n{name}: Model file not found')
    
    return True

if __name__ == "__main__":
    test_new_models()
