import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, precision_score, recall_score

# Load data
data_dir = 'ml_data copy'
X_train = pd.read_csv(f'{data_dir}/X_train.csv')
y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
X_test = pd.read_csv(f'{data_dir}/X_test.csv')
y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
X_val = pd.read_csv(f'{data_dir}/X_val.csv')
y_val = pd.read_csv(f'{data_dir}/y_val.csv').values.ravel()

# Load feature names
with open(f'{data_dir}/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

X_train.columns = feature_names
X_test.columns = feature_names
X_val.columns = feature_names

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")
print(f"Validation data: {X_val.shape}")

# Load models
print("\nLoading models...")
try:
    original_model = joblib.load('models/rf_deposit_model.joblib')
    print("✓ Original Random Forest loaded")
except:
    original_model = None
    print("✗ Original model not found")

try:
    optimized_model = joblib.load('models/rf_deposit_model_optimized.joblib')
    print("✓ Optimized Random Forest loaded")
except:
    optimized_model = None
    print("✗ Optimized RF not found")

try:
    xgb_model = joblib.load('models/xgb_deposit_model.joblib')
    print("✓ XGBoost loaded")
except:
    xgb_model = None
    print("✗ XGBoost not found")

try:
    lgb_model = joblib.load('models/lgb_deposit_model.joblib')
    print("✓ LightGBM loaded")
except:
    lgb_model = None
    print("✗ LightGBM not found")

try:
    ensemble_model = joblib.load('models/ensemble_deposit_model.joblib')
    print("✓ Ensemble loaded")
except:
    ensemble_model = None
    print("✗ Ensemble not found")

# Evaluate function
def evaluate(model, name):
    if model is None:
        return None
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{name} Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    return metrics

print("\n" + "="*50)
print("MODEL COMPARISON - TEST SET")
print("="*50)

results = {}
if original_model:
    results['Original RF'] = evaluate(original_model, "Original RF")
if optimized_model:
    results['Optimized RF'] = evaluate(optimized_model, "Optimized RF")
if xgb_model:
    results['XGBoost'] = evaluate(xgb_model, "XGBoost")
if lgb_model:
    results['LightGBM'] = evaluate(lgb_model, "LightGBM")
if ensemble_model:
    results['Ensemble'] = evaluate(ensemble_model, "Ensemble")

# Create comparison table
print("\n" + "="*50)
print("COMPARISON TABLE")
print("="*50)
print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'AUC':<10}")
print("-"*65)
for name, metrics in results.items():
    if metrics:
        print(f"{name:<15} {metrics['accuracy']:.4f}     {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1']:.4f}     {metrics['auc']:.4f}")

# Find best performing model
if results:
    best_model = max(results.items(), key=lambda x: x[1]['recall'])  # Prioritize recall for deposit detection
    print("\n" + "="*50)
    print(f"Best Model: {best_model[0]}")
    print(f"Best Recall: {best_model[1]['recall']:.4f}")
    print(f"Best AUC: {best_model[1]['auc']:.4f}")
    print(f"Best F1: {best_model[1]['f1']:.4f}")
    
    # Print classification report for best model
    print("\nClassification Report:")
    if best_model[0] == 'Original RF':
        y_pred = original_model.predict(X_test)
    elif best_model[0] == 'Optimized RF':
        y_pred = optimized_model.predict(X_test)
    elif best_model[0] == 'XGBoost':
        y_pred = xgb_model.predict(X_test)
    elif best_model[0] == 'LightGBM':
        y_pred = lgb_model.predict(X_test)
    elif best_model[0] == 'Ensemble':
        y_pred = ensemble_model.predict(X_test)
    
    print(classification_report(y_test, y_pred))

# Save comparison results
if results:
    comp_df = pd.DataFrame.from_dict(results, orient='index')
    comp_df.to_csv('models/model_comparison.csv')
    print("\nComparison saved to models/model_comparison.csv")

# Feature importances comparison
print("\n" + "="*50)
print("FEATURE IMPORTANCES")
print("="*50)

if original_model and hasattr(original_model, 'feature_importances_'):
    original_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': original_model.feature_importances_
    }).sort_values('importance', ascending=False).head(5)
    print("\nOriginal RF Top 5 Features:")
    for _, row in original_importance.iterrows():
        print(f"{row['feature']:<20} {row['importance']:.4f}")

if optimized_model and hasattr(optimized_model, 'feature_importances_'):
    optimized_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': optimized_model.feature_importances_
    }).sort_values('importance', ascending=False).head(5)
    print("\nOptimized RF Top 5 Features:")
    for _, row in optimized_importance.iterrows():
        print(f"{row['feature']:<20} {row['importance']:.4f}")

if xgb_model and hasattr(xgb_model, 'feature_importances_'):
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False).head(5)
    print("\nXGBoost Top 5 Features:")
    for _, row in xgb_importance.iterrows():
        print(f"{row['feature']:<20} {row['importance']:.4f}")

if lgb_model and hasattr(lgb_model, 'feature_importances_'):
    lgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False).head(5)
    print("\nLightGBM Top 5 Features:")
    for _, row in lgb_importance.iterrows():
        print(f"{row['feature']:<20} {row['importance']:.4f}")
