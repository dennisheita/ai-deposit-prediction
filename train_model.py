import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

# Load data
data_dir = 'ml_data copy'
X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv')).values.ravel()

# Load feature names
with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
    feature_names = [line.strip() for line in f]

# Ensure columns match
X_train.columns = feature_names
X_test.columns = feature_names
X_val.columns = feature_names

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Features: {feature_names}")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
y_pred_val = rf.predict(X_val)

print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))

print("Test AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
print("Validation AUC:", roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1]))

print("Classification Report (Test):")
print(classification_report(y_test, y_pred_test))

# Save model
os.makedirs('models', exist_ok=True)
model_path = 'models/rf_deposit_model.joblib'
joblib.dump(rf, model_path)
print(f"Model saved to {model_path}")

# Metrics
test_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
metrics = {
    'best_params': {'n_estimators': 100, 'random_state': 42},
    'best_score': test_auc,
    'model_path': model_path,
    'n_features': len(feature_names)
}

# Insert into database
from src.data_architecture import insert_model
import datetime
version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
insert_model(version, metrics)
print(f"Model inserted into database with version {version}")

# Feature importances
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
print("Top 10 Feature Importances:")
print(feature_importance_df.head(10))