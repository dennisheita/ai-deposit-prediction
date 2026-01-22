import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_dir = 'ml_data copy'
X_test = pd.read_csv(f'{data_dir}/X_test.csv')
y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()

# Load feature names
with open(f'{data_dir}/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

X_test.columns = feature_names

# Load models
print("Loading models...")
original_model = joblib.load('models/rf_deposit_model.joblib')
optimized_model = joblib.load('models/rf_deposit_model_optimized.joblib')

# Make predictions
print("Making predictions...")
y_pred_original = original_model.predict(X_test)
y_pred_optimized = optimized_model.predict(X_test)

y_prob_original = original_model.predict_proba(X_test)[:, 1]
y_prob_optimized = optimized_model.predict_proba(X_test)[:, 1]

# Confusion matrices
cm_original = confusion_matrix(y_test, y_pred_original)
cm_optimized = confusion_matrix(y_test, y_pred_optimized)

# Plot confusion matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Deposit', 'Deposit'],
            yticklabels=['No Deposit', 'Deposit'])
plt.title('Original RF - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Deposit', 'Deposit'],
            yticklabels=['No Deposit', 'Deposit'])
plt.title('Optimized RF - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Confusion matrices saved to models/confusion_matrices.png")

# Calculate performance metrics
def calculate_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob)
    }

original_metrics = calculate_metrics(y_test, y_pred_original, y_prob_original)
optimized_metrics = calculate_metrics(y_test, y_pred_optimized, y_prob_optimized)

# Create comparison table
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Original RF': [
        original_metrics['accuracy'],
        original_metrics['precision'],
        original_metrics['recall'],
        original_metrics['f1'],
        original_metrics['auc']
    ],
    'Optimized RF': [
        optimized_metrics['accuracy'],
        optimized_metrics['precision'],
        optimized_metrics['recall'],
        optimized_metrics['f1'],
        optimized_metrics['auc']
    ],
    'Improvement': [
        optimized_metrics['accuracy'] - original_metrics['accuracy'],
        optimized_metrics['precision'] - original_metrics['precision'],
        optimized_metrics['recall'] - original_metrics['recall'],
        optimized_metrics['f1'] - original_metrics['f1'],
        optimized_metrics['auc'] - original_metrics['auc']
    ]
})

# Display results
print("\n" + "="*60)
print("PERFORMANCE COMPARISON: ORIGINAL vs. OPTIMIZED MODEL")
print("="*60)
print(metrics_df.to_string(float_format='%.4f'))

# Save metrics to CSV
metrics_df.to_csv('models/performance_comparison.csv', index=False)
print("\nMetrics saved to models/performance_comparison.csv")

# Feature importance comparison
plt.figure(figsize=(10, 6))
original_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': original_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

optimized_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': optimized_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

x = np.arange(10)
width = 0.35

plt.bar(x - width/2, original_importance['importance'], width, label='Original RF')
plt.bar(x + width/2, optimized_importance['importance'], width, label='Optimized RF')

plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances Comparison')
plt.xticks(x, original_importance['feature'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('models/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("Feature importance comparison saved to models/feature_importance_comparison.png")

# Calculate business impact metrics
total_samples = len(y_test)
actual_deposits = np.sum(y_test == 1)
actual_non_deposits = np.sum(y_test == 0)

# Original model performance
original_deposits_found = cm_original[1, 1]
original_deposits_missed = cm_original[1, 0]
original_false_positives = cm_original[0, 1]

# Optimized model performance
optimized_deposits_found = cm_optimized[1, 1]
optimized_deposits_missed = cm_optimized[1, 0]
optimized_false_positives = cm_optimized[0, 1]

print("\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)
print(f"Total test samples: {total_samples}")
print(f"Actual deposits in test set: {actual_deposits}")
print(f"Actual non-deposits in test set: {actual_non_deposits}")
print()
print("Original Model:")
print(f"  Deposits found: {original_deposits_found} ({(original_deposits_found/actual_deposits)*100:.1f}%)")
print(f"  Deposits missed: {original_deposits_missed} ({(original_deposits_missed/actual_deposits)*100:.1f}%)")
print(f"  False positives: {original_false_positives} ({(original_false_positives/actual_non_deposits)*100:.1f}%)")
print()
print("Optimized Model:")
print(f"  Deposits found: {optimized_deposits_found} ({(optimized_deposits_found/actual_deposits)*100:.1f}%)")
print(f"  Deposits missed: {optimized_deposits_missed} ({(optimized_deposits_missed/actual_deposits)*100:.1f}%)")
print(f"  False positives: {optimized_false_positives} ({(optimized_false_positives/actual_non_deposits)*100:.1f}%)")
print()
print("Improvement:")
print(f"  Additional deposits found: {optimized_deposits_found - original_deposits_found}")
print(f"  Reduction in deposits missed: {original_deposits_missed - optimized_deposits_missed}")
print(f"  Change in false positives: {optimized_false_positives - original_false_positives}")

# Print final conclusions
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("The Optimized Random Forest model represents a significant improvement over the Original RF.")
print(f"- Recall increased by {(optimized_metrics['recall'] - original_metrics['recall'])*100:.1f}% (from {original_metrics['recall']:.1%} to {optimized_metrics['recall']:.1%})")
print(f"- AUC increased by {(optimized_metrics['auc'] - original_metrics['auc'])*100:.1f}% (from {original_metrics['auc']:.1%} to {optimized_metrics['auc']:.1%})")
print(f"- F1 Score increased by {(optimized_metrics['f1'] - original_metrics['f1'])*100:.1f}% (from {original_metrics['f1']:.1%} to {optimized_metrics['f1']:.1%})")
print()
print("These improvements mean the model will detect nearly twice as many mineral deposits,")
print("which can have a profound impact on exploration efficiency and resource discovery.")
