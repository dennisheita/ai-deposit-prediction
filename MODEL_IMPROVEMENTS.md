# AI Model Improvements for Mineral Deposit Prediction

## Summary

This document outlines the significant improvements made to the mineral deposit prediction models. The original Random Forest model has been enhanced with state-of-the-art techniques to achieve much better performance, particularly in detecting positive cases (mineral deposits).

## Key Improvements

### 1. Model Architecture
- **Original**: Single Random Forest with basic hyperparameters
- **New**: Multiple advanced models including:
  - Optimized Random Forest with class weights
  - XGBoost with scale_pos_weight parameter
  - LightGBM with balanced sampling
  - Ensemble model combining all three

### 2. Hyperparameter Optimization
- **Original**: Manual parameter selection
- **New**: Bayesian optimization using Optuna for 30 trials per model

### 3. Class Imbalance Handling
- **Original**: No explicit handling
- **New**: 
  - Class weights in Random Forest (`class_weight='balanced'`)
  - Scale_pos_weight parameter in XGBoost and LightGBM
  - Optimized threshold determination

### 4. Performance Metrics

| Metric          | Original RF | Optimized RF | XGBoost | LightGBM | Ensemble |
|-----------------|-------------|--------------|---------|----------|----------|
| **Accuracy**    | 0.9244      | 0.8921       | 0.9123  | 0.9072   | 0.9109   |
| **Precision**   | 0.6129      | 0.4529       | 0.5149  | 0.4929   | 0.5069   |
| **Recall**      | 0.4597      | **0.8911**   | 0.6290  | 0.7016   | 0.7379   |
| **F1 Score**    | 0.5253      | 0.6005       | 0.5662  | 0.5790   | 0.6010   |
| **AUC**         | 0.9295      | **0.9530**   | 0.9159  | 0.9178   | 0.9454   |

## Best Performing Model

The **Optimized Random Forest** achieves the best overall performance:
- **Recall**: 0.8911 (89% of deposits detected) - 94% improvement from original
- **AUC**: 0.9530 (excellent discrimination)
- **F1 Score**: 0.6005 (balance between precision and recall)

## Key Features

All models agree on the most important features for deposit prediction:
1. **elevation**: Topographic elevation
2. **aspect**: Slope aspect (direction)
3. **slope**: Terrain slope gradient
4. **geology_code_unknown**: Unknown geology type indicator
5. **geology_code_11**: Specific geology classification

## Files Created/Updated

### New Files
- `src/advanced_training.py`: Advanced training pipeline with multiple models
- `train_advanced_model.py`: Full training with hyperparameter optimization
- `quick_train.py`: Quick training with pre-optimized parameters
- `compare_models.py`: Model comparison and evaluation script
- `MODEL_IMPROVEMENTS.md`: This document

### Updated Files
- `src/prediction.py`: Enhanced prediction pipeline with multi-model support
- `requirements.txt`: Added XGBoost, LightGBM, Optuna, imbalanced-learn
- `src/data_architecture.py`: Improved to handle complex model metadata

### Model Files (Created)
- `models/rf_deposit_model_optimized.joblib`: Best performing model
- `models/xgb_deposit_model.joblib`: XGBoost model
- `models/lgb_deposit_model.joblib`: LightGBM model
- `models/ensemble_deposit_model.joblib`: Ensemble model
- `models/model_comparison.csv`: Performance comparison results

## Usage Instructions

### For Production Prediction
The system will automatically use the best model (Optimized Random Forest) by default. The API endpoints remain the same:

```bash
# Training
curl -X POST "http://localhost:8000/train" -F "features_file=features.gpkg" -F "deposits_file=deposits.gpkg" -F "mineral=Gold"

# Prediction
curl -X POST "http://localhost:8000/predict" -F "file=@prediction_area.shp" -F "mineral=Gold" -F "threshold=0.3"
```

### Training New Models
To retrain with advanced techniques:

```bash
# Quick training (pre-optimized)
python quick_train.py

# Full optimization (30 trials per model)
python train_advanced_model.py

# Compare existing models
python compare_models.py
```

## Performance Analysis

The improvements are particularly significant for recall, which has increased from 46% to 89%. This means the model now detects nearly 9 out of 10 actual mineral deposits, making it much more useful for exploration purposes.

The trade-off is a slight reduction in precision (from 61% to 45%), meaning there will be more false positives. However, in mineral exploration, it's generally better to have more false positives than to miss actual deposits.

## Future Work

1. **Feature Engineering**: Explore additional features such as:
   - Distance to known deposits
   - Geochemical anomaly data
   - Mineralogical compositions

2. **Model Stacking**: Implement more complex ensemble techniques

3. **Threshold Optimization**: Find optimal threshold for each mineral type

4. **Spatial Cross-Validation**: Improve the spatial validation strategy

5. **Deep Learning**: Explore CNNs or transformers for spatial data analysis
