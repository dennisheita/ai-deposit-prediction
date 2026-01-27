
import pandas as pd
import os
import shutil
from src.training_pipeline import run_training_pipeline

def prepare_and_train(mineral, file_path, label_col):
    print(f"Processing {mineral} from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        return

    df = pd.read_csv(file_path)
    
    # Rename label column to 'label'
    if label_col in df.columns:
        df = df.rename(columns={label_col: 'label'})
    else:
        print(f"Warning: Column {label_col} not found in {file_path}. Skipping.")
        return

    # Rename latitude/longitude to lat/lon if needed
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        
    # Ensure directories
    os.makedirs('data/deposits', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    
    filename = f"{mineral.lower()}_training_data.csv"
    
    # Save to deposits (this will be the main driver since it has labels)
    deposit_path = os.path.join('data/deposits', filename)
    df.to_csv(deposit_path, index=False)
    
    # Save to features (just to satisfy the pipeline loader, it won't be used for X if deposits has label)
    feature_path = os.path.join('data/features', filename)
    df.to_csv(feature_path, index=False)
    
    print(f"Training model for {mineral}...")
    try:
        # Run pipeline
        # Note: We pass the same filename for both because the pipeline logic 
        # uses deposits_file as the primary source if it contains 'label'
        result = run_training_pipeline(filename, filename, mineral=mineral)
        print(f"Success: {result}")
    except Exception as e:
        print(f"Error training {mineral}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Uranium
    prepare_and_train('Uranium', 'data/uranium_grid_with_features.csv', 'uranium_present')
    
    # Gold
    prepare_and_train('Gold', 'data/gold grid_with_all_features.csv', 'gold_present')
    
    # Copper
    prepare_and_train('Copper', 'data/copper_grid_with_features.csv', 'copper_present')
