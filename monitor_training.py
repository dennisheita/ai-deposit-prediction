#!/usr/bin/env python3
"""
Script to monitor training progress and provide updates.
"""

import os
import time
import sqlite3
from src.data_architecture import get_models, get_training_runs, get_unique_minerals
from src.data_architecture import initialize_database

def print_summary():
    """Print a summary of training progress"""
    try:
        # Get all models from database
        models = get_models()
        training_runs = get_training_runs()
        minerals = get_unique_minerals()
        
        print(f"\n{'='*60}")
        print("Training Summary")
        print('='*60)
        print(f"Total models created: {len(models)}")
        print(f"Total training runs: {len(training_runs)}")
        print(f"Minerals trained: {', '.join(minerals) if minerals else 'None'}")
        
        # Print per-mineral statistics
        if minerals:
            print("\nPer-mineral breakdown:")
            for mineral in minerals:
                mineral_models = [m for m in models if m[4] == mineral]
                print(f"  {mineral}: {len(mineral_models)} models")
        
        # Check for completed runs
        completed_runs = [run for run in training_runs if run[6]]
        print(f"Completed runs: {len(completed_runs)}/{len(training_runs)}")
        
        # Print accuracy and AUC for completed runs
        if completed_runs:
            print("\nPerformance statistics:")
            accuracies = []
            aucs = []
            for run in completed_runs:
                if run[8] is not None:  # Index 8 is accuracy
                    accuracies.append(run[8])
                if run[9] is not None:  # Index 9 is AUC
                    aucs.append(run[9])
            
            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"  Average accuracy: {avg_acc:.4f}")
            
            if aucs:
                # Filter out non-numeric values (some AUC entries may be JSON strings)
                numeric_aucs = []
                for auc in aucs:
                    if isinstance(auc, (int, float)):
                        numeric_aucs.append(float(auc))
                    elif isinstance(auc, str):
                        # Try to convert string to float, skip if it fails
                        try:
                            numeric_aucs.append(float(auc))
                        except (ValueError, TypeError):
                            # Skip non-numeric strings (like JSON feature importances)
                            pass
                
                if numeric_aucs:
                    avg_auc = sum(numeric_aucs) / len(numeric_aucs)
                    print(f"  Average AUC: {avg_auc:.4f}")
                else:
                    print(f"  Average AUC: N/A (no valid numeric values)")
        
        print('='*60)
        
    except Exception as e:
        print(f"Error getting summary: {str(e)}")

def main():
    # Ensure database is initialized
    initialize_database()
    
    print("Training monitor started. Press Ctrl+C to stop.")
    print("Monitoring training progress every 30 seconds...")
    
    try:
        while True:
            print_summary()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        print_summary()

if __name__ == "__main__":
    main()
