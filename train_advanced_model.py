#!/usr/bin/env python3
"""
Advanced Train Model Script - Train XGBoost, LightGBM, Random Forest, and Ensemble models

This script uses the advanced training pipeline with Optuna hyperparameter optimization,
spatial cross-validation, and ensemble learning to train multiple models and select the best one.
"""

import os
import sys
import logging
import argparse
import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.advanced_training import run_advanced_training_pipeline

# Configure logging
LOGS_DIR = 'logs/'
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'advanced_training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train advanced machine learning models for mineral deposit prediction'
    )
    
    parser.add_argument(
        '--features', '-f', 
        required=True, 
        help='Path to features GeoParquet file'
    )
    
    parser.add_argument(
        '--deposits', '-d', 
        required=True, 
        help='Path to deposits GeoParquet file'
    )
    
    parser.add_argument(
        '--mineral', '-m', 
        type=str, 
        default=None, 
        help='Mineral type (e.g., gold, copper)'
    )
    
    parser.add_argument(
        '--negatives', '-n', 
        type=int, 
        default=1, 
        help='Number of negative samples per positive sample (default: 1)'
    )
    
    parser.add_argument(
        '--folds', '-k', 
        type=int, 
        default=10, 
        help='Number of spatial cross-validation folds (default: 10)'
    )
    
    parser.add_argument(
        '--trials', '-t', 
        type=int, 
        default=50, 
        help='Number of Optuna hyperparameter optimization trials (default: 50)'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the advanced training pipeline."""
    args = parse_args()
    
    logger.info(f"Starting advanced training with parameters: {vars(args)}")
    
    try:
        # Run the advanced training pipeline
        result = run_advanced_training_pipeline(
            features_file=args.features,
            deposits_file=args.deposits,
            mineral=args.mineral,
            n_negatives_per_positive=args.negatives,
            k=args.folds,
            n_trials=args.trials
        )
        
        logger.info(f"Training completed successfully: {result}")
        print(f"✅ {result}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
