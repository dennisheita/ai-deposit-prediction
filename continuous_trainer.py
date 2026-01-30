#!/usr/bin/env python3
"""
Enhanced Continuous Training System

This script provides perpetual model training with:
- Random hyperparameter search
- Smart stopping criteria (plateau detection, max iterations)
- Resource management (memory, disk)
- Multiple mineral support
- Best model tracking
- Graceful shutdown handling

Usage:
    python continuous_trainer.py --minerals Gold Copper --max-runs 100
    python continuous_trainer.py --mode exploration --stop-on-plateau
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import datetime
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training_pipeline import run_training_pipeline
from src.advanced_training import run_advanced_training_pipeline
from src.data_architecture import get_models, get_training_runs, insert_model

# Configuration
LOGS_DIR = 'logs/'
MODELS_DIR = 'models/'
STATE_FILE = 'continuous_trainer_state.json'
LIVE_STATUS_FILE = 'training_status_live.json'  # For real-time WebSocket updates

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def write_live_status(status: dict):
    """Write live status to file for WebSocket monitoring"""
    try:
        with open(LIVE_STATUS_FILE, 'w') as f:
            json.dump(status, f)
    except Exception as e:
        logger.warning(f"Could not write live status: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'continuous_trainer.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a single training run"""
    mineral: str
    features_file: str
    deposits_file: str
    mode: str  # 'basic' or 'advanced'
    hyperparams: Dict
    timestamp: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingResult:
    """Result of a training run"""
    success: bool
    config: TrainingConfig
    score: Optional[float]
    model_path: Optional[str]
    error: Optional[str]
    duration: float


class ContinuousTrainer:
    """
    Manages continuous model training with smart stopping and resource management.
    """
    
    def __init__(
        self,
        minerals: List[str] = None,
        mode: str = 'advanced',
        max_runs: int = None,
        stop_on_plateau: bool = False,
        plateau_patience: int = 10,
        plateau_threshold: float = 0.001,
        memory_limit_gb: float = 8.0,
        disk_limit_gb: float = 10.0,
        save_best_only: bool = True,
        parallel_jobs: int = 1,
        exploration_ratio: float = 0.3
    ):
        """
        Initialize the continuous trainer.
        
        Args:
            minerals: List of minerals to train on (e.g., ['Gold', 'Copper'])
            mode: 'basic' or 'advanced' training mode
            max_runs: Maximum number of training runs (None = infinite)
            stop_on_plateau: Stop if no improvement for N runs
            plateau_patience: Number of runs without improvement before stopping
            plateau_threshold: Minimum improvement to reset plateau counter
            memory_limit_gb: Stop if memory usage exceeds this (GB)
            disk_limit_gb: Stop if disk usage exceeds this (GB)
            save_best_only: Only keep models that improve on previous best
            parallel_jobs: Number of parallel training jobs (1 = sequential)
            exploration_ratio: Ratio of random vs guided hyperparameter search
        """
        self.minerals = minerals or ['Gold', 'Copper', 'Uranium']
        self.mode = mode
        self.max_runs = max_runs
        self.stop_on_plateau = stop_on_plateau
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.memory_limit_gb = memory_limit_gb
        self.disk_limit_gb = disk_limit_gb
        self.save_best_only = save_best_only
        self.parallel_jobs = parallel_jobs
        self.exploration_ratio = exploration_ratio
        
        # State tracking
        self.run_count = 0
        self.best_scores = {mineral: 0.0 for mineral in self.minerals}
        self.plateau_counter = 0
        self.run_history = []
        self.running = True
        self.current_workers = []
        
        # Load previous state if exists
        self._load_state()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"ContinuousTrainer initialized: mode={mode}, minerals={minerals}, max_runs={max_runs}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self._save_state()
    
    def _load_state(self):
        """Load previous training state"""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                self.run_count = state.get('run_count', 0)
                self.best_scores = state.get('best_scores', self.best_scores)
                self.run_history = state.get('run_history', [])
                logger.info(f"Loaded state: {self.run_count} previous runs, best scores: {self.best_scores}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def _save_state(self):
        """Save current training state"""
        state = {
            'run_count': self.run_count,
            'best_scores': self.best_scores,
            'run_history': self.run_history[-100:],  # Keep last 100 runs
            'timestamp': datetime.datetime.now().isoformat()
        }
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info("State saved")
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def _check_resources(self) -> bool:
        """Check if resource limits are exceeded"""
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024 ** 3)
        if memory_gb > self.memory_limit_gb:
            logger.warning(f"Memory limit exceeded: {memory_gb:.1f}GB > {self.memory_limit_gb}GB")
            return False
        
        # Check disk - only check the models directory usage, not entire disk
        if os.path.exists(MODELS_DIR):
            # Calculate actual usage of models directory only
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
            disk_gb = total_size / (1024 ** 3)
        else:
            disk_gb = 0
        
        if disk_gb > self.disk_limit_gb:
            logger.warning(f"Models directory size limit exceeded: {disk_gb:.1f}GB > {self.disk_limit_gb}GB")
            return False
        
        return True
    
    def _should_stop(self) -> bool:
        """Check if training should stop"""
        if not self.running:
            return True
        
        if self.max_runs and self.run_count >= self.max_runs:
            logger.info(f"Reached max runs: {self.max_runs}")
            return True
        
        if self.stop_on_plateau and self.plateau_counter >= self.plateau_patience:
            logger.info(f"Plateau detected: no improvement for {self.plateau_patience} runs")
            return True
        
        if not self._check_resources():
            return True
        
        return False
    
    def _generate_hyperparams(self, mineral: str) -> Dict:
            """Generate random hyperparameters for a training run"""
            import random
            
            # Exploration vs exploitation
            if random.random() < self.exploration_ratio or not self.run_history:
                # Random exploration - OPTIMIZED FOR SPEED
                if self.mode == 'advanced':
                    return {
                        'n_negatives_per_positive': random.choice([1, 2]),
                        'k': random.choice([3, 5]),  # Reduced from [5, 10, 15]
                        'n_trials': random.randint(5, 15),  # Reduced from [20, 100]
                        'min_distance': random.choice([0, 500, 1000])
                    }
                else:
                    return {
                        'n_negatives_per_positive': random.choice([1, 2]),
                        'k': random.choice([3, 5]),  # Reduced from [5, 10]
                        'param_grid': {
                            'n_estimators': random.choice([[50, 200], [100, 300]]),
                            'max_depth': random.choice([[5, 10], [10, 20]]),
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2]
                        }
                    }
            else:
                # Guided search based on best performing configs
                best_configs = [
                    run['config'] for run in self.run_history
                    if run.get('success') and run.get('score', 0) > 0.8
                ]
                if best_configs:
                    # Mutate a good config
                    base = random.choice(best_configs)
                    mutated = base.copy()
                    mutated['n_trials'] = max(5, mutated.get('n_trials', 10) + random.randint(-3, 5))  # Reduced from 50
                    return mutated
                else:
                    return self._generate_hyperparams(mineral)  # Fallback to random
    
    def _get_data_files(self, mineral: str) -> Tuple[str, str]:
        """Get feature and deposit files for a mineral"""
        mineral_lower = mineral.lower()
        
        # Try different file patterns
        possible_files = [
            (f'data/{mineral_lower}_complete_real.csv', f'data/{mineral_lower}_complete_real.csv'),
            (f'data/{mineral_lower}_grid_with_features.csv', f'data/{mineral_lower}_deposits.csv'),
            (f'data/features/{mineral_lower}_features.parquet', f'data/deposits/{mineral_lower}_deposits.geojson'),
        ]
        
        for features_file, deposits_file in possible_files:
            if os.path.exists(features_file):
                return features_file, deposits_file
        
        # Default fallback - use the same file for both features and deposits
        return f'data/{mineral_lower}_complete_real.csv', f'data/{mineral_lower}_complete_real.csv'
    
    def _run_single_training(self, config: TrainingConfig) -> TrainingResult:
        """Execute a single training run"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting training: {config.mineral} (run {self.run_count + 1})")
            
            if self.mode == 'advanced':
                result = run_advanced_training_pipeline(
                    config.features_file,
                    config.deposits_file,
                    mineral=config.mineral,
                    **config.hyperparams
                )
            else:
                result = run_training_pipeline(
                    config.features_file,
                    config.deposits_file,
                    mineral=config.mineral,
                    **config.hyperparams
                )
            
            # Parse result to extract score
            score = self._extract_score_from_result(result)
            model_path = self._extract_model_path_from_result(result)
            
            duration = time.time() - start_time
            
            logger.info(f"Training completed: {config.mineral}, score={score:.4f}, duration={duration:.1f}s")
            
            return TrainingResult(
                success=True,
                config=config,
                score=score,
                model_path=model_path,
                error=None,
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Training failed: {config.mineral}, error={str(e)}")
            
            return TrainingResult(
                success=False,
                config=config,
                score=None,
                model_path=None,
                error=str(e),
                duration=duration
            )
    
    def _extract_score_from_result(self, result: str) -> float:
        """Extract AUC score from training result string"""
        try:
            # Look for patterns like "AUC: 0.9234" or "score: 0.9234"
            import re
            match = re.search(r'(?:AUC|score)[\s:=]+([0-9.]+)', result, re.IGNORECASE)
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return 0.0
    
    def _extract_model_path_from_result(self, result: str) -> Optional[str]:
        """Extract model path from training result string"""
        try:
            import re
            match = re.search(r'saved at\s+(\S+)', result)
            if match:
                return match.group(1)
            return None
        except:
            return None
    
    def _update_best_score(self, mineral: str, score: float) -> bool:
        """Update best score and return True if improved"""
        if score > self.best_scores[mineral] + self.plateau_threshold:
            improvement = score - self.best_scores[mineral]
            self.best_scores[mineral] = score
            self.plateau_counter = 0
            logger.info(f"New best score for {mineral}: {score:.4f} (+{improvement:.4f})")
            return True
        else:
            self.plateau_counter += 1
            return False
    
    def _cleanup_poor_models(self, result: TrainingResult):
        """Remove models that don't improve on best score"""
        if self.save_best_only and result.model_path and result.score:
            mineral = result.config.mineral
            if result.score < self.best_scores[mineral] - self.plateau_threshold:
                try:
                    if os.path.exists(result.model_path):
                        os.remove(result.model_path)
                        logger.info(f"Removed poor model: {result.model_path}")
                except Exception as e:
                    logger.warning(f"Could not remove model: {e}")
    
    def run(self):
        """Main training loop"""
        logger.info("Starting continuous training loop...")
        logger.info(f"Target minerals: {self.minerals}")
        logger.info(f"Press Ctrl+C to stop gracefully")

        print("🚀 Starting continuous training", flush=True)
        print(f"🎯 Target minerals: {', '.join(self.minerals)}", flush=True)
        print(f"⚙️  Mode: {self.mode}", flush=True)
        print("⏹️  Press Ctrl+C to stop gracefully\n", flush=True)

        # Write initial live status
        write_live_status({
            'active': True,
            'mineral': self.minerals[0] if self.minerals else '',
            'current_run': 0,
            'total_runs': self.max_runs or 999999,
            'current_score': None,
            'best_score': max(self.best_scores.values()) if self.best_scores else 0,
            'message': 'Starting continuous training...',
            'timestamp': datetime.datetime.now().isoformat()
        })

        try:
            while not self._should_stop():
                # Cycle through minerals - only train if in allowed list
                for mineral in self.minerals:
                    if self._should_stop():
                        break

                    print(f"\n🪨 Training mineral: {mineral} (Run #{self.run_count + 1})", flush=True)

                    # Update live status before training
                    write_live_status({
                        'active': True,
                        'mineral': mineral,
                        'current_run': self.run_count + 1,
                        'total_runs': self.max_runs or 999999,
                        'current_score': None,
                        'best_score': max(self.best_scores.values()) if self.best_scores else 0,
                        'message': f'Training {mineral} - Run #{self.run_count + 1}',
                        'timestamp': datetime.datetime.now().isoformat()
                    })

                    # Generate configuration
                    features_file, deposits_file = self._get_data_files(mineral)
                    print(f"   📁 Features: {features_file}", flush=True)

                    hyperparams = self._generate_hyperparams(mineral)
                    print(f"   🔧 Hyperparams: {hyperparams}", flush=True)

                    config = TrainingConfig(
                        mineral=mineral,
                        features_file=features_file,
                        deposits_file=deposits_file,
                        mode=self.mode,
                        hyperparams=hyperparams,
                        timestamp=datetime.datetime.now().isoformat()
                    )

                    print(f"   ⚙️  Running {self.mode} pipeline...", flush=True)
                    # Run training
                    result = self._run_single_training(config)
                    self.run_count += 1

                    # Update tracking
                    if result.success and result.score:
                        self._update_best_score(mineral, result.score)
                        print(f"   ✅ Finished {mineral} - Score: {result.score:.4f} (Duration: {result.duration:.1f}s)", flush=True)

                        # Update live status with success
                        write_live_status({
                            'active': True,
                            'mineral': mineral,
                            'current_run': self.run_count,
                            'total_runs': self.max_runs or 999999,
                            'current_score': result.score,
                            'best_score': max(self.best_scores.values()) if self.best_scores else result.score,
                            'message': f'✅ {mineral} completed - Score: {result.score:.4f}',
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                    else:
                        print(f"   ❌ Failed {mineral} - Error: {result.error}", flush=True)

                        # Update live status with failure
                        write_live_status({
                            'active': True,
                            'mineral': mineral,
                            'current_run': self.run_count,
                            'total_runs': self.max_runs or 999999,
                            'current_score': None,
                            'best_score': max(self.best_scores.values()) if self.best_scores else 0,
                            'message': f'❌ {mineral} failed - {result.error}',
                            'timestamp': datetime.datetime.now().isoformat()
                        })

                    # Record history
                    self.run_history.append({
                        'run': self.run_count,
                        'mineral': mineral,
                        'success': result.success,
                        'score': result.score,
                        'duration': result.duration,
                        'error': result.error,
                        'config': config.hyperparams
                    })

                    # Cleanup poor models
                    if result.success:
                        self._cleanup_poor_models(result)

                    # Save state periodically
                    if self.run_count % 10 == 0:
                        self._save_state()
                        self._print_summary()
                        print(f"\n📊 Progress: {self.run_count} runs completed", flush=True)
                        print(f"🏆 Best scores: {self.best_scores}", flush=True)

                    # Small delay between runs
                    time.sleep(1)

        except Exception as e:
            logger.error(f"Unexpected error in training loop: {e}")
            print(f"\n💥 Error: {e}", flush=True)

            # Update live status with error
            write_live_status({
                'active': False,
                'mineral': '',
                'current_run': self.run_count,
                'total_runs': self.max_runs or 999999,
                'current_score': None,
                'best_score': max(self.best_scores.values()) if self.best_scores else 0,
                'message': f'Error: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            })

        finally:
            self._save_state()

            # Write final live status
            write_live_status({
                'active': False,
                'mineral': '',
                'current_run': self.run_count,
                'total_runs': self.max_runs or 999999,
                'current_score': None,
                'best_score': max(self.best_scores.values()) if self.best_scores else 0,
                'message': 'Training completed',
                'timestamp': datetime.datetime.now().isoformat()
            })
            self._print_summary()
            logger.info("Continuous training stopped")
            print("\n🛑 Continuous training stopped", flush=True)
            print(f"📈 Total runs: {self.run_count}", flush=True)
            print(f"🏆 Best scores: {self.best_scores}", flush=True)
    
    def _print_summary(self):
        """Print training summary"""
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total runs: {self.run_count}")
        logger.info(f"Best scores by mineral:")
        for mineral, score in self.best_scores.items():
            logger.info(f"  {mineral}: {score:.4f}")
        
        recent_runs = [r for r in self.run_history[-20:] if r.get('success')]
        if recent_runs:
            avg_score = np.mean([r['score'] for r in recent_runs if r.get('score')])
            logger.info(f"Average score (last 20 runs): {avg_score:.4f}")
        
        logger.info(f"Plateau counter: {self.plateau_counter}/{self.plateau_patience}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Continuous Model Training')
    parser.add_argument('--minerals', nargs='+', default=['Gold', 'Copper', 'Uranium'],
                       help='Minerals to train on')
    parser.add_argument('--mode', choices=['basic', 'advanced'], default='advanced',
                       help='Training mode')
    parser.add_argument('--max-runs', type=int, default=None,
                       help='Maximum number of training runs (default: infinite)')
    parser.add_argument('--stop-on-plateau', action='store_true',
                       help='Stop when performance plateaus')
    parser.add_argument('--plateau-patience', type=int, default=10,
                       help='Runs without improvement before stopping')
    parser.add_argument('--memory-limit', type=float, default=8.0,
                       help='Memory limit in GB')
    parser.add_argument('--disk-limit', type=float, default=10.0,
                       help='Disk limit in GB')
    parser.add_argument('--save-all', action='store_true',
                       help='Save all models (not just best)')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel training jobs')
    parser.add_argument('--exploration', type=float, default=0.3,
                       help='Ratio of random exploration (0-1)')
    
    args = parser.parse_args()
    
    trainer = ContinuousTrainer(
        minerals=args.minerals,
        mode=args.mode,
        max_runs=args.max_runs,
        stop_on_plateau=args.stop_on_plateau,
        plateau_patience=args.plateau_patience,
        memory_limit_gb=args.memory_limit,
        disk_limit_gb=args.disk_limit,
        save_best_only=not args.save_all,
        parallel_jobs=args.parallel,
        exploration_ratio=args.exploration
    )
    
    trainer.run()


if __name__ == "__main__":
    main()
