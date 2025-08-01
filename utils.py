"""Utility functions for training and evaluation."""
import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from config import GatedBertConfig

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def setup_logging(config: GatedBertConfig):
    """Setup logging configuration."""
    log_path = Path(config.log_file)
    
    # Create backup if log file exists
    if log_path.exists() and not config.continue_run:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # When backing up log files, save .bak files in 'bak' directory
        bak_dir = log_path.parent / 'bak'
        bak_dir.mkdir(exist_ok=True)
        backup_path = bak_dir / f"{log_path.stem}_{timestamp}.bak"
        log_path.rename(backup_path)
        print(f"Backed up existing log to {backup_path}")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file, mode='a' if config.continue_run else 'w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"=== Starting Gated BERT Training ===")
    logging.info(f"Task: {config.task}")
    logging.info(f"Configuration: {config}")


def save_training_history(history_dict: Dict[str, Any], filepath: str):
    """Save training history to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"Training history saved to {filepath}")
    except Exception as e:
        print(f"Failed to save training history: {e}")


def plot_training_history(history_dict: Dict[str, Any], save_path: str):
    """Plot training history."""
    if not HAS_MATPLOTLIB:
        return
    
    try:
        epochs = history_dict['epoch']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Train Loss')
        if history_dict['val_loss'] and any(x is not None for x in history_dict['val_loss']):
            axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, history_dict['accuracy'], 'b-', label='Train Accuracy')
        if history_dict['val_accuracy'] and any(x is not None for x in history_dict['val_accuracy']):
            axes[0, 1].plot(epochs, history_dict['val_accuracy'], 'r-', label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Epoch time plot
        if history_dict['epoch_time']:
            axes[1, 0].plot(epochs, history_dict['epoch_time'], 'g-')
            axes[1, 0].set_title('Epoch Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].grid(True)
        
        # Leave the last subplot empty
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {save_path}")
        
    except Exception as e:
        print(f"Failed to create plots: {e}")


def setup_tensorflow():
    """Setup TensorFlow configuration with support for NVIDIA CUDA and Apple MPS."""
    import tensorflow as tf
    import os
    import platform
    
    # Configure TensorFlow for optimal performance
    try:
        tf.config.optimizer.set_jit(False)
        tf.config.optimizer.set_experimental_options({
            'disable_meta_optimizer': True,
            'disable_model_pruning': True,
            'arithmetic_optimization': False
        })
        
        # Detect and configure GPU devices
        gpus = tf.config.experimental.list_physical_devices('GPU')
        cpus = tf.config.experimental.list_physical_devices('CPU')
        
        # Minimal logging for performance
        print(f"GPU Configuration: {len(gpus)} GPU(s), {len(cpus)} CPU(s)")
        
        if gpus:
            # Configure GPU memory growth and device placement
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable soft device placement for better compatibility
            tf.config.set_soft_device_placement(True)
            
            # Quick GPU type detection (minimal overhead)
            if platform.system() == 'Darwin':
                # Apple Metal optimizations
                os.environ['TF_DISABLE_MPS'] = '0'
                os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
                os.environ['METAL_DEVICE_WRAPPER_TYPE_1'] = '1'
                print("✅ Apple Metal GPU configured")
            else:
                # Generic GPU optimizations
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                print("✅ GPU configured")
        else:
            print("No GPU devices found, using CPU")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
    except Exception as e:
        print(f"⚠️  TensorFlow setup error: {e}")
        print("Continuing with default configuration")
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)