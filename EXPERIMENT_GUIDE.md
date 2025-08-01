# Automated Pruning Experiments Guide

## Overview

The `run_experiments.sh` script automates training experiments across different pruning methods with configurable parameters. It systematically tests each pruning method and organizes results for easy comparison.

## Quick Start

```bash
# Make sure you're in the project directory
cd /path/to/your/project

# Run all experiments (default configuration)
./run_experiments.sh
```

## Configuration

Edit the script to customize parameters:

```bash
# Training parameters
EPOCHS=10                    # Number of training epochs
BATCH_SIZE=32               # Batch size for training
LEARNING_RATE="2e-5"        # Learning rate

# Pruning parameters  
START_SPARSITY=0.0          # Initial sparsity (0.0 = no pruning)
END_SPARSITY=0.7            # Target sparsity (0.7 = 70% pruned)
PRUNING_START_EPOCH=2       # Epoch to start pruning
PRUNING_END_EPOCH=8         # Epoch to reach target sparsity (will be capped at EPOCHS)

# Pruning methods to test
PRUNING_METHODS=("magnitude" "movement" "spt" "baseline")

# Task configuration
TASK="mnli"                 # GLUE task (sst2, mnli, qnli, etc.)
MAX_LEN=128                 # Maximum sequence length
```

### ⚠️ **Important: Epochs vs Pruning Schedule**

The script automatically ensures `PRUNING_END_EPOCH ≤ EPOCHS` to prevent training restart issues. 

**Example:**
- If you set `EPOCHS=8` and `PRUNING_END_EPOCH=23`
- The script will automatically use `pruning_end_epoch=8` 
- This prevents the training from restarting after completion

**Why this matters:** Previously, mismatched epochs could cause training to complete normally, then unexpectedly restart for additional epochs due to pruning schedule conflicts.

## What the Script Does

### 1. **Preparation**
- Backs up your original `config.py`
- Creates experiment logs directory
- Validates required files exist

### 2. **For Each Method**
- Updates `config.py` with method-specific parameters
- Runs `python train.py` with full logging
- Captures both stdout and stderr to log files
- Organizes results in method-specific directories

### 3. **Results Organization**
Each method creates its own directory:
```
results_magnitude/
├── backup_magnitude_20250101_120000.zip
├── results_magnitude_20250101_120000.csv
├── metadata_magnitude_20250101_120000.json
├── loss_sparsity_magnitude_20250101_120000.png
├── accuracy_sparsity_magnitude_20250101_120000.png
├── best_model_weights_magnitude_20250101_120000.h5
└── config_magnitude_20250101_120000.json

results_movement/
├── ... (similar structure)

results_spt/
├── ... (similar structure)
├── reward_sparsity_spt_20250101_120000.png  # SPT-specific plot

results_baseline/
├── ... (baseline without pruning)
```

### 4. **Logging**
```
experiment_logs/
├── magnitude_training_20250101_120000.log
├── movement_training_20250101_120000.log
├── spt_training_20250101_120000.log
├── baseline_training_20250101_120000.log
└── experiment_summary_20250101_120000.md
```

## Methods Tested

| Method | Description | Pruning Enabled |
|--------|-------------|-----------------|
| **magnitude** | Prunes weights with smallest magnitudes | ✅ Yes |
| **movement** | Prunes weights with least movement during training | ✅ Yes |
| **spt** | Structured Pruning with Training (reward-based) | ✅ Yes |
| **baseline** | No pruning (for comparison) | ❌ No |

## Expected Runtime

For default configuration (10 epochs, batch_size=32):
- **Per method**: ~15-30 minutes (depending on hardware)
- **Total experiment**: ~1-2 hours for all 4 methods
- **With GPU**: Significantly faster
- **CPU only**: Longer but still manageable

## Output and Analysis

### 1. **Real-time Progress**
The script provides colored output showing:
- 🔵 Headers and progress
- 🟢 Success messages
- 🟡 Warnings
- 🔴 Errors
- 🔷 Information

### 2. **Results Files**
Each method produces:
- **CSV**: Epoch-by-epoch metrics (loss, accuracy, sparsity, rewards)
- **Plots**: Training curves and sparsity progression
- **Model**: Best weights saved during training
- **Config**: Exact configuration used
- **Metadata**: Run information and final metrics

### 3. **Experiment Summary**
Automatically generated markdown report with:
- Configuration table
- Results directory structure
- File listings
- Usage instructions

## Advanced Features

### 🛡️ **Automatic Backup & Restore**

The script provides comprehensive protection for your `config.py`:

#### **Automatic Backup:**
- Creates `config.py.backup` before any modifications
- Creates timestamped backup `config.py.backup_YYYYMMDD_HHMMSS` for extra safety
- Verifies backup exists before making any changes

#### **Automatic Restore:**
- **On normal completion**: Restores original config automatically
- **On error/failure**: Trap handlers restore config immediately  
- **On interruption** (Ctrl+C): Restores config before exit
- **On any signal**: SIGTERM, SIGINT, ERR all trigger restoration

#### **Error Recovery:**
```bash
# If script is interrupted:
Script interrupted or failed. Restoring original config.py...
✅ Original config.py has been restored.

# If individual experiment fails:
❌ spt failed
⚠️  Continuing with remaining methods...
```

#### **Safety Features:**
- **Atomic file updates**: Uses temporary files to prevent corruption
- **Validation checks**: Ensures backup exists before modifications
- **Multiple backups**: Timestamped copies for debugging
- **Graceful degradation**: Continues other experiments if one fails

### 🔄 **What Happens During Interruption:**

1. **Ctrl+C pressed** → Trap catches signal
2. **Restores original config.py** from backup
3. **Shows clear message** about restoration
4. **Exits cleanly** with appropriate code

### 🐛 **Debugging Failed Runs:**

If experiments fail, you'll find:
```bash
config.py.modified_YYYYMMDD_HHMMSS  # Config that was being used when it failed
config.py.backup_YYYYMMDD_HHMMSS   # Original config for comparison
experiment_logs/                    # Detailed error logs
```

## Advanced Usage

### Custom Method Selection
```bash
# Edit the script to test only specific methods
PRUNING_METHODS=("magnitude" "spt")  # Only test these two
```

### Different Tasks
```bash
# Change task in the script
TASK="sst2"     # Binary sentiment classification
TASK="qnli"     # Question-answer inference
TASK="rte"      # Recognizing textual entailment
```

### Quick Testing
```bash
# For quick testing, reduce epochs
EPOCHS=2
BATCH_SIZE=16
```

## Troubleshooting

### Common Issues

1. **Script Permission Denied**
   ```bash
   chmod +x run_experiments.sh
   ```

2. **Python/TensorFlow Not Found**
   - Ensure Python environment is activated
   - Install required dependencies

3. **Out of Memory**
   - Reduce `BATCH_SIZE`
   - Reduce `MAX_LEN`
   - Use CPU-only if GPU memory insufficient

4. **Training Fails**
   - Check individual log files in `experiment_logs/`
   - Verify dataset access
   - Check GPU/CUDA compatibility

### Recovery from Interruption

If the script is interrupted:
```bash
# Restore original config
mv config.py.backup config.py

# Resume from specific method (edit PRUNING_METHODS)
PRUNING_METHODS=("spt" "baseline")  # Skip completed methods
```

## Results Analysis

### Compare Methods
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results from each method
magnitude_df = pd.read_csv('results_magnitude/results_magnitude_*.csv')
movement_df = pd.read_csv('results_movement/results_movement_*.csv')
spt_df = pd.read_csv('results_spt/results_spt_*.csv')
baseline_df = pd.read_csv('results_baseline/results_baseline_*.csv')

# Plot comparison
plt.figure(figsize=(12, 8))
plt.plot(magnitude_df['epoch'], magnitude_df['val_accuracy'], label='Magnitude')
plt.plot(movement_df['epoch'], movement_df['val_accuracy'], label='Movement')
plt.plot(spt_df['epoch'], spt_df['val_accuracy'], label='SPT')
plt.plot(baseline_df['epoch'], baseline_df['val_accuracy'], label='Baseline')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Pruning Methods Comparison')
plt.show()
```

### Key Metrics to Compare
- **Final accuracy**: Which method maintains best performance?
- **Sparsity achieved**: How much was actually pruned?
- **Training stability**: Smooth vs. volatile learning curves?
- **SPT rewards**: How well did SPT identify important weights?

## Tips for Success

1. **Start Small**: Test with fewer epochs first
2. **Monitor Resources**: Watch GPU memory and CPU usage
3. **Save Frequently**: Results auto-saved, but monitor disk space
4. **Compare Fairly**: Use same random seeds for reproducibility
5. **Document Changes**: Note any parameter modifications

## Next Steps

After running experiments:
1. Analyze the CSV files for quantitative comparison
2. Examine the plots for training dynamics
3. Compare final model sizes (sparsity levels)
4. Test best models on additional datasets
5. Fine-tune hyperparameters for best-performing method

---

**Happy Experimenting!** 🚀 