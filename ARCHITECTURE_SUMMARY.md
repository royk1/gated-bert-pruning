# Configuration Architecture Summary

## Overview
This document describes the configuration architecture for the Gated BERT pruning experiments. The architecture ensures proper separation of concerns and allows for flexible parameter management.

## Architecture Components

### 1. `config.py` - Default Configuration
- **Purpose**: Defines all default parameters and configuration logic
- **Default State**: SPT pruning mode (`enable_pruning=True`, `pruning_method='spt'`)
- **Key Parameters**:
  - `enable_pruning=True` (default)
  - `pruning_method='spt'` (default)
  - `task='sst2'` (default)
  - `epochs=25`, `batch_size=96`, `learning_rate=2e-5`
  - `model_size='base'` (default)

### 2. `train.py` - Training Script
- **Purpose**: Main training script that uses configuration defaults
- **Behavior**: 
  - Uses `GatedBertConfig()` with defaults from `config.py`
  - When run directly: Uses SPT mode (config.py defaults)
  - When run via experiment script: Uses overridden parameters

### 3. `run_experiments.sh` - Experiment Orchestrator
- **Purpose**: Manages experiments across different pruning methods
- **Behavior**:
  - Backs up original `config.py`
  - Dynamically modifies `config.py` for each method
  - Runs training with modified configuration
  - Restores original `config.py` after completion

## Supported Methods

### 1. Baseline
- **Configuration**: `enable_pruning=False`, `pruning_method='none'`
- **Purpose**: No pruning, serves as baseline for comparison
- **Expected Behavior**: Stable performance across all epochs

### 2. Magnitude Pruning
- **Configuration**: `enable_pruning=True`, `pruning_method='magnitude'`
- **Purpose**: Prunes weights based on magnitude
- **Parameters**: Uses linear sparsity schedule

### 3. Movement Pruning
- **Configuration**: `enable_pruning=True`, `pruning_method='movement'`
- **Purpose**: Prunes weights based on movement scores
- **Parameters**: Uses step-based frequency and linear schedule

### 4. SPT Pruning
- **Configuration**: `enable_pruning=True`, `pruning_method='spt'`
- **Purpose**: Prunes weights based on SPT exploration
- **Parameters**: Uses step-based sparsity schedule and epsilon exploration

## Parameter Flow

```
config.py (defaults) 
    ↓
run_experiments.sh (overrides)
    ↓
train.py (uses overridden config)
    ↓
Training execution
```

## Key Benefits

1. **No Manual Config Changes**: `config.py` remains unchanged, all modifications are temporary
2. **Proper Baseline**: Baseline method correctly disables pruning
3. **Flexible Overrides**: Experiment script can override any parameter
4. **Clean Separation**: Each component has a clear responsibility
5. **Reproducible**: All experiments use the same base configuration

## Usage

### Direct Training (SPT mode)
```bash
python train.py
```

### Full Experiment Suite
```bash
./run_experiments.sh
```

### Single Method Testing
```bash
# Modify PRUNING_METHODS in run_experiments.sh to test specific methods
PRUNING_METHODS=("baseline" "spt")
./run_experiments.sh
```

## Configuration Parameters

### Training Parameters
- `EPOCHS=25`
- `BATCH_SIZE=96`
- `LEARNING_RATE=2e-5`
- `TASK=sst2`

### Pruning Parameters
- `START_SPARSITY=0.0`
- `END_SPARSITY=0.99`
- `PRUNING_START_EPOCH=4`
- `PRUNING_END_EPOCH=23`
- `SPARSITY_STEPS="10 25 50 70 75 80 84 88 90 92 93 94 95 96 97 97.5 98 98.5 99 99.5"`

### SPT Parameters
- `SPT_EXPLORE_PER_LAYER=False`

## Verification

The architecture has been tested and verified to work correctly:
- ✅ All 4 methods configure properly
- ✅ Baseline disables pruning correctly
- ✅ Config.py remains in SPT mode by default
- ✅ Experiment script properly overrides parameters
- ✅ No modifications needed to config.py 