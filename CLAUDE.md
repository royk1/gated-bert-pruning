# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of a BERT-like transformer model with gated dense layers for GLUE tasks. The project supports multiple pruning methods including magnitude-based pruning, movement pruning, and structured pruning with training (SPT). The codebase is designed to experiment with neural network pruning techniques on transformer architectures with support for multiple model sizes (tiny, small, base, large) ranging from ~9M to ~333M parameters.

## Key Architecture Components

### Core Model Structure
- **GatedDense layers**: Custom dense layers with multiplicative gates for pruning
- **Three pruning variants**: Regular GatedDense, MovementGatedDense, SptGatedDense
- **Transformer blocks**: Multi-head attention with gated feed-forward networks
- **BERT-like architecture**: Token + positional embeddings with transformer blocks

### Model Size Variants (BART-style)
- **Tiny**: ~9M parameters (2 layers, 256 d_model, 4 heads, 512 dff)
- **Small**: ~28M parameters (6 layers, 512 d_model, 8 heads, 1024 dff)
- **Base**: ~108M parameters (12 layers, 768 d_model, 12 heads, 3072 dff)
- **Large**: ~333M parameters (24 layers, 1024 d_model, 16 heads, 4096 dff)

### Model Variants
- **Standard Gated BERT** (`create_gated_bert_model`): Uses GatedDense layers with fixed gates
- **Movement Pruning** (`create_movement_gated_bert_model`): Uses MovementGatedDense for gradient-based pruning
- **SPT Pruning** (`create_spt_gated_bert_model`): Uses SptGatedDense for structured pruning during training

### Key Files and Their Roles
- `train.py`: Main training script with custom training loop for SPT and comprehensive results backup
- `config.py`: Configuration class with task-specific settings, pruning parameters, and model size presets
- `model_builder.py`: Model creation functions for different pruning variants
- `model_layers.py`: Core GatedDense, GatedMultiHeadAttention, and transformer components
- `movement_layers.py`: Movement pruning specific layers and logic
- `spt_layers.py`: SPT (Structured Pruning with Training) specific implementations
- `pruning.py`: Pruning callbacks and methods (magnitude, movement, SPT)
- `data_utils.py`: GLUE dataset loading and preprocessing utilities
- `callbacks.py`: Custom training callbacks for monitoring gated layers
- `utils.py`: Utility functions for logging, plotting, and TensorFlow setup
- `run_experiments.sh`: Automated bash script with command line interface for running experiments across all pruning methods
- `backup.py`: Automated backup script for project files including shell scripts
- `EXPERIMENT_GUIDE.md`: Documentation for the automated experiment script
- `MODEL_SIZE_SCALING_FIXES.md`: Documentation of model size scaling fixes

## Development Commands

### Running Training
```bash
# Basic training with default config (SPT pruning enabled by default)
python train.py

# Run with specific environment variables
BENCHMARK_EPOCHS=5 python train.py

# Automated experiments across all pruning methods (with defaults)
./run_experiments.sh

# Automated experiments with command line parameters
./run_experiments.sh -e 10 -b 64 -m tiny -p baseline,spt

# Parallel execution (all methods simultaneously)
./run_experiments.sh -parallel

# Parallel with custom job limit
./run_experiments.sh -parallel -max-parallel-jobs 2

# Quick parallel test
./run_experiments.sh -parallel -e 5 -b 32 -m tiny -p baseline,spt

# Show command line help
./run_experiments.sh -h

# Create project backup
python backup.py -m "Your backup message here"
```

### Testing
```bash
# Test gated dense layer functionality
python test_gated_dense.py

# Test pruning methods
python test_pruning_methods.py

# Test configuration and model sizes
python test_config_only.py
```

### Configuration
- Training configuration is handled through `GatedBertConfig` class in `config.py`
- Default task is 'sst2', supports all GLUE tasks (sst2, mnli, qnli, rte, wnli, mrpc, cola, stsb)
- Default model size is 'base' (~108M parameters)
- Pruning is enabled by default with SPT method, 0.0→1.0 sparsity from epoch 4→23

### Key Configuration Parameters
- `model_size`: Model size variant ('tiny', 'small', 'base', 'large')
- `enable_pruning`: Controls if pruning is active (default: True)
- `pruning_method`: 'magnitude', 'movement', or 'spt' (default: 'spt')
- `start_sparsity`/`end_sparsity`: Pruning schedule (default: 0.0→1.0)
- `pruning_start_epoch`/`pruning_end_epoch`: When to start/end pruning (default: 4→23)
- `sparsity_steps`: Step-based pruning schedule (alternative to linear start/end sparsity)
  - Format: List of sparsity percentages per epoch after pruning_start_epoch
  - **Model-size-adaptive**: Different schedules for different model sizes
  - If provided, overrides linear start_sparsity → end_sparsity schedule
  - After last step, sparsity remains constant at the final value

## Model Size Scaling and Adaptive Parameters

### **Critical Fix: Model-Size-Adaptive Pruning Parameters**

The implementation now includes **model-size-adaptive parameters** to prevent accuracy collapse in larger models during pruning:

#### **1. SPT Epsilon Scaling**
```python
# Scale epsilon inversely with model size to prevent over-exploration
if self.model_size == 'tiny':
    self.spt_epsilon = 0.01      # Base exploration
elif self.model_size == 'small':
    self.spt_epsilon = 0.005     # Half exploration
elif self.model_size == 'base':
    self.spt_epsilon = 0.0025    # Quarter exploration
elif self.model_size == 'large':
    self.spt_epsilon = 0.001     # Tenth exploration
```

#### **2. Movement Pruning Frequency Scaling**
```python
# Scale frequency with model size
if self.model_size == 'tiny':
    self.movement_pruning_frequency_steps = 100
elif self.model_size == 'small':
    self.movement_pruning_frequency_steps = 200
elif self.model_size == 'base':
    self.movement_pruning_frequency_steps = 400
elif self.model_size == 'large':
    self.movement_pruning_frequency_steps = 800
```

#### **3. Model-Size-Adaptive Pruning Schedules**
```python
# Tiny model: Aggressive schedule (can handle it)
sparsity_steps = [10,25,50,70,75,80,84,88,90,92,93,94,95,96,97,97.5,98,98.5,99,99.5]

# Base model: Conservative schedule
sparsity_steps = [3,10,20,30,40,50,60,70,75,80,85,88,90,92,94,95,96,97,98,99]

# Large model: Very conservative schedule
sparsity_steps = [2,8,15,25,35,45,55,65,70,75,80,85,88,90,92,94,95,96,97,98]
```

### **Parameter Comparison Table**
| Model Size | SPT Epsilon | Movement Frequency | Early Sparsity (5 epochs) |
|------------|-------------|-------------------|---------------------------|
| Tiny (9M)  | 0.01        | 100 steps         | 75%                       |
| Small (28M)| 0.005       | 200 steps         | 55%                       |
| Base (108M)| 0.0025      | 400 steps         | 40%                       |
| Large (333M)| 0.001      | 800 steps         | 35%                       |

## Important Implementation Details

### Custom Training Loop
- SPT method uses `CustomFitModel` with specialized `train_step()` that performs reward-based exploration
- Movement pruning uses `MovementPruningModel` wrapper for gradient-based pruning
- Standard gated model uses regular Keras training

### Comprehensive Results Backup System
The training script automatically creates comprehensive results after each experiment:

#### **Results Organization**
- **Method-specific directories**: `results_{method}/` (spt, magnitude, movement, baseline)
- **Automatic backup**: Previous results zipped before new results
- **Timestamped files**: All files include timestamps for versioning

#### **Results Contents**
- **CSV tables**: Epoch-wise metrics (loss, accuracy, sparsity, rewards)
- **Plots**: Loss vs sparsity, accuracy vs sparsity, reward vs sparsity (SPT)
- **Model files**: Best model weights and configuration
- **Metadata**: Complete experiment parameters and final results

#### **Backup Function**
- `create_comprehensive_results_backup()`: Centralized results creation
- **Always runs**: Executes for all methods, even if pruning is disabled
- **Error handling**: Robust with extensive try-catch blocks
- **Pandas optional**: Gracefully handles missing pandas installation

### Enhanced Automated Experiment Script (`run_experiments.sh`)

#### **New Command Line Interface Features**
The script now supports comprehensive command line parameter control:

##### **Available Parameters**
- **Training Parameters**:
  - `-e, --epochs NUM`: Number of training epochs (default: 25)
  - `-b, --batch-size NUM`: Batch size (default: 96)
  - `-l, --learning-rate RATE`: Learning rate (default: 2e-5)
  - `-m, --model-size SIZE`: Model size: tiny, small, base, large (default: base)

- **Pruning Parameters**:
  - `-p, --pruning-methods LIST`: Comma-separated list of methods to test
  - `-s, --start-sparsity NUM`: Starting sparsity (0.0-1.0, default: 0.0)
  - `-f, --end-sparsity NUM`: Final sparsity (0.0-1.0, default: 0.99)
  - `-ps, --pruning-start NUM`: Epoch to start pruning (default: 4)
  - `-pe, --pruning-end NUM`: Epoch to end pruning (default: 23)
  - `-ss, --sparsity-steps LIST`: Space-separated sparsity steps
  - `-se, --spt-explore BOOL`: SPT per-layer exploration: true/false

- **Task Configuration**:
  - `-t, --task TASK`: Task selection (default: sst2)
  - `-ml, --max-len NUM`: Maximum sequence length (default: 128)

##### **Usage Examples**
```bash
# Show help
./run_experiments.sh -h

# Run with all defaults (same as before)
./run_experiments.sh

# Quick test with custom parameters
./run_experiments.sh -e 10 -b 64 -m tiny

# Test specific methods only
./run_experiments.sh -p baseline,spt -t mnli

# Quick baseline test with tiny model
./run_experiments.sh -m tiny -e 5 -p baseline

# Custom sparsity schedule
./run_experiments.sh -s 0.1 -f 0.8 -ps 2 -pe 15

# Multiple parameter combination
./run_experiments.sh -e 20 -b 48 -m small -p magnitude,movement -t sst2 -s 0.05 -f 0.9
```

##### **Parameter Validation**
- **Comprehensive validation** for all parameters
- **Smart epoch capping**: Pruning end epoch automatically capped to not exceed total epochs
- **Configuration display**: Shows current configuration before running experiments
- **Error handling**: Clear error messages for invalid parameters

##### **Backward Compatibility**
- **No parameters**: Runs with current defaults (same behavior as before)
- **Partial parameters**: Only specified parameters are overridden
- **Help system**: Comprehensive help with examples and notes

#### **Original Features (Still Available)**
- **Multi-method testing**: Runs magnitude, movement, SPT, and baseline methods
- **Config management**: Automatically modifies `config.py` for each method
- **Error handling**: Robust cleanup and restoration on any failure
- **Performance optimization**: Configurable logging and delays
- **Results verification**: Checks for successful results directory creation

#### **Configuration (Now Overridable via CLI)**
```bash
# Training parameters (now command line configurable)
EPOCHS=25
BATCH_SIZE=96
LEARNING_RATE="2e-5"

# Pruning parameters (now command line configurable)
START_SPARSITY=0.0
END_SPARSITY=0.99
PRUNING_START_EPOCH=4
PRUNING_END_EPOCH=23  # Automatically capped at EPOCHS

# Step-based pruning schedule (now command line configurable)
SPARSITY_STEPS="10 25 50 70 75 80 84 88 90 92 93 94 95 96 97 97.5 98 98.5 99 99.5"

# Task configuration (now command line configurable)
TASK="sst2"  # Can be changed to any GLUE task
MAX_LEN=128

# Performance settings
SLEEP_BETWEEN_RUNS=1  # Reduced from 5 seconds
PERFORMANCE_MODE=false  # Set to true for maximum speed
```

#### **Usage**
```bash
# Run all experiments with defaults
./run_experiments.sh

# Run with custom parameters
./run_experiments.sh -e 10 -b 64 -m tiny -p baseline,spt

# For maximum performance (minimal logging)
# Edit script: PERFORMANCE_MODE=true, SLEEP_BETWEEN_RUNS=0
```

#### **Safety Features**
- **Config backup**: Original `config.py` backed up before modification
- **Automatic restoration**: Config restored even on errors/interruptions
- **Cleanup**: Backup files automatically deleted after restoration
- **Trap handling**: Ensures cleanup on Ctrl+C, errors, or normal exit

### Enhanced Backup System (`backup.py`)

#### **Features**
- **Shell script support**: Now includes `.sh` files in backups
- **Comprehensive coverage**: `.py`, `.sh`, `.log`, `.png`, `.json`, `.bak` files
- **File structure**: Generates tree-like structure in readme
- **Timestamped backups**: Automatic naming with timestamps

#### **Usage**
```bash
python backup.py -m "Added new features and bug fixes"
```

### SPT (Structured Pruning with Training) Detailed Implementation

**SPT is a two-phase pruning method that combines per-batch exploration with epoch-end pruning:**

#### **Phase 1: Per-Batch Exploration Process**
After each standard training step, SPT performs additional exploration:

1. **Standard Training Step**: Normal forward pass with dropout, backpropagation, and weight updates
2. **Exploration Phase** (two additional inference-mode forward passes):
   - **Sample epsilon% weights**: `sample_and_prune_epsilon_weights(spt_epsilon)` randomly samples epsilon% of currently unpruned weights
   - **First Forward Pass**: Temporarily gate sampled weights to 0, run inference (`training=False`) to get `pruned_loss`
   - **Restore weights**: `restore_gates()` returns sampled weights to original values
   - **Second Forward Pass**: Run inference (`training=False`) with all weights to get `normal_loss`
3. **Reward Calculation**: `reward = normal_loss - pruned_loss`
   - **Positive reward**: Weight is important (loss increases when pruned)
   - **Negative reward**: Weight is less important (loss decreases when pruned)
4. **Reward Storage**: Update exponential moving average for explored weights:
   - **EMA Formula**: `new_score = alpha * new_reward + (1-alpha) * old_score`
   - **First exploration**: Uses reward directly for new weights
   - **Per-weight tracking**: Each weight maintains independent EMA history

#### **Phase 2: Epoch-End Pruning Process**
At the end of each epoch, SPT performs permanent pruning based on accumulated scores:

1. **Pruning Schedule**: Linear interpolation from `start_sparsity` to `end_sparsity`
   - Applied between `pruning_start_epoch` and `pruning_end_epoch`
   - Default: 0% → 100% sparsity from epoch 4 → 23
2. **Score-Based Selection**: 
   - **Importance Score**: `avg_reward = reward_buffer / reward_count` for explored weights
   - **Unexplored weights**: Given neutral score (0.0) for medium priority
   - **Selection criterion**: Weights with **lowest average rewards** are pruned first (minimal loss degradation)
3. **Permanent Pruning**: Selected weights have their gates permanently set to 0.0
4. **Threshold handling**: Uses `np.partition()` for efficient k-th element selection with tie-breaking

#### **Key SPT Configuration Parameters**
- **`spt_epsilon`**: Fraction of weights to explore per batch (default: 0.01 = 1%)
  - **Model-size-adaptive**: Scales inversely with model size to prevent over-exploration
  - **Per-layer calculation**: When `spt_explore_per_layer=True`, epsilon is calculated PER LAYER TYPE, not across all layers
  - **Example**: With epsilon=0.01, if a layer has 1000 weights, 10 weights will be explored from that layer
  - **Total exploration**: When `spt_explore_per_layer=False`, epsilon is applied to each layer independently
- **`spt_reward_alpha`**: EMA smoothing factor for reward accumulation (default: 0.125 = 1/8)
  - Example: `new_score = 1/8 * new_diff + 7/8 * prev_score`
- **`spt_explore_per_layer`**: Per-layer exploration mode (default: False)
  - If True: Each batch explores only one layer type in cyclic manner
  - Creates competition between layer types and allows more sensitive layers to be pruned less
  - Layer types: Query, Key, Value, Attention Output, FFN Hidden, FFN Output, Regressor, Classifier
- **`start_sparsity`/`end_sparsity`**: Pruning schedule targets (default: 0.0 → 1.0)
- **`pruning_start_epoch`/`pruning_end_epoch`**: Pruning schedule timing (default: 4 → 23)
- **`sparsity_steps`**: Step-based pruning schedule (alternative to linear schedule)
  - Format: List of sparsity percentages per epoch after pruning_start_epoch
  - **Model-size-adaptive**: Different schedules for different model sizes
  - If provided, overrides linear start_sparsity → end_sparsity schedule

#### **SPT Implementation Details**
- **Boolean masking**: Uses `tf.boolean_mask` instead of `tf.gather_nd` for Apple Metal GPU compatibility
- **Reward buffers**: `reward_buffer` and `reward_count` track per-weight exploration history on each `SptGatedDense` layer
- **Inference-only exploration**: All exploration happens with `training=False` to avoid random dropout effects
- **Independent layer processing**: Each `SptGatedDense` layer independently tracks and updates its own weight rewards
- **Per-layer exploration**: When enabled, explores one layer type per batch in cyclic manner (Query → Key → Value → Attention Output → FFN Hidden → FFN Output → Regressor → Classifier)
- **Layer competition**: Per-layer exploration creates competition between layer types, allowing more sensitive layers to be pruned less
- **Layer reward table**: At the end of each epoch before pruning, displays a 2-line table showing average rewards per layer type
- **Efficient selection**: Uses `np.partition()` for O(n) k-th element finding during pruning
- **Incremental pruning**: Only prunes additional weights to reach target sparsity, not recalculating total sparsity

## Critical Bug Fixes and Optimizations

### **Model Size Scaling Fixes**
- **Fixed accuracy collapse**: Implemented model-size-adaptive parameters to prevent larger models from collapsing during pruning
- **SPT epsilon scaling**: Larger models use smaller epsilon values to prevent over-exploration
- **Movement frequency scaling**: Larger models use higher step frequencies to reduce pruning aggressiveness
- **Conservative pruning schedules**: Larger models use more gradual pruning schedules
- **Automatic parameter adjustment**: All scaling is automatic based on `model_size` parameter

### **TensorFlow/Keras Compatibility Issues**
- **Keras 3.x migration**: Updated all imports from `tf.keras` to `keras` for compatibility with TensorFlow 2.15+
- **KerasTensor errors**: Fixed by using direct TensorFlow operations instead of `keras.ops`
- **Device placement**: Replaced `tf.gather_nd` with `tf.boolean_mask` for Apple Metal GPU compatibility
- **Matrix multiplication**: Used `@` operator instead of `tf.matmul` for Keras Functional API
- **Shape handling**: Proper use of `tf.shape` and `tf.reshape` for Keras compatibility
- **Compiled metrics**: Fixed `AttributeError: 'DeprecatedCompiledMetric'` by using `self.metrics` directly
- **Variable initialization**: Fixed `tf.Variable` initialization in `tf.function` contexts using callable initializers

### **Platform Registration Errors (Metal GPU)**
- **Persistent Metal GPU issues**: Resolved platform registration errors on Apple Silicon
- **CPU training fallback**: Implemented `tf.device('/CPU:0')` context for training to avoid GPU issues
- **Environment variables**: Proper setup for Metal GPU stability
- **TensorFlow version**: Downgraded to TensorFlow 2.15.0 for better compatibility
- **Conda environment**: Resolved `pyarrow` and `libiconv` library conflicts

### **Step-Based Pruning Implementation**
- **New pruning schedule**: Added support for step-based pruning schedules as alternative to linear schedules
- **Backward compatibility**: Maintains support for existing linear start_sparsity → end_sparsity schedules
- **Flexible configuration**: Can specify exact sparsity targets per epoch after pruning_start_epoch
- **Automatic conversion**: Handles percentage values (0-100) and converts to decimals (0.0-1.0)
- **Fallback behavior**: After all steps, maintains the final sparsity value

### **Training Loop Fixes**
- **Duplicate training**: Removed duplicate `model.fit()` calls that caused training to run twice
- **Results backup**: Ensured comprehensive results are always saved before returning
- **Error handling**: Added robust error handling throughout the training pipeline
- **CPU training**: Forced CPU-only execution during training to avoid platform registration errors

### **Enhanced Bash Script with Command Line Interface**
- **Command line parameters**: Added comprehensive parameter support for all major configuration options
- **Parameter validation**: Robust validation for all command line arguments
- **Backward compatibility**: Maintains existing behavior when no parameters are provided
- **Help system**: Comprehensive help with examples and usage notes
- **Configuration display**: Shows current configuration before running experiments
- **Smart epoch capping**: Automatically caps pruning end epoch to not exceed total epochs
- **Variable substitution**: Fixed bash variable expansion in heredoc for config updates
- **Performance mode**: Added option to disable logging for maximum speed
- **Sleep reduction**: Reduced delays between experiments from 5 to 1 second
- **Tee overhead**: Optimized logging to reduce I/O overhead
- **Comprehensive cleanup**: Enhanced cleanup to remove all backup and temporary files from previous runs
  - Cleans up `config.py.backup*`, `config.py.modified*`, `config.py.tmp`, `update_config.py`
  - Runs at script start and on any exit (success, error, or interruption)
  - Prevents accumulation of leftover files from previous runs
- **Environment activation**: Removed explicit conda environment activation to use current shell environment

### **SPT Pruning Fixes**
- **Incremental pruning**: Fixed logic to only prune additional weights, not recalculate total sparsity
- **Reward calculation**: Corrected EMA formula for proper reward accumulation
- **Unexplored weights**: Set neutral score (0.0) instead of infinity for unexplored weights
- **Callable initializers**: Fixed `tf.Variable` initialization in `SptGatedDense` using callable initializers

### **Import and Dependency Fixes**
- **Keras imports**: Fixed `ImportError: Keras cannot be imported` by updating all imports
- **Layer inheritance**: Updated all custom layers to inherit from `keras.layers.Layer`
- **Callback inheritance**: Updated all callbacks to inherit from `keras.callbacks.Callback`
- **Model inheritance**: Updated model classes to inherit from `keras.Model`

### **Configuration and Backward Compatibility**
- **Default model size**: Changed from 'tiny' to 'base' for better performance
- **Backward compatibility**: Ensured existing configurations continue to work
- **Parameter validation**: Added validation for model size and other parameters
- **Configuration serialization**: Enhanced save/load functionality for configurations

## Gate Mechanism
- All models use multiplicative gates on weight matrices: `gated_kernel = kernel * gates`
- Gates are initialized to 1.0 (no pruning) and can be made trainable for pruning methods
- Each layer tracks gate statistics for monitoring sparsity

## Dataset Support
- Supports all GLUE tasks with automatic configuration
- Uses HuggingFace datasets and BERT tokenizer
- Handles both classification and regression tasks (STS-B)

## Checkpointing and Logging
- Model weights saved to `checkpoints_gated_bert_*/best_model_weights.h5`
- Training logs written to `gated_bert_*.log`
- Training history and plots automatically generated
- Configuration saved as JSON for reproducibility
- Comprehensive results automatically organized in method-specific directories

## Performance Considerations

### **Bash Script Performance**
- **Command line interface**: New parameter parsing adds minimal overhead
- **Tee overhead**: Removed `tee` command that was causing 20-30% slowdown
- **Config operations**: File I/O for config updates adds overhead
- **Sleep delays**: Reduced from 5 to 1 second between experiments
- **Performance mode**: Option to disable logging for maximum speed

### **GPU Compatibility**
- **Apple Metal**: Uses `tf.boolean_mask` instead of `tf.gather_nd` for GPU compatibility
- **Soft device placement**: Enabled to handle CPU/GPU operation mixing
- **Environment variables**: Proper setup for TensorFlow optimization
- **CPU training**: Training loop runs on CPU to avoid Metal GPU issues

### **Memory Management**
- **Incremental pruning**: Prevents memory spikes during pruning operations
- **Efficient selection**: Uses `np.partition()` for O(n) pruning selection
- **Reward buffers**: Efficient per-weight tracking without excessive memory usage
- **Model size scaling**: Adaptive parameters prevent memory issues in larger models

### **Model Size Performance**
- **Tiny model**: Fastest training, can handle aggressive pruning
- **Small model**: Moderate training speed, balanced pruning
- **Base model**: Standard training speed, conservative pruning
- **Large model**: Slower training, very conservative pruning

## Environment Setup

### **Required Dependencies**
- TensorFlow 2.15.0 (recommended for stability)
- Keras 3.x
- HuggingFace datasets and transformers
- NumPy, Pandas (optional for plotting)
- Matplotlib (optional for plotting)

### **Environment Variables**
- `TF_CPP_MIN_LOG_LEVEL=3`: Suppress TensorFlow warnings
- `TF_DISABLE_MPS=0`: Enable Metal Performance Shaders
- `TF_ENABLE_ONEDNN_OPTS=0`: Disable OneDNN optimizations
- `TF_USE_LEGACY_KERAS=false`: Use new Keras 3.x
- `TF_ENABLE_DEPRECATION_WARNINGS=false`: Suppress deprecation warnings

### **Conda Environment Issues**
- **pyarrow/libiconv conflicts**: Common on macOS, resolved by reinstalling libraries
- **Library conflicts**: Use `conda install libiconv` then `conda install pyarrow`

## Notes for Development

- The codebase is actively developed with comprehensive backup and versioning
- TensorFlow/Keras with CPU training (Metal GPU issues resolved via CPU fallback)
- Heavy use of custom layers means model serialization requires special handling
- SPT pruning is the most complex method, requiring custom training loop and reward tracking
- Enhanced automated experiment script provides comprehensive testing across all pruning methods with command line interface
- Comprehensive results backup ensures no experimental data is lost
- Enhanced backup system includes shell scripts and provides complete project snapshots
- Model size scaling fixes ensure stable performance across all model variants
- All pruning methods now scale appropriately with model size to prevent accuracy collapse
- Command line interface provides maximum flexibility for experiment customization

## Recent Major Updates

### **GPU Support (Latest)**
- Added comprehensive GPU support for both NVIDIA CUDA and Apple MPS
- Implemented automatic GPU detection and configuration in `utils.setup_tensorflow()`
- Added platform-specific optimizations for different GPU types
- Implemented CPU fallback when no GPU is available
- Added test script `test_gpu_support.py` for verifying GPU functionality
- Maintained backward compatibility with existing MPS support
- Added GPU memory growth configuration and soft device placement

### **Parallel Execution (Latest)**
- Added `-parallel` flag for simultaneous execution of all pruning methods
- Implemented real-time progress monitoring with multiple output modes:
  - **Progress Bars**: Clean visual progress bars for each job
  - **Real-time Logs**: Live log output from each job
  - **Both**: Progress bars + key log messages
- Added job isolation with separate working directories
- Implemented configurable parallel job limits (`-max-parallel-jobs`)
- Added comprehensive error handling and cleanup
- Maintained full backward compatibility (no `-parallel` = same as before)

### **Command Line Interface (Latest)**
- Added comprehensive command line parameter support to `run_experiments.sh`
- Implemented parameter validation and error handling
- Added help system with examples and usage notes
- Maintained backward compatibility with existing functionality
- Added configuration display before running experiments

### **Model Size Support (Latest)**
- Added support for 4 model sizes: tiny, small, base, large
- Implemented model-size-adaptive pruning parameters
- Fixed accuracy collapse issues in larger models
- Added comprehensive model size testing

### **TensorFlow/Keras Compatibility (Latest)**
- Migrated to Keras 3.x API
- Fixed all import and inheritance issues
- Resolved Metal GPU platform registration errors
- Implemented CPU training fallback

### **Comprehensive Testing (Latest)**
- Added `test_config_only.py` for configuration validation
- Enhanced experiment automation
- Improved error handling and debugging
- Added model size parameter validation

### **Documentation (Latest)**
- Created `MODEL_SIZE_SCALING_FIXES.md` for detailed explanation
- Updated `CLAUDE.md` with comprehensive information
- Added parameter comparison tables
- Documented all recent fixes and improvements
- Updated README.md with command line interface documentation