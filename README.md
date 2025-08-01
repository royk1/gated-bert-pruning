# Gated BERT: Neural Network Pruning Framework

A comprehensive framework for training and pruning BERT-style transformer models using various pruning methods. This project implements gated neural networks with support for magnitude-based, movement-based, and SPT (Structured Pruning with Training) pruning techniques.

## 🚀 Features

- **Multiple Model Sizes**: Support for tiny, small, base, and large BART-style models
- **Advanced Pruning Methods**: Magnitude, Movement, and SPT pruning
- **Flexible Configuration**: Easy parameter tuning and experiment management
- **Comprehensive Logging**: Detailed training logs and result visualization
- **Automated Experiments**: Batch experiment execution with result organization
- **Command Line Interface**: Full parameter control via command line arguments
- **Parallel Execution**: Run all pruning methods simultaneously with real-time progress monitoring

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Pruning Methods](#pruning-methods)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Experiment Management](#experiment-management)
- [Command Line Interface](#command-line-interface)
- [Parallel Execution](#parallel-execution)
- [Results and Analysis](#results-and-analysis)
- [File Structure](#file-structure)
- [Contributing](#contributing)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- HuggingFace datasets
- NumPy, Matplotlib, Pandas

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd new_design2

# Install dependencies
pip install tensorflow datasets numpy matplotlib pandas

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## 🚀 Quick Start

### Basic Training (SPT Mode)

```bash
# Train with default SPT pruning
python train.py
```

### Run All Experiments (Default)

```bash
# Run all pruning methods (baseline, magnitude, movement, spt) with defaults
./run_experiments.sh
```

### Command Line Experiments

```bash
# Quick test with custom parameters
./run_experiments.sh -e 10 -b 64 -m tiny

# Test specific methods only
./run_experiments.sh -p baseline,spt -t mnli

# Custom sparsity schedule
./run_experiments.sh -s 0.1 -f 0.8 -ps 2 -pe 15
```

### Parallel Execution

```bash
# Run all methods in parallel (much faster!)
./run_experiments.sh -parallel

# Parallel with custom job limit
./run_experiments.sh -parallel -max-parallel-jobs 2

# Quick parallel test
./run_experiments.sh -parallel -e 5 -b 32 -m tiny -p baseline,spt
```

## 🏗️ Model Architecture

### BART-Style Model Variants

The framework supports four model sizes based on BART architecture:

| Model Size | Layers | Hidden Size | Attention Heads | FFN Size | Parameters |
|------------|--------|-------------|-----------------|----------|------------|
| **Tiny**   | 2      | 256         | 4               | 512      | ~9M        |
| **Small**  | 6      | 512         | 8               | 1024     | ~28M       |
| **Base**   | 12     | 768         | 12              | 3072     | ~108M      |
| **Large**  | 24     | 1024        | 16              | 4096     | ~333M      |

### Gated Architecture

- **Gated Dense Layers**: Each weight matrix is multiplied by learnable gates
- **Selective Pruning**: Gates control which weights are active during training
- **Flexible Sparsity**: Dynamic sparsity patterns based on pruning method

## ✂️ Pruning Methods

### 1. Baseline (No Pruning)
- **Purpose**: Control experiment with no pruning applied
- **Configuration**: `enable_pruning=False`, `pruning_method='none'`
- **Use Case**: Establish baseline performance for comparison

### 2. Magnitude Pruning
- **Principle**: Prune weights with smallest absolute values
- **Schedule**: Linear sparsity increase from start to end epoch
- **Configuration**: `enable_pruning=True`, `pruning_method='magnitude'`
- **Advantages**: Simple, effective for static pruning

### 3. Movement Pruning
- **Principle**: Prune weights based on movement scores during training
- **Schedule**: Step-based frequency with linear sparsity increase
- **Configuration**: `enable_pruning=True`, `pruning_method='movement'`
- **Advantages**: Dynamic, adapts to training process

### 4. SPT (Structured Pruning with Training)
- **Principle**: Exploration-based pruning with reward mechanisms
- **Schedule**: Step-based sparsity schedule with epsilon exploration
- **Configuration**: `enable_pruning=True`, `pruning_method='spt'`
- **Advantages**: Intelligent, reward-driven pruning decisions

## ⚙️ Configuration

### Core Parameters

#### Model Configuration
```python
# Model size selection
model_size = 'base'  # 'tiny', 'small', 'base', 'large'

# Task configuration
task = 'sst2'  # 'sst2', 'mnli', 'qnli', 'rte', 'wnli', 'mrpc', 'cola', 'stsb'
max_len = 128  # Maximum sequence length
```

#### Training Parameters
```python
# Training hyperparameters
epochs = 25
batch_size = 96
learning_rate = 2e-5
dropout_rate = 0.0
```

#### Pruning Parameters
```python
# Pruning configuration
enable_pruning = True  # Enable/disable pruning
pruning_method = 'spt'  # 'magnitude', 'movement', 'spt', 'none'
start_sparsity = 0.0  # Initial sparsity (0.0 = no pruning)
end_sparsity = 0.99   # Final sparsity (0.99 = 99% pruned)
pruning_start_epoch = 4   # Epoch to start pruning
pruning_end_epoch = 23    # Epoch to reach target sparsity
```

### Method-Specific Parameters

#### SPT Parameters
```python
# SPT-specific configuration
spt_epsilon = 0.0025  # Exploration rate (scaled by model size)
spt_explore_per_layer = False  # Per-layer exploration
spt_reward_alpha = 0.125  # Reward smoothing factor

# Step-based sparsity schedule
sparsity_steps = [10, 25, 50, 70, 75, 80, 84, 88, 90, 92, 93, 94, 95, 96, 97, 97.5, 98, 98.5, 99, 99.5]
```

#### Movement Parameters
```python
# Movement-specific configuration
movement_pruning_frequency_steps = 400  # Steps between pruning (scaled by model size)
movement_schedule = 'linear'  # 'linear' or 'cubic'
```

### Task-Specific Configuration

| Task | Dataset | Classes | Type | Description |
|------|---------|---------|------|-------------|
| **SST-2** | GLUE/SST-2 | 2 | Classification | Sentiment analysis |
| **MNLI** | GLUE/MNLI | 3 | Classification | Natural language inference |
| **QNLI** | GLUE/QNLI | 2 | Classification | Question-answering NLI |
| **RTE** | GLUE/RTE | 2 | Classification | Recognizing textual entailment |
| **WNLI** | GLUE/WNLI | 2 | Classification | Winograd NLI |
| **MRPC** | GLUE/MRPC | 2 | Classification | Microsoft Research Paraphrase |
| **CoLA** | GLUE/CoLA | 2 | Classification | Corpus of Linguistic Acceptability |
| **STS-B** | GLUE/STS-B | 1 | Regression | Semantic Textual Similarity |

## 📖 Usage Examples

### Example 1: Quick Baseline Test

```python
from config import GatedBertConfig

# Create baseline configuration
config = GatedBertConfig(
    model_size='tiny',  # Fast testing
    task='sst2',
    epochs=5,
    enable_pruning=False,
    pruning_method='none'
)

# Train baseline model
python train.py
```

### Example 2: SPT Pruning Experiment

```python
# SPT configuration for base model
config = GatedBertConfig(
    model_size='base',
    task='sst2',
    epochs=25,
    enable_pruning=True,
    pruning_method='spt',
    start_sparsity=0.0,
    end_sparsity=0.99,
    pruning_start_epoch=4,
    pruning_end_epoch=23
)
```

### Example 3: Custom Magnitude Pruning

```python
# Magnitude pruning with custom schedule
config = GatedBertConfig(
    model_size='small',
    task='mnli',
    epochs=20,
    enable_pruning=True,
    pruning_method='magnitude',
    start_sparsity=0.1,
    end_sparsity=0.8,
    pruning_start_epoch=2,
    pruning_end_epoch=15
)
```

## 🧪 Experiment Management

### Automated Experiment Suite

The `run_experiments.sh` script provides comprehensive experiment management:

```bash
# Run all methods sequentially
./run_experiments.sh

# Run all methods in parallel (much faster!)
./run_experiments.sh -parallel

# Results are organized in:
# - results_baseline_sst2/
# - results_magnitude_sst2/
# - results_movement_sst2/
# - results_spt_sst2/
```

### Experiment Configuration

Edit `run_experiments.sh` to customize experiments:

```bash
# Training parameters
EPOCHS=25
BATCH_SIZE=96
LEARNING_RATE="2e-5"

# Pruning parameters
START_SPARSITY=0.0
END_SPARSITY=0.99
PRUNING_START_EPOCH=4
PRUNING_END_EPOCH=23

# Methods to test
PRUNING_METHODS=("magnitude" "movement" "spt" "baseline")

# Task configuration
TASK="sst2"
```

### Environment Variables

```bash
# Benchmark mode (override epochs)
export BENCHMARK_EPOCHS=10
python train.py
```

## 🖥️ Command Line Interface

### Overview

The `run_experiments.sh` script supports comprehensive command line parameter control, allowing you to customize all major parameters without editing the script.

### Available Parameters

#### **Training Parameters**
- **`-e, --epochs NUM`**: Number of training epochs (default: 25)
- **`-b, --batch-size NUM`**: Batch size (default: 96)
- **`-l, --learning-rate RATE`**: Learning rate (default: 2e-5)
- **`-m, --model-size SIZE`**: Model size: tiny, small, base, large (default: base)

#### **Pruning Parameters**
- **`-p, --pruning-methods LIST`**: Comma-separated list of methods to test
  - Options: baseline, magnitude, movement, spt
  - Default: magnitude,movement,spt,baseline
- **`-s, --start-sparsity NUM`**: Starting sparsity (0.0-1.0, default: 0.0)
- **`-f, --end-sparsity NUM`**: Final sparsity (0.0-1.0, default: 0.99)
- **`-ps, --pruning-start NUM`**: Epoch to start pruning (default: 4)
- **`-pe, --pruning-end NUM`**: Epoch to end pruning (default: 23)
- **`-ss, --sparsity-steps LIST`**: Space-separated sparsity steps (default: auto)
- **`-se, --spt-explore BOOL`**: SPT per-layer exploration: true/false (default: false)

#### **Task Configuration**
- **`-t, --task TASK`**: Task: sst2, mnli, qnli, rte, wnli, mrpc, cola, stsb (default: sst2)
- **`-ml, --max-len NUM`**: Maximum sequence length (default: 128)

### Usage Examples

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

### Parameter Validation

The script includes comprehensive parameter validation:

- **Epochs**: Must be positive integer
- **Batch Size**: Must be positive integer
- **Learning Rate**: Must be valid number format
- **Model Size**: Must be one of: tiny, small, base, large
- **Pruning Methods**: Must be valid methods (baseline, magnitude, movement, spt)
- **Sparsity Values**: Must be between 0.0 and 1.0
- **Task**: Must be valid GLUE task
- **SPT Explore**: Must be true/false

### Smart Features

- **Backward Compatibility**: If no parameters are provided, the script runs with current defaults
- **Epoch Capping**: Pruning end epoch is automatically capped to not exceed total epochs
- **Configuration Display**: Shows current configuration before running experiments
- **Error Handling**: Clear error messages for invalid parameters

## ⚡ Parallel Execution

### Overview

The parallel execution feature allows you to run all pruning methods simultaneously, significantly reducing total experiment time. Each method runs in its own isolated process with real-time progress monitoring.

### Key Features

- **Simultaneous Execution**: All methods run at the same time instead of sequentially
- **Real-time Progress Monitoring**: See progress bars for all jobs simultaneously
- **Job Isolation**: Each job runs in its own directory to prevent conflicts
- **Configurable Limits**: Control how many jobs run in parallel
- **Multiple Output Modes**: Choose your preferred way to view progress

### Usage

```bash
# Basic parallel execution
./run_experiments.sh -parallel

# Parallel with custom job limit
./run_experiments.sh -parallel -max-parallel-jobs 2

# Quick parallel test
./run_experiments.sh -parallel -e 5 -b 32 -m tiny -p baseline,spt

# Parallel with other parameters
./run_experiments.sh -parallel -e 10 -b 64 -m small -t mnli
```

### Output Modes

When you run parallel execution, you'll be prompted to choose your preferred output display:

#### **1. Progress Bars (Default)**
Shows clean progress bars for each job:
```
============================================
PARALLEL TRAINING PROGRESS MONITOR
============================================

[JOB 1 ] baseline     [████████████████████████████████████████████████████████████████████░░░░░░░░░░]  88% (Epoch 22/25)
[JOB 2 ] magnitude    [██████████████████████████████████████████████████████████████░░░░░░░░░░░░░░░░]  80% (Epoch 20/25)
[JOB 3 ] movement     [████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  32% (Epoch 8/25)
[JOB 4 ] spt          [█████████████████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░]  68% (Epoch 17/25)
```

#### **2. Real-time Logs**
Shows live log output from each job with job identifiers.

#### **3. Both**
Shows progress bars + recent important log messages.

### Performance Benefits

- **Faster Execution**: All methods run simultaneously instead of sequentially
- **Better Resource Utilization**: Uses multiple CPU cores/GPUs effectively
- **Real-time Monitoring**: See all jobs' progress at once
- **Flexible Control**: Choose output mode and job limits

### Safety Features

- **Job Isolation**: Each job has its own working directory
- **Error Handling**: Failed jobs don't stop other jobs
- **Cleanup**: Automatic cleanup of temporary files
- **Logging**: Each job gets its own detailed log file

### GPU Considerations

- **Single GPU**: All jobs will share the GPU, which may cause memory contention
- **Multi-GPU**: Jobs can be assigned to different GPUs (requires additional configuration)
- **Memory Management**: Larger models may require reducing parallel job count

## 📊 Results and Analysis

### Output Structure

Each experiment generates:

```
results_{method}_{task}/
├── accuracy_sparsity_{method}_{timestamp}.png
├── loss_sparsity_{method}_{timestamp}.png
├── metadata_{method}_{timestamp}.json
└── results_{method}_{timestamp}.csv
```

### Key Metrics

- **Accuracy vs Sparsity**: Performance degradation analysis
- **Loss vs Sparsity**: Training stability assessment
- **Metadata**: Complete experiment configuration
- **CSV Results**: Detailed epoch-by-epoch data

### Visualization

```python
# Plot comparison across methods
python plot_comparison.py

# Generates: pruning_methods_comparison_dual_axis.png
```

## 📁 File Structure

```
new_design2/
├── config.py              # Configuration management
├── train.py               # Main training script
├── run_experiments.sh     # Experiment orchestration with CLI and parallel execution
├── model_layers.py        # Gated layer implementations
├── pruning.py             # Pruning method implementations
├── callbacks.py           # Training callbacks
├── data_utils.py          # Data loading utilities
├── utils.py               # Utility functions
├── experiment_logs/       # Training logs
├── results_*/             # Experiment results
├── checkpoints_*/         # Model checkpoints
└── saved_results/         # Archived results
```

### Key Files

- **`config.py`**: Central configuration with BART presets and pruning parameters
- **`train.py`**: Main training loop with method-specific model creation
- **`run_experiments.sh`**: Automated experiment execution with command line interface and parallel execution
- **`pruning.py`**: Implementation of magnitude, movement, and SPT pruning
- **`model_layers.py`**: Gated dense layers and transformer components

## 🔬 Advanced Usage

### Custom Model Sizes

```python
# Custom model configuration
config = GatedBertConfig(
    num_layers=8,
    d_model=512,
    num_heads=8,
    dff=2048,
    vocab_size=30522,
    max_len=128
)
```

### Custom Pruning Schedules

```python
# Custom step-based schedule
config = GatedBertConfig(
    sparsity_steps=[5, 15, 30, 50, 70, 85, 95, 98, 99, 99.5]
)
```

### Multi-Task Training

```python
# Switch between tasks
config = GatedBertConfig(task='mnli')  # 3-class classification
config = GatedBertConfig(task='stsb')  # Regression
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**: Use smaller model size (`tiny` or `small`)
2. **Training Instability**: Reduce learning rate or increase batch size
3. **Pruning Too Aggressive**: Adjust sparsity schedule or start/end epochs
4. **Configuration Errors**: Verify parameter values in `config.py`
5. **Parallel Execution Issues**: Reduce `-max-parallel-jobs` if running out of memory

### Debug Mode

```bash
# Enable verbose logging
export VERBOSE=2
python train.py
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where appropriate
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on BART architecture from HuggingFace
- Pruning methods inspired by research in neural network compression
- Built with TensorFlow 2.x and HuggingFace datasets

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the experiment logs for detailed error information
- Review the configuration parameters in `config.py`

---

**Happy Pruning! 🎯**
