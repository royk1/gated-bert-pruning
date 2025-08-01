# BART Size Variants

This document describes the new BART size variants feature that allows you to easily switch between different model sizes ranging from tiny (~9M parameters) to large (~333M parameters).

## Overview

The system now supports four different BART model sizes, similar to HuggingFace's BART variants:

- **Tiny**: ~9M parameters (2 layers, 256 hidden size)
- **Small**: ~28M parameters (6 layers, 512 hidden size)  
- **Base**: ~108M parameters (12 layers, 768 hidden size)
- **Large**: ~333M parameters (24 layers, 1024 hidden size)

## Model Specifications

| Model Size | Parameters | Layers | Hidden Size | Attention Heads | FFN Size | Vocab Size | Max Length |
|------------|------------|--------|-------------|-----------------|----------|------------|------------|
| Tiny       | ~9M        | 2      | 256         | 4               | 512      | 30,522     | 128        |
| Small      | ~28M       | 6      | 512         | 8               | 1024     | 30,522     | 128        |
| Base       | ~108M      | 12     | 768         | 12              | 3072     | 30,522     | 128        |
| Large      | ~333M      | 24     | 1024        | 16              | 4096     | 30,522     | 128        |

## Usage

### Basic Usage

To use different model sizes, simply specify the `model_size` parameter when creating a `GatedBertConfig`:

```python
from config import GatedBertConfig

# Tiny model (~9M parameters)
config = GatedBertConfig(model_size='tiny')

# Small model (~28M parameters)
config = GatedBertConfig(model_size='small')

# Base model (~108M parameters)
config = GatedBertConfig(model_size='base')

# Large model (~333M parameters)
config = GatedBertConfig(model_size='large')
```

### Advanced Configuration

You can combine model sizes with different tasks and pruning methods:

```python
# Tiny model for SST-2 without pruning
config = GatedBertConfig(
    model_size='tiny',
    task='sst2',
    enable_pruning=False
)

# Small model for MNLI with magnitude pruning
config = GatedBertConfig(
    model_size='small',
    task='mnli',
    enable_pruning=True,
    pruning_method='magnitude'
)

# Base model for QNLI with movement pruning
config = GatedBertConfig(
    model_size='base',
    task='qnli',
    enable_pruning=True,
    pruning_method='movement'
)

# Large model for RTE with SPT pruning
config = GatedBertConfig(
    model_size='large',
    task='rte',
    enable_pruning=True,
    pruning_method='spt'
)
```

### Parameter Override

You can override specific parameters while keeping the preset values for others:

```python
# Use base model size but override training parameters
config = GatedBertConfig(
    model_size='base',
    epochs=50,
    batch_size=16,
    learning_rate=1e-5
)

# Use large model size but override model parameters
config = GatedBertConfig(
    model_size='large',
    dropout_rate=0.1,
    max_len=256
)
```

## Model Information

You can get detailed information about a model configuration using the `get_model_info()` method:

```python
config = GatedBertConfig(model_size='base')
model_info = config.get_model_info()

print(f"Model Size: {model_info['model_size']}")
print(f"Parameters: {model_info['estimated_params_millions']}")
print(f"Layers: {model_info['num_layers']}")
print(f"Hidden Size: {model_info['d_model']}")
print(f"Attention Heads: {model_info['num_heads']}")
print(f"FFN Size: {model_info['dff']}")
```

## Supported Tasks

All model sizes support the following GLUE tasks:

- **SST-2**: Sentiment analysis (2 classes)
- **MNLI**: Natural language inference (3 classes)
- **QNLI**: Question-answering NLI (2 classes)
- **RTE**: Recognizing textual entailment (2 classes)
- **WNLI**: Winograd NLI (2 classes)
- **MRPC**: Microsoft Research Paraphrase Corpus (2 classes)
- **CoLA**: Corpus of Linguistic Acceptability (2 classes)
- **STS-B**: Semantic Textual Similarity Benchmark (regression)

## Supported Pruning Methods

All model sizes support the following pruning methods:

- **None**: No pruning (baseline)
- **Magnitude**: Magnitude-based pruning
- **Movement**: Movement pruning
- **SPT**: Structured Pruning with Training

## Training Considerations

### Memory Requirements

- **Tiny**: ~2-4GB GPU memory
- **Small**: ~4-8GB GPU memory
- **Base**: ~8-16GB GPU memory
- **Large**: ~16-32GB GPU memory

### Training Time

- **Tiny**: Fastest training, good for prototyping
- **Small**: Moderate training time, good balance
- **Base**: Longer training, better performance
- **Large**: Longest training, best performance

### Batch Size Recommendations

- **Tiny**: 32-64
- **Small**: 16-32
- **Base**: 8-16
- **Large**: 4-8

## Examples

### Quick Testing

For quick testing and prototyping, use the tiny model:

```python
config = GatedBertConfig(
    model_size='tiny',
    task='sst2',
    epochs=10,
    batch_size=32
)
```

### Production Training

For production training with good performance, use the base model:

```python
config = GatedBertConfig(
    model_size='base',
    task='mnli',
    epochs=25,
    batch_size=16,
    enable_pruning=True,
    pruning_method='spt'
)
```

### Research Experiments

For research experiments requiring maximum performance, use the large model:

```python
config = GatedBertConfig(
    model_size='large',
    task='qnli',
    epochs=50,
    batch_size=8,
    enable_pruning=True,
    pruning_method='movement'
)
```

## Testing

You can test the different model sizes using the provided test scripts:

```bash
# Test model size configurations
python test_model_sizes_simple.py

# Test model creation (requires TensorFlow)
python test_model_creation.py
```

## Migration from Previous Versions

If you were using the previous configuration system, you can migrate by:

1. **Before**: `GatedBertConfig(num_layers=2, d_model=256, ...)`
2. **After**: `GatedBertConfig(model_size='tiny', ...)`

The new system maintains backward compatibility, so existing configurations will continue to work.

## Troubleshooting

### Memory Issues

If you encounter memory issues with larger models:

1. Reduce the batch size
2. Use gradient accumulation
3. Enable mixed precision training
4. Use a smaller model size

### Performance Issues

If training is too slow:

1. Use a smaller model size for initial experiments
2. Reduce the number of epochs
3. Use a larger batch size if memory allows
4. Consider using a smaller task for testing

## Future Enhancements

Potential future enhancements include:

- Custom model size presets
- Dynamic model sizing based on available resources
- Automatic hyperparameter tuning for different model sizes
- Model size-specific learning rate schedules 