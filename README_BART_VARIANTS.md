# BART Size Variants - Quick Start

This project now supports four different BART model sizes, allowing you to easily scale from tiny (~9M parameters) to large (~333M parameters) models.

## Quick Usage

```python
from config import GatedBertConfig

# Choose your model size
config = GatedBertConfig(model_size='base')  # 'tiny', 'small', 'base', 'large'

# Use in training
# train_model(config)
```

## Model Sizes

| Size  | Parameters | Layers | Hidden | Heads | Use Case |
|-------|------------|--------|--------|-------|----------|
| Tiny  | ~9M        | 2      | 256    | 4     | Prototyping |
| Small | ~28M       | 6      | 512    | 8     | Quick experiments |
| Base  | ~108M      | 12     | 768    | 12    | Standard training |
| Large | ~333M      | 24     | 1024   | 16    | Research/Production |

## Examples

### Quick Test
```python
config = GatedBertConfig(
    model_size='tiny',
    task='sst2',
    epochs=5
)
```

### Standard Training
```python
config = GatedBertConfig(
    model_size='base',
    task='mnli',
    enable_pruning=True,
    pruning_method='spt'
)
```

### Research Experiment
```python
config = GatedBertConfig(
    model_size='large',
    task='qnli',
    epochs=50,
    batch_size=4
)
```

## Testing

```bash
# Test configurations
python test_model_sizes_simple.py

# See examples
python example_usage.py
```

## Documentation

For detailed documentation, see `BART_SIZE_VARIANTS.md`.

## Backward Compatibility

Existing configurations continue to work. The new system is additive and doesn't break existing code. 