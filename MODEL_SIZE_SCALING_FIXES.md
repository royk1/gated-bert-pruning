# Model Size Scaling Fixes for Pruning Methods

## Problem Analysis

The accuracy collapse in larger models during pruning was caused by **fixed parameters across all model sizes**, which didn't scale appropriately with model complexity.

### Root Causes

1. **Fixed SPT Epsilon (0.01)**: 
   - Tiny model: 0.01 × 9M = 90K weights explored per batch
   - Base model: 0.01 × 108M = 1.08M weights explored per batch
   - Large model: 0.01 × 333M = 3.33M weights explored per batch

2. **Fixed Movement Pruning Frequency (100 steps)**:
   - Same pruning frequency regardless of model size
   - Larger models pruned too aggressively

3. **Fixed Aggressive Pruning Schedule**:
   - Same aggressive schedule for all models
   - Larger models couldn't handle the rapid capacity reduction

## Implemented Fixes

### 1. Model-Size-Adaptive SPT Epsilon

```python
# Scale epsilon inversely with model size
if self.model_size == 'tiny':
    self.spt_epsilon = 0.01      # Base exploration
elif self.model_size == 'small':
    self.spt_epsilon = 0.005     # Half exploration
elif self.model_size == 'base':
    self.spt_epsilon = 0.0025    # Quarter exploration
elif self.model_size == 'large':
    self.spt_epsilon = 0.001     # Tenth exploration
```

**Result**: Larger models explore proportionally fewer weights per batch, preventing over-exploration.

### 2. Model-Size-Adaptive Movement Pruning Frequency

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

**Result**: Larger models are pruned less frequently, allowing more stable training.

### 3. Model-Size-Adaptive Pruning Schedules

```python
# Tiny model: Aggressive schedule (can handle it)
sparsity_steps = [10,25,50,70,75,80,84,88,90,92,93,94,95,96,97,97.5,98,98.5,99,99.5]

# Base model: Conservative schedule
sparsity_steps = [3,10,20,30,40,50,60,70,75,80,85,88,90,92,94,95,96,97,98,99]

# Large model: Very conservative schedule
sparsity_steps = [2,8,15,25,35,45,55,65,70,75,80,85,88,90,92,94,95,96,97,98]
```

**Result**: Larger models use more gradual pruning schedules, preventing rapid capacity loss.

## Parameter Comparison

| Model Size | SPT Epsilon | Movement Frequency | Early Sparsity (5 epochs) |
|------------|-------------|-------------------|---------------------------|
| Tiny (9M)  | 0.01        | 100 steps         | 75%                       |
| Small (28M)| 0.005       | 200 steps         | 55%                       |
| Base (108M)| 0.0025      | 400 steps         | 40%                       |
| Large (333M)| 0.001      | 800 steps         | 35%                       |

## Expected Improvements

1. **Stable Training**: Larger models should maintain accuracy longer during pruning
2. **Gradual Degradation**: More controlled accuracy decline instead of sudden collapse
3. **Better Exploration**: SPT methods will explore weights more appropriately for each model size
4. **Consistent Performance**: All model sizes should show similar relative performance patterns

## Testing Recommendations

1. **Compare pruning curves** across model sizes
2. **Monitor accuracy stability** during early pruning phases
3. **Check exploration patterns** in SPT methods
4. **Verify movement pruning frequency** doesn't cause instability

## Usage

The fixes are automatically applied based on the `model_size` parameter:

```python
# Automatically uses appropriate parameters for each model size
config = GatedBertConfig(model_size='base', enable_pruning=True, pruning_method='spt')
```

The adaptive parameters are logged during training for verification. 