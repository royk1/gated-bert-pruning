"""Configuration classes for Gated BERT model."""
import json
import os
from typing import Dict, Any, Optional


class GatedBertConfig:
    """Configuration for Gated BERT and pruning methods."""
    
    # BART size presets (similar to HuggingFace BART variants)
    BART_PRESETS = {
        'tiny': {
            'num_layers': 2,
            'd_model': 256,
            'num_heads': 4,
            'dff': 512,
            'vocab_size': 30522,
            'max_len': 128,
            'estimated_params': '~9M'
        },
        'small': {
            'num_layers': 6,
            'd_model': 512,
            'num_heads': 8,
            'dff': 1024,
            'vocab_size': 30522,
            'max_len': 128,
            'estimated_params': '~28M'
        },
        'base': {
            'num_layers': 12,
            'd_model': 768,
            'num_heads': 12,
            'dff': 3072,
            'vocab_size': 30522,
            'max_len': 128,
            'estimated_params': '~108M'
        },
        'large': {
            'num_layers': 24,
            'd_model': 1024,
            'num_heads': 16,
            'dff': 4096,
            'vocab_size': 30522,
            'max_len': 128,
            'estimated_params': '~333M'
        }
    }
    
    def __init__(self, **kwargs):
        # Model size configuration
        self.model_size = kwargs.get('model_size', 'base')  # 'tiny', 'small', 'base', 'large'
        
        # Apply BART preset if specified
        if self.model_size in self.BART_PRESETS:
            preset = self.BART_PRESETS[self.model_size]
            # Override with preset values unless explicitly provided
            self.num_layers = kwargs.get('num_layers', preset['num_layers'])
            self.d_model = kwargs.get('d_model', preset['d_model'])
            self.num_heads = kwargs.get('num_heads', preset['num_heads'])
            self.dff = kwargs.get('dff', preset['dff'])
            self.vocab_size = kwargs.get('vocab_size', preset['vocab_size'])
            self.max_len = kwargs.get('max_len', preset['max_len'])
        else:
            # Fallback to original defaults if model_size is not recognized
            self.num_layers = kwargs.get('num_layers', 2)
            self.d_model = kwargs.get('d_model', 256)
            self.num_heads = kwargs.get('num_heads', 4)
            self.dff = kwargs.get('dff', 512)
            self.vocab_size = kwargs.get('vocab_size', 30522)
            self.max_len = kwargs.get('max_len', 128)
        
        # Task configuration
        self.task = kwargs.get('task', 'sst2')  # sst2, mnli, qnli, rte, wnli, mrpc, cola, stsb
        
        # Model hyperparameters (can override preset values)
        self.dropout_rate = kwargs.get('dropout_rate', 0.0)
        
        # Training parameters
        self.epochs = kwargs.get('epochs', 28)  # Quick test
        self.batch_size = kwargs.get('batch_size', 96)
        self.learning_rate = kwargs.get('learning_rate', 1e-5)
        
        # Operational parameters
        self.verbose = kwargs.get('verbose', 1)
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints_gated_bert')
        self.log_file = kwargs.get('log_file', 'gated_bert.log')
        self.save_freq = kwargs.get('save_freq', 2)
        self.continue_run = kwargs.get('continue_run', False)
        
        # Gate configuration (for future pruning)
        self.gate_init_value = kwargs.get('gate_init_value', 1.0)
        self.gate_trainable = kwargs.get('gate_trainable', False)
        
        # Pruning configuration
        self.enable_pruning = kwargs.get('enable_pruning', True)
        self.pruning_method = kwargs.get('pruning_method', 'movement')  # 'magnitude', 'movement', 'spt', or 'none'
        self.start_sparsity = kwargs.get('start_sparsity', 0.0)
        self.end_sparsity = kwargs.get('end_sparsity', 0.99)
        self.pruning_start_epoch = kwargs.get('pruning_start_epoch', 1)  # Start at epoch 2 (1-indexed)
        #pruning_end_epoch: Optional[int] = None,   # If None, use total epochs
        self.pruning_end_epoch = kwargs.get('pruning_end_epoch', 23)                # Reach target sparsity by epoch 12
        
        # Step-based pruning schedule (overrides linear schedule if provided)
        # List of target sparsity percentages for each epoch after pruning_start_epoch
        # Scale aggressiveness based on model size
        if self.model_size == 'tiny':
            # Aggressive schedule for tiny model (can handle it)
            self.sparsity_steps = kwargs.get('sparsity_steps', [40,60,80,85,90,91,92,93,94,95,96,97,98,99])
        elif self.model_size == 'small':
            # Moderate schedule for small model
            self.sparsity_steps = kwargs.get('sparsity_steps', [40,60,80,85,90,91,92,93,94,95,96,97,98,99])
        elif self.model_size == 'base':
            # Conservative schedule for base model
            self.sparsity_steps = kwargs.get('sparsity_steps', [40,60,80,85,90,91,92,93,94,95,96,97,98,99])
        elif self.model_size == 'large':
            # Very conservative schedule for large model
            self.sparsity_steps = kwargs.get('sparsity_steps', [40,60,80,85,90,91,92,93,94,95,96,97,98,99])
        else:
            # Default to conservative schedule
            self.sparsity_steps = kwargs.get('sparsity_steps', [40,60,80,85,90,91,92,93,94,95,96,97,98,99])
            
        # SPT pruning specific parameters (per-weight SPT only)
        # Scale epsilon inversely with model size to prevent over-exploration
        base_epsilon = 0.01  # Base epsilon for tiny model
        if self.model_size == 'tiny':
            self.spt_epsilon = kwargs.get('spt_epsilon', base_epsilon)
        elif self.model_size == 'small':
            self.spt_epsilon = kwargs.get('spt_epsilon', base_epsilon)  # Half exploration
        elif self.model_size == 'base':
            self.spt_epsilon = kwargs.get('spt_epsilon', base_epsilon)  # Quarter exploration
        elif self.model_size == 'large':
            self.spt_epsilon = kwargs.get('spt_epsilon', base_epsilon)  # Tenth exploration
        else:
            self.spt_epsilon = kwargs.get('spt_epsilon', base_epsilon)
            
        self.spt_reward_alpha = kwargs.get('spt_reward_alpha', 0.125)  # EMA smoothing for reward
        
        # SPT per-layer exploration (cyclic exploration of layer types)
        # If True, each batch explores only one layer type in cyclic manner
        # This creates competition between layer types and allows more sensitive layers to be pruned less
        self.spt_explore_per_layer = kwargs.get('spt_explore_per_layer', False)

        # Movement pruning specific parameters - scale frequency with model size
        base_frequency = 100  # Base frequency for tiny model
        if self.model_size == 'tiny':
            self.movement_pruning_frequency_steps = kwargs.get('movement_pruning_frequency_steps', base_frequency)
        elif self.model_size == 'small':
            self.movement_pruning_frequency_steps = kwargs.get('movement_pruning_frequency_steps', base_frequency)
        elif self.model_size == 'base':
            self.movement_pruning_frequency_steps = kwargs.get('movement_pruning_frequency_steps', base_frequency)
        elif self.model_size == 'large':
            self.movement_pruning_frequency_steps = kwargs.get('movement_pruning_frequency_steps', base_frequency)
        else:
            self.movement_pruning_frequency_steps = kwargs.get('movement_pruning_frequency_steps', base_frequency)
            
        self.movement_schedule = kwargs.get('movement_schedule', 'linear')  # 'linear' or 'cubic'
        
        # Restore original behavior: set up task config, create directories, validate
        self._setup_task_config()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._validate()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameter count estimate."""
        # Calculate approximate parameter count
        # Embedding layers
        embedding_params = self.vocab_size * self.d_model
        
        # Transformer layers
        # Self-attention: query, key, value projections + output projection
        attention_params_per_layer = 4 * self.d_model * self.d_model
        
        # Feed-forward network
        ffn_params_per_layer = 2 * self.d_model * self.dff
        
        # Layer normalization parameters (2 per layer: attention + ffn)
        ln_params_per_layer = 2 * self.d_model * 2  # mean and variance parameters
        
        # Total per transformer layer
        layer_params = attention_params_per_layer + ffn_params_per_layer + ln_params_per_layer
        
        # All transformer layers
        total_transformer_params = layer_params * self.num_layers
        
        # Classification head
        classifier_params = self.d_model * self.num_classes
        
        # Total parameters
        total_params = embedding_params + total_transformer_params + classifier_params
        
        return {
            'model_size': self.model_size,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'estimated_params': f"{total_params:,}",
            'estimated_params_millions': f"{total_params / 1_000_000:.1f}M"
        }
  
    def _setup_task_config(self):
        """Setup task-specific configurations."""
        task_configs = {
            'sst2': {
                'dataset_name': 'glue',
                'dataset_subset': 'sst2',
                'text_key1': 'sentence',
                'text_key2': None,
                'num_classes': 2,
                'val_key': 'validation'
            },
            'mnli': {
                'dataset_name': 'glue',
                'dataset_subset': 'mnli',
                'text_key1': 'premise',
                'text_key2': 'hypothesis',
                'num_classes': 3,
                'val_key': 'validation_matched'
            },
            'qnli': {
                'dataset_name': 'glue',
                'dataset_subset': 'qnli',
                'text_key1': 'question',
                'text_key2': 'sentence',
                'num_classes': 2,
                'val_key': 'validation'
            },
            'rte': {
                'dataset_name': 'glue',
                'dataset_subset': 'rte',
                'text_key1': 'sentence1',
                'text_key2': 'sentence2',
                'num_classes': 2,
                'val_key': 'validation'
            },
            'wnli': {
                'dataset_name': 'glue',
                'dataset_subset': 'wnli',
                'text_key1': 'sentence1',
                'text_key2': 'sentence2',
                'num_classes': 2,
                'val_key': 'validation'
            },
            'mrpc': {
                'dataset_name': 'glue',
                'dataset_subset': 'mrpc',
                'text_key1': 'sentence1',
                'text_key2': 'sentence2',
                'num_classes': 2,
                'val_key': 'validation'
            },
            'cola': {
                'dataset_name': 'glue',
                'dataset_subset': 'cola',
                'text_key1': 'sentence',
                'text_key2': None,
                'num_classes': 2,
                'val_key': 'validation'
            },
            'stsb': {
                'dataset_name': 'glue',
                'dataset_subset': 'stsb',
                'text_key1': 'sentence1',
                'text_key2': 'sentence2',
                'num_classes': 1,
                'val_key': 'validation'
            }
        }
        
        if self.task not in task_configs:
            raise ValueError(f"Unsupported task: {self.task}. Supported tasks: {list(task_configs.keys())}")
        
        config = task_configs[self.task]
        self.dataset_name = config['dataset_name']
        self.dataset_subset = config['dataset_subset']
        self.text_key1 = config['text_key1']
        self.text_key2 = config['text_key2']
        self.num_classes = config['num_classes']
        self.val_key = config['val_key']
        self.is_regression = (self.task == 'stsb')
    
    def _validate(self):
        """Validate configuration parameters."""
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert 0 <= self.dropout_rate <= 1, "dropout_rate must be between 0 and 1"
        assert self.model_size in ['tiny', 'small', 'base', 'large'] or self.model_size in self.BART_PRESETS, f"Unsupported model_size: {self.model_size}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath: str):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __str__(self):
        model_info = self.get_model_info()
        return f"GatedBertConfig(model_size={self.model_size}, task={self.task}, epochs={self.epochs}, batch_size={self.batch_size}, params={model_info['estimated_params_millions']})"