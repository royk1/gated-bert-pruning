"""
BERT-like model with custom gated dense layers for GLUE tasks.
Gates are initially set to 1 (no pruning) - foundation for future pruning methods.
"""
import os

# Suppress TensorFlow warnings and MPS errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['TF_DISABLE_MPS'] = '0'  # Keep MPS enabled
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import tensorflow as tf
import logging

# Additional TensorFlow configuration to avoid MPS issues
tf.get_logger().setLevel('ERROR')

# Configure MPS properly
try:
    # Disable problematic graph optimizations that cause MPS errors
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({
        'disable_meta_optimizer': True,
        'disable_model_pruning': True,
        'arithmetic_optimization': False
    })
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus, 'GPU')
        print(f"Using GPU: {gpus}")
except RuntimeError as e:
    print(f"GPU setup error: {e}")

# Disable problematic optimizations
tf.config.optimizer.set_jit(False)

from tensorflow.keras.layers import Input, Embedding, LayerNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from datasets import load_dataset
from transformers import BertTokenizer

import numpy as np
import os
import sys
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
try:
    import transformers
    transformers.logging.set_verbosity_error()
except (ImportError, AttributeError):
    pass

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting disabled")

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# --- Configuration ---
class GatedBertConfig:
    """Configuration for BERT model with gated dense layers."""
    
    def __init__(
        self,
        # Task configuration
        task: str = 'sst2',  # sst2, mnli, qnli, rte, wnli, mrpc, cola, stsb
        
        # Model hyperparameters
        vocab_size: int = 30522,
        max_len: int = 128,
        num_layers: int = 2,
        d_model: int = 256,
        num_heads: int = 4,
        dff: int = 512,
        dropout_rate: float = 0.1,
        
        # Training parameters
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        
        # Operational parameters
        verbose: int = 1,
        checkpoint_dir: str = 'checkpoints_gated_bert',
        log_file: str = 'gated_bert.log',
        save_freq: int = 2,  # Save every N epochs
        continue_run: bool = False,
        
        # Gate configuration (for future pruning)
        gate_init_value: float = 1.0,  # Initially all gates are open
        gate_trainable: bool = False,  # Gates are not trainable yet
    ):
        self.task = task
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file
        self.save_freq = save_freq
        self.continue_run = continue_run
        self.gate_init_value = gate_init_value
        self.gate_trainable = gate_trainable
        
        # Task-specific configurations
        self._setup_task_config()
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Validation
        self._validate()
    
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
                'num_classes': 1,  # Regression task
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
    
    def to_dict(self):
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
        return f"GatedBertConfig(task={self.task}, epochs={self.epochs}, batch_size={self.batch_size})"


# --- Custom Gated Dense Layer ---
class GatedDense(tf.keras.layers.Layer):
    """Dense layer with gates for future pruning."""
    
    def __init__(self, units, activation=None, name=None, role: str = None, config: GatedBertConfig = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.role = role or 'Unknown'
        self.config = config
        
        if self.config is None:
            raise ValueError("GatedBertConfig must be provided to GatedDense layer")
    
    def build(self, input_shape):
        # Weight matrix
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Bias vector
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # Gates (initially all 1s, not trainable yet)
        self.gates = self.add_weight(
            name='gates',
            shape=self.kernel.shape,
            initializer=tf.keras.initializers.Constant(self.config.gate_init_value),
            trainable=self.config.gate_trainable
        )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Apply gates to weights
        gated_kernel = self.kernel * self.gates
        
        # Standard dense computation
        output = tf.matmul(inputs, gated_kernel) + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def get_gate_stats(self):
        """Get statistics about the gates."""
        gates_np = self.gates.numpy()
        total_gates = gates_np.size
        active_gates = np.sum(gates_np > 0.5)
        sparsity = 1.0 - (active_gates / total_gates)
        
        return {
            'total_gates': total_gates,
            'active_gates': active_gates,
            'sparsity': sparsity,
            'sparsity_percent': sparsity * 100
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'role': self.role,
            # Store only primitive values, not the entire config object
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Remove non-constructor arguments
        activation = tf.keras.activations.deserialize(config.pop('activation'))
        role = config.pop('role', None)
        gate_init_value = config.pop('gate_init_value', 1.0)
        gate_trainable = config.pop('gate_trainable', False)
        
        # Create a minimal config for the layer
        from types import SimpleNamespace
        layer_config = SimpleNamespace()
        layer_config.gate_init_value = gate_init_value
        layer_config.gate_trainable = gate_trainable
        
        # Create the layer
        return cls(config.pop('units'), activation=activation, role=role, config=layer_config, **config)


class GatedMultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention with gated dense layers."""
    
    def __init__(self, d_model: int, num_heads: int, config: GatedBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.config = config
        self.depth = d_model // num_heads
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        layer_name = kwargs.get('name', 'gated_mha')
        
        # Query, Key, Value projections
        self.wq = GatedDense(
            d_model, 
            name=f'{layer_name}_wq',
            role='Query',
            config=config
        )
        self.wk = GatedDense(
            d_model,
            name=f'{layer_name}_wk', 
            role='Key',
            config=config
        )
        self.wv = GatedDense(
            d_model,
            name=f'{layer_name}_wv',
            role='Value', 
            config=config
        )
        
        # Output projection
        self.dense = GatedDense(
            d_model,
            name=f'{layer_name}_output',
            role='Attention Output',
            config=config
        )
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Linear transformations
        q = self.wq(inputs, training=training)
        k = self.wk(inputs, training=training)
        v = self.wv(inputs, training=training)
        
        # Split into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        # Final linear transformation
        output = self.dense(concat_attention, training=training)
        
        return output
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate attention weights and apply to values."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, v)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            # Store only primitive values
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        d_model = config.pop('d_model')
        num_heads = config.pop('num_heads')
        gate_init_value = config.pop('gate_init_value', 1.0)
        gate_trainable = config.pop('gate_trainable', False)
        
        # Create a minimal config for the layer
        from types import SimpleNamespace
        layer_config = SimpleNamespace()
        layer_config.gate_init_value = gate_init_value
        layer_config.gate_trainable = gate_trainable
        
        return cls(d_model, num_heads, layer_config, **config)


class GatedTransformerBlock(tf.keras.layers.Layer):
    """Transformer block with gated dense layers."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float, config: GatedBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.config = config
        
        layer_name = kwargs.get('name', 'gated_transformer')
        
        # Multi-head attention
        self.mha = GatedMultiHeadAttention(
            d_model, 
            num_heads, 
            config,
            name=f'{layer_name}_mha'
        )
        
        # Feed-forward network
        self.ffn_layer1 = GatedDense(
            dff,
            activation='relu',
            name=f'{layer_name}_ffn1',
            role='FFN Hidden',
            config=config
        )
        self.ffn_layer2 = GatedDense(
            d_model,
            name=f'{layer_name}_ffn2',
            role='FFN Output',
            config=config
        )
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name=f'{layer_name}_ln1')
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name=f'{layer_name}_ln2')
        
        # Dropout
        self.dropout1 = Dropout(dropout_rate, name=f'{layer_name}_dropout1')
        self.dropout2 = Dropout(dropout_rate, name=f'{layer_name}_dropout2')
    
    def call(self, inputs, mask=None, training=None):
        # Multi-head attention
        attn_output = self.mha(inputs, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn_layer1(out1, training=training)
        ffn_output = self.ffn_layer2(ffn_output, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            # Store only primitive values
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        d_model = config.pop('d_model')
        num_heads = config.pop('num_heads')
        dff = config.pop('dff')
        dropout_rate = config.pop('dropout_rate')
        gate_init_value = config.pop('gate_init_value', 1.0)
        gate_trainable = config.pop('gate_trainable', False)
        
        # Create a minimal config for the layer
        from types import SimpleNamespace
        layer_config = SimpleNamespace()
        layer_config.gate_init_value = gate_init_value
        layer_config.gate_trainable = gate_trainable
        
        return cls(d_model, num_heads, dff, dropout_rate, layer_config, **config)

class PositionalEmbedding(tf.keras.layers.Layer):
    """Positional embedding layer."""
    
    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
    def build(self, input_shape):
        self.pos_embedding = Embedding(
            input_dim=self.max_len,
            output_dim=self.d_model,
            name='pos_embedding'
        )
        super().build(input_shape)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return self.pos_embedding(positions)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model,
        })
        return config


# --- Main Model ---
def create_gated_bert_model(config: GatedBertConfig) -> tf.keras.Model:
    """Create a BERT model with gated dense layers."""
    
    # Input layers
    input_ids = Input(shape=(config.max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(config.max_len,), dtype=tf.int32, name='attention_mask')
    
    # Create attention mask for multi-head attention
    mask_expanded = Lambda(
        lambda x: tf.cast(tf.reshape(x, (-1, 1, 1, config.max_len)), tf.float32) * -1e9,
        name='mask_expansion'
    )(attention_mask)
    
    # Token embeddings
    token_emb = Embedding(
        config.vocab_size,
        config.d_model,
        mask_zero=False,
        name='token_embedding'
    )(input_ids)
    
    # Position embeddings
    pos_emb = PositionalEmbedding(
        config.max_len,
        config.d_model,
        name='position_embedding'
    )(input_ids)
    
    # Combine embeddings
    embeddings = token_emb + pos_emb
    embeddings = LayerNormalization(epsilon=1e-6, name='embedding_layernorm')(embeddings)
    embeddings = Dropout(config.dropout_rate, name='embedding_dropout')(embeddings)
    
    # Transformer blocks
    hidden_states = embeddings
    for i in range(config.num_layers):
        hidden_states = GatedTransformerBlock(
            config.d_model,
            config.num_heads,
            config.dff,
            config.dropout_rate,
            config,
            name=f'transformer_block_{i}'
        )(hidden_states, mask=mask_expanded)
    
    # Classification head
    # Use CLS token (first token) for classification
    cls_token = Lambda(lambda x: x[:, 0, :], name='cls_token')(hidden_states)
    cls_token = Dropout(config.dropout_rate, name='cls_dropout')(cls_token)
    
    # Final classifier
    if config.is_regression:
        # For regression tasks like STS-B
        logits = GatedDense(
            1,
            name='regressor',
            role='Regressor',
            config=config
        )(cls_token)
    else:
        # For classification tasks
        logits = GatedDense(
            config.num_classes,
            name='classifier',
            role='Classifier',
            config=config
        )(cls_token)
    
    # Create model
    model = Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name=f'GatedBERT_{config.task}'
    )
    
    return model


# --- Callbacks ---
class GatedBertCallback(tf.keras.callbacks.Callback):
    """Custom callback for monitoring gated BERT training."""
    
    def __init__(self, config: GatedBertConfig):
        super().__init__()
        self.config = config
        self.epoch_start_time = None
        self.training_history = defaultdict(list)
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        if self.config.verbose >= 1:
            print(f"\n=== Epoch {epoch + 1}/{self.config.epochs} ===")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Record training history
        self.training_history['epoch'].append(epoch + 1)
        self.training_history['loss'].append(logs.get('loss'))
        self.training_history['accuracy'].append(logs.get('accuracy'))
        self.training_history['val_loss'].append(logs.get('val_loss'))
        self.training_history['val_accuracy'].append(logs.get('val_accuracy'))
        
        # Calculate epoch time
        epoch_time = time.time() - self.epoch_start_time
        self.training_history['epoch_time'].append(epoch_time)
        
        if self.config.verbose >= 1:
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            
            # Print metrics
            if 'loss' in logs:
                print(f"  Train Loss: {logs['loss']:.4f}")
            if 'accuracy' in logs:
                print(f"  Train Accuracy: {logs['accuracy']:.4f}")
            if 'val_loss' in logs:
                print(f"  Val Loss: {logs['val_loss']:.4f}")
            if 'val_accuracy' in logs:
                print(f"  Val Accuracy: {logs['val_accuracy']:.4f}")
            
            # Print gate statistics
            self._print_gate_stats()
    
    def _print_gate_stats(self):
        """Print statistics about gates in the model."""
        if self.config.verbose >= 2:
            print("\n  Gate Statistics:")
            
            total_gates = 0
            total_active = 0
            
            for layer in self.model.layers:
                if isinstance(layer, GatedDense):
                    stats = layer.get_gate_stats()
                    total_gates += stats['total_gates']
                    total_active += stats['active_gates']
                    
                    print(f"    {layer.name} ({layer.role}): "
                          f"{stats['active_gates']}/{stats['total_gates']} "
                          f"({stats['sparsity_percent']:.1f}% sparse)")
            
            if total_gates > 0:
                overall_sparsity = 1.0 - (total_active / total_gates)
                print(f"    Overall: {total_active}/{total_gates} "
                      f"({overall_sparsity * 100:.1f}% sparse)")
    
    def on_train_end(self, logs=None):
        if self.config.verbose >= 1:
            print(f"\n=== Training Completed ===")
            self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final training summary."""
        if not self.training_history['epoch']:
            return
        
        print("\nTraining Summary:")
        print("-" * 60)
        
        # Best metrics
        if self.training_history['val_accuracy'] and any(x is not None for x in self.training_history['val_accuracy']):
            val_accs = [x for x in self.training_history['val_accuracy'] if x is not None]
            if val_accs:
                best_val_acc = max(val_accs)
                best_epoch = self.training_history['val_accuracy'].index(best_val_acc) + 1
                print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
        
        if self.training_history['val_loss'] and any(x is not None for x in self.training_history['val_loss']):
            val_losses = [x for x in self.training_history['val_loss'] if x is not None]
            if val_losses:
                best_val_loss = min(val_losses)
                best_epoch = self.training_history['val_loss'].index(best_val_loss) + 1
                print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
        
        # Training time
        if self.training_history['epoch_time']:
            total_time = sum(self.training_history['epoch_time'])
            avg_time = total_time / len(self.training_history['epoch_time'])
            print(f"Total Training Time: {total_time:.2f}s")
            print(f"Average Time per Epoch: {avg_time:.2f}s")
        
        print("-" * 60)


def save_training_history(history_dict: dict, filepath: str):
    """Save training history to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"Training history saved to {filepath}")
    except Exception as e:
        print(f"Failed to save training history: {e}")


def plot_training_history(history_dict: dict, save_path: str):
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
        
        # Leave the last subplot empty or add model summary
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {save_path}")
        
    except Exception as e:
        print(f"Failed to create plots: {e}")


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


def load_and_prepare_data(config: GatedBertConfig):
    """Load and prepare dataset."""
    logging.info(f"Loading dataset: {config.dataset_name}/{config.dataset_subset}")
    
    # Load dataset
    dataset = load_dataset(config.dataset_name, config.dataset_subset)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenization function
    def tokenize_function(examples):
        if config.text_key2:
            # Sentence pair tasks
            return tokenizer(
                examples[config.text_key1],
                examples[config.text_key2],
                truncation=True,
                padding='max_length',
                max_length=config.max_len,
                return_tensors='tf'
            )
        else:
            # Single sentence tasks
            return tokenizer(
                examples[config.text_key1],
                truncation=True,
                padding='max_length',
                max_length=config.max_len,
                return_tensors='tf'
            )
    
    # Apply tokenization
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing dataset"
    )
    
    # Convert to tensorflow datasets
    train_dataset = tokenized_datasets['train'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['label'],
        shuffle=True,
        batch_size=config.batch_size
    )
    
    val_dataset = tokenized_datasets[config.val_key].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['label'],
        shuffle=False,
        batch_size=config.batch_size
    )
    
    # Print dataset info
    train_size = len(tokenized_datasets['train'])
    val_size = len(tokenized_datasets[config.val_key])
    
    logging.info(f"Dataset loaded successfully:")
    logging.info(f"  Train samples: {train_size}")
    logging.info(f"  Validation samples: {val_size}")
    logging.info(f"  Text keys: {config.text_key1}" + (f", {config.text_key2}" if config.text_key2 else ""))
    logging.info(f"  Number of classes: {config.num_classes}")
    
    return train_dataset, val_dataset


def main():
    """Main training function."""
    
    # Configuration
    config = GatedBertConfig(
        task='mnli',  # Change this to test different tasks
        epochs=10,
        batch_size=32,
        learning_rate=2e-5,
        verbose=2,
        checkpoint_dir='checkpoints_gated_bert_mnli',
        log_file='gated_bert_mnli.log',
        continue_run=False
    )
    
    # Check for benchmark mode
    benchmark_epochs = os.getenv('BENCHMARK_EPOCHS')
    if benchmark_epochs:
        try:
            config.epochs = int(benchmark_epochs)
            logging.info(f"Running in benchmark mode for {config.epochs} epochs")
        except ValueError:
            logging.error(f"Invalid BENCHMARK_EPOCHS value: {benchmark_epochs}")
    
    # Setup logging
    setup_logging(config)
    
    # Load and prepare data
    train_dataset, val_dataset = load_and_prepare_data(config)
    
    # Create model
    logging.info("Creating model...")
    model = create_gated_bert_model(config)
    
    # Compile model
    if config.is_regression:
        # For regression tasks (STS-B)
        model.compile(
            optimizer=Adam(learning_rate=config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    else:
        # For classification tasks
        model.compile(
            optimizer=Adam(learning_rate=config.learning_rate),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy(name='accuracy')]
        )
    
    # Model summary
    model.summary(print_fn=logging.info)
    
    # Count gated layers
    gated_layers = [layer for layer in model.layers if isinstance(layer, GatedDense)]
    logging.info(f"Model has {len(gated_layers)} gated dense layers")
    
    # Callbacks
    callbacks = [
        GatedBertCallback(config),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.checkpoint_dir, 'best_model_weights.h5'),
            monitor='val_accuracy' if not config.is_regression else 'val_mae',
            mode='max' if not config.is_regression else 'min',
            save_best_only=True,
            save_weights_only=True,  # <-- Change this to True
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy' if not config.is_regression else 'val_mae',
            mode='max' if not config.is_regression else 'min',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
        
    # Train model
    logging.info("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1 if config.verbose >= 1 else 0
    )
    
    # Save training history
    if callbacks[0].training_history:
        history_path = os.path.join(config.checkpoint_dir, 'training_history.json')
        save_training_history(dict(callbacks[0].training_history), history_path)
        
        # Create plots
        if HAS_MATPLOTLIB:
            plot_path = os.path.join(config.checkpoint_dir, 'training_plots.png')
            plot_training_history(dict(callbacks[0].training_history), plot_path)
    
    # Save configuration
    config_path = os.path.join(config.checkpoint_dir, 'config.json')
    config.save(config_path)
    
    # Final evaluation
    logging.info("Final evaluation...")
    final_results = model.evaluate(val_dataset, verbose=0)
    
    if config.is_regression:
        logging.info(f"Final validation MSE: {final_results[0]:.4f}")
        logging.info(f"Final validation MAE: {final_results[1]:.4f}")
    else:
        logging.info(f"Final validation loss: {final_results[0]:.4f}")
        logging.info(f"Final validation accuracy: {final_results[1]:.4f}")
    
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()