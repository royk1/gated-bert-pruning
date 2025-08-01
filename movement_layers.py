"""Movement pruning layers for Gated BERT."""
import tensorflow as tf
from keras.layers import Layer
import numpy as np
from typing import Optional
from model_layers import GatedDense, GatedTransformerBlock
from config import GatedBertConfig


class MovementGatedDense(GatedDense):
    """GatedDense layer with movement pruning capabilities."""
    
    def __init__(self, units, activation=None, name=None, role: str = None, config: GatedBertConfig = None, **kwargs):
        super().__init__(units, activation, name, role, config, **kwargs)
        self.movement_scores = None
        self.update_count = 0
        
    def build(self, input_shape):
        super().build(input_shape)
        
        # Initialize movement scores (will be updated during training)
        self.movement_scores = self.add_weight(
            name='movement_scores',
            shape=self.kernel.shape,
            initializer='zeros',
            trainable=False
        )
        
    def update_movement_scores(self, gradients):
        """Update movement scores based on weight * gradient with momentum."""
        if gradients is not None:
            # Movement score = -weight * gradient (negative because we want to prune weights that hurt performance)
            current_movement = -self.kernel * gradients
            
            # Apply exponential moving average for stability
            momentum = 0.9
            if self.update_count == 0:
                self.movement_scores.assign(current_movement)
            else:
                self.movement_scores.assign(
                    momentum * self.movement_scores + (1 - momentum) * current_movement
                )
            
            self.update_count += 1
            
            # Debug: Print occasionally with more detail
            if self.update_count % 100 == 0:  # Less frequent
                score_mean = tf.reduce_mean(self.movement_scores)
                score_std = tf.math.reduce_std(self.movement_scores)
                score_min = tf.reduce_min(self.movement_scores)
                score_max = tf.reduce_max(self.movement_scores)
                print(f"DEBUG {self.name}: Movement scores updated {self.update_count} times")
                print(f"  Mean: {score_mean:.6f}, Std: {score_std:.6f}")
                print(f"  Range: [{score_min:.6f}, {score_max:.6f}]")
    
    def update_gates_from_movement(self, target_sparsity):
        """Update gates based on movement scores and target sparsity."""
        if self.movement_scores is None:
            print(f"DEBUG {self.name}: No movement scores available, using random pruning")
            self._random_prune(target_sparsity)
            return
        
        # Get current gates and movement scores
        current_gates = self.gates.numpy()
        scores = self.movement_scores.numpy()
        
        # Only consider currently active weights
        active_mask = current_gates > 0.5
        if not np.any(active_mask):
            print(f"DEBUG {self.name}: No active weights to prune")
            return
        
        # Calculate sparsity only among active weights
        total_active = np.sum(active_mask)
        target_active = int(total_active * (1.0 - target_sparsity))
        
        if target_active <= 0:
            # Prune all active weights
            new_gates = np.zeros_like(current_gates)
            print(f"DEBUG {self.name}: Pruning all {total_active} active weights")
        elif target_active >= total_active:
            # Keep all active weights
            new_gates = current_gates.copy()
            print(f"DEBUG {self.name}: Keeping all {total_active} active weights")
        else:
            # Prune based on movement scores among active weights
            active_scores = scores[active_mask]
            
            # Find threshold among active weights
            threshold = np.partition(active_scores, -target_active)[-target_active]
            
            # Create new gates
            new_gates = current_gates.copy()
            
            # Set gates to 0 for weights with scores below threshold (among active weights)
            prune_mask = active_mask & (scores < threshold)
            new_gates[prune_mask] = 0.0
            
            # Handle ties at threshold
            tie_mask = active_mask & (scores == threshold)
            num_ties = np.sum(tie_mask)
            num_kept_above = np.sum(active_mask & (scores > threshold))
            
            if num_kept_above < target_active and num_ties > 0:
                remaining_to_keep = target_active - num_kept_above
                tie_indices = np.where(tie_mask)
                
                if remaining_to_keep < num_ties:
                    # Randomly select which tied weights to keep
                    keep_indices = np.random.choice(
                        len(tie_indices[0]), 
                        size=remaining_to_keep, 
                        replace=False
                    )
                    # Prune all tied weights first
                    new_gates[tie_mask] = 0.0
                    # Then keep selected ones
                    for i in keep_indices:
                        new_gates[tie_indices[0][i], tie_indices[1][i]] = 1.0
        
        # Update gates and log changes
        old_active = np.sum(current_gates)
        new_active = np.sum(new_gates)
        
        print(f"DEBUG {self.name}: Active weights: {old_active} -> {new_active} "
              f"(pruned {old_active - new_active})")
        
        self.gates.assign(new_gates)

    def _random_prune(self, target_sparsity):
        """Fallback random pruning when movement scores unavailable."""
        current_gates = self.gates.numpy()
        active_mask = current_gates > 0.5
        total_active = np.sum(active_mask)
        
        num_to_prune = int(total_active * target_sparsity)
        
        if num_to_prune > 0:
            active_indices = np.where(active_mask)
            prune_selection = np.random.choice(
                len(active_indices[0]), 
                size=min(num_to_prune, len(active_indices[0])), 
                replace=False
            )
            
            new_gates = current_gates.copy()
            for i in prune_selection:
                new_gates[active_indices[0][i], active_indices[1][i]] = 0.0
            
            self.gates.assign(new_gates)
            print(f"DEBUG {self.name}: Random pruning: {num_to_prune} weights")

    def get_config(self):
        """Get configuration for layer serialization."""
        config = super().get_config()
        config.update({
            'update_count': self.update_count
        })
        return config

    @classmethod  
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)


class MovementGatedMultiHeadAttention(Layer):
    """Multi-head attention with movement-pruned gated dense layers."""
    
    def __init__(self, d_model: int, num_heads: int, config: GatedBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.config = config
        self.depth = d_model // num_heads
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        layer_name = kwargs.get('name', 'movement_mha')
        
        # Use MovementGatedDense instead of GatedDense
        self.wq = MovementGatedDense(d_model, name=f'{layer_name}_wq', role='Query', config=config)
        self.wk = MovementGatedDense(d_model, name=f'{layer_name}_wk', role='Key', config=config)
        self.wv = MovementGatedDense(d_model, name=f'{layer_name}_wv', role='Value', config=config)
        self.dense = MovementGatedDense(d_model, name=f'{layer_name}_output', role='Attention Output', config=config)
    
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


class MovementGatedTransformerBlock(Layer):
    """Transformer block with movement-pruned gated dense layers."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float, config: GatedBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.config = config
        
        layer_name = kwargs.get('name', 'movement_transformer')
        
        # Multi-head attention with movement pruning
        self.mha = MovementGatedMultiHeadAttention(d_model, num_heads, config, name=f'{layer_name}_mha')
        
        # Feed-forward network with movement pruning
        self.ffn_layer1 = MovementGatedDense(dff, activation='relu', name=f'{layer_name}_ffn1', role='FFN Hidden', config=config)
        self.ffn_layer2 = MovementGatedDense(d_model, name=f'{layer_name}_ffn2', role='FFN Output', config=config)
        
        # Layer normalization and dropout (unchanged)
        from tensorflow.keras.layers import LayerNormalization, Dropout
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name=f'{layer_name}_ln1')
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name=f'{layer_name}_ln2')
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
        """Get configuration for layer serialization."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else None
        })
        return config

    @classmethod  
    def from_config(cls, config):
        """Create layer from configuration."""
        gated_bert_config = config.pop('config', None)
        if gated_bert_config and isinstance(gated_bert_config, dict):
            from config import GatedBertConfig
            gated_bert_config = GatedBertConfig(**gated_bert_config)
        return cls(config=gated_bert_config, **config)