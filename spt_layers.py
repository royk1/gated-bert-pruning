"""Structured Pruning with Training (SPT) layers for Gated BERT."""
import tensorflow as tf
import keras
from keras.layers import Layer
import numpy as np
from typing import Optional
from config import GatedBertConfig


class GatedDense(Layer):
    """Dense layer with gates for future pruning (standalone, identical to original GatedDense)."""
    def __init__(self, units, activation=None, name=None, role: str = None, config: GatedBertConfig = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        from keras import activations
        self.activation = activations.get(activation)
        self.role = role or 'Unknown'
        self.config = config
        if self.config is None:
            raise ValueError("GatedBertConfig must be provided to GatedDense layer")

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        from keras import initializers
        self.gates = self.add_weight(
            name='gates',
            shape=self.kernel.shape,
            initializer=initializers.Constant(self.config.gate_init_value),
            trainable=self.config.gate_trainable
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        gated_kernel = self.kernel * self.gates
        output = tf.matmul(inputs, gated_kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_gate_stats(self):
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
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config

    @classmethod
    def from_config(cls, config):
        activation = tf.keras.activations.deserialize(config.pop('activation'))
        role = config.pop('role', None)
        gate_init_value = config.pop('gate_init_value', 1.0)
        gate_trainable = config.pop('gate_trainable', False)
        from types import SimpleNamespace
        layer_config = SimpleNamespace()
        layer_config.gate_init_value = gate_init_value
        layer_config.gate_trainable = gate_trainable
        return cls(config.pop('units'), activation=activation, role=role, config=layer_config, **config)

    #def sample_and_prune_epsilon_weights(self, epsilon):
    #    import tensorflow as tf
    #    return tf.zeros((0, 2), dtype=tf.int64), tf.constant([])

    #def restore_gates(self, indices, old_values):
    #    return

    #def permanently_prune_indices(self, indices):
    #    return

    #def update_reward(self, indices, reward):
    #    return


class SptGatedDense(Layer):
    """Dense layer with gates for future pruning (SPT version)."""
    def __init__(self, units, activation=None, name=None, role: str = None, config: GatedBertConfig = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        from keras import activations
        self.activation = activations.get(activation)
        self.role = role or 'Unknown'
        self.config = config
        if self.config is None:
            raise ValueError("GatedBertConfig must be provided to SptGatedDense layer")
        self._reward_buffers_initialized = False
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        from keras import initializers
        self.gates = self.add_weight(
            name='gates',
            shape=self.kernel.shape,
            initializer=initializers.Constant(self.config.gate_init_value),
            trainable=self.config.gate_trainable
        )
        # Add reward buffer variables as tf.Variable (not Keras weights)
        if not self._reward_buffers_initialized:
            # Use callable initializers to avoid tf.function lifting issues
            def reward_buffer_initializer():
                return tf.zeros(self.gates.shape, dtype=tf.float32)
            
            def reward_count_initializer():
                return tf.zeros(self.gates.shape, dtype=tf.float32)
            
            self.reward_buffer = tf.Variable(
                initial_value=reward_buffer_initializer,
                trainable=False,
                name='reward_buffer'
            )
            self.reward_count = tf.Variable(
                initial_value=reward_count_initializer,
                trainable=False,
                name='reward_count'
            )
            self._reward_buffers_initialized = True
        super().build(input_shape)
    def call(self, inputs, training=None):
        gated_kernel = self.kernel * self.gates
        output = tf.matmul(inputs, gated_kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
    def get_gate_stats(self):
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
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config
    @classmethod
    def from_config(cls, config):
        activation = tf.keras.activations.deserialize(config.pop('activation'))
        role = config.pop('role', None)
        gate_init_value = config.pop('gate_init_value', 1.0)
        gate_trainable = config.pop('gate_trainable', False)
        from types import SimpleNamespace
        layer_config = SimpleNamespace()
        layer_config.gate_init_value = gate_init_value
        layer_config.gate_trainable = gate_trainable
        return cls(config.pop('units'), activation=activation, role=role, config=layer_config, **config)

    def sample_and_prune_epsilon_weights(self, epsilon):
        """Sample epsilon fraction of unpruned weights and temporarily prune them."""

        gates = self.gates
        unpruned_mask = tf.cast(gates > 0.5, tf.float32)
        unpruned_indices = tf.where(unpruned_mask > 0)
        num_unpruned = tf.shape(unpruned_indices)[0]
        num_to_sample = tf.cast(tf.math.ceil(epsilon * tf.cast(num_unpruned, tf.float32)), tf.int32)
        
        def no_sample():
            return tf.zeros((0, 2), dtype=tf.int64), tf.constant([])
        def do_sample():
            shuffled = tf.random.shuffle(unpruned_indices)
            selected = shuffled[:num_to_sample]
            
            # Create boolean mask for selected indices instead of using gather_nd
            shape = tf.shape(gates)
            mask = tf.zeros(shape, dtype=tf.bool)
            mask = tf.tensor_scatter_nd_update(mask, selected, tf.ones(tf.shape(selected)[0], dtype=tf.bool))
            
            # Get old values using boolean masking
            old_values = tf.boolean_mask(gates, mask)
            
            # Apply pruning using tensor_scatter_nd_update
            pruned_gates = tf.tensor_scatter_nd_update(gates, selected, tf.zeros_like(old_values))
            self.gates.assign(pruned_gates)
            
            return selected, old_values
        indices, old_values = tf.cond(num_to_sample > 0, do_sample, no_sample)
        return indices, old_values

    def restore_gates(self, indices, old_values):
        def do_restore():
            gates = self.gates
            restored_gates = tf.tensor_scatter_nd_update(gates, indices, old_values)
            self.gates.assign(restored_gates)
            return
        tf.cond(tf.size(indices) > 0, do_restore, lambda: None)

    def update_reward(self, indices, reward):
        def do_update():
            # Get alpha from config for EMA smoothing
            alpha = getattr(self.config, 'spt_reward_alpha', 0.125)
            
            # Create boolean mask for indices instead of using gather_nd
            shape = tf.shape(self.reward_buffer)
            mask = tf.zeros(shape, dtype=tf.bool)
            mask = tf.tensor_scatter_nd_update(mask, indices, tf.ones(tf.shape(indices)[0], dtype=tf.bool))
            
            # Get current values using boolean masking
            current_buffer = tf.boolean_mask(self.reward_buffer, mask)
            current_count = tf.boolean_mask(self.reward_count, mask)
            
            # Exponential Moving Average: new_score = alpha * new_reward + (1 - alpha) * old_score
            # For first time (count == 0), use the reward directly
            is_first_time = tf.equal(current_count, 0.0)
            new_buffer = tf.where(
                is_first_time,
                reward,  # First time: use reward directly
                alpha * reward + (1.0 - alpha) * current_buffer  # EMA update
            )
            new_count = current_count + 1.0
            
            updated_reward_buffer = tf.tensor_scatter_nd_update(self.reward_buffer, indices, new_buffer)
            updated_reward_count = tf.tensor_scatter_nd_update(self.reward_count, indices, new_count)
            self.reward_buffer.assign(updated_reward_buffer)
            self.reward_count.assign(updated_reward_count)
        tf.cond(tf.size(indices) > 0, do_update, lambda: None)


class SptGatedMultiHeadAttention(Layer):
    """Multi-head attention with gated dense layers (identical to GatedMultiHeadAttention for SPT stepwise integration)."""
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
        self.wq = SptGatedDense(d_model, name=f'{layer_name}_wq', role='Query', config=config)
        self.wk = SptGatedDense(d_model, name=f'{layer_name}_wk', role='Key', config=config)
        self.wv = SptGatedDense(d_model, name=f'{layer_name}_wv', role='Value', config=config)
        # Output projection
        self.dense = SptGatedDense(d_model, name=f'{layer_name}_output', role='Attention Output', config=config)
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        q = self.wq(inputs, training=training)
        k = self.wk(inputs, training=training)
        v = self.wv(inputs, training=training)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        attention_output = self.scaled_dot_product_attention(q, k, v, mask)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention, training=training)
        return output
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)


class SptGatedTransformerBlock(Layer):
    """Transformer block with gated dense layers (identical to GatedTransformerBlock for SPT stepwise integration)."""
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float, config: GatedBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.config = config
        layer_name = kwargs.get('name', 'gated_transformer')
        # Multi-head attention
        self.mha = SptGatedMultiHeadAttention(d_model, num_heads, config, name=f'{layer_name}_mha')
        # Feed-forward network
        self.ffn_layer1 = SptGatedDense(dff, activation='relu', name=f'{layer_name}_ffn1', role='FFN Hidden', config=config)
        self.ffn_layer2 = SptGatedDense(d_model, name=f'{layer_name}_ffn2', role='FFN Output', config=config)
        # Layer normalization
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{layer_name}_ln1')
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{layer_name}_ln2')
        # Dropout
        self.dropout1 = keras.layers.Dropout(dropout_rate, name=f'{layer_name}_dropout1')
        self.dropout2 = keras.layers.Dropout(dropout_rate, name=f'{layer_name}_dropout2')
    def call(self, inputs, mask=None, training=None):
        attn_output = self.mha(inputs, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
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
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config 