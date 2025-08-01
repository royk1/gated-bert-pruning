"""Custom layers for Gated BERT model."""
import tensorflow as tf
from keras.layers import Embedding, LayerNormalization, Dropout, Layer
import numpy as np
from config import GatedBertConfig


class GatedDense(Layer):
    """Dense layer with gates for future pruning."""
    
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
        from keras import initializers
        self.gates = self.add_weight(
            name='gates',
            shape=self.kernel.shape,
            initializer=initializers.Constant(self.config.gate_init_value),
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
        from keras import activations
        config.update({
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'role': self.role,
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        from keras import activations
        activation = activations.deserialize(config.pop('activation'))
        role = config.pop('role', None)
        gate_init_value = config.pop('gate_init_value', 1.0)
        gate_trainable = config.pop('gate_trainable', False)
        
        from types import SimpleNamespace
        layer_config = SimpleNamespace()
        layer_config.gate_init_value = gate_init_value
        layer_config.gate_trainable = gate_trainable
        
        return cls(config.pop('units'), activation=activation, role=role, config=layer_config, **config)


class PositionalEmbedding(Layer):
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
        # Use TensorFlow operations directly
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(0, seq_len, 1)
        return self.pos_embedding(positions)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model,
        })
        return config


class GatedMultiHeadAttention(Layer):
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
        self.wq = GatedDense(d_model, name=f'{layer_name}_wq', role='Query', config=config)
        self.wk = GatedDense(d_model, name=f'{layer_name}_wk', role='Key', config=config)
        self.wv = GatedDense(d_model, name=f'{layer_name}_wv', role='Value', config=config)
        
        # Output projection
        self.dense = GatedDense(d_model, name=f'{layer_name}_output', role='Attention Output', config=config)
    
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
        matmul_qk = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        
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
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config


class GatedTransformerBlock(Layer):
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
        self.mha = GatedMultiHeadAttention(d_model, num_heads, config, name=f'{layer_name}_mha')
        
        # Feed-forward network
        self.ffn_layer1 = GatedDense(dff, activation='relu', name=f'{layer_name}_ffn1', role='FFN Hidden', config=config)
        self.ffn_layer2 = GatedDense(d_model, name=f'{layer_name}_ffn2', role='FFN Output', config=config)
        
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
            'gate_init_value': self.config.gate_init_value if self.config else 1.0,
            'gate_trainable': self.config.gate_trainable if self.config else False,
        })
        return config