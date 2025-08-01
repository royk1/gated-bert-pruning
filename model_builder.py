"""Model creation functions."""
import tensorflow as tf
from keras.layers import Input, Embedding, LayerNormalization, Dropout, Lambda
from keras.models import Model
from config import GatedBertConfig
from model_layers import GatedDense, PositionalEmbedding, GatedTransformerBlock
from spt_layers import SptGatedDense, SptGatedTransformerBlock
from movement_layers import MovementGatedDense, MovementGatedTransformerBlock


def create_gated_bert_model(config: GatedBertConfig) -> Model:
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
        logits = GatedDense(1, name='regressor', role='Regressor', config=config)(cls_token)
    else:
        # For classification tasks
        logits = GatedDense(config.num_classes, name='classifier', role='Classifier', config=config)(cls_token)
    
    # Create model
    model = Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name=f'GatedBERT_{config.task}'
    )
    
    return model


def create_spt_gated_bert_model(config: GatedBertConfig) -> Model:
    """Create a BERT model with structured pruning with training (SPT) capabilities."""
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
    # Transformer blocks with SPT pruning
    hidden_states = embeddings
    for i in range(config.num_layers):
        hidden_states = SptGatedTransformerBlock(
            config.d_model,
            config.num_heads,
            config.dff,
            config.dropout_rate,
            config,
            name=f'spt_transformer_block_{i}'
        )(hidden_states, mask=mask_expanded)
    # Classification head
    cls_token = Lambda(lambda x: x[:, 0, :], name='cls_token')(hidden_states)
    cls_token = Dropout(config.dropout_rate, name='cls_dropout')(cls_token)
    # Final classifier with SPT pruning
    if config.is_regression:
        logits = SptGatedDense(1, name='regressor', role='Regressor', config=config)(cls_token)
    else:
        logits = SptGatedDense(config.num_classes, name='classifier', role='Classifier', config=config)(cls_token)
    # Create base model
    model = Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name=f'SptGatedBERT_{config.task}'
    )
    return model


def create_movement_gated_bert_model(config: GatedBertConfig) -> Model:
    """Create a BERT model with movement pruning capabilities."""
    
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
    
    # Transformer blocks with movement pruning
    hidden_states = embeddings
    for i in range(config.num_layers):
        hidden_states = MovementGatedTransformerBlock(
            config.d_model,
            config.num_heads,
            config.dff,
            config.dropout_rate,
            config,
            name=f'movement_transformer_block_{i}'
        )(hidden_states, mask=mask_expanded)
    
    # Classification head
    cls_token = Lambda(lambda x: x[:, 0, :], name='cls_token')(hidden_states)
    cls_token = Dropout(config.dropout_rate, name='cls_dropout')(cls_token)
    
    # Final classifier with movement pruning
    if config.is_regression:
        logits = MovementGatedDense(1, name='regressor', role='Regressor', config=config)(cls_token)
    else:
        logits = MovementGatedDense(config.num_classes, name='classifier', role='Classifier', config=config)(cls_token)
    
    # Create base model (not wrapped in MovementPruningModel here)
    model = Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name=f'MovementGatedBERT_{config.task}'
    )
    
    return model