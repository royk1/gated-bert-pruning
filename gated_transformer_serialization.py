def get_config(self):
    """Get configuration for layer serialization."""
    config = super().get_config()
    config.update({
        'd_model': self.d_model,
        'num_heads': self.num_heads,
        'dff': self.dff,
        'dropout_rate': self.dropout_rate,
        'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
    })
    return config

@classmethod
def from_config(cls, config):
    """Create layer from configuration."""
    gated_bert_config = config.pop('config')
    if isinstance(gated_bert_config, dict):
        gated_bert_config = GatedBertConfig(**gated_bert_config)
    return cls(config=gated_bert_config, **config)