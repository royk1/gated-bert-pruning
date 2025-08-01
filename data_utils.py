"""Data loading and preparation utilities."""
import logging
from datasets import load_dataset
from transformers import BertTokenizer
from config import GatedBertConfig


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