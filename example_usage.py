#!/usr/bin/env python3
"""Example usage of BART size variants."""

from config import GatedBertConfig

def example_basic_usage():
    """Example of basic usage for different model sizes."""
    
    print("=== Basic Usage Examples ===\n")
    
    # Example 1: Quick prototyping with tiny model
    print("1. Quick Prototyping (Tiny Model)")
    config_tiny = GatedBertConfig(
        model_size='tiny',
        task='sst2',
        epochs=5,
        batch_size=32
    )
    model_info = config_tiny.get_model_info()
    print(f"   Model: {model_info['model_size']} ({model_info['estimated_params_millions']})")
    print(f"   Task: {config_tiny.task}")
    print(f"   Epochs: {config_tiny.epochs}")
    print(f"   Batch Size: {config_tiny.batch_size}")
    print()
    
    # Example 2: Standard training with small model
    print("2. Standard Training (Small Model)")
    config_small = GatedBertConfig(
        model_size='small',
        task='mnli',
        epochs=15,
        batch_size=16,
        enable_pruning=True,
        pruning_method='magnitude'
    )
    model_info = config_small.get_model_info()
    print(f"   Model: {model_info['model_size']} ({model_info['estimated_params_millions']})")
    print(f"   Task: {config_small.task}")
    print(f"   Pruning: {config_small.pruning_method}")
    print(f"   Epochs: {config_small.epochs}")
    print()
    
    # Example 3: Research with base model
    print("3. Research Experiment (Base Model)")
    config_base = GatedBertConfig(
        model_size='base',
        task='qnli',
        epochs=25,
        batch_size=8,
        enable_pruning=True,
        pruning_method='spt'
    )
    model_info = config_base.get_model_info()
    print(f"   Model: {model_info['model_size']} ({model_info['estimated_params_millions']})")
    print(f"   Task: {config_base.task}")
    print(f"   Pruning: {config_base.pruning_method}")
    print(f"   Epochs: {config_base.epochs}")
    print()
    
    # Example 4: Production with large model
    print("4. Production Training (Large Model)")
    config_large = GatedBertConfig(
        model_size='large',
        task='rte',
        epochs=50,
        batch_size=4,
        enable_pruning=True,
        pruning_method='movement'
    )
    model_info = config_large.get_model_info()
    print(f"   Model: {model_info['model_size']} ({model_info['estimated_params_millions']})")
    print(f"   Task: {config_large.task}")
    print(f"   Pruning: {config_large.pruning_method}")
    print(f"   Epochs: {config_large.epochs}")
    print()

def example_parameter_override():
    """Example of overriding specific parameters."""
    
    print("=== Parameter Override Examples ===\n")
    
    # Example 1: Override training parameters
    print("1. Override Training Parameters")
    config = GatedBertConfig(
        model_size='base',
        epochs=100,  # Override default epochs
        batch_size=4,  # Override default batch size
        learning_rate=1e-5  # Override default learning rate
    )
    model_info = config.get_model_info()
    print(f"   Model: {model_info['model_size']} ({model_info['estimated_params_millions']})")
    print(f"   Epochs: {config.epochs} (overridden)")
    print(f"   Batch Size: {config.batch_size} (overridden)")
    print(f"   Learning Rate: {config.learning_rate} (overridden)")
    print()
    
    # Example 2: Override model parameters
    print("2. Override Model Parameters")
    config = GatedBertConfig(
        model_size='large',
        dropout_rate=0.2,  # Override default dropout
        max_len=256  # Override default max length
    )
    model_info = config.get_model_info()
    print(f"   Model: {model_info['model_size']} ({model_info['estimated_params_millions']})")
    print(f"   Dropout: {config.dropout_rate} (overridden)")
    print(f"   Max Length: {config.max_len} (overridden)")
    print()

def example_task_comparison():
    """Example of using different tasks with the same model size."""
    
    print("=== Task Comparison Examples ===\n")
    
    tasks = ['sst2', 'mnli', 'qnli', 'rte']
    
    for task in tasks:
        config = GatedBertConfig(
            model_size='small',
            task=task,
            epochs=10,
            batch_size=16
        )
        model_info = config.get_model_info()
        print(f"Task: {task.upper()}")
        print(f"   Model: {model_info['model_size']} ({model_info['estimated_params_millions']})")
        print(f"   Classes: {config.num_classes}")
        print(f"   Regression: {config.is_regression}")
        print()

def example_pruning_comparison():
    """Example of using different pruning methods with the same model size."""
    
    print("=== Pruning Method Comparison ===\n")
    
    pruning_methods = [
        {'name': 'No Pruning', 'enabled': False, 'method': 'magnitude'},
        {'name': 'Magnitude Pruning', 'enabled': True, 'method': 'magnitude'},
        {'name': 'Movement Pruning', 'enabled': True, 'method': 'movement'},
        {'name': 'SPT Pruning', 'enabled': True, 'method': 'spt'}
    ]
    
    for method in pruning_methods:
        config = GatedBertConfig(
            model_size='base',
            task='sst2',
            enable_pruning=method['enabled'],
            pruning_method=method['method'],
            epochs=15,
            batch_size=8
        )
        model_info = config.get_model_info()
        print(f"Method: {method['name']}")
        print(f"   Model: {model_info['model_size']} ({model_info['estimated_params_millions']})")
        print(f"   Pruning Enabled: {config.enable_pruning}")
        print(f"   Pruning Method: {config.pruning_method}")
        print(f"   Start Sparsity: {config.start_sparsity}")
        print(f"   End Sparsity: {config.end_sparsity}")
        print()

def main():
    """Run all examples."""
    
    print("BART Size Variants - Usage Examples")
    print("=" * 50)
    print()
    
    example_basic_usage()
    example_parameter_override()
    example_task_comparison()
    example_pruning_comparison()
    
    print("=== Summary ===")
    print("These examples demonstrate how to:")
    print("1. Use different model sizes for different use cases")
    print("2. Override specific parameters while keeping presets")
    print("3. Compare different tasks with the same model size")
    print("4. Compare different pruning methods with the same model size")
    print()
    print("To use these in your training script, simply copy the relevant")
    print("configuration and pass it to your training function.")

if __name__ == "__main__":
    main() 