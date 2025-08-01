"""Main training script for Gated BERT."""
import os
import logging
import sys

# Set environment variables BEFORE importing TensorFlow
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Enable Metal GPU for performance
os.environ['TF_DISABLE_MPS'] = '0'  # Enable Metal Performance Shaders
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Use new Keras 3.x instead of legacy tf.keras
os.environ['TF_USE_LEGACY_KERAS'] = 'false'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Additional environment variables for Metal GPU stability
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Metal-specific environment variables
os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
os.environ['METAL_DEVICE_WRAPPER_TYPE_1'] = '1'

print("Environment variables set. TF_DISABLE_MPS =", os.environ.get('TF_DISABLE_MPS'))
print("CUDA_VISIBLE_DEVICES =", os.environ.get('CUDA_VISIBLE_DEVICES'))
print("TF_USE_LEGACY_KERAS =", os.environ.get('TF_USE_LEGACY_KERAS'))

import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from pruning import MovementPruningModel  # Import here to avoid circular imports
from movement_layers import MovementGatedDense
from keras.callbacks import Callback
from keras import Model
 
# Setup TensorFlow
tf.get_logger().setLevel('ERROR')

# Enable soft device placement to handle CPU/GPU operation mixing
tf.config.set_soft_device_placement(True)
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")

try:
    import transformers
    transformers.logging.set_verbosity_error()
except (ImportError, AttributeError):
    pass

# Additional imports for results processing
try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed. Results CSV functionality will be limited.")
    pd = None

# Import our modules
from config import GatedBertConfig
from data_utils import load_and_prepare_data
from model_builder import create_gated_bert_model, create_movement_gated_bert_model, create_spt_gated_bert_model
from model_layers import GatedDense
from movement_layers import MovementGatedDense
from spt_layers import SptGatedDense
from callbacks import GatedBertCallback
from utils import setup_logging, save_training_history, plot_training_history, setup_tensorflow
from pruning import create_magnitude_pruning_callback, create_movement_pruning_callback, create_spt_pruning_callback, save_pruning_history, plot_pruning_history

# Global variable to track cycling index
_current_layer_type_index = 0
# Global variable to track training steps
_training_step_counter = 4  # Start with FFN Hidden layers (index 4)

# Global batch index for cycling
global_batch_index = 0

class BatchIndexTracker(Callback):
    def on_train_batch_begin(self, batch, logs=None):
        global global_batch_index
        global_batch_index = batch

class CustomFitModel(Model):
    # Class variable to track cycling index - start with FFN Hidden (index 4)
    _class_layer_type_index = 4
    # Class variable to track batch counter across all instances - start with Value layers (index 2)
    _class_batch_counter = 2
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use a simple Python variable that we can control
        self._cycling_counter = 0
        # Add a batch counter that increments with each train_step call
        self._batch_counter = 0
    
    def train_step(self, data):
        x, y = data
        # 1. Standard forward/backward pass (with dropout)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        # Updated logging for Keras 3
        logs = {m.name: m.result() for m in self.metrics}
        logs['loss'] = loss
        # SPT reward calculation (after backward/update, before validation)
        if hasattr(self, 'is_spt') and self.is_spt:
            from spt_layers import SptGatedDense
            spt_layers = []
            def find_spt_layers(layer):
                if isinstance(layer, SptGatedDense):
                    spt_layers.append(layer)
                elif hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        find_spt_layers(sublayer)
                else:
                    # Check if layer has other layer attributes
                    for attr_name in dir(layer):
                        if not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(layer, attr_name)
                            except Exception:
                                continue
                            from keras.layers import Layer
                            if isinstance(attr_value, Layer) and attr_value is not layer:
                                find_spt_layers(attr_value)
            find_spt_layers(self)
            
            # Get SPT exploration parameters
            spt_epsilon = getattr(self, 'spt_epsilon', 0.95)
            spt_explore_per_layer = getattr(self, 'spt_explore_per_layer', True)
            
            # Within-batch exploration: explore all layer types in each batch
            if spt_explore_per_layer:
                # Define all layer types (including Query and Key)
                layer_types = ['Query', 'Key', 'Value', 'Attention Output', 'FFN Hidden', 'FFN Output', 'Regressor', 'Classifier']
                
                # For each layer type, do a separate exploration
                for layer_type in layer_types:
                    layers_to_explore = [layer for layer in spt_layers if getattr(layer, 'role', None) == layer_type]
                    
                    if layers_to_explore:  # Only explore if this layer type exists
                        # 1. Sample and prune epsilon of the gates for this layer type
                        explored_indices = {}
                        for layer in layers_to_explore:
                            indices, old_values = layer.sample_and_prune_epsilon_weights(spt_epsilon)
                            explored_indices[layer] = (indices, old_values)
                        
                        # 2. Forward pass (inference mode) to get pruned loss
                        pruned_logits = self(x, training=False)
                        pruned_loss = self.compiled_loss(y, pruned_logits, regularization_losses=self.losses)
                        
                        # 3. Restore gates
                        for layer in explored_indices:
                            indices, old_values = explored_indices[layer]
                            layer.restore_gates(indices, old_values)
                        
                        # 4. Forward pass (inference mode) to get normal loss
                        logits_eval = self(x, training=False)
                        loss_eval = self.compiled_loss(y, logits_eval, regularization_losses=self.losses)
                        
                        # Calculate reward as difference between normal and pruned loss
                        reward = loss_eval - pruned_loss
                        
                        # 5. Update reward for this layer type only
                        for layer in explored_indices:
                            indices, _ = explored_indices[layer]
                            layer.update_reward(indices, reward)
                
                # Calculate average reward across all explored layers for logging
                total_reward = tf.constant(0.0)
                total_count = 0
                for layer in spt_layers:
                    if hasattr(layer, 'reward_buffer'):
                        layer_reward = tf.reduce_mean(layer.reward_buffer)
                        total_reward += layer_reward
                        total_count += 1
                
                avg_reward = total_reward / tf.cast(total_count, tf.float32) if total_count > 0 else tf.constant(0.0)
                
            else:
                # Original single exploration logic (kept for backward compatibility)
                layers_to_explore = spt_layers
                
                # 1. Sample and prune epsilon of the gates for each selected SPT layer
                explored_indices = {}
                for layer in layers_to_explore:
                    indices, old_values = layer.sample_and_prune_epsilon_weights(spt_epsilon)
                    explored_indices[layer] = (indices, old_values)
                
                # 2. Forward pass (inference mode) to get pruned loss
                pruned_logits = self(x, training=False)
                pruned_loss = self.compiled_loss(y, pruned_logits, regularization_losses=self.losses)
                
                # 3. Restore gates
                for layer in explored_indices:
                    indices, old_values = explored_indices[layer]
                    layer.restore_gates(indices, old_values)
                
                # 4. Forward pass (inference mode) to get normal loss
                logits_eval = self(x, training=False)
                loss_eval = self.compiled_loss(y, logits_eval, regularization_losses=self.losses)
                
                # Calculate reward as difference between normal and pruned loss
                reward = loss_eval - pruned_loss
                
                # 6. For each explored SPT layer, update_reward for the explored indices
                for layer in explored_indices:
                    indices, _ = explored_indices[layer]
                    layer.update_reward(indices, reward)
                
                avg_reward = tf.cast(tf.reduce_mean(reward), tf.float32)
            
            # 7. Add reward to logs (only essential metric)
            logs['reward'] = avg_reward
            return logs
        
        # For non-SPT models, return logs without reward
        return logs


def main():
    """Main training function."""
    
    # Setup TensorFlow
    setup_tensorflow()
    
    # Configuration - Use defaults from config.py
    config = GatedBertConfig()
    
    # Display model information
    model_info = config.get_model_info()
    print(f"=== Model Configuration ===")
    print(f"Model Size: {model_info['model_size']}")
    print(f"Parameters: {model_info['estimated_params_millions']} ({model_info['estimated_params']})")
    print(f"Layers: {model_info['num_layers']}")
    print(f"Hidden Size: {model_info['d_model']}")
    print(f"Attention Heads: {model_info['num_heads']}")
    print(f"FFN Size: {model_info['dff']}")
    print(f"Task: {config.task}")
    print(f"==========================")
    
    # Add debug logging
    print(f"DEBUG: enable_pruning = {config.enable_pruning}")
    print(f"DEBUG: pruning_method = {config.pruning_method}")
    print(f"DEBUG: start_sparsity = {config.start_sparsity}")
    print(f"DEBUG: end_sparsity = {config.end_sparsity}")
    print(f"DEBUG: pruning_start_epoch = {config.pruning_start_epoch}")
    print(f"DEBUG: pruning_end_epoch = {config.pruning_end_epoch}")
    
    # Add model-size-adaptive parameter logging
    if config.enable_pruning:
        print(f"DEBUG: Model-size-adaptive parameters:")
        print(f"DEBUG:   spt_epsilon = {config.spt_epsilon} (scaled for {config.model_size})")
        print(f"DEBUG:   movement_frequency = {config.movement_pruning_frequency_steps} steps (scaled for {config.model_size})")
        print(f"DEBUG:   sparsity_schedule = {config.sparsity_steps[:5]}... (scaled for {config.model_size})")
    
    # Check for benchmark mode
    benchmark_epochs = os.getenv('BENCHMARK_EPOCHS')
    if benchmark_epochs:
        try:
            config.epochs = int(benchmark_epochs)
            print(f"Running in benchmark mode for {config.epochs} epochs")
        except ValueError:
            print(f"Invalid BENCHMARK_EPOCHS value: {benchmark_epochs}")
    
    # Setup logging
    setup_logging(config)
    
    # Load and prepare data
    train_dataset, val_dataset = load_and_prepare_data(config)
    
    # Create model based on pruning method
    logging.info("Creating model...")
    if config.enable_pruning and config.pruning_method == 'movement':
        from pruning import MovementPruningModel  # Import here to avoid circular imports
        base_model = create_movement_gated_bert_model(config)
        model = MovementPruningModel(base_model, config, name=f'MovementPruningBERT_{config.task}')
        
        # Build the wrapper model by calling it with dummy data
        logging.info("Building movement pruning model...")
        dummy_input_ids = tf.zeros((1, config.max_len), dtype=tf.int32)
        dummy_attention_mask = tf.ones((1, config.max_len), dtype=tf.int32)
        _ = model([dummy_input_ids, dummy_attention_mask], training=False)
        logging.info("Movement pruning model built successfully")
        
        # Compile model (FIX: was missing)
        if config.is_regression:
            model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss='mse',
                metrics=['mae']
            )
        else:
            model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[SparseCategoricalAccuracy(name='accuracy')]
            )
        
        # NOW calculate and set steps per epoch
        try:
            # Calculate dataset size dynamically
            import math
            # Get the actual dataset size from the dataset
            train_size = sum(1 for _ in train_dataset)
            steps_per_epoch = math.ceil(train_size / config.batch_size)
            print(f"DEBUG: Dataset size: {train_size}, Steps per epoch: {steps_per_epoch}")
            model.steps_per_epoch = steps_per_epoch  # Set it on the model
        except Exception as e:
            print(f"DEBUG: Could not set steps per epoch: {e}")
        
        # Train the movement pruning model
        callbacks = [GatedBertCallback(config)]
        if config.enable_pruning:
            pruning_callback = create_movement_pruning_callback(
                config=config,
                start_sparsity=config.start_sparsity,
                end_sparsity=config.end_sparsity,
                start_epoch=config.pruning_start_epoch,
                end_epoch=config.pruning_end_epoch,
                verbose=config.verbose
            )
            callbacks.append(pruning_callback)
        
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.epochs,
            callbacks=callbacks
        )
        
        # Store results for centralized backup
        final_results = model.evaluate(val_dataset, verbose=0)
        
        # Create comprehensive results backup before returning
        create_comprehensive_results_backup(model, config, callbacks, final_results)
        
        # Skip the general training section since movement training is complete
        return
    
    elif config.enable_pruning and config.pruning_method == 'spt':
        base_model = create_spt_gated_bert_model(config)
        model = CustomFitModel(inputs=base_model.inputs, outputs=base_model.outputs)
        model.is_spt = True
        model.spt_epsilon = getattr(config, 'spt_epsilon', 0.01)
        model.spt_explore_per_layer = getattr(config, 'spt_explore_per_layer', False)
        logging.info("Building SPT pruning model...")
        dummy_input_ids = tf.zeros((1, config.max_len), dtype=tf.int32)
        dummy_attention_mask = tf.ones((1, config.max_len), dtype=tf.int32)
        _ = model([dummy_input_ids, dummy_attention_mask], training=False)
        logging.info("SPT pruning model built successfully")
        model.summary()
        
        # def print_dense_layers(layer, prefix=""):
        #     if isinstance(layer, (GatedDense, SptGatedDense)):
        #         print(f"{prefix}{layer.name}: {layer.__class__} from {layer.__class__.__module__}", file=sys.stderr)
        #     if hasattr(layer, 'layers'):
        #         for sublayer in layer.layers:
        #             print_dense_layers(sublayer, prefix=prefix + layer.name + "/")
        #     for attr in dir(layer):
        #         if not attr.startswith('_'):
        #             try:
        #                 sub = getattr(layer, attr)
        #             except Exception:
        #                 continue
        #             if isinstance(sub, tf.keras.layers.Layer) and sub is not layer:
        #                 print_dense_layers(sub, prefix=prefix + layer.name + "." + attr + "/")
        # print("=== GatedDense/SptGatedDense layer diagnostic ===", file=sys.stderr)
        # print_dense_layers(model)
        # print("=== End diagnostic ===", file=sys.stderr)
        if config.is_regression:
            model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss='mse',
                metrics=['mae']
            )
        else:
            model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[SparseCategoricalAccuracy(name='accuracy')]
            )
        callbacks = [GatedBertCallback(config)]
        if config.enable_pruning:
            pruning_callback = create_spt_pruning_callback(
                config=config,
                start_sparsity=config.start_sparsity,
                end_sparsity=config.end_sparsity,
                start_epoch=config.pruning_start_epoch,
                end_epoch=config.pruning_end_epoch,
                verbose=config.verbose
            )
            callbacks.append(pruning_callback)
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.epochs,
            callbacks=callbacks
        )
        
        # Store results for centralized backup
        final_results = model.evaluate(val_dataset, verbose=0)
        
        # Create comprehensive results backup before returning
        create_comprehensive_results_backup(model, config, callbacks, final_results)
        
        # Skip the general training section since SPT training is complete
        return
        
    else:
        # Default path: standard gated model or magnitude pruning
        base_model = create_gated_bert_model(config)
        model = CustomFitModel(inputs=base_model.inputs, outputs=base_model.outputs)
        model.is_spt = False  # Explicitly set for non-SPT models
        if config.is_regression:
            model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss='mse',
                metrics=['mae']
            )
        else:
            model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[SparseCategoricalAccuracy(name='accuracy')]
            )
        callbacks = [GatedBertCallback(config)]
        if config.enable_pruning:
            pruning_callback = create_magnitude_pruning_callback(
                config=config,
                start_sparsity=config.start_sparsity,
                end_sparsity=config.end_sparsity,
                start_epoch=config.pruning_start_epoch,
                end_epoch=config.pruning_end_epoch,
                verbose=config.verbose
            )
            callbacks.append(pruning_callback)
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.epochs,
            callbacks=callbacks
        )
        
        # Store results for centralized backup
        final_results = model.evaluate(val_dataset, verbose=0)
        
        # Create comprehensive results backup before returning
        create_comprehensive_results_backup(model, config, callbacks, final_results)
        
        # Skip the general training section since magnitude/default training is complete
        return
    
    # Model summary
    try:
        if hasattr(model, 'base_model'):
            # For wrapper models, show both wrapper and base model info
            logging.info("=== Wrapper Model Info ===")
            logging.info(f"Model type: {type(model).__name__}")
            logging.info(f"Base model type: {type(model.base_model).__name__}")
            logging.info("=== Base Model Summary ===")
            model.base_model.summary(print_fn=logging.info)
        else:
            # For regular models
            model.summary(print_fn=logging.info)
    except Exception as e:
        logging.warning(f"Could not print model summary: {e}")

    # Count gated layers
    if hasattr(model, 'base_model'):
        # For wrapper models, count layers in base model
        gated_layers = [layer for layer in model.base_model.layers if isinstance(layer, (GatedDense, MovementGatedDense, SptGatedDense))]
    else:
        gated_layers = [layer for layer in model.layers if isinstance(layer, (GatedDense, MovementGatedDense, SptGatedDense))]

    logging.info(f"Model has {len(gated_layers)} gated dense layers")
    if hasattr(model, 'movement_layers'):
        logging.info(f"Movement pruning tracking {len(model.movement_layers)} layers")
    
    # Callbacks
    callbacks = [
        GatedBertCallback(config),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.checkpoint_dir, 'best_model_weights.h5'),
            monitor='val_accuracy' if not config.is_regression else 'val_mae',
            mode='max' if not config.is_regression else 'min',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        #tf.keras.callbacks.EarlyStopping(
        #    monitor='val_accuracy' if not config.is_regression else 'val_mae',
        #    mode='max' if not config.is_regression else 'min',
        #    patience=5,  # Increase patience for pruning
        #    restore_best_weights=True,
        #    verbose=1
        #)
    ]
    
    # Add debug logging before adding pruning callback
    print(f"DEBUG: About to check pruning: enable_pruning={config.enable_pruning}")
    
    # Add pruning callback if enabled
    if config.enable_pruning:
        print(f"DEBUG: Adding pruning callback with method: {config.pruning_method}")
        if config.pruning_method == 'movement':
            pruning_callback = create_movement_pruning_callback(
                config=config,
                start_sparsity=config.start_sparsity,
                end_sparsity=config.end_sparsity,
                start_epoch=config.pruning_start_epoch,
                end_epoch=config.pruning_end_epoch,
                verbose=config.verbose
            )
        elif config.pruning_method == 'spt':
            pruning_callback = create_spt_pruning_callback(
                config=config,
                start_sparsity=config.start_sparsity,
                end_sparsity=config.end_sparsity,
                start_epoch=config.pruning_start_epoch,
                end_epoch=config.pruning_end_epoch,
                verbose=config.verbose
            )
        else:  # magnitude pruning
            pruning_callback = create_magnitude_pruning_callback(
                config=config,
                start_sparsity=config.start_sparsity,
                end_sparsity=config.end_sparsity,
                start_epoch=config.pruning_start_epoch,
                end_epoch=config.pruning_end_epoch,
                verbose=config.verbose
            )
        
        callbacks.append(pruning_callback)
        print(f"DEBUG: Pruning callback added. Total callbacks: {len(callbacks)}")
        logging.info(f"Pruning enabled: {config.pruning_method} - {config.start_sparsity:.1%} -> {config.end_sparsity:.1%} "
                    f"from epoch {config.pruning_start_epoch} to {config.pruning_end_epoch}")
    else:
        print("DEBUG: Pruning is disabled")
    
    # Train the model
    logging.info("Starting training...")
    
    # Force CPU-only execution during training to avoid platform registration errors
    with tf.device('/CPU:0'):
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=config.verbose
        )
    
    # Save training history
    if callbacks and len(callbacks) > 0 and hasattr(callbacks[0], 'training_history') and callbacks[0].training_history:
        history_path = os.path.join(config.checkpoint_dir, 'training_history.json')
        save_training_history(dict(callbacks[0].training_history), history_path)
        
        # Create plots
        plot_path = os.path.join(config.checkpoint_dir, 'training_plots.png')
        plot_training_history(dict(callbacks[0].training_history), plot_path)
    
    # Save configuration
    config_path = os.path.join(config.checkpoint_dir, 'config.json')
    config.save(config_path)
    
    # === CENTRALIZED FINAL EVALUATION AND RESULTS BACKUP ===
    # This section always runs regardless of pruning method or enable_pruning setting
    
    # Final evaluation and logging
    logging.info("Final evaluation...")
    if final_results and len(final_results) > 0:
        if config.is_regression:
            logging.info(f"Final validation MSE: {final_results[0]:.4f}")
            if len(final_results) > 1:
                logging.info(f"Final validation MAE: {final_results[1]:.4f}")
        else:
            logging.info(f"Final validation loss: {final_results[0]:.4f}")
            if len(final_results) > 1:
                logging.info(f"Final validation accuracy: {final_results[1]:.4f}")
    else:
        logging.info("Final evaluation results not available")
    
    logging.info("Training completed successfully!")
    
    # Always save pruning history if pruning is enabled and callbacks exist
    if config.enable_pruning and callbacks and len(callbacks) > 1:  # At least GatedBertCallback + pruning callback
        pruning_callback = callbacks[-1]  # Last callback should be pruning callback
        if hasattr(pruning_callback, 'get_pruning_history'):
            # Save pruning history
            pruning_history_path = os.path.join(config.checkpoint_dir, 'pruning_history.json')
            save_pruning_history(pruning_callback.get_pruning_history(), pruning_history_path)
            
            # Create pruning plot
            pruning_plot_path = os.path.join(config.checkpoint_dir, 'pruning_plot.png')
            plot_pruning_history(pruning_callback.get_pruning_history(), pruning_plot_path)
    
    # === ALWAYS CREATE COMPREHENSIVE RESULTS BACKUP ===
    # This runs for ALL cases: pruning enabled/disabled, all methods, all configurations
    create_comprehensive_results_backup(model, config, callbacks, final_results)


def create_comprehensive_results_backup(model, config, callbacks, final_results):
    """Create comprehensive backup and results logging with method-specific organization."""
    try:
        import zipfile
        import matplotlib.pyplot as plt
        from datetime import datetime
        import shutil
        import glob
        
        print(f"\n=== Creating Comprehensive Results Backup ===")
        print(f"Pruning enabled: {config.enable_pruning}")
        print(f"Pruning method: {config.pruning_method if config.enable_pruning else 'N/A'}")
        
        # Check if pandas is available
        try:
            import pandas as pd
            pandas_available = True
        except ImportError:
            print("Warning: pandas not available. CSV export will be limited.")
            pandas_available = False
        
        # Determine pruning method for folder organization
        if config.enable_pruning:
            method_name = config.pruning_method
        else:
            method_name = 'baseline'
        
        # Create method-specific results directory
        results_dir = f'results_{method_name}_{config.task}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # === BACKUP PREVIOUS RESULTS ===
        if os.path.exists(results_dir) and os.listdir(results_dir):
            backup_filename = f'backup_{method_name}_{timestamp}.zip'
            backup_path = os.path.join(results_dir, backup_filename)
            
            try:
                print(f"Backing up previous results to {backup_path}")
                with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(results_dir):
                        for file in files:
                            if not file.startswith('backup_'):  # Don't backup other backups
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, results_dir)
                                zipf.write(file_path, arcname)
                
                # Remove old files (except backups)
                for item in os.listdir(results_dir):
                    item_path = os.path.join(results_dir, item)
                    if not item.startswith('backup_'):
                        try:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except Exception as e:
                            print(f"Warning: Could not remove {item_path}: {e}")
            except Exception as e:
                print(f"Warning: Backup creation failed: {e}")
                print("Continuing with results creation...")
        
        # === COLLECT TRAINING DATA ===
        training_data = []
        
        # Get basic training history
        training_history = {}
        if callbacks and len(callbacks) > 0 and hasattr(callbacks[0], 'training_history'):
            training_history = callbacks[0].training_history or {}
        
        # Get pruning history if available
        pruning_history = []
        if config.enable_pruning and callbacks and len(callbacks) > 1:  # At least GatedBertCallback + pruning callback
            pruning_callback = callbacks[-1]  # Last callback should be pruning callback
            if hasattr(pruning_callback, 'pruning_history'):
                pruning_history = pruning_callback.pruning_history
        
        # Build comprehensive epoch data
        for epoch in range(config.epochs):
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': 0.0,
                'train_accuracy': 0.0,
                'val_loss': 0.0,
                'val_accuracy': 0.0,
                'sparsity_start': 0.0,
                'sparsity_end': 0.0,
                'active_weights': 0,
                'total_weights': 0,
                'avg_reward': 0.0
            }
            
            # Safely extract training history data
            if 'loss' in training_history and epoch < len(training_history['loss']):
                epoch_data['train_loss'] = float(training_history['loss'][epoch])
            if 'accuracy' in training_history and epoch < len(training_history['accuracy']):
                epoch_data['train_accuracy'] = float(training_history['accuracy'][epoch])
            if 'val_loss' in training_history and epoch < len(training_history['val_loss']):
                epoch_data['val_loss'] = float(training_history['val_loss'][epoch])
            if 'val_accuracy' in training_history and epoch < len(training_history['val_accuracy']):
                epoch_data['val_accuracy'] = float(training_history['val_accuracy'][epoch])
            
            # Add pruning-specific data
            if pruning_history and epoch < len(pruning_history):
                pruning_data = pruning_history[epoch]
                epoch_data.update({
                    'sparsity_start': pruning_data.get('target_sparsity', 0.0),
                    'sparsity_end': pruning_data.get('overall_sparsity', 0.0),
                    'active_weights': pruning_data.get('total_active', 0),
                    'total_weights': pruning_data.get('total_weights', 0)
                })
            
            # Add SPT-specific reward data
            if config.enable_pruning and config.pruning_method == 'spt':
                if 'explored_avg_reward' in training_history and epoch < len(training_history['explored_avg_reward']):
                    epoch_data['avg_reward'] = float(training_history['explored_avg_reward'][epoch])
            
            training_data.append(epoch_data)
        
        # === CREATE RESULTS TABLE ===
        if pandas_available:
            df = pd.DataFrame(training_data)
            
            # Add run metadata
            metadata = {
                'run_timestamp': timestamp,
                'task': config.task,
                'method': method_name,
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'final_val_loss': final_results[0] if final_results and len(final_results) > 0 else 0,
                'final_val_accuracy': final_results[1] if final_results and len(final_results) > 1 else 0
            }
            
            # Add method-specific parameters
            if config.enable_pruning:
                metadata.update({
                    'start_sparsity': config.start_sparsity,
                    'end_sparsity': config.end_sparsity,
                    'pruning_start_epoch': config.pruning_start_epoch,
                    'pruning_end_epoch': config.pruning_end_epoch
                })
                
                if config.pruning_method == 'spt':
                    metadata.update({
                        'spt_epsilon': getattr(config, 'spt_epsilon', 0.01),
                        'spt_reward_alpha': getattr(config, 'spt_reward_alpha', 0.125)
                    })
            
            # Save results table
            results_csv_path = os.path.join(results_dir, f'results_{method_name}_{timestamp}.csv')
            df.to_csv(results_csv_path, index=False)
            
            # Save metadata
            metadata_path = os.path.join(results_dir, f'metadata_{method_name}_{timestamp}.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            print(f"Skipping CSV export due to pandas not being available.")
            # Still save metadata even without pandas
            metadata_path = os.path.join(results_dir, f'metadata_{method_name}_{timestamp}.json')
            import json
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save metadata: {e}")
        
        # === CREATE PLOTS ===
        plt.style.use('default')
        
        try:
            # Extract data for plotting
            if pandas_available:
                epochs = df['epoch']
                train_loss = df['train_loss']
                val_loss = df['val_loss']
                train_accuracy = df['train_accuracy']
                val_accuracy = df['val_accuracy']
                sparsity_end = df['sparsity_end']
                avg_reward = df['avg_reward']
            else:
                epochs = [d['epoch'] for d in training_data]
                train_loss = [d['train_loss'] for d in training_data]
                val_loss = [d['val_loss'] for d in training_data]
                train_accuracy = [d['train_accuracy'] for d in training_data]
                val_accuracy = [d['val_accuracy'] for d in training_data]
                sparsity_end = [d['sparsity_end'] for d in training_data]
                avg_reward = [d['avg_reward'] for d in training_data]
        
            # Plot 1: Train/Test Loss and Sparsity
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
                ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title(f'{method_name.upper()} - Training Progress: Loss vs Sparsity')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(epochs, sparsity_end, 'g-', label='Sparsity', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Sparsity')
                ax2.set_ylim(0, 1)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot1_path = os.path.join(results_dir, f'loss_sparsity_{method_name}_{timestamp}.png')
                plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create loss/sparsity plot: {e}")
            
            # Plot 2: Train/Test Accuracy and Sparsity
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                ax1.plot(epochs, train_accuracy, 'b-', label='Train Accuracy', linewidth=2)
                ax1.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy', linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.set_title(f'{method_name.upper()} - Training Progress: Accuracy vs Sparsity')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(epochs, sparsity_end, 'g-', label='Sparsity', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Sparsity')
                ax2.set_ylim(0, 1)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot2_path = os.path.join(results_dir, f'accuracy_sparsity_{method_name}_{timestamp}.png')
                plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create accuracy/sparsity plot: {e}")
            
            # Plot 3: Reward vs Sparsity (SPT only)
            if config.enable_pruning and config.pruning_method == 'spt' and sum(avg_reward) != 0:
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    
                    # Create scatter plot with color gradient
                    scatter = ax.scatter(sparsity_end, avg_reward, c=epochs, cmap='viridis', s=50, alpha=0.7)
                    ax.plot(sparsity_end, avg_reward, 'k-', alpha=0.3, linewidth=1)
                    
                    ax.set_xlabel('Sparsity')
                    ax.set_ylabel('Average Reward')
                    ax.set_title(f'SPT - Reward vs Sparsity Progression')
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                    ax.grid(True, alpha=0.3)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Epoch')
                    
                    plt.tight_layout()
                    plot3_path = os.path.join(results_dir, f'reward_sparsity_{method_name}_{timestamp}.png')
                    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Warning: Could not create reward/sparsity plot: {e}")
        
        except Exception as e:
            print(f"Warning: Plot creation failed: {e}")
        
        # Copy important files to results directory
        if os.path.exists(config.checkpoint_dir):
            # Copy best model weights
            best_weights_src = os.path.join(config.checkpoint_dir, 'best_model_weights.h5')
            if os.path.exists(best_weights_src):
                try:
                    best_weights_dst = os.path.join(results_dir, f'best_model_weights_{method_name}_{timestamp}.h5')
                    shutil.copy2(best_weights_src, best_weights_dst)
                except Exception as e:
                    print(f"Warning: Could not copy model weights: {e}")
            
            # Copy config
            config_src = os.path.join(config.checkpoint_dir, 'config.json')
            if os.path.exists(config_src):
                try:
                    config_dst = os.path.join(results_dir, f'config_{method_name}_{timestamp}.json')
                    shutil.copy2(config_src, config_dst)
                except Exception as e:
                    print(f"Warning: Could not copy config file: {e}")
        
        print(f"\n=== Results Summary ===")
        print(f"Method: {method_name.upper()}")
        print(f"Results saved to: {results_dir}/")
        print(f"Timestamp: {timestamp}")
        if final_results and len(final_results) > 1:
            print(f"Final validation accuracy: {final_results[1]:.4f}")
        else:
            print(f"Final validation accuracy: N/A")
        if config.enable_pruning and pruning_history:
            final_sparsity = pruning_history[-1].get('overall_sparsity', 0) if pruning_history else 0
            print(f"Final sparsity: {final_sparsity:.1%}")
        if pandas_available:
            csv_filename = f'results_{method_name}_{timestamp}.csv'
            print(f"Results table: {csv_filename}")
        else:
            print("Results table: CSV export skipped due to pandas not being available.")
        print(f"Plots created: loss_sparsity, accuracy_sparsity" + (", reward_sparsity" if config.enable_pruning and config.pruning_method == 'spt' else ""))
    except Exception as e:
        logging.error(f"Error during comprehensive results backup: {e}")


if __name__ == "__main__":
    main()