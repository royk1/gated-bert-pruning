"""Training callbacks for Gated BERT."""
import tensorflow as tf
import time
from collections import defaultdict
from config import GatedBertConfig
from model_layers import GatedDense
from keras.callbacks import Callback


class GatedBertCallback(Callback):
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
        print(f"[DEBUG] on_epoch_end called for epoch {epoch+1}")
        if hasattr(self, 'epoch_start_time') and self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
        else:
            epoch_time = None
        logs = logs or {}
        
        # Record training history
        self.training_history['epoch'].append(epoch + 1)
        self.training_history['loss'].append(logs.get('loss'))
        self.training_history['accuracy'].append(logs.get('accuracy'))
        self.training_history['val_loss'].append(logs.get('val_loss'))
        self.training_history['val_accuracy'].append(logs.get('val_accuracy'))
        
        # Calculate epoch time
        if epoch_time is not None:
            self.training_history['epoch_time'].append(epoch_time)
        
        if self.config.verbose >= 1:
            if epoch_time is not None:
                print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            else:
                print(f"Epoch {epoch + 1} completed (timing unavailable)")
            
            # Print metrics
            if 'loss' in logs:
                print(f"  Train Loss: {logs['loss']:.4f}")
            if 'accuracy' in logs:
                print(f"  Train Accuracy: {logs['accuracy']:.4f}")
            if 'val_loss' in logs:
                print(f"  Val Loss: {logs['val_loss']:.4f}")
            if 'val_accuracy' in logs:
                print(f"  Val Accuracy: {logs['val_accuracy']:.4f}")
            
            # Print average reward if present
            if 'reward' in logs:
                print(f"  Avg Reward (batch): {logs['reward']:.6f}")
                if logs['reward'] == 0.0:
                    print("  [Warning] Reward is always zero. Check SPT logic.")
            if 'explored_avg_reward' in logs:
                print(f"  Avg Explored Reward (batch): {logs['explored_avg_reward']:.6f}")
            
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