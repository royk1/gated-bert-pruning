"""Pruning methods for Gated BERT models."""
import tensorflow as tf
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable
from config import GatedBertConfig
from model_layers import GatedDense
from movement_layers import MovementGatedDense
from spt_layers import SptGatedDense
from keras.callbacks import Callback
from keras import Model


class PruningMethod:
    """Base class for pruning methods."""
    
    def __init__(self, config: GatedBertConfig):
        self.config = config
    
    def prune_layer(self, layer: GatedDense, target_sparsity: float) -> Dict:
        """Prune a single layer and return statistics."""
        raise NotImplementedError
    
    def get_pruning_schedule(self, epoch: int) -> float:
        """Get target sparsity for current epoch."""
        raise NotImplementedError


class StepBasedPruningMethod(PruningMethod):
    """Base class for pruning methods that support step-based scheduling."""
    
    def __init__(self, config: GatedBertConfig, 
                 start_sparsity: float = 0.0,
                 end_sparsity: float = 0.5,
                 start_epoch: int = 0,
                 end_epoch: int = 10,
                 sparsity_steps: Optional[List[float]] = None):
        super().__init__(config)
        self.start_sparsity = start_sparsity
        self.end_sparsity = end_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.sparsity_steps = sparsity_steps
        
        # Convert sparsity steps from percentages to decimals if needed
        if self.sparsity_steps is not None:
            self.sparsity_steps = [step / 100.0 if step > 1.0 else step for step in self.sparsity_steps]
    
    def get_pruning_schedule(self, epoch: int) -> float:
        """Get target sparsity for current epoch using step-based or linear schedule."""
        if epoch < self.start_epoch:
            return self.start_sparsity
        
        # Use step-based schedule if provided
        if self.sparsity_steps is not None:
            step_index = epoch - self.start_epoch
            if step_index < len(self.sparsity_steps):
                return self.sparsity_steps[step_index]
            else:
                # After all steps, maintain the last sparsity value
                return self.sparsity_steps[-1]
        
        # Fall back to linear schedule
        if epoch >= self.end_epoch:
            return self.end_sparsity
        else:
            # Linear interpolation
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return self.start_sparsity + progress * (self.end_sparsity - self.start_sparsity)


class MagnitudePruning(StepBasedPruningMethod):
    """Magnitude-based pruning - prunes weights with smallest absolute values."""
    
    def __init__(self, config: GatedBertConfig, 
                 start_sparsity: float = 0.0,
                 end_sparsity: float = 0.5,
                 start_epoch: int = 0,
                 end_epoch: int = 10,
                 sparsity_steps: Optional[List[float]] = None):
        super().__init__(config, start_sparsity, end_sparsity, start_epoch, end_epoch, sparsity_steps)
    
    def prune_layer(self, layer: GatedDense, target_sparsity: float) -> Dict:
        """Prune layer based on weight magnitudes."""
        if not hasattr(layer, 'kernel') or not hasattr(layer, 'gates'):
            return {'error': 'Layer does not have kernel or gates'}
        
        # Get current weights and gates
        weights = layer.kernel.numpy()
        gates = layer.gates.numpy()
        
        # Calculate absolute magnitudes
        weight_magnitudes = np.abs(weights)
        
        # Calculate number of weights to prune
        total_weights = weights.size
        num_to_prune = int(total_weights * target_sparsity)
        
        # Find threshold for pruning (k-th smallest magnitude)
        if num_to_prune > 0:
            # Flatten weights and find threshold
            flat_magnitudes = weight_magnitudes.flatten()
            threshold = np.partition(flat_magnitudes, num_to_prune - 1)[num_to_prune - 1]
            
            # Create new gate values (0 for pruned weights, 1 for kept weights)
            new_gates = np.where(weight_magnitudes <= threshold, 0.0, 1.0)
            
            # If we have ties at the threshold, we need to be more careful
            num_at_threshold = np.sum(weight_magnitudes == threshold)
            num_already_pruned = np.sum(weight_magnitudes < threshold)
            
            if num_already_pruned < num_to_prune and num_at_threshold > 1:
                # We have ties at the threshold, prune some but not all
                remaining_to_prune = num_to_prune - num_already_pruned
                at_threshold_indices = np.where(weight_magnitudes == threshold)
                
                # Randomly select which ones to prune among the tied values
                if len(at_threshold_indices[0]) > 0:
                    total_at_threshold = len(at_threshold_indices[0])
                    prune_indices = np.random.choice(
                        total_at_threshold, 
                        size=min(remaining_to_prune, total_at_threshold), 
                        replace=False
                    )
                    
                    # Set selected tied weights to be pruned
                    for i in prune_indices:
                        row_idx = at_threshold_indices[0][i]
                        col_idx = at_threshold_indices[1][i]
                        new_gates[row_idx, col_idx] = 0.0
                    
                    # Keep the rest of the tied weights
                    for i in range(total_at_threshold):
                        if i not in prune_indices:
                            row_idx = at_threshold_indices[0][i]
                            col_idx = at_threshold_indices[1][i]
                            new_gates[row_idx, col_idx] = 1.0
            
            # Update the layer's gates
            layer.gates.assign(new_gates)
        
        # Calculate final statistics
        final_gates = layer.gates.numpy()
        active_weights = np.sum(final_gates)
        actual_sparsity = 1.0 - (active_weights / total_weights)
        
        return {
            'layer_name': layer.name,
            'layer_role': layer.role,
            'total_weights': total_weights,
            'active_weights': int(active_weights),
            'pruned_weights': int(total_weights - active_weights),
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'threshold': threshold if num_to_prune > 0 else 0.0
        }


class MovementPruning(StepBasedPruningMethod):
    """Movement-based pruning - prunes weights that hurt performance most."""
    
    def __init__(self, config: GatedBertConfig, 
                 start_sparsity: float = 0.0,
                 end_sparsity: float = 0.5,
                 start_epoch: int = 0,
                 end_epoch: int = 10,
                 sparsity_steps: Optional[List[float]] = None):
        super().__init__(config, start_sparsity, end_sparsity, start_epoch, end_epoch, sparsity_steps)
    
    def get_pruning_schedule(self, epoch: int) -> float:
        """Get target sparsity based on schedule (step-based, linear, or cubic)."""
        if epoch < self.start_epoch:
            return self.start_sparsity
        
        # Use step-based schedule if provided
        if self.sparsity_steps is not None:
            step_index = epoch - self.start_epoch
            if step_index < len(self.sparsity_steps):
                return self.sparsity_steps[step_index]
            else:
                # After all steps, maintain the last sparsity value
                return self.sparsity_steps[-1]
        
        # Fall back to linear/cubic schedule
        if epoch >= self.end_epoch:
            return self.end_sparsity
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            
            if self.config.movement_schedule == 'cubic':
                # Cubic schedule: s_f + (s_i - s_f) * (1 - progress)^3
                current_sparsity = self.end_sparsity + (self.start_sparsity - self.end_sparsity) * ((1.0 - progress) ** 3)
            else:  # linear
                current_sparsity = self.start_sparsity + progress * (self.end_sparsity - self.start_sparsity)
            
            return current_sparsity
    
    def prune_layer(self, layer: MovementGatedDense, target_sparsity: float) -> Dict:
        """Use movement scores for pruning instead of random."""
        if not hasattr(layer, 'gates'):
            return {'error': 'Layer does not have gates'}
        
        # Check if we have movement scores to use
        if target_sparsity > 0 and hasattr(layer, 'movement_scores') and layer.movement_scores is not None:
            # Use movement scores for pruning
            weights = layer.kernel.numpy()
            gates = layer.gates.numpy()
            movement_scores = layer.movement_scores.numpy()
            
            total_weights = weights.size
            num_to_keep = int(total_weights * (1.0 - target_sparsity))
            
            print(f"DEBUG MOVEMENT: {layer.name} using movement scores for pruning")
            print(f"  Movement score range: [{np.min(movement_scores):.6f}, {np.max(movement_scores):.6f}]")
            
            if num_to_keep <= 0:
                # Prune everything
                new_gates = np.zeros_like(gates)
                print(f"DEBUG MOVEMENT: {layer.name} pruning everything")
            elif num_to_keep >= total_weights:
                # Keep everything
                new_gates = np.ones_like(gates)
                print(f"DEBUG MOVEMENT: {layer.name} keeping everything")
            else:
                # Find the threshold for keeping top-k weights by movement score
                flat_scores = movement_scores.flatten()
                threshold = np.partition(flat_scores, -num_to_keep)[-num_to_keep]
                
                # Create new gates (1 for keep, 0 for prune)
                new_gates = np.where(movement_scores >= threshold, 1.0, 0.0)
                
                # Handle ties at threshold (similar to the layer's method)
                num_at_threshold = np.sum(movement_scores == threshold)
                num_kept_above = np.sum(movement_scores > threshold)
                
                if num_kept_above < num_to_keep and num_at_threshold > 1:
                    remaining_to_keep = num_to_keep - num_kept_above
                    at_threshold_indices = np.where(movement_scores == threshold)
                    
                    total_at_threshold = len(at_threshold_indices[0])
                    keep_indices = np.random.choice(
                        total_at_threshold,
                        size=remaining_to_keep,
                        replace=False
                    )
                    
                    # Set all tied weights to 0 first
                    for i in range(total_at_threshold):
                        row_idx = at_threshold_indices[0][i]
                        col_idx = at_threshold_indices[1][i]
                        new_gates[row_idx, col_idx] = 0.0
                
                    # Then set selected ones to 1
                    for i in keep_indices:
                        row_idx = at_threshold_indices[0][i]
                        col_idx = at_threshold_indices[1][i]
                        new_gates[row_idx, col_idx] = 1.0
            
                print(f"DEBUG MOVEMENT: {layer.name} threshold: {threshold:.6f}")
        
            # Update gates
            old_active = np.sum(gates)
            layer.gates.assign(new_gates)
            new_active = np.sum(new_gates)
        
            print(f"DEBUG MOVEMENT: {layer.name} active weights: {old_active} -> {new_active}")
        
        elif target_sparsity > 0:
            # Fallback to random pruning if no movement scores available
            print(f"DEBUG MOVEMENT: {layer.name} no movement scores, using random pruning")
            weights = layer.kernel.numpy()
            gates = layer.gates.numpy()
        
            total_weights = weights.size
            num_to_prune = int(total_weights * target_sparsity)
        
            if num_to_prune > 0:
                # Random pruning as fallback
                flat_indices = np.arange(total_weights)
                prune_indices = np.random.choice(flat_indices, size=num_to_prune, replace=False)
                
                flat_gates = gates.flatten()
                flat_gates[prune_indices] = 0.0
                new_gates = flat_gates.reshape(gates.shape)
                
                layer.gates.assign(new_gates)
                print(f"DEBUG MOVEMENT: Randomly pruned {num_to_prune} weights in {layer.name}")
    
        # Calculate statistics
        gates = layer.gates.numpy()
        total_weights = gates.size
        active_weights = np.sum(gates)
        actual_sparsity = 1.0 - (active_weights / total_weights)
        
        return {
            'layer_name': layer.name,
            'layer_role': layer.role,
            'total_weights': total_weights,
            'active_weights': int(active_weights),
            'pruned_weights': int(total_weights - active_weights),
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'threshold': 0.0
        }


class SptPruning(StepBasedPruningMethod):
    """Structured Pruning with Training (SPT) - global reward-based pruning across all layers."""
    def __init__(self, config: GatedBertConfig, 
                 start_sparsity: float = 0.0,
                 end_sparsity: float = 0.5,
                 start_epoch: int = 0,
                 end_epoch: int = 10,
                 sparsity_steps: Optional[List[float]] = None):
        super().__init__(config, start_sparsity, end_sparsity, start_epoch, end_epoch, sparsity_steps)
    
    def get_pruning_schedule(self, epoch: int) -> float:
        """Get target sparsity based on schedule (step-based or linear)."""
        if epoch < self.start_epoch:
            return self.start_sparsity
        
        # Use step-based schedule if provided
        if self.sparsity_steps is not None:
            step_index = epoch - self.start_epoch
            if step_index < len(self.sparsity_steps):
                return self.sparsity_steps[step_index]
            else:
                # After all steps, maintain the last sparsity value
                return self.sparsity_steps[-1]
        
        # Fall back to linear schedule
        if epoch >= self.end_epoch:
            return self.end_sparsity
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            
            # Only linear schedule is supported for SPT now
            current_sparsity = self.start_sparsity + progress * (self.end_sparsity - self.start_sparsity)
            
            return current_sparsity
    
    def prune_globally(self, spt_layers: List[SptGatedDense], target_sparsity: float) -> Dict:
        """Prune weights globally across all SPT layers based on reward scores."""
        if not spt_layers:
            return {'error': 'No SPT layers provided'}
        
        print(f"DEBUG SPT Global: Starting global pruning with target sparsity {target_sparsity:.1%}")
        
        # Collect all weights and their reward scores across all layers
        all_weights_info = []  # List of (layer, flat_index, reward_score, current_gate_value)
        
        total_weights = 0
        total_active = 0
        
        for layer in spt_layers:
            if not hasattr(layer, 'reward_buffer') or not hasattr(layer, 'reward_count'):
                print(f"WARNING: Layer {layer.name} does not have reward_buffer, skipping.")
                continue
                
            # Get reward scores for this layer
            reward_buffer = layer.reward_buffer.numpy()
            reward_count = layer.reward_count.numpy()
            current_gates = layer.gates.numpy()
            
            # Calculate average reward scores (0.0 for unexplored weights)
            avg_reward = np.where(reward_count > 0, reward_buffer / reward_count, 0.0)
            
            # Flatten for global comparison
            flat_rewards = avg_reward.flatten()
            flat_gates = current_gates.flatten()
            
            layer_total = len(flat_gates)
            layer_active = np.sum(flat_gates > 0)
            
            total_weights += layer_total
            total_active += layer_active
            
            # Add this layer's weights to global list
            for i in range(len(flat_rewards)):
                all_weights_info.append((layer, i, flat_rewards[i], flat_gates[i]))
            
            print(f"DEBUG SPT Global: Layer {layer.name} ({getattr(layer, 'role', 'Unknown')}): "
                  f"{layer_active}/{layer_total} active, avg_reward_range=[{np.min(flat_rewards):.6f}, {np.max(flat_rewards):.6f}]")
        
        if total_weights == 0:
            return {'error': 'No weights found in SPT layers'}
        
        current_sparsity = 1.0 - (total_active / total_weights)
        print(f"DEBUG SPT Global: Current global sparsity: {current_sparsity:.1%}")
        
        # Calculate target number of active weights globally
        target_active = int(total_weights * (1.0 - target_sparsity))
        
        if total_active <= target_active:
            print(f"DEBUG SPT Global: Already at target sparsity, no pruning needed")
            
            # Still need to return layer_stats for compatibility
            layer_stats = []
            for layer in spt_layers:
                current_gates = layer.gates.numpy()
                layer_total = current_gates.size
                layer_active = np.sum(current_gates > 0)
                layer_sparsity = 1.0 - (layer_active / layer_total)
                
                layer_stats.append({
                    'layer_name': layer.name,
                    'layer_role': getattr(layer, 'role', 'Unknown'),
                    'total_weights': layer_total,
                    'active_weights': layer_active,
                    'pruned_weights': 0,
                    'actual_sparsity': layer_sparsity,
                    'importance_metric': 'reward'
                })
            
            return {
                'total_weights': total_weights,
                'total_active': total_active,
                'target_sparsity': target_sparsity,
                'actual_sparsity': current_sparsity,
                'weights_pruned': 0,
                'layer_stats': layer_stats
            }
        
        # Calculate how many weights to prune globally
        num_to_prune = total_active - target_active
        print(f"DEBUG SPT Global: Need to prune {num_to_prune} weights globally")
        
        # Filter to only currently active weights and sort by reward score (ascending = worst first)
        active_weights = [(layer, idx, reward, gate) for layer, idx, reward, gate in all_weights_info if gate > 0]
        active_weights.sort(key=lambda x: x[2])  # Sort by reward score (ascending)
        
        print(f"DEBUG SPT Global: Active weights range from {active_weights[0][2]:.6f} to {active_weights[-1][2]:.6f}")
        
        # Select the worst weights to prune
        weights_to_prune = active_weights[:num_to_prune]
        
        # Group by layer for efficient pruning
        layer_prune_indices = {}
        for layer, idx, reward, gate in weights_to_prune:
            if layer not in layer_prune_indices:
                layer_prune_indices[layer] = []
            layer_prune_indices[layer].append(idx)
        
        # Apply pruning to each layer
        layer_stats = []
        total_pruned = 0
        
        for layer in spt_layers:
            if layer in layer_prune_indices:
                indices_to_prune = layer_prune_indices[layer]
                
                # Get current gates
                current_gates = layer.gates.numpy()
                original_shape = current_gates.shape
                flat_gates = current_gates.flatten()
                
                # Prune the selected indices
                for idx in indices_to_prune:
                    flat_gates[idx] = 0.0
                
                # Reshape and update gates
                new_gates = flat_gates.reshape(original_shape)
                layer.gates.assign(new_gates)
                
                # Calculate layer statistics
                layer_total = current_gates.size
                layer_active_before = np.sum(current_gates > 0)
                layer_active_after = np.sum(new_gates > 0)
                layer_pruned = layer_active_before - layer_active_after
                layer_sparsity = 1.0 - (layer_active_after / layer_total)
                
                total_pruned += layer_pruned
                
                layer_stats.append({
                    'layer_name': layer.name,
                    'layer_role': getattr(layer, 'role', 'Unknown'),
                    'total_weights': layer_total,
                    'active_weights': layer_active_after,
                    'pruned_weights': layer_pruned,
                    'actual_sparsity': layer_sparsity,
                    'importance_metric': 'reward'
                })
                
                print(f"DEBUG SPT Global: Pruned {layer_pruned} weights from {layer.name} ({getattr(layer, 'role', 'Unknown')})")
            else:
                # No weights pruned from this layer
                current_gates = layer.gates.numpy()
                layer_total = current_gates.size
                layer_active = np.sum(current_gates > 0)
                layer_sparsity = 1.0 - (layer_active / layer_total)
                
                layer_stats.append({
                    'layer_name': layer.name,
                    'layer_role': getattr(layer, 'role', 'Unknown'),
                    'total_weights': layer_total,
                    'active_weights': layer_active,
                    'pruned_weights': 0,
                    'actual_sparsity': layer_sparsity,
                    'importance_metric': 'reward'
                })
        
        final_active = total_active - total_pruned
        final_sparsity = 1.0 - (final_active / total_weights)
        
        print(f"DEBUG SPT Global: Pruned {total_pruned} weights globally, final sparsity: {final_sparsity:.1%}")
        
        return {
            'total_weights': total_weights,
            'total_active': final_active,
            'target_sparsity': target_sparsity,
            'actual_sparsity': final_sparsity,
            'weights_pruned': total_pruned,
            'layer_stats': layer_stats
        }
    
    def prune_layer(self, layer: SptGatedDense, target_sparsity: float) -> Dict:
        """Legacy per-layer pruning method - now just returns current state without pruning."""
        # This method is kept for compatibility but doesn't actually prune
        # The real pruning is done by prune_globally()
        
        if not hasattr(layer, 'gates'):
            return {'error': 'Layer does not have gates'}
            
        current_gates = layer.gates.numpy()
        total_weights = current_gates.size
        active_weights = int(np.sum(current_gates > 0))
        actual_sparsity = 1.0 - (active_weights / total_weights)
        
        return {
            'layer_name': layer.name,
            'layer_role': getattr(layer, 'role', 'Unknown'),
            'total_weights': total_weights,
            'active_weights': active_weights,
            'pruned_weights': 0,  # No pruning done here
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'threshold': 0.0,
            'importance_metric': 'reward'
        }


class SparsePruningCallback(Callback):
    """Callback for sparse pruning at the end of each epoch."""
    
    def __init__(self, 
                 config: GatedBertConfig,
                 pruning_method: PruningMethod,
                 verbose: int = 1):
        super().__init__()
        self.config = config
        self.pruning_method = pruning_method
        self.verbose = verbose
        self.pruning_history = []
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Prune the model at the end of each epoch."""
        logs = logs or {}
        
        # Get target sparsity for this epoch
        target_sparsity = self.pruning_method.get_pruning_schedule(epoch)
        
        print(f"\nDEBUG: Pruning callback called for epoch {epoch + 1}")
        print(f"DEBUG: Target sparsity: {target_sparsity:.1%}")
        
        if self.verbose >= 1:
            print(f"\n=== Pruning at Epoch {epoch + 1} ===")
            print(f"Target sparsity: {target_sparsity:.1%}")
        
        # Find all GatedDense layers
        gated_layers = []
        
        # Check if model has base_model attribute (for MovementPruningModel)
        model_to_search = self.model
        if hasattr(self.model, 'base_model'):
            model_to_search = self.model.base_model
            print("DEBUG: Using base_model for layer search")
        
        for layer in model_to_search.layers:
            if isinstance(layer, (GatedDense, MovementGatedDense, SptGatedDense)):
                gated_layers.append(layer)
                print(f"DEBUG: Found top-level gated layer: {layer.name}")
    
        # Also check nested layers (for transformer blocks)
        def find_gated_layers(layer):
            layers = []
            if isinstance(layer, (GatedDense, MovementGatedDense, SptGatedDense)):
                layers.append(layer)
            elif hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    layers.extend(find_gated_layers(sublayer))
            elif hasattr(layer, '__dict__'):
                for attr_name, attr_value in layer.__dict__.items():
                    if isinstance(attr_value, tf.keras.layers.Layer):
                        layers.extend(find_gated_layers(attr_value))
            return layers
        
        all_gated_layers = []
        for layer in model_to_search.layers:
            found_layers = find_gated_layers(layer)
            all_gated_layers.extend(found_layers)
            if found_layers:
                print(f"DEBUG: Found {len(found_layers)} gated layers in {layer.name}")
    
        # Remove duplicates
        all_gated_layers = list(set(all_gated_layers))
    
        print(f"DEBUG: Total unique gated layers found: {len(all_gated_layers)}")
        for layer in all_gated_layers:
            print(f"  - {layer.name} ({type(layer).__name__})")
    
        if self.verbose >= 1:
            print(f"Found {len(all_gated_layers)} gated layers to prune")
        
        # Print SPT layer reward table before pruning (only for SPT layers)
        spt_layers = [layer for layer in all_gated_layers if isinstance(layer, SptGatedDense)]
        if spt_layers and self.verbose >= 1:
            self._print_spt_layer_reward_table(spt_layers, epoch)
        
        # Use global pruning for SPT layers if this is an SPT pruning method
        if isinstance(self.pruning_method, SptPruning) and spt_layers:
            print(f"DEBUG: Using global SPT pruning for {len(spt_layers)} SPT layers")
            
            # Do global pruning across all SPT layers
            global_result = self.pruning_method.prune_globally(spt_layers, target_sparsity)
            
            if 'error' in global_result:
                print(f"ERROR: Global SPT pruning failed: {global_result['error']}")
                return
            
            # Create epoch stats from global result
            epoch_stats = {
                'epoch': epoch + 1,
                'target_sparsity': target_sparsity,
                'layer_stats': global_result['layer_stats'],
                'total_weights': global_result['total_weights'],
                'total_active': global_result['total_active'],
                'overall_sparsity': global_result['actual_sparsity']
            }
            
            if self.verbose >= 1:
                print(f"Global SPT pruning completed:")
                print(f"  Total weights: {global_result['total_weights']}")
                print(f"  Weights pruned: {global_result['weights_pruned']}")
                print(f"  Final sparsity: {global_result['actual_sparsity']:.1%}")
                
            if self.verbose >= 2:
                for layer_stat in global_result['layer_stats']:
                    if layer_stat['pruned_weights'] > 0:
                        print(f"  {layer_stat['layer_name']} ({layer_stat['layer_role']}): "
                              f"pruned {layer_stat['pruned_weights']} weights, "
                              f"{layer_stat['active_weights']}/{layer_stat['total_weights']} "
                              f"({layer_stat['actual_sparsity']:.1%} sparse)")
        
        else:
            # Use traditional per-layer pruning for non-SPT layers
            print(f"DEBUG: Using per-layer pruning for {len(all_gated_layers)} layers")
            
            # Determine if per-layer-type SPT pruning is enabled
            spt_explore_per_layer = getattr(self.config, 'spt_explore_per_layer', False)
            if spt_explore_per_layer:
                # Set up layer types and cycling index
                if not hasattr(self, '_current_layer_type_index'):
                    self._current_layer_type_index = 0
                    self._layer_types = ['Query', 'Key', 'Value', 'Attention Output', 'FFN Hidden', 'FFN Output', 'Regressor', 'Classifier']
                current_layer_type = self._layer_types[self._current_layer_type_index]
                print(f"[SPT] Pruning only layer type: {current_layer_type}")
            
            # Prune each layer individually
            epoch_stats = {
                'epoch': epoch + 1,
                'target_sparsity': target_sparsity,
                'layer_stats': [],
                'total_weights': 0,
                'total_active': 0,
                'overall_sparsity': 0.0
            }
            
            for layer in all_gated_layers:
                # If per-layer-type SPT is enabled, only prune SptGatedDense layers of the current type
                if spt_explore_per_layer and isinstance(layer, SptGatedDense):
                    if getattr(layer, 'role', None) != current_layer_type:
                        continue  # Skip this layer
                layer_stats = self.pruning_method.prune_layer(layer, target_sparsity)
                epoch_stats['layer_stats'].append(layer_stats)
                epoch_stats['total_weights'] += layer_stats['total_weights']
                epoch_stats['total_active'] += layer_stats['active_weights']
                if self.verbose >= 2:
                    print(f"  {layer_stats['layer_name']} ({layer_stats['layer_role']}): "
                          f"{layer_stats['active_weights']}/{layer_stats['total_weights']} "
                          f"({layer_stats['actual_sparsity']:.1%} sparse)")
            
            # Increment the layer type index for next epoch if per-layer-type SPT is enabled
            if spt_explore_per_layer:
                self._current_layer_type_index = (self._current_layer_type_index + 1) % len(self._layer_types)

            # Calculate overall statistics
            if epoch_stats['total_weights'] > 0:
                epoch_stats['overall_sparsity'] = 1.0 - (epoch_stats['total_active'] / epoch_stats['total_weights'])

        # Store history
        self.pruning_history.append(epoch_stats)
        
        if self.verbose >= 1:
            print(f"Overall sparsity: {epoch_stats['overall_sparsity']:.1%} "
                  f"({epoch_stats['total_active']}/{epoch_stats['total_weights']})")
        
        # Log to training logs
        if logs is not None:
            logs['sparsity'] = epoch_stats['overall_sparsity']
            logs['active_weights'] = epoch_stats['total_active']
            logs['total_weights'] = epoch_stats['total_weights']
    
    def _print_spt_layer_reward_table(self, spt_layers, epoch):
        """Print a 2-line table with layer types as columns and average rewards."""
        # Define layer types in order
        layer_types = ['Query', 'Key', 'Value', 'Attention Output', 'FFN Hidden', 'FFN Output', 'Regressor', 'Classifier']
        
        # Calculate average rewards per layer type
        layer_type_rewards = {}
        for layer_type in layer_types:
            rewards = []
            for layer in spt_layers:
                if layer.role == layer_type:
                    # Calculate average reward for this layer
                    reward_buffer = layer.reward_buffer.numpy()
                    reward_count = layer.reward_count.numpy()
                    
                    # Only consider weights that have been explored (count > 0)
                    explored_mask = reward_count > 0
                    if np.any(explored_mask):
                        avg_rewards = reward_buffer[explored_mask] / reward_count[explored_mask]
                        rewards.extend(avg_rewards)
            
            if rewards:
                layer_type_rewards[layer_type] = np.mean(rewards)
            else:
                layer_type_rewards[layer_type] = 0.0
        
        # Print the table
        print(f"\n=== SPT Layer Rewards (Epoch {epoch + 1}) ===")
        
        # Header line
        header = "Layer Type:"
        for layer_type in layer_types:
            header += f" {layer_type:>12}"
        print(header)
        
        # Values line
        values = "Avg Reward: "
        for layer_type in layer_types:
            reward = layer_type_rewards[layer_type]
            values += f" {reward:>12.6f}"
        print(values)
        print("=" * (13 + 12 * len(layer_types)))
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Print pruning summary at the end of training."""
        if self.verbose >= 1:
            print(f"\n=== Pruning Summary ===")
            print("-" * 50)
            
            if self.pruning_history:
                final_stats = self.pruning_history[-1]
                print(f"Final overall sparsity: {final_stats['overall_sparsity']:.1%}")
                print(f"Final active weights: {final_stats['total_active']}/{final_stats['total_weights']}")
                
                # Print per-layer summary
                print("\nPer-layer final sparsity:")
                for layer_stat in final_stats['layer_stats']:
                    print(f"  {layer_stat['layer_name']} ({layer_stat['layer_role']}): "
                          f"{layer_stat['actual_sparsity']:.1%}")
            
            print("-" * 50)
    
    def get_pruning_history(self) -> List[Dict]:
        """Get the pruning history."""
        return self.pruning_history


# Utility functions for pruning
def create_magnitude_pruning_callback(config: GatedBertConfig,
                                    start_sparsity: float = 0.0,
                                    end_sparsity: float = 0.5,
                                    start_epoch: int = 0,
                                    end_epoch: int = None,
                                    sparsity_steps: Optional[List[float]] = None,
                                    verbose: int = 1) -> SparsePruningCallback:
    """Create a magnitude-based pruning callback."""
    if end_epoch is None:
        end_epoch = config.epochs
    
    # Use sparsity_steps from config if not provided
    if sparsity_steps is None:
        sparsity_steps = getattr(config, 'sparsity_steps', None)
    
    pruning_method = MagnitudePruning(
        config=config,
        start_sparsity=start_sparsity,
        end_sparsity=end_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        sparsity_steps=sparsity_steps
    )
    
    return SparsePruningCallback(
        config=config,
        pruning_method=pruning_method,
        verbose=verbose
    )


def create_movement_pruning_callback(config: GatedBertConfig,
                                   start_sparsity: float = 0.0,
                                   end_sparsity: float = 0.5,
                                   start_epoch: int = 0,
                                   end_epoch: int = None,
                                   sparsity_steps: Optional[List[float]] = None,
                                   verbose: int = 1) -> SparsePruningCallback:
    """Create a movement-based pruning callback."""
    if end_epoch is None:
        end_epoch = config.epochs
    
    # Use sparsity_steps from config if not provided
    if sparsity_steps is None:
        sparsity_steps = getattr(config, 'sparsity_steps', None)
    
    pruning_method = MovementPruning(
        config=config,
        start_sparsity=start_sparsity,
        end_sparsity=end_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        sparsity_steps=sparsity_steps
    )
    
    return SparsePruningCallback(
        config=config,
        pruning_method=pruning_method,
        verbose=verbose
    )


def create_spt_pruning_callback(config: GatedBertConfig,
                              start_sparsity: float = 0.0,
                              end_sparsity: float = 0.5,
                              start_epoch: int = 0,
                              end_epoch: int = None,
                              sparsity_steps: Optional[List[float]] = None,
                              verbose: int = 1) -> SparsePruningCallback:
    """Create a per-weight SPT pruning callback."""
    if end_epoch is None:
        end_epoch = config.epochs
    
    # Use sparsity_steps from config if not provided
    if sparsity_steps is None:
        sparsity_steps = getattr(config, 'sparsity_steps', None)
    
    pruning_method = SptPruning(
        config=config,
        start_sparsity=start_sparsity,
        end_sparsity=end_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        sparsity_steps=sparsity_steps
    )
    return SparsePruningCallback(
        config=config,
        pruning_method=pruning_method,
        verbose=verbose
    )


def save_pruning_history(pruning_history: List[Dict], filepath: str):
    """Save pruning history to JSON file."""
    import json
    try:
        with open(filepath, 'w') as f:
            json.dump(pruning_history, f, indent=2)
        print(f"Pruning history saved to {filepath}")
    except Exception as e:
        print(f"Failed to save pruning history: {e}")


def plot_pruning_history(pruning_history: List[Dict], save_path: str):
    """Plot pruning history."""
    try:
        import matplotlib.pyplot as plt
        
        epochs = [stat['epoch'] for stat in pruning_history]
        sparsities = [stat['overall_sparsity'] for stat in pruning_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, sparsities, 'b-', linewidth=2, marker='o')
        plt.title('Pruning Progress', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Sparsity', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Pruning plot saved to {save_path}")
        
    except ImportError:
        print("matplotlib not available, skipping pruning plot")
    except Exception as e:
        print(f"Failed to create pruning plot: {e}")


class MovementPruningModel(Model):
    """Custom model for movement pruning that overrides train_step."""
    
    def __init__(self, base_model, config: GatedBertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.config = config
        self.step_counter = 0
        self.movement_layers = []
        self.total_steps = 0
        self.steps_per_epoch = 1000  # Default, will be updated
        
        # Find all MovementGatedDense layers
        self._find_movement_layers()
        
        # Debug: Print pruning configuration
        print(f"DEBUG MovementPruning: start_sparsity={config.start_sparsity}")
        print(f"DEBUG MovementPruning: end_sparsity={config.end_sparsity}")
        print(f"DEBUG MovementPruning: pruning_start_epoch={config.pruning_start_epoch}")
        print(f"DEBUG MovementPruning: pruning_end_epoch={config.pruning_end_epoch}")
        print(f"DEBUG MovementPruning: frequency_steps={config.movement_pruning_frequency_steps}")
    
    def _find_movement_layers(self):
        """Find all MovementGatedDense layers in the model."""
        def find_layers(layer):
            layers = []
            if isinstance(layer, MovementGatedDense):
                layers.append(layer)
            elif hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    layers.extend(find_layers(sublayer))
            elif hasattr(layer, '__dict__'):
                for attr_name, attr_value in layer.__dict__.items():
                    if isinstance(attr_value, tf.keras.layers.Layer):
                        layers.extend(find_layers(attr_value))
            return layers
        
        self.movement_layers = find_layers(self.base_model)
        print(f"Found {len(self.movement_layers)} MovementGatedDense layers")
    
    def call(self, inputs, training=None, **kwargs):
        return self.base_model(inputs, training=training, **kwargs)
    
    @property
    def layers(self):
        """Expose base model layers."""
        return self.base_model.layers
    
    def _should_prune_this_step(self):
        """Check if we should prune on this training step."""
        if not self.config.enable_pruning or self.config.pruning_method != 'movement':
            return False
        
        # Use step-based logic, but make it more reasonable
        current_epoch = self.step_counter // self.steps_per_epoch
        
        # Only prune during the pruning window
        if current_epoch < self.config.pruning_start_epoch:
            return False
        if current_epoch >= self.config.pruning_end_epoch:
            return False
        
        # Prune every N steps during the pruning window
        frequency = getattr(self.config, 'movement_pruning_frequency_steps', 100)
        should_prune = self.step_counter % frequency == 0
        
        # Debug output
        if should_prune:
            print(f"DEBUG: Should prune at step {self.step_counter} (epoch ~{current_epoch})")
        
        return should_prune
    
    def _calculate_target_sparsity(self):
        """Calculate target sparsity for current step."""
        current_epoch = self.step_counter // self.steps_per_epoch
        
        if current_epoch < self.config.pruning_start_epoch:
            return self.config.start_sparsity
        elif current_epoch >= self.config.pruning_end_epoch:
            return self.config.end_sparsity
        else:
            progress = (current_epoch - self.config.pruning_start_epoch) / (self.config.pruning_end_epoch - self.config.pruning_start_epoch)
            
            if getattr(self.config, 'movement_schedule', 'linear') == 'cubic':
                return self.config.end_sparsity + (self.config.start_sparsity - self.config.end_sparsity) * ((1.0 - progress) ** 3)
            else:  # linear
                return self.config.start_sparsity + progress * (self.config.end_sparsity - self.config.start_sparsity)
    
    def train_step(self, data):
        x, y = data
        self.step_counter += 1
        self.total_steps += 1
        
        # DEBUG: Print step counter every 25 steps for immediate feedback
        if self.step_counter % 25 == 0:
            print(f"DEBUG TRAIN_STEP: Step {self.step_counter}")
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)
        
        # Get gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update movement scores and prune if needed
        if self.config.enable_pruning and self.config.pruning_method == 'movement':
            var_to_grad = {var.ref(): grad for var, grad in zip(trainable_vars, gradients) if grad is not None}
            
            # Always update movement scores
            movement_updates = 0
            for layer in self.movement_layers:
                if layer.kernel.ref() in var_to_grad:
                    layer.update_movement_scores(var_to_grad[layer.kernel.ref()])
                    movement_updates += 1
        
            # Debug: Print movement updates occasionally
            if self.step_counter % 25 == 0:
                print(f"DEBUG TRAIN_STEP: Updated movement scores for {movement_updates}/{len(self.movement_layers)} layers")
        
            # Check if we should prune this step
            should_prune = self._should_prune_this_step()
            if should_prune:
                target_sparsity = self._calculate_target_sparsity()
                print(f"DEBUG TRAIN_STEP: PRUNING at step {self.step_counter}, target sparsity: {target_sparsity:.1%}")
                
                pruned_layers = 0
                for layer in self.movement_layers:
                    old_gates = tf.reduce_sum(layer.gates).numpy()
                    layer.update_gates_from_movement(target_sparsity)
                    new_gates = tf.reduce_sum(layer.gates).numpy()
                    
                    if old_gates != new_gates:
                        pruned_layers += 1
                        print(f"DEBUG TRAIN_STEP: {layer.name}: {old_gates} -> {new_gates} active gates")
                
                print(f"DEBUG TRAIN_STEP: Pruned {pruned_layers}/{len(self.movement_layers)} layers")
    
        # Mask gradients (zero out gradients for pruned weights)
        masked_gradients = []
        var_to_gate = {layer.kernel.ref(): layer.gates for layer in self.movement_layers}
        
        for var, grad in zip(trainable_vars, gradients):
            if grad is None:
                masked_gradients.append(None)
            elif var.ref() in var_to_gate:
                # Mask the gradient
                masked_gradients.append(grad * var_to_gate[var.ref()])
            else:
                masked_gradients.append(grad)
        
        # Apply masked gradients
        self.optimizer.apply_gradients(zip(masked_gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, predictions)
        return {m.name: m.result() for m in self.metrics}