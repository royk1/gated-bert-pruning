#!/usr/bin/env python3
"""
Plot comparison script for pruning methods results.
Reads CSV files from results directories and creates comparison plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def read_results_csv(directory_path):
    """Read the results CSV file from a directory."""
    csv_files = list(Path(directory_path).glob('results_*.csv'))
    if not csv_files:
        print(f"Warning: No results CSV found in {directory_path}")
        return None
    
    # Use the first CSV file found
    csv_file = csv_files[0]
    print(f"Reading {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def create_dual_axis_plots():
    """Create 4 plots with dual y-axes - left for metrics, right for sparsity."""
    
    # Define the results directories and their colors
    #methods = {
    #    'baseline': {'dir': 'results_baseline_sst2', 'color': 'black', 'label': 'Baseline'},
    #    'magnitude': {'dir': 'results_magnitude_sst2', 'color': 'blue', 'label': 'Magnitude'},
    #    'movement': {'dir': 'results_movement_sst2', 'color': 'red', 'label': 'Movement'},
    #    'spt': {'dir': 'results_spt_sst2', 'color': 'green', 'label': 'SPT'}
    #}
    methods = {
        #'baseline': {'dir': 'results_baseline_mnli', 'color': 'black', 'label': 'Baseline'},
        'magnitude': {'dir': 'results_magnitude_mnli', 'color': 'blue', 'label': 'Magnitude'},
        'movement': {'dir': 'results_movement_mnli', 'color': 'red', 'label': 'Movement'},
        'spt': {'dir': 'results_spt_mnli', 'color': 'green', 'label': 'SPT'}
    }
    
    # Read data from all methods
    data = {}
    for method_name, method_info in methods.items():
        if os.path.exists(method_info['dir']):
            df = read_results_csv(method_info['dir'])
            if df is not None:
                data[method_name] = df
                print(f"✓ Loaded {method_name}: {len(df)} epochs")
            else:
                print(f"✗ Failed to load {method_name}")
        else:
            print(f"✗ Directory not found: {method_info['dir']}")
    
    if not data:
        print("No data found! Please ensure the results directories exist.")
        return
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pruning Methods Comparison - SST-2 Dataset', fontsize=16, fontweight='bold')
    
    # Plot 1: Validation Accuracy (left) vs Sparsity (right)
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    for method_name, df in data.items():
        method_info = methods[method_name]
        # Plot accuracy on left y-axis
        line1 = ax1.plot(df['epoch'], df['val_accuracy'], 
                        color=method_info['color'], label=method_info['label'], 
                        linewidth=2, marker='o', markersize=4)
        # Plot sparsity on right y-axis (same color, dashed)
        line2 = ax1_twin.plot(df['epoch'], df['sparsity_end'] * 100, 
                             color=method_info['color'], linestyle='--', alpha=0.7,
                             linewidth=1.5, marker='s', markersize=3)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy', color='black')
    ax1_twin.set_ylabel('Sparsity (%)', color='gray')
    ax1.set_title('Validation Accuracy vs Sparsity')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Set y-axis limits
    ax1.set_ylim(0, 1)
    ax1_twin.set_ylim(0, 100)
    
    # Plot 2: Validation Loss (left) vs Sparsity (right)
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    for method_name, df in data.items():
        method_info = methods[method_name]
        # Plot loss on left y-axis
        line1 = ax2.plot(df['epoch'], df['val_loss'], 
                        color=method_info['color'], label=method_info['label'], 
                        linewidth=2, marker='s', markersize=4)
        # Plot sparsity on right y-axis (same color, dashed)
        line2 = ax2_twin.plot(df['epoch'], df['sparsity_end'] * 100, 
                             color=method_info['color'], linestyle='--', alpha=0.7,
                             linewidth=1.5, marker='o', markersize=3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss', color='black')
    ax2_twin.set_ylabel('Sparsity (%)', color='gray')
    ax2.set_title('Validation Loss vs Sparsity')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Set y-axis limits
    ax2_twin.set_ylim(0, 100)
    
    # Plot 3: Training Accuracy (left) vs Sparsity (right)
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    for method_name, df in data.items():
        method_info = methods[method_name]
        # Plot accuracy on left y-axis
        line1 = ax3.plot(df['epoch'], df['train_accuracy'], 
                        color=method_info['color'], label=method_info['label'], 
                        linewidth=2, marker='^', markersize=4)
        # Plot sparsity on right y-axis (same color, dashed)
        line2 = ax3_twin.plot(df['epoch'], df['sparsity_end'] * 100, 
                             color=method_info['color'], linestyle='--', alpha=0.7,
                             linewidth=1.5, marker='d', markersize=3)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Accuracy', color='black')
    ax3_twin.set_ylabel('Sparsity (%)', color='gray')
    ax3.set_title('Training Accuracy vs Sparsity')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # Set y-axis limits
    ax3.set_ylim(0, 1)
    ax3_twin.set_ylim(0, 100)
    
    # Plot 4: Training Loss (left) vs Sparsity (right)
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    for method_name, df in data.items():
        method_info = methods[method_name]
        # Plot loss on left y-axis
        line1 = ax4.plot(df['epoch'], df['train_loss'], 
                        color=method_info['color'], label=method_info['label'], 
                        linewidth=2, marker='d', markersize=4)
        # Plot sparsity on right y-axis (same color, dashed)
        line2 = ax4_twin.plot(df['epoch'], df['sparsity_end'] * 100, 
                             color=method_info['color'], linestyle='--', alpha=0.7,
                             linewidth=1.5, marker='^', markersize=3)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training Loss', color='black')
    ax4_twin.set_ylabel('Sparsity (%)', color='gray')
    ax4.set_title('Training Loss vs Sparsity')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    # Set y-axis limits
    ax4_twin.set_ylim(0, 100)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = 'pruning_methods_comparison_dual_axis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Dual-axis comparison plot saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for method_name, df in data.items():
        method_info = methods[method_name]
        print(f"\n{method_info['label']} ({method_name}):")
        print(f"  Final Validation Accuracy: {df['val_accuracy'].iloc[-1]:.4f}")
        print(f"  Final Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
        print(f"  Final Training Accuracy: {df['train_accuracy'].iloc[-1]:.4f}")
        print(f"  Final Training Loss: {df['train_loss'].iloc[-1]:.4f}")
        print(f"  Final Sparsity: {df['sparsity_end'].iloc[-1]:.1f}%")
        print(f"  Epochs: {len(df)}")

if __name__ == "__main__":
    print("="*60)
    print("PRUNING METHODS COMPARISON PLOTTER")
    print("="*60)
    
    # Create dual-axis comparison plots
    print("\nCreating dual-axis comparison plots...")
    create_dual_axis_plots()
    
    print("\n" + "="*60)
    print("PLOTTING COMPLETE!")
    print("="*60) 