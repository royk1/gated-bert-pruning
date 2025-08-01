#!/bin/bash

# =============================================================================
# Automated Pruning Experiments Script
# =============================================================================
# This script runs training experiments across different pruning methods
# with configurable parameters and automatic results organization.
#
# Usage: ./run_experiments.sh
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# ERROR HANDLING AND CLEANUP
# =============================================================================

# Global flag to track if config is backed up
CONFIG_BACKED_UP=false

# Function to restore config on any exit (success, error, or interruption)
cleanup_and_restore() {
    local exit_code=$?
    
    if [ "$CONFIG_BACKED_UP" = true ]; then
        print_warning "Script interrupted or failed. Restoring original config.py..."
        restore_config
        print_success "Original config.py has been restored."
    else
        # Even if no backup was made, clean up any leftover files
        print_warning "Cleaning up any leftover files..."
        rm -f config.py.backup
        rm -f config.py.backup_*
        rm -f config.py.modified_*
        rm -f config.py.tmp
        rm -f update_config.py
        print_success "Cleanup completed."
    fi
    
    if [ $exit_code -ne 0 ]; then
        print_error "Script failed with exit code $exit_code"
        print_info "Check the experiment logs for details"
    fi
    
    exit $exit_code
}

# Set up traps to ensure cleanup happens on any exit
trap cleanup_and_restore EXIT      # Normal exit
trap cleanup_and_restore SIGINT    # Ctrl+C
trap cleanup_and_restore SIGTERM   # Termination signal
trap cleanup_and_restore ERR       # Any error (when set -e is active)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Training parameters
EPOCHS=25
BATCH_SIZE=96
LEARNING_RATE="2e-5"

# Pruning parameters  
START_SPARSITY=0.0
END_SPARSITY=0.99
PRUNING_START_EPOCH=4       # Epoch to start pruning
PRUNING_END_EPOCH=23         # Epoch to reach target sparsity (will be capped at EPOCHS)

# NOTE: PRUNING_END_EPOCH will be automatically limited to not exceed EPOCHS
# This prevents training restart issues when pruning_end_epoch > total epochs

# Step-based pruning schedule (alternative to linear start_sparsity -> end_sparsity)
# If provided, this overrides the linear schedule
# Format: space-separated list of sparsity percentages (0-100)
# Default: "10 25 50 70 75 80 84 88 90 92 93 94 95 96 97 97.5 98 98.5 99 99.5"
# Leave empty to use linear schedule
SPARSITY_STEPS="10 25 50 70 75 80 84 88 90 92 93 94 95 96 97 97.5 98 98.5 99 99.5"

# SPT per-layer exploration (cyclic exploration of layer types)
# If true, each batch explores only one layer type in cyclic manner
# This creates competition between layer types and allows more sensitive layers to be pruned less
SPT_EXPLORE_PER_LAYER="False"  # Python boolean literal

# Pruning methods to test
PRUNING_METHODS=("magnitude" "movement" "spt" "baseline")
#PRUNING_METHODS=("spt" "baseline")

# Task configuration
#TASK="mnli"
TASK="sst2"
MAX_LEN=128

# Experiment metadata
EXPERIMENT_NAME="pruning_comparison_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="experiment_logs"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to clean up any leftover files from previous runs
cleanup_previous_runs() {
    print_info "Cleaning up any leftover files from previous runs..."
    rm -f config.py.backup
    rm -f config.py.backup_*
    rm -f config.py.modified_*
    rm -f config.py.tmp
    rm -f update_config.py
    print_success "Previous run cleanup completed"
}

# Function to backup original config
backup_config() {
    if [ -f "config.py" ]; then
        # Create backup with timestamp for extra safety
        local timestamp=$(date +%Y%m%d_%H%M%S)
        cp config.py config.py.backup
        cp config.py "config.py.backup_${timestamp}"
        print_info "Backed up original config.py to config.py.backup"
        print_info "Extra backup created: config.py.backup_${timestamp}"
        CONFIG_BACKED_UP=true
    else
        print_error "config.py not found! Cannot create backup."
        exit 1
    fi
}

# Function to restore original config
restore_config() {
    if [ -f "config.py.backup" ]; then
        if [ -f "config.py" ]; then
            # Save current modified config for debugging if needed
            local timestamp=$(date +%Y%m%d_%H%M%S)
            cp config.py "config.py.modified_${timestamp}"
            print_info "Saved modified config as config.py.modified_${timestamp}"
        fi
        
        mv config.py.backup config.py
        print_success "Restored original config.py"
        CONFIG_BACKED_UP=false
        
        # Clean up ALL backup and temporary files (including from previous runs)
        print_info "Cleaning up ALL backup and temporary files..."
        rm -f config.py.backup
        rm -f config.py.backup_*
        rm -f config.py.modified_*
        rm -f config.py.tmp
        rm -f update_config.py
        print_success "All backup and temporary files cleaned up"
    else
        print_warning "No backup file found (config.py.backup)"
    fi
}

# Enhanced function to safely update config.py with error handling
update_config() {
    local method="$1"
    local enable_pruning="$2"
    
    print_info "Updating config.py for method: $method"
    
    # Verify backup exists before modifying
    if [ ! -f "config.py.backup" ]; then
        print_error "No config backup found! This is unsafe. Aborting."
        exit 1
    fi
    
    # Create a temporary Python script to update config
    cat > update_config.py << EOFPYTHON
import re
import sys
import os

try:
    # Read the current config file
    if not os.path.exists('config.py'):
        print("ERROR: config.py not found!")
        sys.exit(1)
        
    with open('config.py', 'r') as f:
        content = f.read()

    # Update parameters with validation
    content = re.sub(r"self\.epochs = kwargs\.get\('epochs', \d+\)", 
                    f"self.epochs = kwargs.get('epochs', ${EPOCHS})", content)
    content = re.sub(r"self\.batch_size = kwargs\.get\('batch_size', \d+\)", 
                    f"self.batch_size = kwargs.get('batch_size', ${BATCH_SIZE})", content)
    content = re.sub(r"self\.learning_rate = kwargs\.get\('learning_rate', [^)]+\)", 
                    f"self.learning_rate = kwargs.get('learning_rate', ${LEARNING_RATE})", content)
    content = re.sub(r"self\.task = kwargs\.get\('task', '[^']+'\)", 
                    f"self.task = kwargs.get('task', '${TASK}')", content)
    content = re.sub(r"self\.max_len = kwargs\.get\('max_len', \d+\)", 
                    f"self.max_len = kwargs.get('max_len', ${MAX_LEN})", content)

    # Update pruning parameters
    content = re.sub(r"self\.enable_pruning = kwargs\.get\('enable_pruning', [^)]+\)", 
                    f"self.enable_pruning = kwargs.get('enable_pruning', ${enable_pruning})", content)
    content = re.sub(r"self\.pruning_method = kwargs\.get\('pruning_method', '[^']+'\)", 
                    f"self.pruning_method = kwargs.get('pruning_method', '${method}')", content)
    content = re.sub(r"self\.start_sparsity = kwargs\.get\('start_sparsity', [^)]+\)", 
                    f"self.start_sparsity = kwargs.get('start_sparsity', ${START_SPARSITY})", content)
    content = re.sub(r"self\.end_sparsity = kwargs\.get\('end_sparsity', [^)]+\)", 
                    f"self.end_sparsity = kwargs.get('end_sparsity', ${END_SPARSITY})", content)
    content = re.sub(r"self\.pruning_start_epoch = kwargs\.get\('pruning_start_epoch', \d+\)", 
                    f"self.pruning_start_epoch = kwargs.get('pruning_start_epoch', ${PRUNING_START_EPOCH})", content)

    # CRITICAL FIX: Ensure pruning_end_epoch doesn't exceed total epochs
    pruning_end_epoch = min(${PRUNING_END_EPOCH}, ${EPOCHS})
    content = re.sub(r"self\.pruning_end_epoch = kwargs\.get\('pruning_end_epoch', \d+\)", 
                    f"self.pruning_end_epoch = kwargs.get('pruning_end_epoch', {pruning_end_epoch})", content)

    # Update sparsity_steps parameter
    if "${SPARSITY_STEPS}" != "":
        # Convert space-separated string to comma-separated Python list
        steps_list="[$(echo "${SPARSITY_STEPS}" | sed 's/ /,/g')]"
        content = re.sub(r"self\.sparsity_steps = kwargs\.get\('sparsity_steps', [^)]+\)", 
                        f"self.sparsity_steps = kwargs.get('sparsity_steps', {steps_list})", content)
    else:
        # Keep default None value
        content = re.sub(r"self\.sparsity_steps = kwargs\.get\('sparsity_steps', [^)]+\)", 
                        f"self.sparsity_steps = kwargs.get('sparsity_steps', None)", content)

    # Update SPT per-layer exploration parameter
    content = re.sub(r"self\.spt_explore_per_layer = kwargs\.get\('spt_explore_per_layer', [^)]+\)", 
                    f"self.spt_explore_per_layer = kwargs.get('spt_explore_per_layer', ${SPT_EXPLORE_PER_LAYER})", content)

    # Write the updated config with atomic operation
    temp_file = 'config.py.tmp'
    with open(temp_file, 'w') as f:
        f.write(content)
    
    # Atomic move to prevent corruption
    os.rename(temp_file, 'config.py')

    print(f"✅ Config updated for method: ${method} (pruning: ${enable_pruning})")
    print(f"✅ Epochs: ${EPOCHS}, Pruning end epoch: {pruning_end_epoch}")

except Exception as e:
    print(f"ERROR updating config: {e}")
    sys.exit(1)
EOFPYTHON

    # Run the Python script with error handling
    if python update_config.py; then
        rm update_config.py
        print_success "Config updated successfully"
    else
        rm -f update_config.py
        print_error "Failed to update config.py"
        return 1
    fi
}

# Function to run training for a specific method
run_training() {
    local method=$1
    local enable_pruning=$2
    local results_suffix=${3:-$method}  # Use method name as default if not provided
    
    print_header "TRAINING: $method"
    
    # Update configuration
    update_config "$method" "$enable_pruning"
    
    # Create method-specific log file using results_suffix for consistency
    local log_file="${LOG_DIR}/${results_suffix}_training_$(date +%Y%m%d_%H%M%S).log"
    local results_dir="results_${results_suffix}_${TASK}"
    
    print_info "Starting training with method: $method"
    print_info "Logs will be saved to: $log_file"
    print_info "Results will be organized in: ${results_dir}/"
    
    # Run training with logging
    if python train.py 2>&1 | tee "$log_file"; then
        print_success "Training completed successfully for method: $method"
        
        # Check if results directory was created
        if [ -d "$results_dir" ]; then
            local result_count=$(ls -1 "$results_dir/" | wc -l)
            print_success "Results saved: $result_count files in $results_dir/"
        else
            print_warning "Results directory $results_dir/ not found"
        fi
    else
        print_error "Training failed for method: $method"
        return 1
    fi
}

# Function to create experiment summary
create_summary() {
    local summary_file="${LOG_DIR}/experiment_summary_$(date +%Y%m%d_%H%M%S).md"
    
    print_info "Creating experiment summary: $summary_file"
    
    cat > "$summary_file" << EOF
# Pruning Methods Comparison Experiment

**Experiment Name:** $EXPERIMENT_NAME  
**Date:** $(date)  
**Host:** $(hostname)  

## Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | $EPOCHS |
| Batch Size | $BATCH_SIZE |
| Learning Rate | $LEARNING_RATE |
| Task | $TASK |
| Max Length | $MAX_LEN |
| Start Sparsity | $START_SPARSITY |
| End Sparsity | $END_SPARSITY |
| Pruning Start Epoch | $PRUNING_START_EPOCH |
| Pruning End Epoch | $PRUNING_END_EPOCH |
| Sparsity Steps | ${SPARSITY_STEPS:-"Linear schedule"} |
| SPT Per-Layer Exploration | ${SPT_EXPLORE_PER_LAYER} |

## Methods Tested

EOF

    for method in "${PRUNING_METHODS[@]}"; do
        echo "- $method" >> "$summary_file"
    done
    
    cat >> "$summary_file" << EOF

## Results Structure

Each method creates its own results directory:

EOF

    for method in "${PRUNING_METHODS[@]}"; do
        local results_dir="results_${method}_${TASK}"
        if [ -d "$results_dir" ]; then
            echo "### $results_dir/" >> "$summary_file"
            echo "\`\`\`" >> "$summary_file"
            ls -la "$results_dir/" >> "$summary_file"
            echo "\`\`\`" >> "$summary_file"
            echo "" >> "$summary_file"
        fi
    done
    
    cat >> "$summary_file" << EOF

## Log Files

All training logs are saved in: \`${LOG_DIR}/\`

## Usage

To analyze results, check the CSV files and plots in each \`results_*\` directory.
Each directory contains:
- Training metrics CSV
- Loss/accuracy plots  
- Model weights
- Configuration files
- Metadata

EOF

    print_success "Experiment summary created: $summary_file"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    print_header "AUTOMATED PRUNING EXPERIMENTS"
    
    print_info "Experiment: $EXPERIMENT_NAME"
    print_info "Methods to test: ${PRUNING_METHODS[*]}"
    print_info "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE"
    
    # Clean up any leftover files from previous runs
    cleanup_previous_runs
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Backup original configuration (this sets CONFIG_BACKED_UP=true)
    print_info "Creating backup of original config.py..."
    backup_config
    
    # Track successful/failed runs
    local successful_runs=()
    local failed_runs=()
    local experiment_failed=false
    
    # Run experiments for each method
    for method in "${PRUNING_METHODS[@]}"; do
        print_header "EXPERIMENT: $method"
        
        # Determine if pruning should be enabled and the actual method to use
        local enable_pruning="True"
        local actual_method="$method"
        local results_suffix="$method"
        local enable_pruning="True"  # Default to True for pruning methods
        
        if [ "$method" = "baseline" ]; then
            enable_pruning="False"
            actual_method="none"  # Use none as the underlying method for baseline
            results_suffix="baseline"  # But keep baseline for results directory
        fi
        
        print_info "Method: $method, Actual Method: $actual_method, Pruning Enabled: $enable_pruning"
        
        # Run training with error handling
        if run_training "$actual_method" "$enable_pruning" "$results_suffix"; then
            successful_runs+=("$method")
            print_success "✅ $method completed successfully"
        else
            failed_runs+=("$method")
            print_error "❌ $method failed"
            experiment_failed=true
            
            # Continue with other methods even if one fails
            print_warning "Continuing with remaining methods..."
        fi
        
        print_info "Waiting 5 seconds before next experiment..."
        sleep 5
    done
    
    # Create experiment summary (even if some experiments failed)
    print_info "Creating experiment summary..."
    create_summary
    
    # Final report
    print_header "EXPERIMENT COMPLETE"
    
    if [ ${#successful_runs[@]} -gt 0 ]; then
        print_success "Successful runs (${#successful_runs[@]}): ${successful_runs[*]}"
    fi
    
    if [ ${#failed_runs[@]} -gt 0 ]; then
        print_error "Failed runs (${#failed_runs[@]}): ${failed_runs[*]}"
    fi
    
    print_info "Total experiments: ${#PRUNING_METHODS[@]}"
    print_info "Success rate: $((${#successful_runs[@]} * 100 / ${#PRUNING_METHODS[@]}))%"
    
    # Show results directories
    print_info "\nResults directories created:"
    for method in "${PRUNING_METHODS[@]}"; do
        local display_method=$method
        local results_dir="results_${display_method}_${TASK}"
        if [ -d "$results_dir" ]; then
            local file_count=$(ls -1 "$results_dir/" | wc -l)
            print_success "  $results_dir/  ($file_count files)"
        else
            print_warning "  $results_dir/  (not found)"
        fi
    done
    
    print_header "EXPERIMENT SUMMARY"
    print_info "Check the results_* directories for detailed results"
    print_info "Check $LOG_DIR for training logs and experiment summary"
    
    # Manual restore before exit (trap will also restore, but this is explicit)
    print_info "Restoring original configuration..."
    restore_config
    
    if [ "$experiment_failed" = true ]; then
        print_warning "Some experiments failed. Check the logs for details."
        print_success "However, successful experiments have been completed and results saved."
    else
        print_success "All experiments completed successfully!"
    fi
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "train.py" ]; then
    print_error "train.py not found in current directory"
    exit 1
fi

if [ ! -f "config.py" ]; then
    print_error "config.py not found in current directory"
    exit 1
fi

# Make sure script is executable
chmod +x "$0"

# Run main function
main "$@" 