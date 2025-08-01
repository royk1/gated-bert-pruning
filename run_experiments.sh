#!/bin/bash

# =============================================================================
# Automated Pruning Experiments Script
# =============================================================================
# This script runs training experiments across different pruning methods
# with configurable parameters and automatic results organization.
#
# Usage: ./run_experiments.sh [OPTIONS]
# =============================================================================

set -e  # Exit on any error

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

# =============================================================================
# COMMAND LINE PARAMETER HANDLING
# =============================================================================

# Default values (will be overridden by command line arguments)
EPOCHS=25
BATCH_SIZE=96
LEARNING_RATE="2e-5"
START_SPARSITY=0.0
END_SPARSITY=0.99
PRUNING_START_EPOCH=4
PRUNING_END_EPOCH=23
SPARSITY_STEPS="10 25 50 70 75 80 84 88 90 92 93 94 95 96 97 97.5 98 98.5 99 99.5"
SPT_EXPLORE_PER_LAYER="False"
PRUNING_METHODS=("magnitude" "movement" "spt" "baseline")
TASK="sst2"
MAX_LEN=128
MODEL_SIZE="base"
PARALLEL_MODE=false
MAX_PARALLEL_JOBS=4

# Help function
show_help() {
    echo -e "\n${BLUE}Gated BERT Pruning Experiments Script${NC}"
    echo -e "${BLUE}============================================${NC}\n"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help                    Show this help message and exit"
    echo ""
    echo "  Training Parameters:"
    echo "  -e, --epochs NUM              Number of training epochs (default: 25)"
    echo "  -b, --batch-size NUM          Batch size (default: 96)"
    echo "  -l, --learning-rate RATE      Learning rate (default: 2e-5)"
    echo "  -m, --model-size SIZE         Model size: tiny, small, base, large (default: base)"
    echo ""
    echo "  Pruning Parameters:"
    echo "  -p, --pruning-methods LIST    Comma-separated list of methods to test"
    echo "                                Options: baseline, magnitude, movement, spt"
    echo "                                (default: magnitude,movement,spt,baseline)"
    echo "  -s, --start-sparsity NUM      Starting sparsity (0.0-1.0, default: 0.0)"
    echo "  -f, --end-sparsity NUM        Final sparsity (0.0-1.0, default: 0.99)"
    echo "  -ps, --pruning-start NUM      Epoch to start pruning (default: 4)"
    echo "  -pe, --pruning-end NUM        Epoch to end pruning (default: 23)"
    echo "  -ss, --sparsity-steps LIST    Space-separated sparsity steps (default: auto)"
    echo "  -se, --spt-explore BOOL       SPT per-layer exploration: true/false (default: false)"
    echo ""
    echo "  Task Configuration:"
    echo "  -t, --task TASK               Task: sst2, mnli, qnli, rte, wnli, mrpc, cola, stsb (default: sst2)"
    echo "  -ml, --max-len NUM            Maximum sequence length (default: 128)"
    echo ""
    echo "  Parallel Execution:"
    echo "  -parallel                     Enable parallel execution of methods"
    echo "  -max-parallel-jobs NUM        Maximum parallel jobs (default: 4)"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Run with all defaults"
    echo "  $0 -h                                 # Show this help"
    echo "  $0 -e 10 -b 64                       # Quick test with 10 epochs, batch size 64"
    echo "  $0 -p baseline,spt -t mnli           # Test only baseline and SPT on MNLI"
    echo "  $0 -m tiny -e 5 -p baseline          # Quick baseline test with tiny model"
    echo "  $0 -s 0.1 -f 0.8 -ps 2 -pe 15        # Custom sparsity schedule"
    echo "  $0 -parallel                          # Run all methods in parallel"
    echo "  $0 -parallel -max-parallel-jobs 2     # Run with max 2 parallel jobs"
    echo ""
    echo "NOTES:"
    echo "  - If no parameters are provided, the script runs with current defaults"
    echo "  - Pruning end epoch will be automatically capped to not exceed total epochs"
    echo "  - Results are saved in results_{method}_{task}/ directories"
    echo "  - Logs are saved in experiment_logs/ directory"
    echo "  - Parallel mode runs methods simultaneously (faster but uses more resources)"
    echo "  - Each parallel job gets its own log file and terminal output"
    echo ""
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -e|--epochs)
                EPOCHS="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -l|--learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            -m|--model-size)
                MODEL_SIZE="$2"
                shift 2
                ;;
            -p|--pruning-methods)
                IFS=',' read -ra PRUNING_METHODS <<< "$2"
                shift 2
                ;;
            -s|--start-sparsity)
                START_SPARSITY="$2"
                shift 2
                ;;
            -f|--end-sparsity)
                END_SPARSITY="$2"
                shift 2
                ;;
            -ps|--pruning-start)
                PRUNING_START_EPOCH="$2"
                shift 2
                ;;
            -pe|--pruning-end)
                PRUNING_END_EPOCH="$2"
                shift 2
                ;;
            -ss|--sparsity-steps)
                SPARSITY_STEPS="$2"
                shift 2
                ;;
            -se|--spt-explore)
                SPT_EXPLORE_PER_LAYER="$2"
                shift 2
                ;;
            -t|--task)
                TASK="$2"
                shift 2
                ;;
            -ml|--max-len)
                MAX_LEN="$2"
                shift 2
                ;;
            -parallel)
                PARALLEL_MODE=true
                shift 1
                ;;
            -max-parallel-jobs)
                MAX_PARALLEL_JOBS="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Validate parameters
validate_parameters() {
    # Validate epochs
    if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [ "$EPOCHS" -lt 1 ]; then
        echo "Error: Epochs must be a positive integer"
        exit 1
    fi
    
    # Validate batch size
    if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
        echo "Error: Batch size must be a positive integer"
        exit 1
    fi
    
    # Validate learning rate
    if ! [[ "$LEARNING_RATE" =~ ^[0-9]+\.?[0-9]*e?[+-]?[0-9]*$ ]]; then
        echo "Error: Learning rate must be a valid number"
        exit 1
    fi
    
    # Validate model size
    if [[ ! "$MODEL_SIZE" =~ ^(tiny|small|base|large)$ ]]; then
        echo "Error: Model size must be one of: tiny, small, base, large"
        exit 1
    fi
    
    # Validate pruning methods
    for method in "${PRUNING_METHODS[@]}"; do
        if [[ ! "$method" =~ ^(baseline|magnitude|movement|spt)$ ]]; then
            echo "Error: Invalid pruning method: $method"
            echo "Valid methods: baseline, magnitude, movement, spt"
            exit 1
        fi
    done
    
    # Validate sparsity values
    if ! [[ "$START_SPARSITY" =~ ^[0-9]+\.?[0-9]*$ ]] || (( $(echo "$START_SPARSITY < 0" | bc -l) )) || (( $(echo "$START_SPARSITY > 1" | bc -l) )); then
        echo "Error: Start sparsity must be between 0.0 and 1.0"
        exit 1
    fi
    
    if ! [[ "$END_SPARSITY" =~ ^[0-9]+\.?[0-9]*$ ]] || (( $(echo "$END_SPARSITY < 0" | bc -l) )) || (( $(echo "$END_SPARSITY > 1" | bc -l) )); then
        echo "Error: End sparsity must be between 0.0 and 1.0"
        exit 1
    fi
    
    # Validate pruning epochs
    if ! [[ "$PRUNING_START_EPOCH" =~ ^[0-9]+$ ]] || [ "$PRUNING_START_EPOCH" -lt 1 ]; then
        echo "Error: Pruning start epoch must be a positive integer"
        exit 1
    fi
    
    if ! [[ "$PRUNING_END_EPOCH" =~ ^[0-9]+$ ]] || [ "$PRUNING_END_EPOCH" -lt 1 ]; then
        echo "Error: Pruning end epoch must be a positive integer"
        exit 1
    fi
    
    # Validate task
    if [[ ! "$TASK" =~ ^(sst2|mnli|qnli|rte|wnli|mrpc|cola|stsb)$ ]]; then
        echo "Error: Task must be one of: sst2, mnli, qnli, rte, wnli, mrpc, cola, stsb"
        exit 1
    fi
    
    # Validate max length
    if ! [[ "$MAX_LEN" =~ ^[0-9]+$ ]] || [ "$MAX_LEN" -lt 1 ]; then
        echo "Error: Max length must be a positive integer"
        exit 1
    fi
    
    # Validate SPT explore
    if [[ ! "$SPT_EXPLORE_PER_LAYER" =~ ^(true|false|True|False)$ ]]; then
        echo "Error: SPT explore must be true or false"
        exit 1
    fi
    
    # Validate parallel parameters
    if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then
        echo "Error: Max parallel jobs must be a positive integer"
        exit 1
    fi
    
    # Cap pruning end epoch to not exceed total epochs
    if [ "$PRUNING_END_EPOCH" -gt "$EPOCHS" ]; then
        echo "Warning: Pruning end epoch ($PRUNING_END_EPOCH) exceeds total epochs ($EPOCHS)"
        echo "Capping pruning end epoch to $EPOCHS"
        PRUNING_END_EPOCH=$EPOCHS
    fi
}

# Parse command line arguments
parse_arguments "$@"

# Validate all parameters
validate_parameters

# Display current configuration
display_configuration() {
    print_header "EXPERIMENT CONFIGURATION"
    echo -e "${CYAN}Training Parameters:${NC}"
    echo -e "  Epochs: ${GREEN}$EPOCHS${NC}"
    echo -e "  Batch Size: ${GREEN}$BATCH_SIZE${NC}"
    echo -e "  Learning Rate: ${GREEN}$LEARNING_RATE${NC}"
    echo -e "  Model Size: ${GREEN}$MODEL_SIZE${NC}"
    echo -e "  Task: ${GREEN}$TASK${NC}"
    echo -e "  Max Length: ${GREEN}$MAX_LEN${NC}"
    echo ""
    echo -e "${CYAN}Pruning Parameters:${NC}"
    echo -e "  Methods: ${GREEN}${PRUNING_METHODS[*]}${NC}"
    echo -e "  Start Sparsity: ${GREEN}$START_SPARSITY${NC}"
    echo -e "  End Sparsity: ${GREEN}$END_SPARSITY${NC}"
    echo -e "  Pruning Start Epoch: ${GREEN}$PRUNING_START_EPOCH${NC}"
    echo -e "  Pruning End Epoch: ${GREEN}$PRUNING_END_EPOCH${NC}"
    echo -e "  SPT Explore Per Layer: ${GREEN}$SPT_EXPLORE_PER_LAYER${NC}"
    echo ""
    echo -e "${CYAN}Execution Mode:${NC}"
    if [ "$PARALLEL_MODE" = true ]; then
        echo -e "  Mode: ${GREEN}PARALLEL${NC}"
        echo -e "  Max Parallel Jobs: ${GREEN}$MAX_PARALLEL_JOBS${NC}"
    else
        echo -e "  Mode: ${GREEN}SEQUENTIAL${NC}"
    fi
    echo ""
    echo -e "${CYAN}Sparsity Steps:${NC}"
    echo -e "  ${GREEN}$SPARSITY_STEPS${NC}"
    echo ""
}

# Display configuration
display_configuration

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
# EXPERIMENT METADATA
# =============================================================================

# Experiment metadata
EXPERIMENT_NAME="pruning_comparison_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="experiment_logs"

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
    content = re.sub(r"self\.model_size = kwargs\.get\('model_size', '[^']+'\)", 
                    f"self.model_size = kwargs.get('model_size', '${MODEL_SIZE}')", content)
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
        # Convert space-separated or comma-separated string to proper Python list
        # Remove duplicates and create clean list
        steps_list="[$(echo "${SPARSITY_STEPS}" | tr ',' ' ' | tr -s ' ' | sed 's/^ *//;s/ *$//' | tr ' ' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')]"
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

# Function to run training in parallel with proper output management
run_training_parallel() {
    local method=$1
    local enable_pruning=$2
    local results_suffix=${3:-$method}  # Use method name as default if not provided
    local job_id=$4
    
    # Create job-specific working directory
    local job_dir="/tmp/job_${job_id}_${method}"
    mkdir -p "$job_dir"
    
    # Copy necessary files to job directory
    cp config.py.backup "$job_dir/config.py.backup"
    cp train.py "$job_dir/"
    cp config.py "$job_dir/"
    cp model_builder.py "$job_dir/" 2>/dev/null || true
    cp model_layers.py "$job_dir/" 2>/dev/null || true
    cp movement_layers.py "$job_dir/" 2>/dev/null || true
    cp spt_layers.py "$job_dir/" 2>/dev/null || true
    cp pruning.py "$job_dir/" 2>/dev/null || true
    cp callbacks.py "$job_dir/" 2>/dev/null || true
    cp data_utils.py "$job_dir/" 2>/dev/null || true
    cp utils.py "$job_dir/" 2>/dev/null || true
    
    # Create method-specific log file (use absolute path from original directory)
local original_dir="$(pwd)"
local log_file="${original_dir}/${LOG_DIR}/${results_suffix}_training_$(date +%Y%m%d_%H%M%S)_job${job_id}.log"
local results_dir="${original_dir}/results_${results_suffix}_${TASK}"
    
    # Create a temporary script for this job
    local job_script="/tmp/training_job_${method}_${job_id}.sh"
    
    # Create a named pipe for real-time output
    local output_pipe="/tmp/job_output_${job_id}.pipe"
    mkfifo "$output_pipe" 2>/dev/null || true
    
    cat > "$job_script" << EOF
#!/bin/bash
# Job script for $method (Job ID: $job_id)

# Set up environment
cd "$job_dir"

# Ensure log directory exists in the original directory
mkdir -p "${original_dir}/${LOG_DIR}"

# Update config for this specific job
echo "[JOB $job_id] Updating config for method: $method"

# Create a simple Python script to update config
cat > update_config_job.py << PYTHON_SCRIPT
import re
import sys
import os

try:
    # Read original config
    with open('config.py.backup', 'r') as f:
        content = f.read()

    # Update parameters
    content = re.sub(r"self\.epochs = kwargs\.get\('epochs', \d+\)", 
                    f"self.epochs = kwargs.get('epochs', $EPOCHS)", content)
    content = re.sub(r"self\.batch_size = kwargs\.get\('batch_size', \d+\)", 
                    f"self.batch_size = kwargs.get('batch_size', $BATCH_SIZE)", content)
    content = re.sub(r"self\.learning_rate = kwargs\.get\('learning_rate', [^)]+\)", 
                    f"self.learning_rate = kwargs.get('learning_rate', '$LEARNING_RATE')", content)
    content = re.sub(r"self\.model_size = kwargs\.get\('model_size', '[^']+'\)", 
                    f"self.model_size = kwargs.get('model_size', '$MODEL_SIZE')", content)
    content = re.sub(r"self\.task = kwargs\.get\('task', '[^']+'\)", 
                    f"self.task = kwargs.get('task', '$TASK')", content)
    content = re.sub(r"self\.max_len = kwargs\.get\('max_len', \d+\)", 
                    f"self.max_len = kwargs.get('max_len', $MAX_LEN)", content)

    # Update pruning parameters
    content = re.sub(r"self\.enable_pruning = kwargs\.get\('enable_pruning', [^)]+\)", 
                    f"self.enable_pruning = kwargs.get('enable_pruning', $enable_pruning)", content)
    content = re.sub(r"self\.pruning_method = kwargs\.get\('pruning_method', '[^']+'\)", 
                    f"self.pruning_method = kwargs.get('pruning_method', '$method')", content)
    content = re.sub(r"self\.start_sparsity = kwargs\.get\('start_sparsity', [^)]+\)", 
                    f"self.start_sparsity = kwargs.get('start_sparsity', $START_SPARSITY)", content)
    content = re.sub(r"self\.end_sparsity = kwargs\.get\('end_sparsity', [^)]+\)", 
                    f"self.end_sparsity = kwargs.get('end_sparsity', $END_SPARSITY)", content)
    content = re.sub(r"self\.pruning_start_epoch = kwargs\.get\('pruning_start_epoch', \d+\)", 
                    f"self.pruning_start_epoch = kwargs.get('pruning_start_epoch', $PRUNING_START_EPOCH)", content)

    # Ensure pruning_end_epoch doesn't exceed total epochs
    pruning_end_epoch = min($PRUNING_END_EPOCH, $EPOCHS)
    content = re.sub(r"self\.pruning_end_epoch = kwargs\.get\('pruning_end_epoch', \d+\)", 
                    f"self.pruning_end_epoch = kwargs.get('pruning_end_epoch', {pruning_end_epoch})", content)

    # Update sparsity_steps parameter
    if "$SPARSITY_STEPS" != "":
        # Convert space-separated or comma-separated string to proper Python list
        # Remove duplicates and create clean list
        steps_list="[$(echo "$SPARSITY_STEPS" | tr ',' ' ' | tr -s ' ' | sed 's/^ *//;s/ *$//' | tr ' ' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')]"
        content = re.sub(r"self\.sparsity_steps = kwargs\.get\('sparsity_steps', [^)]+\)", 
                        f"self.sparsity_steps = kwargs.get('sparsity_steps', {steps_list})", content)
    else:
        content = re.sub(r"self\.sparsity_steps = kwargs\.get\('sparsity_steps', [^)]+\)", 
                        f"self.sparsity_steps = kwargs.get('sparsity_steps', None)", content)

    # Update SPT per-layer exploration parameter
    content = re.sub(r"self\.spt_explore_per_layer = kwargs\.get\('spt_explore_per_layer', [^)]+\)", 
                    f"self.spt_explore_per_layer = kwargs.get('spt_explore_per_layer', $SPT_EXPLORE_PER_LAYER)", content)

    # Write the updated config
    with open('config.py', 'w') as f:
        f.write(content)

    print(f"✅ Config updated for method: $method (pruning: $enable_pruning)")

except Exception as e:
    print(f"ERROR updating config: {e}")
    sys.exit(1)
PYTHON_SCRIPT

# Run the Python script with better error handling
if python update_config_job.py; then
    rm update_config_job.py
    echo "[JOB $job_id] Config updated successfully"
else
    rm -f update_config_job.py
    echo "[JOB $job_id] Failed to update config"
    echo "[JOB $job_id] Python script exit code: $?"
    exit 1
fi

# Run training with logging and real-time output
echo "[JOB $job_id] Starting training for method: $method"
echo "[JOB $job_id] Log file: $log_file"

# Function to send output to both log file and named pipe
send_output() {
    local message="\$1"
    echo "\$message" >> "$log_file"
    echo "[JOB $job_id] \$message" > "$output_pipe"
}

# Start training with real-time output monitoring and better error handling
set -e  # Exit on any error in the job script
if python train.py 2>&1 | while IFS= read -r line; do
    echo "\$line" >> "$log_file"
    echo "[JOB $job_id] \$line" > "$output_pipe"
done; then
    send_output "✅ Training completed successfully for method: $method"
    
    # Check if results directory was created
    if [ -d "$results_dir" ]; then
        local result_count=\$(ls -1 "$results_dir/" | wc -l)
        send_output "✅ Results saved: \$result_count files in $results_dir/"
    else
        send_output "⚠️  Results directory $results_dir/ not found"
    fi
    
    # Clean up temporary files
    rm -f "$job_script"
    rm -f "$output_pipe"
    rm -rf "$job_dir"
    
    # Signal success
    echo "SUCCESS:$method:$job_id" > "/tmp/job_result_${job_id}.txt"
else
    send_output "❌ Training failed for method: $method"
    
    # Clean up temporary files
    rm -f "$job_script"
    rm -f "$output_pipe"
    rm -rf "$job_dir"
    
    # Signal failure
    echo "FAILED:$method:$job_id" > "/tmp/job_result_${job_id}.txt"
    exit 1
fi
EOF

    # Make the job script executable
    chmod +x "$job_script"
    
    # Run the job in background
    "$job_script" &
    local job_pid=$!
    
    echo "[PARALLEL] Started job $job_id for method $method (PID: $job_pid)"
    echo "$job_pid" > "/tmp/job_pid_${job_id}.txt"
    return $job_pid
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
    
    if [ "$PARALLEL_MODE" = true ]; then
        run_experiments_parallel
    else
        run_experiments_sequential
    fi
    
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

# Function to run experiments sequentially (original behavior)
run_experiments_sequential() {
    print_header "SEQUENTIAL EXECUTION MODE"
    
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
}

# Function to run experiments in parallel
run_experiments_parallel() {
    print_header "PARALLEL EXECUTION MODE"
    print_info "Max parallel jobs: $MAX_PARALLEL_JOBS"
    
    # Clean up any existing job result files and pipes
    rm -f /tmp/job_result_*.txt
    rm -f /tmp/job_output_*.pipe
    
    # Array to store job PIDs
    local job_pids=()
    local job_methods=()
    local job_ids=()
    local current_job_id=0
    
    # Set output mode (default to progress for non-interactive)
    if [ -t 0 ]; then
        # Interactive mode - ask user
        echo ""
        echo -e "${CYAN}Choose output display mode:${NC}"
        echo -e "  1) ${GREEN}Progress bars${NC} - Clean progress bars for each job"
        echo -e "  2) ${GREEN}Real-time logs${NC} - Live log output from each job"
        echo -e "  3) ${GREEN}Both${NC} - Progress bars + key log messages"
        echo ""
        read -p "Enter choice (1-3, default: 1): " output_choice
        output_choice=${output_choice:-1}
        
        case $output_choice in
            1) OUTPUT_MODE="progress" ;;
            2) OUTPUT_MODE="logs" ;;
            3) OUTPUT_MODE="both" ;;
            *) OUTPUT_MODE="progress" ;;
        esac
    else
        # Non-interactive mode - use progress bars
        OUTPUT_MODE="progress"
    fi
    
    print_info "Output mode: $OUTPUT_MODE"
    
    # Simple parallel execution without complex monitoring
    print_info "Starting parallel jobs..."
    
    # Function to check job results
    check_job_results() {
        for result_file in /tmp/job_result_*.txt; do
            if [ -f "$result_file" ]; then
                local result=$(cat "$result_file")
                local status=$(echo "$result" | cut -d: -f1)
                local method=$(echo "$result" | cut -d: -f2)
                local job_id=$(echo "$result" | cut -d: -f3)
                
                if [ "$status" = "SUCCESS" ]; then
                    successful_runs+=("$method")
                    print_success "✅ Job $job_id ($method) completed successfully"
                else
                    failed_runs+=("$method")
                    print_error "❌ Job $job_id ($method) failed"
                    experiment_failed=true
                fi
                
                # Remove the result file and PID file
                rm -f "$result_file"
                rm -f "/tmp/job_pid_${job_id}.txt"
            fi
        done
    }
    
    # Function to start a new job
    start_job() {
        local method=$1
        local enable_pruning="True"
        local actual_method="$method"
        local results_suffix="$method"
        
        if [ "$method" = "baseline" ]; then
            enable_pruning="False"
            actual_method="none"
            results_suffix="baseline"
        fi
        
        current_job_id=$((current_job_id + 1))
        local job_id=$current_job_id
        
        print_info "Starting job $job_id for method: $method"
        
        # Start the job
        run_training_parallel "$actual_method" "$enable_pruning" "$results_suffix" "$job_id"
        local job_pid=$?
        
        # Store job information
        job_pids+=($job_pid)
        job_methods+=("$method")
        job_ids+=($job_id)
        
        print_info "Job $job_id (PID: $job_pid) started for method: $method"
    }
    
    # Function to wait for jobs to complete
    wait_for_jobs() {
        local active_jobs=${#job_pids[@]}
        
        while [ $active_jobs -gt 0 ]; do
            # Check for completed jobs
            check_job_results
            
            # Count active jobs
            active_jobs=0
            for i in "${!job_pids[@]}"; do
                local job_id="${job_ids[$i]}"
                local pid_file="/tmp/job_pid_${job_id}.txt"
                
                if [ -f "$pid_file" ]; then
                    local job_pid=$(cat "$pid_file")
                    if kill -0 "$job_pid" 2>/dev/null; then
                        active_jobs=$((active_jobs + 1))
                    else
                        # Job completed, remove PID file
                        rm -f "$pid_file"
                    fi
                fi
            done
            
            if [ $active_jobs -gt 0 ]; then
                print_info "Waiting for $active_jobs jobs to complete..."
                sleep 10
            fi
        done
        
        # Final check for any remaining results
        check_job_results
    }
    
    # Start jobs with parallel limit
    local method_index=0
    local running_jobs=0
    
    while [ $method_index -lt ${#PRUNING_METHODS[@]} ]; do
        # Check if we can start a new job
        if [ $running_jobs -lt $MAX_PARALLEL_JOBS ]; then
            local method="${PRUNING_METHODS[$method_index]}"
            start_job "$method"
            method_index=$((method_index + 1))
            running_jobs=$((running_jobs + 1))
        else
            # Wait for a job to complete
            print_info "Max parallel jobs reached ($MAX_PARALLEL_JOBS). Waiting for jobs to complete..."
            sleep 5
            
            # Check for completed jobs
            for i in "${!job_pids[@]}"; do
                local job_id="${job_ids[$i]}"
                local pid_file="/tmp/job_pid_${job_id}.txt"
                
                if [ -f "$pid_file" ]; then
                    local job_pid=$(cat "$pid_file")
                    if ! kill -0 "$job_pid" 2>/dev/null; then
                        running_jobs=$((running_jobs - 1))
                        print_info "Job ${job_ids[$i]} (${job_methods[$i]}) completed"
                        rm -f "$pid_file"
                    fi
                fi
            done
        fi
    done
    
    # Wait for all remaining jobs to complete
    print_info "All jobs started. Waiting for completion..."
    wait_for_jobs
    
    print_success "All parallel jobs completed!"
    
    # Wait for any remaining background jobs
    wait
    
    # Clean up pipes
    rm -f /tmp/job_output_*.pipe
}

# Function to monitor and display real-time output from all parallel jobs
start_output_monitor() {
    print_info "Starting real-time output monitor..."
    
    # Create a temporary file to store all output
    local output_file="/tmp/parallel_output_$(date +%s).txt"
    
    # Function to monitor a specific job's output
    monitor_job_output() {
        local job_id=$1
        local method=$2
        local pipe="/tmp/job_output_${job_id}.pipe"
        
        if [ -p "$pipe" ]; then
            while IFS= read -r line; do
                # Add timestamp and job info
                local timestamp=$(date '+%H:%M:%S')
                echo "[$timestamp] [JOB $job_id] [$method] $line" >> "$output_file"
                
                # Display important messages
                if echo "$line" | grep -q "Epoch\|accuracy\|loss\|sparsity\|SUCCESS\|FAILED"; then
                    echo -e "${CYAN}[$timestamp] [JOB $job_id] [$method]${NC} $line"
                fi
            done < "$pipe"
        fi
    }
    
    # Function to show progress bars for all jobs
    show_progress_bars() {
        local job_count=${#job_ids[@]}
        local terminal_width=$(tput cols 2>/dev/null || echo 80)
        local progress_width=$((terminal_width - 30))  # Leave space for job info
        
        # Clear screen and show progress header
        clear
        echo -e "${BLUE}============================================${NC}"
        echo -e "${BLUE}PARALLEL TRAINING PROGRESS MONITOR${NC}"
        echo -e "${BLUE}============================================${NC}"
        echo ""
        
        # Show progress for each job
        for i in "${!job_ids[@]}"; do
            local job_id="${job_ids[$i]}"
            local method="${job_methods[$i]}"
            local log_file="${LOG_DIR}/${method}_training_*_job${job_id}.log"
            
            # Get the actual log file
            local actual_log_file=$(ls $log_file 2>/dev/null | head -1)
            
            if [ -f "$actual_log_file" ]; then
                # Extract current epoch from log
                local current_epoch=$(tail -n 50 "$actual_log_file" | grep -o "Epoch [0-9]*/[0-9]*" | tail -1 | grep -o "[0-9]*/[0-9]*" | head -1)
                
                if [ -n "$current_epoch" ]; then
                    local epoch_num=$(echo "$current_epoch" | cut -d'/' -f1)
                    local total_epochs=$(echo "$current_epoch" | cut -d'/' -f2)
                    
                    # Calculate progress percentage
                    local progress_percent=$((epoch_num * 100 / total_epochs))
                    
                    # Create progress bar
                    local filled=$((progress_percent * progress_width / 100))
                    local empty=$((progress_width - filled))
                    
                    local progress_bar=""
                    for ((j=0; j<filled; j++)); do
                        progress_bar+="█"
                    done
                    for ((j=0; j<empty; j++)); do
                        progress_bar+="░"
                    done
                    
                    # Show job progress
                    printf "${CYAN}[JOB %-2s]${NC} ${GREEN}%-12s${NC} [${YELLOW}%s${NC}] ${GREEN}%3d%%${NC} (Epoch %s/%s)\n" \
                           "$job_id" "$method" "$progress_bar" "$progress_percent" "$epoch_num" "$total_epochs"
                else
                    # Job not started or no epoch info yet
                    printf "${CYAN}[JOB %-2s]${NC} ${GREEN}%-12s${NC} [${YELLOW}%s${NC}] ${GREEN}%3d%%${NC} (Starting...)\n" \
                           "$job_id" "$method" "$(printf '%*s' $progress_width '' | tr ' ' '░')" "0"
                fi
            else
                # Log file not found yet
                printf "${CYAN}[JOB %-2s]${NC} ${GREEN}%-12s${NC} [${YELLOW}%s${NC}] ${GREEN}%3d%%${NC} (Initializing...)\n" \
                       "$job_id" "$method" "$(printf '%*s' $progress_width '' | tr ' ' '░')" "0"
            fi
        done
        
        echo ""
        echo -e "${CYAN}Press Ctrl+C to stop monitoring (jobs will continue running)${NC}"
        echo ""
    }
    
    # Function to show real-time logs for all jobs
    show_real_time_logs() {
        # Start tail processes for each job
        for i in "${!job_ids[@]}"; do
            local job_id="${job_ids[$i]}"
            local method="${job_methods[$i]}"
            local log_file="${LOG_DIR}/${method}_training_*_job${job_id}.log"
            
            # Get the actual log file
            local actual_log_file=$(ls $log_file 2>/dev/null | head -1)
            
            if [ -f "$actual_log_file" ]; then
                # Start tail in background with job identifier
                (echo -e "${CYAN}=== JOB $job_id ($method) LOG ===${NC}"; tail -f "$actual_log_file" | sed "s/^/[JOB $job_id] /") &
            fi
        done
        
        # Wait for user to stop
        echo -e "${CYAN}Real-time logs started. Press Ctrl+C to stop monitoring (jobs will continue running)${NC}"
        wait
    }
    
    # Monitor based on output mode
    case "$OUTPUT_MODE" in
        "progress")
            # Monitor all job pipes and update progress
            while true; do
                # Check if all jobs are complete
                local all_complete=true
                for job_id in "${job_ids[@]}"; do
                    if [ -p "/tmp/job_output_${job_id}.pipe" ]; then
                        all_complete=false
                        break
                    fi
                done
                
                if [ "$all_complete" = true ]; then
                    break
                fi
                
                # Show progress bars
                show_progress_bars
                
                # Wait before next update
                sleep 2
            done
            
            # Final progress update
            show_progress_bars
            ;;
            
        "logs")
            # Show real-time logs
            show_real_time_logs
            ;;
            
        "both")
            # Show both progress bars and key log messages
            while true; do
                # Check if all jobs are complete
                local all_complete=true
                for job_id in "${job_ids[@]}"; do
                    if [ -p "/tmp/job_output_${job_id}.pipe" ]; then
                        all_complete=false
                        break
                    fi
                done
                
                if [ "$all_complete" = true ]; then
                    break
                fi
                
                # Show progress bars
                show_progress_bars
                
                # Show recent log messages
                echo -e "${CYAN}Recent Log Messages:${NC}"
                for i in "${!job_ids[@]}"; do
                    local job_id="${job_ids[$i]}"
                    local method="${job_methods[$i]}"
                    local log_file="${LOG_DIR}/${method}_training_*_job${job_id}.log"
                    local actual_log_file=$(ls $log_file 2>/dev/null | head -1)
                    
                    if [ -f "$actual_log_file" ]; then
                        local recent_line=$(tail -n 1 "$actual_log_file" 2>/dev/null)
                        if [ -n "$recent_line" ]; then
                            echo -e "${CYAN}[JOB $job_id]${NC} $recent_line"
                        fi
                    fi
                done
                echo ""
                
                # Wait before next update
                sleep 3
            done
            ;;
    esac
    
    print_info "Output monitor completed. Full output saved to: $output_file"
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