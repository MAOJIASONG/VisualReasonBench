#!/bin/bash

# VLM Interactive Benchmark Runner
# This script runs the interactive VLM benchmark with automatic cleanup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERACTIVE_SCRIPT="${SCRIPT_DIR}/PhyVPuzzle/vlm_interactive_benchmark.py"
RESULTS_DIR="${SCRIPT_DIR}/PhyVPuzzle/interactive_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${SCRIPT_DIR}/.backup_interactive_${TIMESTAMP}"
API_KEY="sk-3fHP5EI7lXxX0GGYpdAurdqjqYLqJ9Sn48PtO3QVHgdtkOP3"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Function to cleanup on script exit
cleanup() {
    local exit_code=$?
    print_status "Cleaning up interactive test environment..."
    
    # Keep test results directory for analysis
    if [ -d "$RESULTS_DIR" ]; then
        print_status "Keeping results directory for analysis: $RESULTS_DIR"
        # Don't remove results directory anymore
        # rm -rf "$RESULTS_DIR"
    fi
    
    # Remove any orphaned screenshot directories in root (keep the ones moved to results)
    for dir in "${SCRIPT_DIR}"/vlm_interactive_screenshots*; do
        if [ -d "$dir" ]; then
            print_status "Removing temporary screenshot directory: $dir"
            rm -rf "$dir"
        fi
    done
    
    # Remove backup directory
    if [ -d "$BACKUP_DIR" ]; then
        print_status "Removing backup directory: $BACKUP_DIR"
        rm -rf "$BACKUP_DIR"
    fi
    
    # Restore original working directory
    cd "$SCRIPT_DIR"
    
    if [ $exit_code -eq 0 ]; then
        print_success "Environment restored successfully!"
        if [ -d "$RESULTS_DIR" ]; then
            print_status "Results preserved at: $RESULTS_DIR"
        fi
    else
        print_warning "Script exited with error code $exit_code, but environment has been restored."
    fi
}

# Set trap to ensure cleanup runs on script exit
trap cleanup EXIT

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if interactive script exists
    if [ ! -f "$INTERACTIVE_SCRIPT" ]; then
        print_error "Interactive script not found: $INTERACTIVE_SCRIPT"
        exit 1
    fi
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Determine Python command
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    
    print_success "Prerequisites check passed"
}

# Function to backup current state
backup_state() {
    print_status "Creating backup of current state..."
    mkdir -p "$BACKUP_DIR"
    
    # Backup any existing screenshot directories
    for dir in "${SCRIPT_DIR}"/vlm_interactive_screenshots*; do
        if [ -d "$dir" ]; then
            cp -r "$dir" "$BACKUP_DIR/"
        fi
    done
    
    print_success "Backup created at: $BACKUP_DIR"
}

# Function to setup test environment
setup_test_environment() {
    print_status "Setting up interactive test environment..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Change to PhyVPuzzle directory
    cd "$SCRIPT_DIR/PhyVPuzzle"
    
    print_success "Interactive test environment ready"
}

# Function to run the interactive benchmark
run_interactive_benchmark() {
    local task_type="$1"
    print_status "Starting VLM Interactive Benchmark..."
    print_status "Task: $task_type"
    print_status "This will involve multiple VLM-environment interactions..."
    
    # Create log file
    local log_file="${RESULTS_DIR}/interactive_${task_type}_${TIMESTAMP}.log"
    local results_file="${RESULTS_DIR}/interactive_results_${task_type}_${TIMESTAMP}.json"
    
    # Run the interactive benchmark
    if $PYTHON_CMD "$INTERACTIVE_SCRIPT" --api_key "$API_KEY" --task "$task_type" --save-results "$results_file" --headless 2>&1 | tee "$log_file"; then
        print_success "Interactive benchmark completed successfully!"
        print_status "Test log saved to: $log_file"
        
        # Check if results were generated
        if [ -f "$results_file" ]; then
            print_success "Results saved to: $results_file"
        else
            print_warning "Results file not found at expected location: $results_file"
        fi
        
        # Move any generated screenshots
        for screenshot_dir in vlm_interactive_screenshots*; do
            if [ -d "$screenshot_dir" ]; then
                mv "$screenshot_dir" "${RESULTS_DIR}/"
                print_status "Screenshots moved to: ${RESULTS_DIR}/$screenshot_dir"
            fi
        done
        
    else
        print_error "Interactive benchmark failed!"
        print_status "Check the log file for details: $log_file"
        return 1
    fi
}

# Function to display test summary
display_summary() {
    local task_type="$1"
    print_status "Interactive Benchmark Summary:"
    echo "==============================================="
    echo "Task Type: $task_type"
    echo "Timestamp: $TIMESTAMP"
    echo "Results Directory: $RESULTS_DIR"
    
    if [ -f "${RESULTS_DIR}/interactive_results_${task_type}_${TIMESTAMP}.json" ]; then
        echo "Results File: interactive_results_${task_type}_${TIMESTAMP}.json"
        
        # Try to extract summary from results if possible
        if command -v python3 &> /dev/null; then
            python3 -c "
import json
import sys
try:
    with open('${RESULTS_DIR}/interactive_results_${task_type}_${TIMESTAMP}.json', 'r') as f:
        results = json.load(f)
    if 'summary' in results:
        print('Success Rate: {:.1%}'.format(results['summary'].get('success_rate', 0)))
        print('Average Score: {:.3f}'.format(results['summary'].get('average_score', 0)))
        print('Total Tasks: {}'.format(results['summary'].get('total_tasks', 0)))
    elif 'final_score' in results:
        print('Final Score: {:.3f}'.format(results['final_score']))
        print('Status: {}'.format(results.get('status', 'unknown')))
        print('Steps Taken: {}/{}'.format(results.get('steps_taken', 0), results.get('max_steps', 0)))
except:
    pass
" 2>/dev/null || true
        fi
    fi
    
    echo "==============================================="
}

# Main execution
main() {
    local task_type="${1:-all}"
    
    print_status "Starting VLM Interactive Benchmark Runner"
    print_status "Task: $task_type"
    print_status "Timestamp: $TIMESTAMP"
    
    check_prerequisites
    backup_state
    setup_test_environment
    
    # Run the benchmark
    if run_interactive_benchmark "$task_type"; then
        display_summary "$task_type"
        print_success "Interactive benchmark completed successfully!"
    else
        print_error "Interactive benchmark failed!"
        exit 1
    fi
}

# Parse command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "VLM Interactive Benchmark Runner"
    echo ""
    echo "Usage: $0 [TASK_TYPE]"
    echo ""
    echo "This script runs the VLM interactive benchmark with automatic cleanup."
    echo ""
    echo "Task Types:"
    echo "  domino    Run only the domino chain reaction task"
    echo "  luban     Run only the Luban lock assembly task"
    echo "  all       Run all interactive tasks (default)"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "The script will:"
    echo "  1. Check prerequisites"
    echo "  2. Backup current state"
    echo "  3. Run the interactive VLM benchmark"
    echo "  4. Save results and screenshots"
    echo "  5. Restore original environment"
    echo ""
    echo "Interactive Flow:"
    echo "  - VLM receives initial environment image and instructions"
    echo "  - VLM selects from available actions"
    echo "  - Action is executed in PyBullet"
    echo "  - VLM receives updated environment image"
    echo "  - Process repeats until task completion or timeout"
    echo "  - Final score is calculated based on task success"
    echo ""
    echo "Note: API key is pre-configured in the script."
    exit 0
fi

# Run main function with task type
main "$@"
