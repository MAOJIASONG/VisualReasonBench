"""
Example usage of PhyVPuzzle environment for VLM evaluation.

This script demonstrates how to set up and run the PhyVPuzzle environment
with observation-action-feedback loops for VLM benchmarking.
"""

import os
import sys
import time
from typing import Dict, Any

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from phyvpuzzle.environment import (
        create_luban_lock_environment,
        create_pagoda_environment,
        VLMBenchmarkController,
        VLMBenchmarkConfig,
        PuzzleType
    )
    from phyvpuzzle.core.translator import EnvironmentCommand
    from phyvpuzzle.core.vllm_processor import VLLMProcessor
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and the project structure is correct.")
    sys.exit(1)


class MockVLLMProcessor(VLLMProcessor):
    """Mock VLM processor for testing purposes."""
    
    def __init__(self, model_name: str = "mock_model"):
        self.model_name = model_name
        self.step_count = 0
    
    def process_input(self, vlm_input: Dict[str, Any]) -> str:
        """Generate mock VLM responses for testing."""
        self.step_count += 1
        
        # Mock decision based on step count
        if self.step_count <= 3:
            return f"I will move piece obj_1 to position (0.1, 0.1, 0.6). Action: move_piece, object: obj_1, position: [0.1, 0.1, 0.6]"
        elif self.step_count <= 6:
            return f"I will rotate piece obj_2 around z-axis by 45 degrees. Action: rotate_piece, object: obj_2, axis: z, angle: 45"
        elif self.step_count <= 9:
            return f"I will check if the puzzle is solved. Action: check_solution"
        else:
            return "I believe the puzzle is solved. Action: check_solution"


def test_luban_lock_environment():
    """Test the Luban lock environment setup and basic operations."""
    print("=== Testing Luban Lock Environment ===")
    
    # Create environment
    models_path = "./phobos_models"
    env = create_luban_lock_environment(models_path, gui=True)
    
    try:
        # Setup environment
        env.setup_environment()
        print("Environment setup completed")
        
        # Get initial observation
        observation = env.get_observation_for_vlm()
        print("Initial observation obtained")
        print(f"State: {observation['state_description'][:200]}...")
        
        # Test a few basic commands
        commands_to_test = [
            EnvironmentCommand("move_piece", {"piece_id": "obj_1", "target_position": [0.1, 0.1, 0.6]}),
            EnvironmentCommand("rotate_piece", {"piece_id": "obj_1", "rotation_axis": "z", "rotation_angle": 45}),
            EnvironmentCommand("check_solution", {})
        ]
        
        for i, cmd in enumerate(commands_to_test):
            print(f"\\nExecuting command {i+1}: {cmd.command_type}")
            success = env.execute_command(cmd)
            print(f"Command result: {'Success' if success else 'Failed'}")
            
            # Get updated state
            state = env.get_state()
            print(f"Current step: {state['step_count']}, Task completed: {state['task_completed']}")
            
            time.sleep(1)  # Brief pause for visualization
        
        # Test success detection
        is_complete = env.is_task_complete()
        success_status, reason = env.get_success_status()
        print(f"\\nTask complete: {is_complete}")
        print(f"Success status: {success_status}, Reason: {reason}")
        
    except Exception as e:
        print(f"Error during Luban lock test: {e}")
    finally:
        env.close()
        print("Environment closed")


def test_pagoda_environment():
    """Test the Pagoda environment setup and basic operations."""
    print("\\n=== Testing Pagoda Environment ===")
    
    # Create environment
    models_path = "./phobos_models"
    env = create_pagoda_environment(models_path, gui=True)
    
    try:
        # Setup environment
        env.setup_environment()
        print("Environment setup completed")
        
        # Get initial observation
        observation = env.get_observation_for_vlm()
        print("Initial observation obtained")
        print(f"Available actions: {observation['available_actions']}")
        
        # Test pagoda-specific commands
        commands_to_test = [
            EnvironmentCommand("move_pole_x", {"direction": 1, "distance": 0.1}),
            EnvironmentCommand("move_pole_z", {"direction": 1, "distance": 0.05}),
            EnvironmentCommand("check_solution", {})
        ]
        
        for i, cmd in enumerate(commands_to_test):
            print(f"\\nExecuting command {i+1}: {cmd.command_type}")
            success = env.execute_command(cmd)
            print(f"Command result: {'Success' if success else 'Failed'}")
            
            time.sleep(1)
        
    except Exception as e:
        print(f"Error during Pagoda test: {e}")
    finally:
        env.close()
        print("Environment closed")


def test_vlm_benchmark():
    """Test the VLM benchmark controller with mock VLM."""
    print("\\n=== Testing VLM Benchmark Controller ===")
    
    # Create mock VLM processor
    mock_vlm = MockVLLMProcessor("test_model")
    
    # Configure benchmark
    config = VLMBenchmarkConfig(
        model_name="test_model",
        puzzle_type=PuzzleType.LUBAN_LOCK,
        max_steps=10,
        time_limit=60.0,
        save_trajectory=True,
        save_images=True,
        output_dir="./test_benchmark_results"
    )
    
    try:
        # Create controller
        controller = VLMBenchmarkController(config, mock_vlm)
        
        # Run benchmark
        print("Starting benchmark...")
        result = controller.run_benchmark("test_run")
        
        # Display results
        print(f"\\nBenchmark Results:")
        print(f"Task ID: {result.task_id}")
        print(f"Success: {result.success}")
        print(f"Total Steps: {result.total_steps}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Efficiency Score: {result.efficiency_score:.3f}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
    except Exception as e:
        print(f"Error during benchmark test: {e}")


def main():
    """Run all tests."""
    print("PhyVPuzzle Environment Integration Tests")
    print("=" * 50)
    
    # Check if models directory exists
    models_path = "./phobos_models"
    if not os.path.exists(models_path):
        print(f"Warning: Models directory not found at {models_path}")
        print("Some tests may fail. Please ensure PhyVPuzzle models are available.")
    
    try:
        # Test individual environments
        test_luban_lock_environment()
        test_pagoda_environment()
        
        # Test benchmark system
        test_vlm_benchmark()
        
        print("\\n" + "=" * 50)
        print("All tests completed!")
        
    except KeyboardInterrupt:
        print("\\nTests interrupted by user")
    except Exception as e:
        print(f"\\nUnexpected error: {e}")


if __name__ == "__main__":
    main()