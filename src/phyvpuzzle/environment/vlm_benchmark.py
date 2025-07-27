"""
VLM Benchmark Controller for PhyVPuzzle

This module implements the main benchmark loop for evaluating Vision-Language Models
on complex physics puzzle tasks with observation-action-feedback cycles.
"""

import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from PIL import Image
import numpy as np

from .phyvpuzzle_env import PhyVPuzzleEnvironment, PuzzleType, create_luban_lock_environment, create_pagoda_environment
from .physics_env import TaskDefinition, ExecutionStep, BenchmarkResult
from ..core.translator import EnvironmentCommand
from ..core.action_descriptor import ActionDescriptor, ParsedAction
from ..core.vllm_processor import VLLMProcessor, DecisionParser


@dataclass
class VLMBenchmarkConfig:
    """Configuration for VLM benchmark evaluation."""
    model_name: str
    puzzle_type: PuzzleType
    max_steps: int = 100
    time_limit: float = 600.0
    save_trajectory: bool = True
    save_images: bool = True
    output_dir: str = "./benchmark_results"
    instruction_template: str = None
    
    def __post_init__(self):
        if self.instruction_template is None:
            self.instruction_template = self._get_default_instruction()
    
    def _get_default_instruction(self) -> str:
        """Get default instruction template for the puzzle type."""
        if self.puzzle_type == PuzzleType.LUBAN_LOCK:
            return """
You are controlling a robotic system to solve a Luban lock puzzle. The goal is to disassemble or assemble the interlocking wooden pieces by moving them in the correct sequence.

Available actions:
- move_piece: Move a piece to a target position
- rotate_piece: Rotate a piece around an axis
- slide_piece: Slide a piece in a direction
- lift_piece: Lift a piece vertically
- insert_piece: Insert a piece into a slot
- remove_piece: Remove a piece from its position
- check_solution: Check if the puzzle is solved

Observe the current state carefully and plan your next action. Consider the physical constraints and interlocking mechanisms.
"""
        elif self.puzzle_type == PuzzleType.PAGODA:
            return """
You are controlling a robotic system to solve a Pagoda stacking puzzle. The goal is to arrange the pieces in a stable tower configuration.

Available actions:
- move_piece: Move a piece to a target position  
- rotate_piece: Rotate a piece around an axis
- slide_piece: Slide a piece in a direction
- lift_piece: Lift a piece vertically
- move_pole_x/y/z: Move the central pole in X/Y/Z direction
- check_solution: Check if the puzzle is solved

Consider balance, stability, and proper stacking order when making moves.
"""
        else:
            return "Solve the physics puzzle by manipulating the objects."


class VLMBenchmarkController:
    """
    Main controller for running VLM benchmarks on PhyVPuzzle environments.
    Manages the observation-action-feedback loop and result collection.
    """
    
    def __init__(self, config: VLMBenchmarkConfig, vlm_processor: VLLMProcessor):
        self.config = config
        self.vlm_processor = vlm_processor
        self.action_descriptor = ActionDescriptor()
        self.decision_parser = DecisionParser()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize environment
        self.environment = self._create_environment()
        
        # Results tracking
        self.execution_steps = []
        self.current_step = 0
        self.start_time = None
        
    def _create_environment(self) -> PhyVPuzzleEnvironment:
        """Create the appropriate puzzle environment."""
        # Use relative path to PhyVPuzzle directory
        base_path = "./PhyVPuzzle/phobos_models"
        
        if self.config.puzzle_type == PuzzleType.LUBAN_LOCK:
            return create_luban_lock_environment(base_path, gui=False)
        elif self.config.puzzle_type == PuzzleType.PAGODA:
            return create_pagoda_environment(base_path, gui=False)
        else:
            raise ValueError(f"Unknown puzzle type: {self.config.puzzle_type}")
    
    def run_benchmark(self, task_id: str = None) -> BenchmarkResult:
        """
        Run a complete benchmark evaluation.
        
        Returns:
            BenchmarkResult with execution details and success metrics
        """
        if task_id is None:
            task_id = f"{self.config.puzzle_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Starting VLM benchmark: {task_id}")
        print(f"Model: {self.config.model_name}")
        print(f"Puzzle: {self.config.puzzle_type.value}")
        
        # Setup environment
        self.environment.setup_environment()
        self.start_time = time.time()
        self.current_step = 0
        self.execution_steps = []
        
        try:
            # Run the main benchmark loop
            success = self._run_benchmark_loop()
            
            # Calculate final metrics
            execution_time = time.time() - self.start_time
            efficiency_score = self._calculate_efficiency_score(success, execution_time)
            
            # Create result
            result = BenchmarkResult(
                task_id=task_id,
                success=success,
                total_steps=self.current_step,
                execution_time=execution_time,
                efficiency_score=efficiency_score,
                steps_history=self.execution_steps,
                final_state=self.environment.get_state()
            )
            
            # Save results
            self._save_benchmark_result(result)
            
            return result
            
        except Exception as e:
            print(f"Benchmark failed with error: {e}")
            return BenchmarkResult(
                task_id=task_id,
                success=False,
                total_steps=self.current_step,
                execution_time=time.time() - self.start_time if self.start_time else 0,
                efficiency_score=0.0,
                steps_history=self.execution_steps,
                error_message=str(e)
            )
        finally:
            self.environment.close()
    
    def _run_benchmark_loop(self) -> bool:
        """
        Run the main observation-action-feedback loop.
        
        Returns:
            bool: True if task completed successfully
        """
        while not self.environment.is_task_complete() and self.current_step < self.config.max_steps:
            self.current_step += 1
            step_start_time = time.time()
            
            try:
                # 1. Get observation for VLM
                observation = self.environment.get_observation_for_vlm()
                
                # 2. Prepare VLM input
                vlm_input = self._prepare_vlm_input(observation)
                
                # 3. Get VLM decision
                vlm_response = self.vlm_processor.process_input(vlm_input)
                
                # 4. Parse action from VLM response
                parsed_action = self.decision_parser.parse_decision(vlm_response)
                
                # 5. Convert to environment command
                env_command = self._action_to_command(parsed_action)
                
                # 6. Execute action in environment
                execution_success = False
                if env_command:
                    execution_success = self.environment.execute_command(env_command)
                
                # 7. Get environment feedback
                feedback = self._generate_feedback(execution_success, observation)
                
                # 8. Record execution step
                step = ExecutionStep(
                    step_id=self.current_step,
                    vlm_input=vlm_input,
                    vlm_response=vlm_response,
                    parsed_action=parsed_action,
                    environment_command=env_command,
                    execution_result=execution_success,
                    environment_state=self.environment.get_state(),
                    feedback=feedback,
                    timestamp=time.time()
                )
                
                self.execution_steps.append(step)
                
                # 9. Save step data if configured
                if self.config.save_trajectory:
                    self._save_step_data(step, observation)
                
                # 10. Check if task is complete
                if self.environment.task_completed:
                    print(f"Task completed successfully in {self.current_step} steps!")
                    return True
                
                print(f"Step {self.current_step}: {execution_success} - {feedback}")
                
                # Small delay for stability
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in step {self.current_step}: {e}")
                continue
        
        # Check final success status
        success, reason = self.environment.get_success_status()
        print(f"Benchmark ended: {reason}")
        return success
    
    def _prepare_vlm_input(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for VLM processing."""
        vlm_input = {
            "image": observation["image"],
            "instruction": self.config.instruction_template,
            "current_state": observation["state_description"],
            "available_actions": observation["available_actions"],
            "step_number": self.current_step,
            "max_steps": self.config.max_steps
        }
        
        # Add context from previous steps
        if len(self.execution_steps) > 0:
            recent_steps = self.execution_steps[-3:]  # Last 3 steps for context
            context = "Recent actions:\\n"
            for step in recent_steps:
                action_desc = step.parsed_action.action_type if step.parsed_action else "unknown"
                context += f"Step {step.step_id}: {action_desc} - {'Success' if step.execution_result else 'Failed'}\\n"
            vlm_input["recent_context"] = context
        
        return vlm_input
    
    def _action_to_command(self, parsed_action: Optional[ParsedAction]) -> Optional[EnvironmentCommand]:
        """Convert parsed action to environment command."""
        if not parsed_action:
            return None
        
        try:
            # Map VLM action to environment command
            command_type = parsed_action.action_type
            parameters = parsed_action.parameters or {}
            
            # Handle common parameter mappings
            if "object" in parameters and "piece_id" not in parameters:
                parameters["piece_id"] = parameters["object"]
            
            if "position" in parameters and "target_position" not in parameters:
                parameters["target_position"] = parameters["position"]
            
            return EnvironmentCommand(
                command_type=command_type,
                parameters=parameters,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"Error converting action to command: {e}")
            return None
    
    def _generate_feedback(self, success: bool, observation: Dict[str, Any]) -> str:
        """Generate feedback message for the VLM."""
        if success:
            feedback = "Action executed successfully. "
        else:
            feedback = "Action failed to execute. "
        
        # Add state-specific feedback
        state = observation["raw_state"]
        
        if state.get("task_completed"):
            feedback += "Puzzle solved! Task completed."
        elif state.get("puzzle_state") == "initial":
            feedback += "Puzzle is in initial state."
        else:
            feedback += f"Puzzle state: {state.get('puzzle_state', 'unknown')}"
        
        # Add piece information
        pieces = state.get("pieces", {})
        if pieces:
            feedback += f" {len(pieces)} pieces in environment."
        
        return feedback
    
    def _calculate_efficiency_score(self, success: bool, execution_time: float) -> float:
        """Calculate efficiency score based on success, steps, and time."""
        if not success:
            return 0.0
        
        # Base score for success
        score = 1.0
        
        # Penalty for excessive steps
        step_penalty = max(0, (self.current_step - 20) * 0.01)
        score -= step_penalty
        
        # Penalty for excessive time
        time_penalty = max(0, (execution_time - 60) * 0.001)
        score -= time_penalty
        
        return max(0.0, min(1.0, score))
    
    def _save_step_data(self, step: ExecutionStep, observation: Dict[str, Any]) -> None:
        """Save step data including images and state."""
        step_dir = os.path.join(self.config.output_dir, f"step_{step.step_id:03d}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Save image
        if self.config.save_images and "image" in observation:
            image_path = os.path.join(step_dir, "observation.png")
            observation["image"].save(image_path)
        
        # Save step data
        step_data = {
            "step_id": step.step_id,
            "vlm_input": {k: v for k, v in step.vlm_input.items() if k != "image"},
            "vlm_response": step.vlm_response,
            "parsed_action": asdict(step.parsed_action) if step.parsed_action else None,
            "environment_command": asdict(step.environment_command) if step.environment_command else None,
            "execution_result": step.execution_result,
            "environment_state": step.environment_state,
            "feedback": step.feedback,
            "timestamp": step.timestamp
        }
        
        step_file = os.path.join(step_dir, "step_data.json")
        with open(step_file, 'w') as f:
            json.dump(step_data, f, indent=2, default=str)
    
    def _save_benchmark_result(self, result: BenchmarkResult) -> None:
        """Save final benchmark result."""
        result_file = os.path.join(self.config.output_dir, "benchmark_result.json")
        
        result_data = {
            "task_id": result.task_id,
            "success": result.success,
            "total_steps": result.total_steps,
            "execution_time": result.execution_time,
            "efficiency_score": result.efficiency_score,
            "error_message": result.error_message,
            "final_state": result.final_state,
            "config": {
                "model_name": self.config.model_name,
                "puzzle_type": self.config.puzzle_type.value,
                "max_steps": self.config.max_steps,
                "time_limit": self.config.time_limit
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_steps_executed": len(result.steps_history)
            }
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        print(f"Benchmark result saved to: {result_file}")


def run_luban_lock_benchmark(model_name: str, vlm_processor: VLLMProcessor, 
                            output_dir: str = "./luban_results") -> BenchmarkResult:
    """Convenience function to run Luban lock benchmark."""
    config = VLMBenchmarkConfig(
        model_name=model_name,
        puzzle_type=PuzzleType.LUBAN_LOCK,
        max_steps=100,
        time_limit=600.0,
        output_dir=output_dir
    )
    
    controller = VLMBenchmarkController(config, vlm_processor)
    return controller.run_benchmark()


def run_pagoda_benchmark(model_name: str, vlm_processor: VLLMProcessor,
                        output_dir: str = "./pagoda_results") -> BenchmarkResult:
    """Convenience function to run Pagoda benchmark."""
    config = VLMBenchmarkConfig(
        model_name=model_name,
        puzzle_type=PuzzleType.PAGODA,
        max_steps=50,
        time_limit=300.0,
        output_dir=output_dir
    )
    
    controller = VLMBenchmarkController(config, vlm_processor)
    return controller.run_benchmark()


def run_full_benchmark_suite(model_name: str, vlm_processor: VLLMProcessor,
                            base_output_dir: str = "./benchmark_results") -> Dict[str, BenchmarkResult]:
    """Run complete benchmark suite with all puzzle types."""
    results = {}
    
    # Run Luban lock benchmark
    luban_output = os.path.join(base_output_dir, "luban_lock")
    results["luban_lock"] = run_luban_lock_benchmark(model_name, vlm_processor, luban_output)
    
    # Run Pagoda benchmark  
    pagoda_output = os.path.join(base_output_dir, "pagoda")
    results["pagoda"] = run_pagoda_benchmark(model_name, vlm_processor, pagoda_output)
    
    # Save summary
    summary = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "results": {
            name: {
                "success": result.success,
                "total_steps": result.total_steps,
                "execution_time": result.execution_time,
                "efficiency_score": result.efficiency_score
            }
            for name, result in results.items()
        }
    }
    
    summary_file = os.path.join(base_output_dir, "benchmark_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Full benchmark suite completed. Summary saved to: {summary_file}")
    return results