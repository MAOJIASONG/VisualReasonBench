"""
Main runner for PhyVPuzzle benchmark execution.

This module coordinates the execution of benchmark tasks, bringing together
environments, agents, tasks, and evaluation components.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .core.base import TaskResult, Action, State, Observation
from .core.config import Config
from .environment.base_env import PhysicsEnvironment
from .environment.domino_env import DominoEnvironment
from .tasks.base_task import PuzzleTask
from .tasks.domino_task import DominoTask
from .tasks.luban_task import LubanTask
from .agents.base_agent import VLMAgent
from .agents.openai_agent import OpenAIAgent
from .agents.vllm_agent import VLLMAgent
from .evaluation.evaluator import BenchmarkEvaluator
from .utils.logger import ExperimentLogger


class BenchmarkRunner:
    """Main runner for PhyVPuzzle benchmark execution."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = ExperimentLogger(
            log_dir=config.runner.log_dir,
            experiment_name=config.runner.experiment_name
        )
        
        # Initialize components
        self.environment: Optional[PhysicsEnvironment] = None
        self.task: Optional[PuzzleTask] = None
        self.agent: Optional[VLMAgent] = None
        self.evaluator: Optional[BenchmarkEvaluator] = None
        
        # Execution state
        self.current_trajectory = []
        self.start_time = None
        
    def setup(self) -> None:
        """Setup all components for benchmark execution."""
        print(f"Setting up benchmark: {self.config.runner.experiment_name}")
        
        # Create environment
        self.environment = self._create_environment()
        
        # Create task
        self.task = self._create_task()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create evaluator
        self.evaluator = self._create_evaluator()
        
        print("Benchmark setup complete")
        
    def _create_environment(self) -> PhysicsEnvironment:
        """Create physics environment based on task type."""
        env_config = {
            "gui": self.config.environment.gui,
            "urdf_base_path": self.config.environment.urdf_base_path,
            "gravity": self.config.environment.gravity,
            "render_width": self.config.environment.render_width,
            "render_height": self.config.environment.render_height,
            "multi_view": self.config.environment.multi_view,
            "max_steps": self.config.runner.max_steps
        }
        
        # Create environment based on task type
        if self.config.task.type.value == "domino":
            from .environment.domino_env import DominoEnvironment, DominoConfig
            domino_config = DominoConfig.from_difficulty(self.config.task.difficulty)
            return DominoEnvironment(env_config, domino_config)
        else:
            # Default to base physics environment
            return PhysicsEnvironment(env_config)
            
    def _create_task(self) -> PuzzleTask:
        """Create task based on configuration."""
        task_config = self.config.task.parameters
        
        if self.config.task.type.value == "domino":
            return DominoTask(self.config.task.difficulty, task_config)
        elif self.config.task.type.value == "luban_lock":
            return LubanTask(self.config.task.difficulty, task_config)
        else:
            raise ValueError(f"Unknown task type: {self.config.task.type}")
            
    def _create_agent(self) -> VLMAgent:
        """Create VLM agent based on configuration."""
        agent_config = {
            "api_key": self.config.agent.api_key,
            "base_url": self.config.agent.base_url,
            "temperature": self.config.agent.temperature,
            "max_tokens": self.config.agent.max_tokens,
            "timeout": self.config.agent.timeout
        }
        
        model_name = self.config.agent.model_name
        
        # Determine agent type based on model name or config
        if "vllm" in model_name.lower() or self.config.agent.base_url and "vllm" in self.config.agent.base_url:
            return VLLMAgent(model_name, agent_config)
        else:
            return OpenAIAgent(model_name, agent_config)
            
    def _create_evaluator(self) -> BenchmarkEvaluator:
        """Create evaluator with judge if configured."""
        eval_config = {}
        
        if self.config.judge_agent:
            eval_config["judge"] = {
                "model_name": self.config.judge_agent.model_name,
                "api_key": self.config.judge_agent.api_key,
                "base_url": self.config.judge_agent.base_url,
                "temperature": self.config.judge_agent.temperature,
                "max_tokens": self.config.judge_agent.max_tokens
            }
            
        return BenchmarkEvaluator(eval_config)
        
    def run_single_task(self) -> TaskResult:
        """Run a single task instance."""
        if not all([self.environment, self.task, self.agent]):
            raise RuntimeError("Components not properly initialized. Call setup() first.")
            
        print(f"Starting task: {self.task.task_id}")
        
        self.start_time = time.time()
        self.current_trajectory = []
        
        try:
            # Setup task environment
            self.task.setup_environment(self.environment)
            
            # Reset environment and get initial observation
            observation = self.environment.reset()
            
            # Get task prompts
            system_prompt = self.task.get_system_prompt()
            initial_prompt = self.task.get_initial_user_prompt()
            
            # Log initial state
            self._log_step(0, {
                "step_type": "initial",
                "observation": observation.to_dict(),
                "system_prompt": system_prompt,
                "user_prompt": initial_prompt,
                "image": observation.image,
                "multi_view_images": observation.multi_view_images
            })
            
            # Execute task loop
            step = 0
            max_steps = self.task.get_max_steps()
            
            while step < max_steps and not observation.state.completed:
                step += 1
                
                # Get agent response
                response, tool_calls = self.agent.process_observation(
                    observation, system_prompt, initial_prompt
                )
                
                # Execute tool calls
                action_executed = False
                final_feedback = response
                
                for tool_call in tool_calls:
                    try:
                        tool_name = tool_call["function"]["name"]
                        arguments = json.loads(tool_call["function"]["arguments"])
                        
                        # Execute tool in environment
                        tool_result = self.environment.execute_tool_call(tool_name, arguments)
                        
                        # Create action object
                        action = Action(action_type=tool_name, parameters=arguments)
                        
                        # Step environment
                        new_observation, feedback, done = self.environment.step(action)
                        
                        # Record in trajectory
                        self.current_trajectory.append((action, new_observation.state, feedback))
                        
                        # Log step
                        self._log_step(step, {
                            "step_type": "action",
                            "agent_response": response,
                            "tool_call": tool_call,
                            "tool_result": tool_result,
                            "action": action.to_dict(),
                            "observation": new_observation.to_dict(),
                            "feedback": feedback,
                            "done": done,
                            "image": new_observation.image,
                            "multi_view_images": new_observation.multi_view_images
                        })
                        
                        observation = new_observation
                        final_feedback = feedback
                        action_executed = True
                        
                        if done:
                            break
                            
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_call.get('function', {}).get('name', 'unknown')}: {e}"
                        print(error_msg)
                        
                        self._log_step(step, {
                            "step_type": "error", 
                            "error": error_msg,
                            "tool_call": tool_call,
                            "agent_response": response
                        })
                        
                        final_feedback = error_msg
                
                # If no tools were executed, log the response
                if not action_executed:
                    self._log_step(step, {
                        "step_type": "response_only",
                        "agent_response": response,
                        "observation": observation.to_dict()
                    })
                
                # Update prompts for next iteration
                initial_prompt = f"Previous step result: {final_feedback}\n\nWhat's your next action?"
                
            # Create task result
            execution_time = time.time() - self.start_time
            
            task_result = TaskResult(
                task_id=self.task.task_id,
                task_type=self.task.task_type,
                success=observation.state.success,
                total_steps=step,
                execution_time=execution_time,
                final_state=observation.state,
                trajectory=self.current_trajectory,
                metadata={
                    "difficulty": self.task.difficulty.value,
                    "optimal_steps": self.task.calculate_optimal_steps(),
                    "total_tokens": self.agent.get_token_count(),
                    "agent_model": self.agent.model_name,
                    "task_config": self.task.to_dict()
                }
            )
            
            # Save final results
            self.logger.save_logs()
            
            print(f"Task completed: {task_result.success} in {step} steps ({execution_time:.2f}s)")
            
            return task_result
            
        except Exception as e:
            execution_time = time.time() - self.start_time if self.start_time else 0
            
            error_result = TaskResult(
                task_id=self.task.task_id,
                task_type=self.task.task_type,
                success=False,
                total_steps=len(self.current_trajectory),
                execution_time=execution_time,
                final_state=observation.state if 'observation' in locals() else State(0, {}, False, False),
                trajectory=self.current_trajectory,
                error_message=str(e),
                metadata={
                    "difficulty": self.task.difficulty.value,
                    "total_tokens": self.agent.get_token_count() if self.agent else 0,
                    "agent_model": self.agent.model_name if self.agent else "unknown"
                }
            )
            
            print(f"Task failed with error: {e}")
            return error_result
            
        finally:
            # Cleanup
            if self.environment:
                self.environment.close()
                
    def run_multiple_tasks(self, num_runs: int = 1) -> List[TaskResult]:
        """Run multiple instances of the same task."""
        results = []
        
        for i in range(num_runs):
            print(f"Running task instance {i+1}/{num_runs}")
            
            # Reinitialize for each run
            self.setup()
            
            result = self.run_single_task()
            results.append(result)
            
            # Reset agent state for next run
            if self.agent:
                self.agent.reset_conversation()
                
        return results
        
    def run_benchmark(self, num_runs: int = 1) -> None:
        """Run complete benchmark with evaluation."""
        print(f"Starting benchmark with {num_runs} runs")
        
        # Run tasks
        task_results = self.run_multiple_tasks(num_runs)
        
        # Evaluate results
        evaluation_result = self.evaluator.evaluate_multiple_tasks(task_results)
        
        # Export results to Excel
        self.evaluator.export_results_to_excel(
            evaluation_result,
            self.config.runner.results_excel_path,
            self.config.agent.model_name
        )
        
        # Export detailed report
        report_dir = os.path.join(self.config.runner.log_dir, "detailed_reports")
        self.evaluator.export_detailed_report(
            evaluation_result,
            report_dir,
            self.config.agent.model_name
        )
        
        # Print summary
        self._print_benchmark_summary(evaluation_result)
        
    def _log_step(self, step: int, data: Dict[str, Any]) -> None:
        """Log a single step of execution."""
        self.logger.log_step(step, data)
        
    def _print_benchmark_summary(self, evaluation_result) -> None:
        """Print benchmark summary to console."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Model: {self.config.agent.model_name}")
        print(f"Task: {self.config.task.name} ({self.config.task.difficulty.value})")
        print(f"Total Tasks: {len(evaluation_result.task_results)}")
        print(f"Successful: {sum(1 for r in evaluation_result.task_results if r.success)}")
        print(f"Accuracy: {evaluation_result.accuracy:.1%}")
        
        if evaluation_result.pass_at_k:
            for k, rate in evaluation_result.pass_at_k.items():
                print(f"Pass@{k}: {rate:.1%}")
                
        if evaluation_result.distance_to_optimal != float('inf'):
            print(f"Distance to Optimal: {evaluation_result.distance_to_optimal:.2f}")
            
        if evaluation_result.token_efficiency != float('inf'):
            print(f"Token Efficiency: {evaluation_result.token_efficiency:.0f} tokens/success")
            
        print(f"Results saved to: {self.config.runner.results_excel_path}")
        print("="*60)
