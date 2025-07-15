"""
Pipeline Module

This module provides the main pipeline that coordinates the interaction between
VLLM, ActionDescriptor, Translator, and Environment components.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import time
from dataclasses import dataclass
from PIL import Image
import logging

from .vllm_processor import VLLMProcessor, DecisionParser, create_vllm_processor
from .action_descriptor import ActionDescriptor, ParsedAction
from .translator import Translator, TranslationResult, create_translator
from ..environment.physics_env import PhysicsEnvironment, create_environment
from ..tasks.base_task import BaseTask, TaskResult


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    vllm_type: str = "openai"
    vllm_model: str = "gpt-4-vision-preview"
    translator_type: str = "rule_based"
    environment_type: str = "pybullet"
    max_iterations: int = 100
    timeout: float = 300.0
    enable_logging: bool = True
    log_level: str = "INFO"
    feedback_history_size: int = 5
    retry_attempts: int = 3


@dataclass
class PipelineStep:
    """Represents a single step in the pipeline execution."""
    step_number: int
    image: Image.Image
    vllm_response: str
    decision_type: str
    decision_description: str
    parsed_action: Optional[ParsedAction] = None
    translation_result: Optional[TranslationResult] = None
    execution_success: bool = False
    execution_time: float = 0.0
    error_message: Optional[str] = None


class PhysicalReasoningPipeline:
    """
    Main pipeline for physical visual reasoning tasks.
    
    This pipeline coordinates the flow:
    Task -> VLLM -> choose: finish/action -> Action Description -> 
    Translator -> Environment -> 3D feedback + history -> VLLM
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_logging()
        
        # Initialize components
        self.vllm_processor = None
        self.action_descriptor = ActionDescriptor()
        self.translator = None
        self.environment = None
        self.decision_parser = DecisionParser()
        
        # Pipeline state
        self.current_task = None
        self.execution_history = []
        self.feedback_history = []
        self.step_count = 0
        self.start_time = None
        
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        if self.config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
    
    def initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Initialize VLLM processor
            self.vllm_processor = create_vllm_processor(
                processor_type=self.config.vllm_type,
                model_name=self.config.vllm_model
            )
            self.vllm_processor.load_model()
            
            # Initialize translator
            self.translator = create_translator(
                translator_type=self.config.translator_type
            )
            
            # Initialize environment
            self.environment = create_environment(
                env_type=self.config.environment_type,
                gui=False
            )
            self.environment.setup_environment()
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def execute_task(self, task: BaseTask) -> TaskResult:
        """
        Execute a complete task using the pipeline.
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult with execution details
        """
        self.current_task = task
        self.start_time = time.time()
        self.step_count = 0
        self.execution_history.clear()
        self.feedback_history.clear()
        
        try:
            # Setup task in environment
            if not task.setup_task(self.environment):
                return TaskResult(
                    success=False,
                    final_score=0.0,
                    steps_taken=0,
                    time_taken=0.0,
                    error_message="Failed to setup task in environment"
                )
            
            # Main execution loop
            while (not task.is_task_finished() and 
                   self.step_count < self.config.max_iterations and
                   self._get_elapsed_time() < self.config.timeout):
                
                step_result = self._execute_single_step()
                
                if step_result.error_message:
                    self.logger.warning(f"Step {self.step_count} failed: {step_result.error_message}")
                    if not self._should_retry():
                        break
                
                self.step_count += 1
                task.state.elapsed_time = self._get_elapsed_time()
            
            # Get final result
            result = task.get_result()
            result.time_taken = self._get_elapsed_time()
            
            self.logger.info(f"Task completed. Success: {result.success}, "
                           f"Score: {result.final_score:.3f}, "
                           f"Steps: {result.steps_taken}, "
                           f"Time: {result.time_taken:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return TaskResult(
                success=False,
                final_score=0.0,
                steps_taken=self.step_count,
                time_taken=self._get_elapsed_time(),
                error_message=str(e)
            )
    
    def _execute_single_step(self) -> PipelineStep:
        """
        Execute a single step of the pipeline.
        
        Returns:
            PipelineStep with execution details
        """
        step_start_time = time.time()
        
        # Step 1: Render current environment state
        try:
            current_image = self.environment.render()
        except Exception as e:
            return PipelineStep(
                step_number=self.step_count,
                image=None,
                vllm_response="",
                decision_type="error",
                decision_description="",
                error_message=f"Failed to render environment: {e}"
            )
        
        # Step 2: Get task context
        task_context = self.current_task.get_context()
        task_description = self.current_task.get_task_description()
        
        # Step 3: Process with VLLM
        try:
            vllm_response = self.vllm_processor.process_input(
                image=current_image,
                task_description=task_description,
                context=task_context
            )
        except Exception as e:
            return PipelineStep(
                step_number=self.step_count,
                image=current_image,
                vllm_response="",
                decision_type="error",
                decision_description="",
                error_message=f"VLLM processing failed: {e}"
            )
        
        # Step 4: Parse decision
        decision_type, decision_description = self.decision_parser.parse_decision(vllm_response)
        
        step = PipelineStep(
            step_number=self.step_count,
            image=current_image,
            vllm_response=vllm_response,
            decision_type=decision_type,
            decision_description=decision_description,
            execution_time=time.time() - step_start_time
        )
        
        # Step 5: Handle decision
        if decision_type == "finish":
            self.current_task.state.is_completed = True
            step.execution_success = True
            self.logger.info(f"Task finished at step {self.step_count}")
        
        elif decision_type == "action":
            # Step 6: Parse action
            try:
                parsed_action = self.action_descriptor.parse_action(decision_description)
                step.parsed_action = parsed_action
            except Exception as e:
                step.error_message = f"Action parsing failed: {e}"
                return step
            
            # Step 7: Translate to environment commands
            try:
                env_state = self.environment.get_state()
                translation_result = self.translator.translate_action(parsed_action, env_state)
                step.translation_result = translation_result
                
                if not translation_result.success:
                    step.error_message = f"Translation failed: {translation_result.error_message}"
                    return step
            except Exception as e:
                step.error_message = f"Translation failed: {e}"
                return step
            
            # Step 8: Execute commands in environment
            try:
                execution_success = True
                for command in translation_result.commands:
                    if not self.environment.execute_command(command):
                        execution_success = False
                        break
                
                step.execution_success = execution_success
                
                if execution_success:
                    self.current_task.update_state(decision_description, True)
                    self.logger.info(f"Step {self.step_count}: Action executed successfully")
                else:
                    self.current_task.update_state(decision_description, False)
                    step.error_message = "Command execution failed"
                    
            except Exception as e:
                step.error_message = f"Command execution failed: {e}"
                step.execution_success = False
        
        # Step 9: Add to history
        self.execution_history.append(step)
        self._update_feedback_history(step)
        
        return step
    
    def _update_feedback_history(self, step: PipelineStep) -> None:
        """Update feedback history for next VLLM call."""
        feedback = {
            "step": step.step_number,
            "action": step.decision_description,
            "success": step.execution_success,
            "result": "Success" if step.execution_success else step.error_message or "Failed"
        }
        
        self.feedback_history.append(feedback)
        
        # Keep only recent feedback
        if len(self.feedback_history) > self.config.feedback_history_size:
            self.feedback_history.pop(0)
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time since task start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def _should_retry(self) -> bool:
        """Determine if we should retry after a failure."""
        # Simple retry logic - can be enhanced
        recent_failures = sum(1 for step in self.execution_history[-3:] 
                            if step.error_message is not None)
        return recent_failures < self.config.retry_attempts
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of pipeline execution.
        
        Returns:
            Dictionary with execution summary
        """
        if not self.execution_history:
            return {"status": "No execution history"}
        
        successful_steps = sum(1 for step in self.execution_history 
                             if step.execution_success)
        failed_steps = len(self.execution_history) - successful_steps
        
        return {
            "total_steps": len(self.execution_history),
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "success_rate": successful_steps / len(self.execution_history) if self.execution_history else 0.0,
            "total_time": self._get_elapsed_time(),
            "avg_step_time": sum(step.execution_time for step in self.execution_history) / len(self.execution_history),
            "task_completed": self.current_task.state.is_completed if self.current_task else False,
            "final_score": self.current_task.state.current_score if self.current_task else 0.0
        }
    
    def get_step_details(self, step_number: int) -> Optional[PipelineStep]:
        """
        Get details of a specific step.
        
        Args:
            step_number: Step number to retrieve
            
        Returns:
            PipelineStep or None if not found
        """
        for step in self.execution_history:
            if step.step_number == step_number:
                return step
        return None
    
    def reset_pipeline(self) -> None:
        """Reset pipeline state for new task."""
        self.current_task = None
        self.execution_history.clear()
        self.feedback_history.clear()
        self.step_count = 0
        self.start_time = None
        
        # Reset component states
        if self.vllm_processor:
            self.vllm_processor.clear_history()
        
        if self.environment:
            self.environment.reset()
        
        self.logger.info("Pipeline reset completed")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.environment:
            self.environment.close()
        
        self.logger.info("Pipeline cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize_components()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class PipelineEvaluator:
    """Evaluator for pipeline performance."""
    
    def __init__(self):
        self.evaluation_metrics = []
    
    def evaluate_pipeline(self, pipeline: PhysicalReasoningPipeline, 
                         tasks: List[BaseTask]) -> Dict[str, Any]:
        """
        Evaluate pipeline performance on multiple tasks.
        
        Args:
            pipeline: Pipeline to evaluate
            tasks: List of tasks to evaluate on
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = []
        
        for task in tasks:
            result = pipeline.execute_task(task)
            results.append(result)
            pipeline.reset_pipeline()
        
        # Calculate overall metrics
        successful_tasks = sum(1 for r in results if r.success)
        total_tasks = len(results)
        
        avg_score = sum(r.final_score for r in results) / total_tasks if results else 0.0
        avg_steps = sum(r.steps_taken for r in results) / total_tasks if results else 0.0
        avg_time = sum(r.time_taken for r in results) / total_tasks if results else 0.0
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "average_score": avg_score,
            "average_steps": avg_steps,
            "average_time": avg_time,
            "task_results": results
        }


def create_pipeline(config: Optional[PipelineConfig] = None) -> PhysicalReasoningPipeline:
    """
    Factory function to create pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        PhysicalReasoningPipeline instance
    """
    if config is None:
        config = PipelineConfig()
    
    return PhysicalReasoningPipeline(config) 