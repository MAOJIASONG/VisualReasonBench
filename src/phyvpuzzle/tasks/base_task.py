"""
Base Task Module

This module provides the abstract base class for all physical visual reasoning tasks.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import numpy as np


class TaskType(Enum):
    """Types of physical visual reasoning tasks."""
    PUZZLE = "puzzle"
    LEGO = "lego"
    DOMINOES = "dominoes"


class TaskDifficulty(Enum):
    """Task difficulty levels."""
    VERY_EASY = "very-easy"  # New: single object tasks
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class TaskConfiguration:
    """Configuration for a task."""
    task_type: TaskType
    difficulty: TaskDifficulty
    max_steps: int = 100
    time_limit: float = 300.0  # seconds
    success_threshold: float = 0.9
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class TaskState:
    """Current state of a task."""
    step_count: int = 0
    is_completed: bool = False
    is_failed: bool = False
    success_rate: float = 0.0
    current_score: float = 0.0
    elapsed_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool
    final_score: float
    steps_taken: int
    time_taken: float
    error_message: Optional[str] = None
    metrics: Dict[str, float] = None
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    action_sequence: Optional[List[str]] = None
    task_id: Optional[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}


class BaseTask(ABC):
    """Abstract base class for all physical visual reasoning tasks."""
    
    def __init__(self, config: TaskConfiguration):
        self.config = config
        self.state = TaskState()
        self.environment = None
        self.initial_state = None
        self.target_state = None
        self.current_objects = {}
        
    @abstractmethod
    def setup_task(self, environment) -> bool:
        """
        Setup the task in the given environment.
        
        Args:
            environment: Physics environment instance
            
        Returns:
            True if setup successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """
        Get natural language description of the task.
        
        Returns:
            Task description string
        """
        pass
    
    @abstractmethod
    def check_completion(self) -> bool:
        """
        Check if the task is completed.
        
        Returns:
            True if task is completed, False otherwise
        """
        pass
    
    @abstractmethod
    def evaluate_state(self) -> float:
        """
        Evaluate the current state and return a score.
        
        Returns:
            Score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def get_optimal_solution(self) -> List[str]:
        """
        Get the optimal solution sequence.
        
        Returns:
            List of optimal action descriptions
        """
        pass
    
    @abstractmethod
    def reset_task(self) -> None:
        """Reset the task to initial state."""
        pass
    
    def update_state(self, action_taken: str, success: bool) -> None:
        """
        Update task state after an action.
        
        Args:
            action_taken: Description of the action taken
            success: Whether the action was successful
        """
        self.state.step_count += 1
        
        if success:
            self.state.current_score = self.evaluate_state()
            self.state.is_completed = self.check_completion()
        
        # Check failure conditions
        if self.state.step_count >= self.config.max_steps:
            self.state.is_failed = True
        
        if self.state.elapsed_time >= self.config.time_limit:
            self.state.is_failed = True
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get current context for VLLM processing.
        
        Returns:
            Context dictionary
        """
        return {
            "task_type": self.config.task_type.value,
            "difficulty": self.config.difficulty.value,
            "step_count": self.state.step_count,
            "max_steps": self.config.max_steps,
            "current_score": self.state.current_score,
            "is_completed": self.state.is_completed,
            "objects": self.current_objects,
            "task_specific": self.get_task_specific_context()
        }
    
    @abstractmethod
    def get_task_specific_context(self) -> Dict[str, Any]:
        """
        Get task-specific context information.
        
        Returns:
            Task-specific context dictionary
        """
        pass
    
    def get_progress(self) -> float:
        """
        Get task progress as percentage.
        
        Returns:
            Progress percentage (0.0 to 1.0)
        """
        return self.state.current_score
    
    def get_remaining_steps(self) -> int:
        """
        Get remaining steps allowed.
        
        Returns:
            Number of remaining steps
        """
        return max(0, self.config.max_steps - self.state.step_count)
    
    def get_hints(self) -> List[str]:
        """
        Get hints for the current state.
        
        Returns:
            List of hint strings
        """
        hints = []
        
        if self.state.step_count == 0:
            hints.append("Start by analyzing the current state of objects.")
        
        if self.state.step_count > self.config.max_steps * 0.5:
            hints.append("You're halfway through the allowed steps.")
        
        if self.state.step_count > self.config.max_steps * 0.8:
            hints.append("You're running out of steps. Focus on the most important actions.")
        
        return hints
    
    def calculate_distance_to_optimal(self, current_actions: List[str]) -> float:
        """
        Calculate distance from current actions to optimal solution.
        
        Args:
            current_actions: List of actions taken so far
            
        Returns:
            Distance metric (0.0 = optimal, higher = further from optimal)
        """
        optimal_actions = self.get_optimal_solution()
        
        if not optimal_actions:
            return 0.0
        
        # Simple edit distance calculation
        current_len = len(current_actions)
        optimal_len = len(optimal_actions)
        
        if current_len == 0:
            return float(optimal_len)
        
        # Calculate similarity based on action sequence
        common_prefix = 0
        for i in range(min(current_len, optimal_len)):
            if current_actions[i] == optimal_actions[i]:
                common_prefix += 1
            else:
                break
        
        # Distance = steps remaining + penalty for wrong actions
        steps_remaining = optimal_len - common_prefix
        wrong_actions = current_len - common_prefix
        
        return float(steps_remaining + wrong_actions)
    
    def get_result(self) -> TaskResult:
        """
        Get final task result.
        
        Returns:
            TaskResult object
        """
        return TaskResult(
            success=self.state.is_completed,
            final_score=self.state.current_score,
            steps_taken=self.state.step_count,
            time_taken=self.state.elapsed_time,
            error_message=None if self.state.is_completed else "Task not completed",
            metrics={
                "success_rate": float(self.state.is_completed),
                "efficiency": 1.0 - (self.state.step_count / self.config.max_steps),
                "time_efficiency": 1.0 - (self.state.elapsed_time / self.config.time_limit),
                "final_score": self.state.current_score
            }
        )
    
    def is_task_finished(self) -> bool:
        """
        Check if task is finished (completed or failed).
        
        Returns:
            True if task is finished, False otherwise
        """
        return self.state.is_completed or self.state.is_failed
    
    def get_visualization_elements(self) -> Dict[str, Any]:
        """
        Get elements for task visualization.
        
        Returns:
            Dictionary with visualization elements
        """
        return {
            "task_info": {
                "type": self.config.task_type.value,
                "difficulty": self.config.difficulty.value,
                "step_count": self.state.step_count,
                "max_steps": self.config.max_steps,
                "score": self.state.current_score
            },
            "progress_bar": {
                "current": self.state.step_count,
                "total": self.config.max_steps,
                "percentage": (self.state.step_count / self.config.max_steps) * 100
            },
            "status": {
                "completed": self.state.is_completed,
                "failed": self.state.is_failed,
                "in_progress": not self.is_task_finished()
            }
        }


class TaskValidator:
    """Validator for task configurations and states."""
    
    @staticmethod
    def validate_config(config: TaskConfiguration) -> bool:
        """
        Validate task configuration.
        
        Args:
            config: Task configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        if config.max_steps <= 0:
            return False
        
        if config.time_limit <= 0:
            return False
        
        if not (0.0 <= config.success_threshold <= 1.0):
            return False
        
        return True
    
    @staticmethod
    def validate_state(state: TaskState) -> bool:
        """
        Validate task state.
        
        Args:
            state: Task state to validate
            
        Returns:
            True if valid, False otherwise
        """
        if state.step_count < 0:
            return False
        
        if not (0.0 <= state.current_score <= 1.0):
            return False
        
        if state.elapsed_time < 0:
            return False
        
        return True


def create_task_config(task_type: str, difficulty: str = "medium", 
                      **kwargs) -> TaskConfiguration:
    """
    Factory function to create task configuration.
    
    Args:
        task_type: Type of task ("puzzle", "lego", "dominoes")
        difficulty: Task difficulty ("easy", "medium", "hard", "expert")
        **kwargs: Additional configuration parameters
        
    Returns:
        TaskConfiguration instance
    """
    return TaskConfiguration(
        task_type=TaskType(task_type),
        difficulty=TaskDifficulty(difficulty),
        **kwargs
    ) 