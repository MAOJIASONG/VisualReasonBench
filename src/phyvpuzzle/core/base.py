"""
Base classes and interfaces for the PhyVPuzzle benchmark system.

This module defines the fundamental abstractions for environments, tasks,
agents, and evaluators that form the backbone of the benchmark.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from PIL import Image


class TaskType(Enum):
    """Types of physics puzzles supported."""
    DOMINO = "domino"
    LUBAN_LOCK = "luban_lock"  
    PAGODA = "pagoda"
    TANGLED_NAILS = "tangled_nails"


class TaskDifficulty(Enum):
    """Difficulty levels for tasks."""
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


@dataclass
class Action:
    """Represents an action to be executed in the environment."""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary representation."""
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
            "timestamp": self.timestamp
        }


@dataclass
class State:
    """Represents the state of the environment at a given time."""
    step: int
    objects: Dict[str, Any]
    completed: bool
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "step": self.step,
            "objects": self.objects,
            "completed": self.completed,
            "success": self.success,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class Observation:
    """Observation data provided to the agent."""
    image: Image.Image
    state: State
    description: str
    available_actions: List[str]
    multi_view_images: Optional[Dict[str, Image.Image]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary (excluding images)."""
        return {
            "state": self.state.to_dict(),
            "description": self.description,
            "available_actions": self.available_actions,
            "has_multi_view": self.multi_view_images is not None
        }


@dataclass
class TaskResult:
    """Results from task execution."""
    task_id: str
    task_type: TaskType
    success: bool
    total_steps: int
    execution_time: float
    final_state: State
    trajectory: List[Tuple[Action, State, str]]  # (action, resulting_state, feedback)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task result to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "success": self.success,
            "total_steps": self.total_steps,
            "execution_time": self.execution_time,
            "final_state": self.final_state.to_dict(),
            "trajectory_length": len(self.trajectory),
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class EvaluationResult:
    """Results from evaluation of task performance."""
    accuracy: float
    pass_at_k: Dict[int, float]  # k -> success rate
    distance_to_optimal: float
    token_efficiency: float
    detailed_metrics: Dict[str, float]
    task_results: List[TaskResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary representation."""
        return {
            "accuracy": self.accuracy,
            "pass_at_k": self.pass_at_k,
            "distance_to_optimal": self.distance_to_optimal,
            "token_efficiency": self.token_efficiency,
            "detailed_metrics": self.detailed_metrics,
            "num_tasks": len(self.task_results)
        }


class BaseEnvironment(ABC):
    """Base class for physics simulation environments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_state = None
        self.step_count = 0
        
    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment to initial state."""
        pass
    
    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, str, bool]:
        """Execute action and return new observation, feedback, and done flag."""
        pass
    
    @abstractmethod
    def render(self, multi_view: bool = True) -> Union[Image.Image, Dict[str, Image.Image]]:
        """Render the current environment state."""
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """Get list of available action types."""
        pass
    
    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for tool functions that VLM can call."""
        pass
    
    @abstractmethod
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call from VLM."""
        pass
    
    @abstractmethod
    def is_success(self) -> bool:
        """Check if task has been completed successfully."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass


class BaseTask(ABC):
    """Base class for benchmark tasks."""
    
    def __init__(self, task_type: TaskType, difficulty: TaskDifficulty, config: Dict[str, Any]):
        self.task_type = task_type
        self.difficulty = difficulty
        self.config = config
        self.task_id = f"{task_type.value}_{difficulty.value}_{int(time.time())}"
        
    @abstractmethod
    def setup_environment(self, environment: BaseEnvironment) -> None:
        """Setup the environment for this specific task."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for this task."""
        pass
    
    @abstractmethod
    def get_initial_user_prompt(self) -> str:
        """Get initial user prompt to start the task."""
        pass
    
    @abstractmethod
    def is_complete(self, state: State) -> bool:
        """Check if task is complete based on current state."""
        pass
    
    @abstractmethod
    def calculate_optimal_steps(self) -> int:
        """Calculate optimal number of steps for this task."""
        pass


class BaseAgent(ABC):
    """Base class for VLM agents."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.total_tokens = 0
        
    @abstractmethod
    def process_observation(self, observation: Observation, system_prompt: str, 
                          user_prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process observation and return response and tool calls.
        
        Returns:
            Tuple of (text_response, tool_calls_list)
        """
        pass
    
    @abstractmethod
    def get_token_count(self) -> int:
        """Get total tokens used by this agent."""
        pass


class BaseEvaluator(ABC):
    """Base class for evaluating task performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def evaluate_single_task(self, task_result: TaskResult) -> Dict[str, float]:
        """Evaluate a single task result."""
        pass
    
    @abstractmethod
    def evaluate_multiple_tasks(self, task_results: List[TaskResult]) -> EvaluationResult:
        """Evaluate multiple task results and aggregate metrics."""
        pass
    
    @abstractmethod
    def calculate_pass_at_k(self, task_results: List[TaskResult], k_values: List[int]) -> Dict[int, float]:
        """Calculate pass@k metrics."""
        pass


class BaseJudge(ABC):
    """Base class for LLM-as-judge evaluation."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        
    @abstractmethod
    def judge_success(self, final_image: Image.Image, task_description: str, 
                     trajectory: List[str]) -> Tuple[bool, float, str]:
        """
        Judge if task was completed successfully.
        
        Returns:
            Tuple of (success, confidence_score, reasoning)
        """
        pass
