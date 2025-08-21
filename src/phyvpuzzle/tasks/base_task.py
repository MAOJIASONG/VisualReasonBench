"""
Base task implementation for PhyVPuzzle benchmarks.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional

from ..core.base import BaseTask, BaseEnvironment, TaskType, TaskDifficulty, State


class PuzzleTask(BaseTask):
    """Base class for physics puzzle tasks."""
    
    def __init__(self, task_type: TaskType, difficulty: TaskDifficulty, config: Dict[str, Any]):
        super().__init__(task_type, difficulty, config)
        self.optimal_steps = self._calculate_optimal_steps()
        
    def setup_environment(self, environment: BaseEnvironment) -> None:
        """Setup the environment for this specific task."""
        # Apply task-specific environment configuration
        self._configure_environment(environment)
        
        # Load task-specific objects and constraints
        self._load_task_objects(environment)
        
        # Set initial state
        self._set_initial_state(environment)
        
    def get_system_prompt(self) -> str:
        """Get system prompt for this task type."""
        base_prompt = self._get_base_system_prompt()
        task_specific = self._get_task_specific_prompt()
        
        return f"{base_prompt}\n\n{task_specific}"
        
    def get_initial_user_prompt(self) -> str:
        """Get initial user prompt to start the task."""
        return self._get_task_description() + "\n\n" + self._get_initial_instruction()
        
    def is_complete(self, state: State) -> bool:
        """Check if task is complete based on current state."""
        return state.completed
        
    def calculate_optimal_steps(self) -> int:
        """Calculate optimal number of steps for this task."""
        return self.optimal_steps
        
    # Abstract methods for subclasses
    @abstractmethod
    def _configure_environment(self, environment: BaseEnvironment) -> None:
        """Configure environment settings for this task."""
        pass
        
    @abstractmethod
    def _load_task_objects(self, environment: BaseEnvironment) -> None:
        """Load task-specific objects into environment."""
        pass
        
    @abstractmethod
    def _set_initial_state(self, environment: BaseEnvironment) -> None:
        """Set initial state for the task."""
        pass
        
    @abstractmethod
    def _get_base_system_prompt(self) -> str:
        """Get base system prompt for this task type."""
        pass
        
    @abstractmethod
    def _get_task_specific_prompt(self) -> str:
        """Get task-specific additions to system prompt."""
        pass
        
    @abstractmethod
    def _get_task_description(self) -> str:
        """Get description of the specific task instance."""
        pass
        
    @abstractmethod
    def _get_initial_instruction(self) -> str:
        """Get initial instruction for starting the task."""
        pass
        
    @abstractmethod
    def _calculate_optimal_steps(self) -> int:
        """Calculate optimal number of steps for this specific task."""
        pass
        
    # Utility methods
    def get_difficulty_multiplier(self) -> float:
        """Get difficulty-based multiplier for various metrics."""
        multipliers = {
            TaskDifficulty.VERY_EASY: 0.5,
            TaskDifficulty.EASY: 0.7, 
            TaskDifficulty.MEDIUM: 1.0,
            TaskDifficulty.HARD: 1.5,
            TaskDifficulty.VERY_HARD: 2.0
        }
        return multipliers.get(self.difficulty, 1.0)
        
    def get_max_steps(self) -> int:
        """Get maximum allowed steps for this task."""
        base_steps = self.config.get("max_steps", 50)
        return int(base_steps * self.get_difficulty_multiplier())
        
    def get_time_limit(self) -> float:
        """Get time limit for this task."""
        base_time = self.config.get("time_limit", 300.0)
        return base_time * self.get_difficulty_multiplier()
        
    def get_success_criteria(self) -> Dict[str, Any]:
        """Get success criteria for this task."""
        return self.config.get("success_criteria", {})
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "difficulty": self.difficulty.value,
            "optimal_steps": self.optimal_steps,
            "max_steps": self.get_max_steps(),
            "time_limit": self.get_time_limit(),
            "config": self.config
        }
