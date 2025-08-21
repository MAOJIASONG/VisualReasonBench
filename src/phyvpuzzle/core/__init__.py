"""
Core modules for PhyVPuzzle benchmark system.

This package contains the fundamental components that drive the physics-based
visual reasoning benchmark, including:
- Base classes for environments, tasks, and agents
- Configuration management
- Action parsing and execution
- Result aggregation and logging
"""

from .base import (
    BaseEnvironment,
    BaseTask, 
    BaseAgent,
    BaseEvaluator,
    Action,
    TaskResult,
    EvaluationResult
)

from .config import Config, load_config

__all__ = [
    "BaseEnvironment",
    "BaseTask",
    "BaseAgent", 
    "BaseEvaluator",
    "Action",
    "TaskResult",
    "EvaluationResult",
    "Config",
    "load_config"
]
