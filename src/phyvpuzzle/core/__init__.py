"""
Core modules for PhyVPuzzle benchmark system.

This package contains the fundamental components that drive the physics-based
visual reasoning benchmark, including:
- Base classes for environments, tasks, and agents
- Configuration management
- Action parsing and execution
- Result aggregation and logging
"""

from phyvpuzzle.core.base import (
    BaseEnvironment,
    BaseTask,
    BaseAgent,
    BaseEvaluator,
    BaseJudge,
)

from phyvpuzzle.core.config import Config, load_config, create_default_config, validate_config, EnvironmentConfig, TaskConfig, AgentConfig, JudgementConfig, CameraConfig

from phyvpuzzle.core.registry import register_environment, register_environment_config, register_agent, register_task_config, register_task, ENVIRONMENT_REGISTRY, TASK_REGISTRY, AGENT_REGISTRY, TASK_CONFIG_REGISTRY

__all__ = [
    "BaseEnvironment",
    "BaseTask",
    "BaseAgent",
    "BaseEvaluator",
    "BaseJudge",
    "Config",
    "load_config",
    "create_default_config", 
    "validate_config",
    "EnvironmentConfig",
    "TaskConfig",
    "AgentConfig",
    "JudgementConfig",
    "register_environment",
    "register_environment_config",
    "register_task_config",
    "register_task",
    "register_agent",
    "CameraConfig",
    "ENVIRONMENT_REGISTRY",
    "TASK_REGISTRY",
    "AGENT_REGISTRY",
    "TASK_CONFIG_REGISTRY"
]
