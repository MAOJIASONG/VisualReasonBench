"""
Configuration management for PhyVPuzzle benchmark.

This module handles loading and validation of configuration files,
environment variables, and provides typed configuration objects.
"""

import os
import yaml
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path

from .base import TaskType, TaskDifficulty


@dataclass
class AgentConfig:
    """Configuration for VLM agents."""
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500
    timeout: float = 30.0
    
    def __post_init__(self):
        # Load from environment variables if not provided
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


@dataclass
class JudgeAgentConfig:
    """Configuration for judge agents (LLM-as-judge)."""
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 50
    timeout: float = 30.0
    
    def __post_init__(self):
        # Load from environment variables if not provided
        if not self.api_key:
            self.api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.base_url:
            self.base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


@dataclass
class EnvironmentConfig:
    """Configuration for physics environment."""
    type: str = "pybullet"
    gui: bool = False
    urdf_base_path: str = "src/phyvpuzzle/environment/phobos_models"
    gravity: float = -9.81
    time_step: float = 1.0/240.0
    render_width: int = 512
    render_height: int = 512
    multi_view: bool = True
    load_table: bool = True
    
    def __post_init__(self):
        # Convert to absolute path if relative
        if not os.path.isabs(self.urdf_base_path):
            self.urdf_base_path = os.path.abspath(self.urdf_base_path)


@dataclass  
class TaskConfig:
    """Configuration for specific tasks."""
    name: str
    type: Optional[TaskType] = None
    difficulty: Optional[TaskDifficulty] = None
    max_steps: int = 50
    time_limit: float = 300.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = TaskType(self.type)
        if isinstance(self.difficulty, str):
            self.difficulty = TaskDifficulty(self.difficulty)


@dataclass
class RunnerConfig:
    """Configuration for experiment runner."""
    experiment_name: str
    log_dir: str = "logs"
    results_excel_path: str = "logs/experiment_results.xlsx"
    max_steps: int = 50
    save_images: bool = True
    save_multi_view: bool = True
    save_trajectory: bool = True
    
    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.results_excel_path), exist_ok=True)


@dataclass
class Config:
    """Main configuration object."""
    runner: RunnerConfig
    agent: AgentConfig
    judge_agent: Optional[JudgeAgentConfig] = None
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        runner_data = data.get("runner", {})
        runner = RunnerConfig(**runner_data)
        
        agent_data = data.get("agent", {})
        agent = AgentConfig(**agent_data)
        
        judge_data = data.get("judge_agent")
        judge_agent = JudgeAgentConfig(**judge_data) if judge_data else None
        
        env_data = data.get("environment", {})
        environment = EnvironmentConfig(**env_data)
        
        task_data = data.get("task", {})
        task = TaskConfig(**task_data)
        
        return cls(
            runner=runner,
            agent=agent,
            judge_agent=judge_agent,
            environment=environment,
            task=task
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        result = {
            "runner": {
                "experiment_name": self.runner.experiment_name,
                "log_dir": self.runner.log_dir,
                "results_excel_path": self.runner.results_excel_path,
                "max_steps": self.runner.max_steps,
                "save_images": self.runner.save_images,
                "save_multi_view": self.runner.save_multi_view,
                "save_trajectory": self.runner.save_trajectory
            },
            "agent": {
                "model_name": self.agent.model_name,
                "temperature": self.agent.temperature,
                "max_tokens": self.agent.max_tokens,
                "timeout": self.agent.timeout
            },
            "environment": {
                "type": self.environment.type,
                "gui": self.environment.gui,
                "urdf_base_path": self.environment.urdf_base_path,
                "gravity": self.environment.gravity,
                "render_width": self.environment.render_width,
                "render_height": self.environment.render_height,
                "multi_view": self.environment.multi_view
            },
            "task": {
                "name": self.task.name,
                "type": self.task.type.value if self.task.type else None,
                "difficulty": self.task.difficulty.value if self.task.difficulty else None,
                "max_steps": self.task.max_steps,
                "time_limit": self.task.time_limit,
                "parameters": self.task.parameters
            }
        }
        
        if self.judge_agent:
            result["judge_agent"] = {
                "model_name": self.judge_agent.model_name,
                "temperature": self.judge_agent.temperature,
                "max_tokens": self.judge_agent.max_tokens,
                "timeout": self.judge_agent.timeout
            }
        
        return result


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
        ValueError: If required fields are missing
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config: {e}")
    
    if not data:
        raise ValueError("Configuration file is empty")
    
    try:
        return Config.from_dict(data)
    except Exception as e:
        raise ValueError(f"Error creating config from data: {e}")


def create_default_config(output_path: str = "config.yaml") -> Config:
    """
    Create a default configuration file.
    
    Args:
        output_path: Path where to save the default config
        
    Returns:
        Default Config object
    """
    config = Config(
        runner=RunnerConfig(
            experiment_name="default_experiment"
        ),
        agent=AgentConfig(
            model_name="gpt-4o"
        ),
        judge_agent=JudgeAgentConfig(
            model_name="gpt-4o"
        ),
        environment=EnvironmentConfig(),
        task=TaskConfig(
            name="domino_simple",
            type=TaskType.DOMINO,
            difficulty=TaskDifficulty.VERY_EASY
        )
    )
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
    
    return config


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    issues = []
    
    # Check required fields
    if not config.agent.model_name:
        issues.append("ERROR: Agent model_name is required")
    
    if not config.runner.experiment_name:
        issues.append("ERROR: Experiment name is required")
    
    # Check paths exist
    if not os.path.exists(config.environment.urdf_base_path):
        issues.append(f"WARNING: URDF base path does not exist: {config.environment.urdf_base_path}")
    
    # Check API configuration
    if not config.agent.api_key:
        issues.append("WARNING: No API key found for agent. Set OPENAI_API_KEY environment variable.")
    
    if config.judge_agent and not config.judge_agent.api_key:
        issues.append("WARNING: No API key found for judge agent.")
    
    # Check numeric ranges
    if config.agent.temperature < 0 or config.agent.temperature > 2:
        issues.append("WARNING: Agent temperature should be between 0 and 2")
    
    if config.agent.max_tokens <= 0:
        issues.append("ERROR: Agent max_tokens must be positive")
    
    if config.runner.max_steps <= 0:
        issues.append("ERROR: max_steps must be positive")
    
    return issues
