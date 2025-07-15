"""
PhyVPuzzle: Physical Visual Reasoning Benchmark

A comprehensive benchmark for evaluating Vision-Language Models (VLMs) 
on physical visual reasoning tasks.
"""

__version__ = "0.1.0"
__author__ = "MAOJIASONG"
__email__ = "maojia_song@mymail.sutd.edu.sg"

# Core components
from .core.pipeline import (
    PhysicalReasoningPipeline,
    PipelineConfig,
    PipelineStep,
    create_pipeline
)

from .core.vllm_processor import (
    VLLMProcessor,
    OpenAIVLLMProcessor,
    HuggingFaceVLLMProcessor,
    DecisionParser,
    create_vllm_processor
)

from .core.action_descriptor import (
    ActionDescriptor,
    ActionType,
    ActionParameters,
    ParsedAction,
    ObjectExtractor
)

from .core.translator import (
    Translator,
    RuleBasedTranslator,
    LLMTranslator,
    EnvironmentCommand,
    TranslationResult,
    create_translator
)

# Environment
from .environment.physics_env import (
    PhysicsEnvironment,
    PyBulletEnvironment,
    CameraConfig,
    RobotConfig,
    ObjectInfo,
    create_environment
)

# Tasks
from .tasks.base_task import (
    BaseTask,
    TaskType,
    TaskDifficulty,
    TaskConfiguration,
    TaskState,
    TaskResult,
    TaskValidator,
    create_task_config
)

# Evaluation
from .evaluation.metrics import (
    MetricType,
    EvaluationResult,
    AccuracyMetric,
    PassAtKMetric,
    DistanceToOptimalMetric,
    EfficiencyMetric,
    RobustnessMetric,
    ComprehensiveEvaluator,
    evaluate_model_performance
)

# Convenience imports
from .cli import main as cli_main

# Default configurations
DEFAULT_PIPELINE_CONFIG = PipelineConfig(
    vllm_type="openai",
    vllm_model="gpt-4-vision-preview",
    translator_type="rule_based",
    environment_type="pybullet",
    max_iterations=100,
    timeout=300.0,
    enable_logging=True,
    log_level="INFO"
)

DEFAULT_TASK_CONFIG = TaskConfiguration(
    task_type=TaskType.PUZZLE,
    difficulty=TaskDifficulty.MEDIUM,
    max_steps=100,
    time_limit=300.0,
    success_threshold=0.9
)

# Utility functions
def create_default_pipeline() -> PhysicalReasoningPipeline:
    """Create a pipeline with default configuration."""
    return create_pipeline(DEFAULT_PIPELINE_CONFIG)

def create_default_task_config(task_type: str = "puzzle", 
                             difficulty: str = "medium") -> TaskConfiguration:
    """Create a task configuration with default values."""
    return create_task_config(task_type, difficulty)

def quick_evaluate(task_type: str = "puzzle", 
                  difficulty: str = "medium",
                  num_tasks: int = 5,
                  num_runs: int = 3) -> EvaluationResult:
    """
    Quick evaluation with minimal setup.
    
    Args:
        task_type: Type of task to evaluate
        difficulty: Task difficulty level
        num_tasks: Number of tasks to create
        num_runs: Number of runs per task
        
    Returns:
        EvaluationResult with metrics
    """
    try:
        # Create pipeline
        pipeline = create_default_pipeline()
        
        # Create sample tasks (placeholder - would need actual task implementations)
        tasks = []
        for i in range(num_tasks):
            config = create_task_config(task_type, difficulty)
            # Note: This would need actual task implementations
            # task = SampleTask(config)
            # tasks.append(task)
        
        # Run evaluation
        if tasks:
            return evaluate_model_performance(pipeline, tasks, num_runs=num_runs)
        else:
            print("Warning: No tasks created for evaluation")
            return EvaluationResult(
                total_tasks=0,
                successful_tasks=0,
                metrics={},
                task_results=[],
                per_task_metrics=[],
                metadata={}
            )
            
    except Exception as e:
        print(f"Error in quick evaluation: {e}")
        raise

# Package metadata
__all__ = [
    # Core
    "PhysicalReasoningPipeline",
    "PipelineConfig", 
    "PipelineStep",
    "create_pipeline",
    
    # VLLM
    "VLLMProcessor",
    "OpenAIVLLMProcessor",
    "HuggingFaceVLLMProcessor",
    "DecisionParser",
    "create_vllm_processor",
    
    # Actions
    "ActionDescriptor",
    "ActionType",
    "ActionParameters", 
    "ParsedAction",
    "ObjectExtractor",
    
    # Translation
    "Translator",
    "RuleBasedTranslator",
    "LLMTranslator",
    "EnvironmentCommand",
    "TranslationResult",
    "create_translator",
    
    # Environment
    "PhysicsEnvironment",
    "PyBulletEnvironment",
    "CameraConfig",
    "RobotConfig",
    "ObjectInfo",
    "create_environment",
    
    # Tasks
    "BaseTask",
    "TaskType",
    "TaskDifficulty",
    "TaskConfiguration",
    "TaskState",
    "TaskResult",
    "TaskValidator",
    "create_task_config",
    
    # Evaluation
    "MetricType",
    "EvaluationResult",
    "AccuracyMetric",
    "PassAtKMetric",
    "DistanceToOptimalMetric",
    "EfficiencyMetric",
    "RobustnessMetric",
    "ComprehensiveEvaluator",
    "evaluate_model_performance",
    
    # Utilities
    "cli_main",
    "create_default_pipeline",
    "create_default_task_config",
    "quick_evaluate",
    
    # Constants
    "DEFAULT_PIPELINE_CONFIG",
    "DEFAULT_TASK_CONFIG",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__"
]

# Package info
def get_package_info():
    """Get package information."""
    return {
        "name": "phyvpuzzle",
        "version": __version__,
        "description": "Physical Visual Reasoning Benchmark",
        "author": __author__,
        "email": __email__,
        "components": {
            "core": ["Pipeline", "VLLM", "Actions", "Translation"],
            "environment": ["PyBullet Physics"],
            "tasks": ["Puzzle", "Lego", "Dominoes"],
            "evaluation": ["Accuracy", "Pass@K", "Distance", "Efficiency"]
        }
    }

def print_package_info():
    """Print package information."""
    info = get_package_info()
    print(f"PhyVPuzzle v{info['version']}")
    print(f"Physical Visual Reasoning Benchmark")
    print(f"Author: {info['author']}")
    print(f"Components: {', '.join(info['components']['core'])}")
    print(f"Tasks: {', '.join(info['components']['tasks'])}")
    print(f"Metrics: {', '.join(info['components']['evaluation'])}")

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
