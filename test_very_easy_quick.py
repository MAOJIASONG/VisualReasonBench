#!/usr/bin/env python3
"""Quick test of very-easy domino task with VLM completion checking."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment
os.environ["OPENAI_API_KEY"] = "sk-s7Vr5Dbjm93xJDZh5mBIQG2PNQ8PanIxS4ECb2fBcJzIM2Xc"
os.environ["base_url"] = "https://openai.app.msh.team/v1"

from src.phyvpuzzle.core.pipeline import PhysicalReasoningPipeline, PipelineConfig
from src.phyvpuzzle.tasks.domino_task import DominoTask
from src.phyvpuzzle.tasks.base_task import TaskConfiguration, TaskType, TaskDifficulty

print("\n" + "="*80)
print("VERY-EASY DOMINO TASK TEST")
print("="*80)
print("\nConfiguration:")
print("- Task: 1 domino (very-easy)")
print("- Model: gpt-4o")
print("- Max iterations: 5")
print("- VLM completion check: Enabled")

# Configure pipeline
config = PipelineConfig(
    vllm_type="openai",
    vllm_model="gpt-4o",
    environment_type="pybullet",
    gui=False,
    max_iterations=5,
    enable_logging=True
)

# Create pipeline
print("\nInitializing pipeline...")
pipeline = PhysicalReasoningPipeline(config)
pipeline.initialize_components()

# Create very-easy task
print("Creating very-easy domino task (1 domino)...")
task_config = TaskConfiguration(
    task_type=TaskType.DOMINOES,
    difficulty=TaskDifficulty.VERY_EASY
)
task = DominoTask(task_config)

# Verify configuration
print(f"- Dominoes: {task.config.parameters.get('num_dominoes', 0)}")
print(f"- Max steps: {task.config.max_steps}")
print(f"- VLM check: {task.use_vlm_completion_check}")

# Execute task
print("\nExecuting task (this may take a moment)...")
print("-"*60)

result = pipeline.execute_task(task)

print("-"*60)
print(f"\nResults:")
print(f"- Success: {result.success}")
print(f"- Score: {result.final_score:.2%}")
print(f"- Steps: {result.steps_taken}")
print(f"- Time: {result.time_taken:.2f}s")

# Show log location
if hasattr(pipeline, 'logger_io') and pipeline.logger_io.trial_dir:
    print(f"\nLogs saved to: {pipeline.logger_io.trial_dir}")

pipeline.cleanup()
print("\n" + "="*80)