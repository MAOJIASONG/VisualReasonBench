"""
Basic tests for PhyVPuzzle components.
"""

import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from phyvpuzzle.core.action_descriptor import ActionDescriptor, ActionType
from phyvpuzzle.core.pipeline import PipelineConfig
from phyvpuzzle.tasks.base_task import create_task_config, TaskType, TaskDifficulty
from phyvpuzzle.evaluation.metrics import AccuracyMetric, PassAtKMetric


class TestActionDescriptor:
    """Test ActionDescriptor functionality."""
    
    def test_action_descriptor_init(self):
        """Test ActionDescriptor initialization."""
        descriptor = ActionDescriptor()
        assert descriptor is not None
        assert descriptor.action_patterns is not None
        assert descriptor.object_extractor is not None
    
    def test_parse_pick_action(self):
        """Test parsing pick action."""
        descriptor = ActionDescriptor()
        action = descriptor.parse_action("pick up the red block")
        
        assert action.action_type == ActionType.PICK
        assert action.parameters.object_id == "red block"
        assert action.confidence > 0.8
    
    def test_parse_place_action(self):
        """Test parsing place action."""
        descriptor = ActionDescriptor()
        action = descriptor.parse_action("place the blue cube on the table")
        
        assert action.action_type == ActionType.PLACE
        assert action.parameters.object_id == "blue cube"
        assert action.parameters.target_object_id == "table"
        assert action.confidence > 0.8
    
    def test_parse_finish_action(self):
        """Test parsing finish action."""
        descriptor = ActionDescriptor()
        action = descriptor.parse_action("finish the task")
        
        assert action.action_type == ActionType.FINISH
        assert action.confidence > 0.8


class TestPipelineConfig:
    """Test PipelineConfig functionality."""
    
    def test_pipeline_config_init(self):
        """Test PipelineConfig initialization."""
        config = PipelineConfig()
        
        assert config.vllm_type == "openai"
        assert config.vllm_model == "gpt-4-vision-preview"
        assert config.translator_type == "rule_based"
        assert config.environment_type == "pybullet"
        assert config.max_iterations == 100
        assert config.timeout == 300.0
        assert config.enable_logging is True
    
    def test_pipeline_config_custom(self):
        """Test PipelineConfig with custom values."""
        config = PipelineConfig(
            vllm_type="huggingface",
            max_iterations=50,
            timeout=120.0,
            enable_logging=False
        )
        
        assert config.vllm_type == "huggingface"
        assert config.max_iterations == 50
        assert config.timeout == 120.0
        assert config.enable_logging is False


class TestTaskConfig:
    """Test task configuration functionality."""
    
    def test_create_task_config(self):
        """Test creating task configuration."""
        config = create_task_config("puzzle", "medium")
        
        assert config.task_type == TaskType.PUZZLE
        assert config.difficulty == TaskDifficulty.MEDIUM
        assert config.max_steps == 100
        assert config.time_limit == 300.0
        assert config.success_threshold == 0.9
    
    def test_create_task_config_custom(self):
        """Test creating task configuration with custom parameters."""
        config = create_task_config(
            "lego", 
            "hard",
            max_steps=50,
            time_limit=180.0,
            success_threshold=0.8
        )
        
        assert config.task_type == TaskType.LEGO
        assert config.difficulty == TaskDifficulty.HARD
        assert config.max_steps == 50
        assert config.time_limit == 180.0
        assert config.success_threshold == 0.8


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_accuracy_metric(self):
        """Test accuracy metric calculation."""
        # Create mock results
        from phyvpuzzle.tasks.base_task import TaskResult
        
        results = [
            TaskResult(success=True, final_score=1.0, steps_taken=10, time_taken=30.0),
            TaskResult(success=False, final_score=0.5, steps_taken=20, time_taken=60.0),
            TaskResult(success=True, final_score=0.8, steps_taken=15, time_taken=45.0),
        ]
        
        metric = AccuracyMetric()
        accuracy = metric.evaluate(results)
        
        assert accuracy == 2.0 / 3.0  # 2 out of 3 successful
    
    def test_pass_at_k_metric(self):
        """Test Pass@K metric calculation."""
        from phyvpuzzle.tasks.base_task import TaskResult
        
        # Create mock result groups
        results_groups = [
            [  # Task 1 attempts
                TaskResult(success=False, final_score=0.0, steps_taken=10, time_taken=30.0),
                TaskResult(success=True, final_score=1.0, steps_taken=8, time_taken=25.0),
            ],
            [  # Task 2 attempts
                TaskResult(success=False, final_score=0.2, steps_taken=15, time_taken=40.0),
                TaskResult(success=False, final_score=0.3, steps_taken=12, time_taken=35.0),
            ],
            [  # Task 3 attempts
                TaskResult(success=True, final_score=0.9, steps_taken=5, time_taken=20.0),
                TaskResult(success=True, final_score=1.0, steps_taken=6, time_taken=22.0),
            ],
        ]
        
        metric = PassAtKMetric(k=2)
        pass_at_k = metric.evaluate(results_groups)
        
        assert pass_at_k == 2.0 / 3.0  # 2 out of 3 tasks had at least one success
    
    def test_empty_results(self):
        """Test metrics with empty results."""
        metric = AccuracyMetric()
        accuracy = metric.evaluate([])
        assert accuracy == 0.0
        
        pass_metric = PassAtKMetric(k=2)
        pass_at_k = pass_metric.evaluate([])
        assert pass_at_k == 0.0


class TestUtilities:
    """Test utility functions."""
    
    def test_task_type_enum(self):
        """Test TaskType enum."""
        assert TaskType.PUZZLE.value == "puzzle"
        assert TaskType.LEGO.value == "lego"
        assert TaskType.DOMINOES.value == "dominoes"
    
    def test_task_difficulty_enum(self):
        """Test TaskDifficulty enum."""
        assert TaskDifficulty.EASY.value == "easy"
        assert TaskDifficulty.MEDIUM.value == "medium"
        assert TaskDifficulty.HARD.value == "hard"
        assert TaskDifficulty.EXPERT.value == "expert"


if __name__ == "__main__":
    pytest.main([__file__]) 