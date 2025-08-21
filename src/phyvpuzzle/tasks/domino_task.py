"""
Domino task implementation for falling domino puzzles.
"""

from typing import Dict, Any
from ..core.base import BaseEnvironment, TaskType, TaskDifficulty
from ..environment.domino_env import DominoEnvironment, DominoConfig
from .base_task import PuzzleTask


class DominoTask(PuzzleTask):
    """Task for domino falling chain reaction puzzles."""
    
    def __init__(self, difficulty: TaskDifficulty, config: Dict[str, Any]):
        super().__init__(TaskType.DOMINO, difficulty, config)
        
        # Create domino-specific configuration
        self.domino_config = DominoConfig.from_difficulty(difficulty)
        
        # Override with any config parameters
        if "num_dominoes" in config:
            self.domino_config.num_dominoes = config["num_dominoes"]
        if "arrangement_pattern" in config:
            self.domino_config.arrangement_pattern = config["arrangement_pattern"]
        if "domino_spacing" in config:
            self.domino_config.domino_spacing = config["domino_spacing"]
            
    def _configure_environment(self, environment: BaseEnvironment) -> None:
        """Configure environment for domino task."""
        # Ensure we have a domino environment
        if not isinstance(environment, DominoEnvironment):
            raise ValueError("DominoTask requires DominoEnvironment")
            
        # Configure domino-specific settings
        environment.domino_config = self.domino_config
        
        # Set camera position for optimal domino viewing
        if hasattr(environment, 'multi_view_renderer'):
            # Adjust camera positions based on domino arrangement
            if self.domino_config.arrangement_pattern == "line":
                environment.multi_view_renderer.set_camera_config(
                    "perspective", [0, -2.0, 1.0], [0, 0, 0.4]
                )
            elif self.domino_config.arrangement_pattern == "circle":
                environment.multi_view_renderer.set_camera_config(
                    "perspective", [2.0, 0, 1.5], [0, 0, 0.4]
                )
                
    def _load_task_objects(self, environment: BaseEnvironment) -> None:
        """Load dominoes into the environment."""
        # This is handled by the DominoEnvironment's _setup_task_environment method
        pass
        
    def _set_initial_state(self, environment: BaseEnvironment) -> None:
        """Set initial state for domino task."""
        # Reset any previous state
        environment.step_count = 0
        
        # Ensure dominoes are in upright position
        if hasattr(environment, '_reset_dominoes'):
            environment._reset_dominoes()
            
    def _get_base_system_prompt(self) -> str:
        """Get base system prompt for domino tasks."""
        return """You are an AI agent controlling a physics simulation to solve domino falling puzzles. 
Your goal is to create a chain reaction by pushing the first domino, causing all dominoes to fall in sequence.

You can observe the environment through images showing the current state of the domino setup from multiple viewpoints.
The environment provides you with several tools to interact with the dominoes:

AVAILABLE TOOLS:
- push_domino(): Push the first domino to start the chain reaction
- push_specific_domino(domino_id, force, direction): Push a specific domino
- observe(angle): Change camera angle to observe from different viewpoints  
- check_solution(): Check if the puzzle has been solved (all dominoes fallen)

IMPORTANT GUIDELINES:
1. Always observe the initial setup carefully before taking action
2. Usually, pushing the first domino with appropriate force is sufficient
3. The chain reaction should propagate through all dominoes automatically
4. Success is measured by the percentage of dominoes that fall (typically >80%)
5. Use observe() to get different viewpoints if needed to assess the situation
6. Be patient - allow time for the physics simulation to propagate the chain reaction"""
        
    def _get_task_specific_prompt(self) -> str:
        """Get domino-specific additions to system prompt."""
        arrangement = self.domino_config.arrangement_pattern
        num_dominoes = self.domino_config.num_dominoes
        
        prompt = f"""
TASK SPECIFICS:
- Number of dominoes: {num_dominoes}
- Arrangement pattern: {arrangement}
- Difficulty: {self.difficulty.value}
"""
        
        if arrangement == "line":
            prompt += "- The dominoes are arranged in a straight line. Push the first domino to start the chain."
        elif arrangement == "curve":
            prompt += "- The dominoes are arranged in a curved pattern. Consider the curve when pushing."
        elif arrangement == "zigzag":
            prompt += "- The dominoes are arranged in a zigzag pattern. The chain reaction follows the zigzag path."
        elif arrangement == "circle":
            prompt += "- The dominoes are arranged in a circle. The chain reaction should propagate around the circle."
            
        # Add difficulty-specific guidance
        if self.difficulty == TaskDifficulty.VERY_EASY:
            prompt += "\n- This is a very easy setup - a gentle push should be sufficient."
        elif self.difficulty == TaskDifficulty.EASY:
            prompt += "\n- This is an easy setup - standard force should work well."
        elif self.difficulty == TaskDifficulty.MEDIUM:
            prompt += "\n- This is a medium difficulty setup - you may need to adjust force or direction."
        elif self.difficulty == TaskDifficulty.HARD:
            prompt += "\n- This is a hard setup - careful observation and precise pushing may be needed."
        elif self.difficulty == TaskDifficulty.VERY_HARD:
            prompt += "\n- This is a very hard setup - you may need multiple pushes or strategic planning."
            
        return prompt
        
    def _get_task_description(self) -> str:
        """Get description of the specific domino task instance."""
        return f"""DOMINO FALLING PUZZLE

You are presented with {self.domino_config.num_dominoes} dominoes arranged in a {self.domino_config.arrangement_pattern} pattern.
Your objective is to create a chain reaction by pushing the first domino, causing all dominoes to fall in sequence.

Task Details:
- Difficulty Level: {self.difficulty.value}
- Number of Dominoes: {self.domino_config.num_dominoes}
- Arrangement: {self.domino_config.arrangement_pattern}
- Success Criteria: At least 80% of dominoes must fall

The physics simulation will handle the chain reaction once you provide the initial push.
Observe the setup carefully and determine the best approach to ensure maximum domino fall rate."""
        
    def _get_initial_instruction(self) -> str:
        """Get initial instruction for starting the domino task."""
        return """Please start by observing the domino setup. Look at the arrangement pattern, spacing, and orientation of the dominoes. 
Once you understand the setup, push the first domino with appropriate force to start the chain reaction.

Remember to:
1. First observe the scene to understand the domino layout
2. Push the first domino (usually domino_1) to start the chain
3. Wait for the chain reaction to complete
4. Check the solution to see how many dominoes fell
5. If needed, you can observe from different angles or make additional pushes"""
        
    def _calculate_optimal_steps(self) -> int:
        """Calculate optimal steps for domino task."""
        # Basic domino tasks typically require:
        # 1. Observe initial setup
        # 2. Push first domino  
        # 3. Wait for chain reaction
        # 4. Check solution
        
        base_steps = 4
        
        # Adjust based on difficulty and arrangement
        if self.difficulty in [TaskDifficulty.HARD, TaskDifficulty.VERY_HARD]:
            base_steps += 2  # May need additional observations or pushes
            
        if self.domino_config.arrangement_pattern in ["zigzag", "circle"]:
            base_steps += 1  # More complex arrangements may need extra step
            
        return base_steps
        
    def get_expected_success_rate(self) -> float:
        """Get expected success rate for this task configuration."""
        base_rates = {
            TaskDifficulty.VERY_EASY: 0.95,
            TaskDifficulty.EASY: 0.90,
            TaskDifficulty.MEDIUM: 0.80,
            TaskDifficulty.HARD: 0.65,
            TaskDifficulty.VERY_HARD: 0.50
        }
        
        base_rate = base_rates.get(self.difficulty, 0.80)
        
        # Adjust based on arrangement complexity
        arrangement_adjustments = {
            "line": 0.0,
            "curve": -0.05,
            "zigzag": -0.10,
            "circle": -0.15
        }
        
        adjustment = arrangement_adjustments.get(self.domino_config.arrangement_pattern, 0.0)
        
        return max(0.1, min(1.0, base_rate + adjustment))
        
    def get_task_metrics(self) -> Dict[str, Any]:
        """Get task-specific metrics for evaluation."""
        return {
            "num_dominoes": self.domino_config.num_dominoes,
            "arrangement_pattern": self.domino_config.arrangement_pattern,
            "domino_spacing": self.domino_config.domino_spacing,
            "expected_success_rate": self.get_expected_success_rate(),
            "optimal_steps": self.optimal_steps,
            "difficulty_multiplier": self.get_difficulty_multiplier()
        }
