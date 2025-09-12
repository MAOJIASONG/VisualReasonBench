"""
Pagoda task implementation for 3D puzzle assembly.
"""

from typing import Dict, Any
from dataclasses import dataclass
from phyvpuzzle.core.base import TaskDifficulty
from phyvpuzzle.core import BaseEnvironment, register_task, register_task_config, EnvironmentConfig, TaskConfig
from phyvpuzzle.tasks.base_task import PhysicsTask
from phyvpuzzle.environment.lego_env import LegoEnvironment

@register_task_config("pagoda")
@dataclass
class PagodaTaskConfig(TaskConfig):
    """Configuration for Pagoda task."""
    assembly_mode: bool = True  # True for assembly, False for disassembly
    tower_style: str = "traditional"  # "simple", "traditional", "complex"
    
    def __post_init__(self):
        super().__post_init__()

@register_task("pagoda")
class PagodaTask(PhysicsTask):
    """Task for Pagoda (tower) puzzle assembly/disassembly."""
    
    def __init__(self, config: TaskConfig):
        # Pagoda-specific configuration (set before calling super)
        self.num_pieces = self._get_num_pieces_for_difficulty(config.difficulty)
        self.assembly_mode = getattr(config, 'assembly_mode', True)  # Default to assembly mode
        self.piece_type = f"{self.num_pieces}-level"
        
        super().__init__(config)
        
    def _get_num_pieces_for_difficulty(self, difficulty: TaskDifficulty = None) -> int:
        """Get number of levels based on difficulty."""
        if difficulty is None:
            difficulty = getattr(self, 'difficulty', TaskDifficulty.EASY)
            
        level_counts = {
            TaskDifficulty.VERY_EASY: 3,   # 3-level simple pagoda
            TaskDifficulty.EASY: 5,        # 5-level pagoda
            TaskDifficulty.MEDIUM: 7,      # 7-level pagoda
            TaskDifficulty.HARD: 9,        # 9-level pagoda
            TaskDifficulty.VERY_HARD: 11   # 11-level complex pagoda
        }
        return level_counts.get(difficulty, 5)
        
    def _configure_environment(self, environment: BaseEnvironment) -> None:
        """Configure environment for Pagoda task."""
        # Set appropriate camera positions for 3D puzzle viewing
        if hasattr(environment, 'multi_view_renderer'):
            # Set camera to capture the 3D structure well
            environment.multi_view_renderer.set_camera_config(
                "perspective", [1.2, -1.2, 1.0], [0, 0, 0.5]
            )
            
        # Configure environment based on assembly mode
        if self.assembly_mode:
            # Start with pieces scattered for assembly task
            self._scatter_levels(environment)
        else:
            # Start with assembled tower for disassembly task  
            self._build_initial_tower(environment)
            
    def _scatter_levels(self, environment: BaseEnvironment) -> None:
        """Scatter pagoda levels for assembly task."""
        # If using LegoEnvironment, disassemble any existing structure
        if hasattr(environment, 'disassemble'):
            # Scatter pieces randomly for assembly challenge
            for obj in getattr(environment, 'lego_bricks', []):
                environment._tool_disassemble(obj.name)
        else:
            print("Environment doesn't support level scattering - levels will be in default positions")
            
    def _build_initial_tower(self, environment: BaseEnvironment) -> None:
        """Build initial tower for disassembly task."""
        # If using LegoEnvironment, build tower automatically
        if hasattr(environment, 'build_tower'):
            # Build tower at center position for disassembly challenge
            environment._tool_build_tower(base_position=[0, 0, 0.42])
        else:
            print("Environment doesn't support auto-building - levels will be in default positions")
        
    def _get_initial_system_prompt(self) -> str:
        """Get base system prompt for Pagoda tasks."""
        base_prompt = """You are an AI agent controlling a physics simulation to solve Pagoda assembly puzzles.
These are 3D tower structures where levels must be stacked in specific ways to create a stable pagoda.

You can observe the environment through images showing the current state from multiple viewpoints.
The environment provides you with tools to manipulate the wooden levels:

AVAILABLE TOOLS:
- pick(object_id): Pick up a pagoda level
- place(object_id, position): Place a level at a specific 3D position
- move(object_id, position): Move a level to a new position
- push(object_id, force, direction): Push a level with force in a direction
- observe(angle): Change camera angle for better viewing
- check_solution(): Check if the pagoda is properly assembled

IMPORTANT GUIDELINES:
1. Pagodas require levels to be stacked with proper alignment and balance
2. Each level must be positioned correctly on top of the previous level
3. The key is understanding the stacking order and structural stability
4. Use multiple viewpoints (front, side, top, perspective) to understand the tower structure
5. Be patient and methodical - pagoda assembly requires careful alignment
6. Success is measured by proper stacking and overall stability of the pagoda"""
        
        mode_text = "assembly" if self.assembly_mode else "disassembly"

        prompt = base_prompt + "\n\n" + f"""
TASK SPECIFICS:
- Puzzle type: {self.piece_type} Pagoda
- Mode: {mode_text}
- Number of levels: {self.num_pieces}
- Difficulty: {self.difficulty.value}

"""

        if self.assembly_mode:
            prompt += "\n\n" + """ASSEMBLY MODE:
- The levels are initially separated
- Your goal is to stack them into a stable pagoda structure
- Look for proper alignment and balance points
- Levels must be stacked in the correct order from base to top"""
        else:
            prompt += "\n\n" + """DISASSEMBLY MODE:
- The pagoda starts in assembled form
- Your goal is to separate the levels without collapsing the structure
- Find the top level that can be removed first
- Often there's a specific sequence for safe disassembly"""

        # Add difficulty-specific guidance
        if self.difficulty == TaskDifficulty.VERY_EASY:
            prompt += "\n\n- This is a simple 3-level pagoda with basic stacking"
        elif self.difficulty == TaskDifficulty.MEDIUM:
            prompt += "\n\n- This 7-level pagoda requires understanding of balance and alignment"
        elif self.difficulty in [TaskDifficulty.HARD, TaskDifficulty.VERY_HARD]:
            prompt += "\n\n- This complex pagoda may require discovering precise stacking techniques"

        return prompt
        
    def _get_initial_instruction(self) -> str:
        """Get description of the specific Pagoda task."""
        mode_text = "assembly" if self.assembly_mode else "disassembly"

        base_prompt = f"""PAGODA {mode_text.upper()} PUZZLE

You are presented with a {self.piece_type} Pagoda puzzle.
Your objective is to {"assemble the levels into a stable tower structure" if self.assembly_mode else "disassemble the pagoda by separating all levels"}.

Task Details:
- Difficulty Level: {self.difficulty.value}
- Number of Levels: {self.num_pieces}
- Puzzle Type: {self.piece_type}
- Mode: {mode_text}

Pagodas are traditional tower structures that require levels to be stacked together
in a specific order to create balance and stability. Success depends on understanding
the stacking sequence and maintaining structural integrity."""

        if self.assembly_mode:
            return base_prompt + "\n\n" + """Please start by carefully observing all the pagoda levels from multiple angles.
Look for base platforms, alignment features, and stacking indicators that show the correct building order.

Assembly steps:
1. Observe each level carefully from all angles
2. Identify the base level and stacking order
3. Start with the foundation level
4. Stack levels systematically from bottom to top
5. Ensure proper alignment and balance at each step
6. Check the solution when you think the pagoda is complete"""
        else:
            return base_prompt + "\n\n" + """Please start by observing the assembled pagoda from multiple angles.
Look for the top level and assess the overall stability of the structure.

Disassembly steps:
1. Observe the assembled tower from all angles
2. Identify the top level that can be safely removed
3. Remove levels from top to bottom carefully
4. Maintain balance to prevent collapse
5. Avoid sudden movements that could destabilize the structure
6. Continue until all levels are separated"""
        
    def _calculate_optimal_steps(self) -> int:
        """Calculate optimal steps for Pagoda task."""
        base_steps = {
            3: 10,   # 3-level: observe + 3 level moves + check
            5: 18,   # 5-level: more complex stacking
            7: 26,   # 7-level: complex multi-step process
            9: 34,   # 9-level: very complex
            11: 42   # 11-level: extremely complex
        }

        optimal = base_steps.get(self.num_pieces, 18)

        # Disassembly is usually faster than assembly
        if not self.assembly_mode:
            optimal = int(optimal * 0.8)

        return optimal
        
    def get_expected_success_rate(self) -> float:
        """Get expected success rate for this pagoda task configuration."""
        base_rates = {
            TaskDifficulty.VERY_EASY: 0.90,   # Simple 3-level pagoda
            TaskDifficulty.EASY: 0.80,        # 5-level pagoda
            TaskDifficulty.MEDIUM: 0.65,      # 7-level pagoda
            TaskDifficulty.HARD: 0.45,        # 9-level pagoda
            TaskDifficulty.VERY_HARD: 0.30    # 11-level complex pagoda
        }

        base_rate = base_rates.get(self.difficulty, 0.65)

        # Assembly is harder than disassembly for pagodas
        if self.assembly_mode:
            base_rate *= 0.85

        return max(0.1, min(1.0, base_rate))
        
    def get_task_metrics(self) -> Dict[str, Any]:
        """Get task-specific metrics for evaluation."""
        return {
            "num_levels": self.num_pieces,
            "pagoda_type": self.piece_type,
            "assembly_mode": self.assembly_mode,
            "expected_success_rate": self.get_expected_success_rate(),
            "optimal_steps": self.optimal_steps,
            "difficulty_multiplier": self.get_difficulty_multiplier()
        }
    
    def _evaluate_success(self, final_state, trajectory) -> str:
        """Return success criteria description for LLM judge evaluation."""
        mode_text = "assembly" if self.assembly_mode else "disassembly"
        
        base_criteria = f"""
PAGODA {mode_text.upper()} SUCCESS CRITERIA:

Task: {self.piece_type} Pagoda {mode_text}
Difficulty: {self.difficulty.value}

SUCCESS REQUIREMENTS:
"""
        
        if self.assembly_mode:
            criteria = base_criteria + """
1. STRUCTURAL STABILITY: Tower must be properly stacked
   - All levels should be stacked vertically without falling
   - The tower should be stable and balanced
   - No pieces should be tilted or about to fall

2. CORRECT STACKING ORDER: Levels should be in proper sequence
   - Base level should be at the bottom
   - Levels should generally decrease in size from bottom to top (if applicable)
   - Each level should be properly centered on the level below

3. VERTICAL ALIGNMENT: Tower should be straight and centered
   - The tower should not lean to one side
   - Each level should be aligned with the center axis
   - Overall structure should appear balanced and symmetric

4. COMPLETION: All levels should be used and properly positioned
   - All available levels should be incorporated into the tower
   - No levels should be left unused on the side
   - The tower should appear complete and finished

EVALUATION INSTRUCTIONS:
- Examine the final state image carefully from multiple angles
- Check if the tower is standing upright and stable
- Verify that all levels are properly stacked and aligned
- Look for proper spacing and contact between levels
- Success = complete, stable, properly aligned pagoda tower
"""
        else:
            criteria = base_criteria + """
1. COMPLETE DISASSEMBLY: All levels must be separated
   - No levels should remain stacked together
   - Each level should be individual and separate
   - The original tower structure should be completely dismantled

2. LEVEL PRESERVATION: Each level should be intact
   - No damaged or broken levels from disassembly
   - Original level geometry should be maintained  
   - Levels should be clearly identifiable as separate pieces

3. SAFE DISMANTLING: Evidence of controlled disassembly
   - Levels should be carefully removed without damage
   - No signs of the tower collapsing or falling over
   - The disassembly should appear methodical and controlled

4. SPATIAL SEPARATION: Levels should be clearly distinguishable
   - Dismantled levels should not be overlapping
   - Each level should be visible and separate from others
   - Clear evidence that the pagoda has been completely taken apart

EVALUATION INSTRUCTIONS:
- Examine the final state image carefully
- Count the separated levels and compare to expected number
- Look for any levels that might still be stacked together
- Check if levels appear undamaged from the disassembly process
- Success = all levels cleanly separated and individually placed
"""
        
        return criteria
