"""
Luban lock task implementation for 3D puzzle assembly.
"""

from typing import Dict, Any
from dataclasses import dataclass
from phyvpuzzle.core.base import TaskDifficulty
from phyvpuzzle.core import BaseEnvironment, register_task, register_task_config, EnvironmentConfig, TaskConfig
from phyvpuzzle.tasks.base_task import PhysicsTask
from phyvpuzzle.environment.puzzle_env import PuzzleEnvironment

@register_task_config("luban")
@dataclass
class LubanTaskConfig(TaskConfig):
    """Configuration for Luban lock task."""
    assembly_mode: bool = True  # True for assembly, False for disassembly
    piece_complexity: str = "standard"  # "simple", "standard", "complex"
    
    def __post_init__(self):
        super().__post_init__()

@register_task("luban")
class LubanTask(PhysicsTask):
    """Task for Luban lock (wooden burr puzzle) assembly/disassembly."""
    
    def __init__(self, config: TaskConfig):
        # Luban-specific configuration (set before calling super)
        self.num_pieces = self._get_num_pieces_for_difficulty(config.difficulty)
        self.assembly_mode = getattr(config, 'assembly_mode', True)  # Default to assembly mode
        self.piece_type = f"{self.num_pieces}-piece"
        
        super().__init__(config)
        
    def _get_num_pieces_for_difficulty(self, difficulty: TaskDifficulty = None) -> int:
        """Get number of pieces based on difficulty."""
        if difficulty is None:
            difficulty = getattr(self, 'difficulty', TaskDifficulty.EASY)
            
        piece_counts = {
            TaskDifficulty.VERY_EASY: 3,
            TaskDifficulty.EASY: 3,
            TaskDifficulty.MEDIUM: 6,
            TaskDifficulty.HARD: 15,
            TaskDifficulty.VERY_HARD: 15
        }
        return piece_counts.get(difficulty, 6)
        
    def _configure_environment(self, environment: BaseEnvironment) -> None:
        """Configure environment for Luban lock task."""
        # Set appropriate camera positions for 3D puzzle viewing
        if hasattr(environment, 'multi_view_renderer'):
            # Set camera to capture the 3D structure well
            environment.multi_view_renderer.set_camera_config(
                "perspective", [1.2, -1.2, 1.0], [0, 0, 0.5]
            )
            
        if self.assembly_mode:
            # Start with pieces separated for assembly task
            self._separate_pieces(environment)
        else:
            # Start with assembled puzzle for disassembly task
            self._assemble_pieces(environment)
            
    def _separate_pieces(self, environment: BaseEnvironment) -> None:
        """Separate Luban pieces for assembly task."""
        # If using PuzzleEnvironment, shuffle pieces for assembly challenge
        if hasattr(environment, 'shuffle_pieces'):
            # Use larger scatter radius for Luban pieces
            environment._tool_shuffle_pieces(scatter_radius=0.4)
        else:
            # Fallback for other environment types
            print("Environment doesn't support piece shuffling - pieces will be in default positions")
        
    def _assemble_pieces(self, environment: BaseEnvironment) -> None:
        """Pre-assemble pieces for disassembly task."""
        # If using PuzzleEnvironment, auto-solve to create assembled state
        if hasattr(environment, 'auto_solve'):
            # Create assembled puzzle for disassembly challenge
            environment._tool_auto_solve(step_by_step=False)
        else:
            # Fallback for other environment types
            print("Environment doesn't support auto-assembly - pieces will be in default positions")
        
    def _get_initial_system_prompt(self) -> str:
        """Get initial system prompt for Luban lock tasks."""
        mode_text = "assembly" if self.assembly_mode else "disassembly"
        
        base_prompt = """You are an AI agent controlling a physics simulation to solve Luban lock puzzles.
These are 3D wooden burr puzzles where pieces interlock in specific ways.

You can observe the environment through images showing the current state from multiple viewpoints.
The environment provides you with tools to manipulate the wooden pieces:

AVAILABLE TOOLS:
- pick(object_id): Pick up a puzzle piece
- place(object_id, position): Place a piece at a specific 3D position
- move(object_id, position): Move a piece to a new position
- push(object_id, force, direction): Push a piece with force in a direction
- observe(angle): Change camera angle for better viewing
- check_solution(): Check if the puzzle is solved

IMPORTANT GUIDELINES:
1. Luban locks require precise positioning and orientation of pieces
2. Pieces must slide into specific slots and orientations to interlock properly
3. The key is understanding the 3D geometry and interlocking mechanisms
4. Use multiple viewpoints (front, side, top, perspective) to understand the structure
5. Be patient and methodical - these puzzles require careful planning
6. Success is measured by proper interlocking and stability of the assembly

"""
        
        task_specifics = f"""TASK SPECIFICS:
- Puzzle type: {self.piece_type} Luban lock
- Mode: {mode_text}
- Number of pieces: {self.num_pieces}
- Difficulty: {self.difficulty.value}

"""
        
        mode_prompt = ""
        if self.assembly_mode:
            mode_prompt = """ASSEMBLY MODE:
- The pieces are initially separated
- Your goal is to assemble them into the interlocked configuration
- Look for grooves, slots, and complementary shapes
- Pieces typically slide together in specific sequences"""
        else:
            mode_prompt = """DISASSEMBLY MODE:
- The puzzle starts in assembled form
- Your goal is to separate the pieces without forcing
- Find the key piece that can be removed first
- Often there's a specific sequence for disassembly"""
            
        # Add difficulty-specific guidance
        difficulty_guidance = ""
        if self.difficulty == TaskDifficulty.VERY_EASY:
            difficulty_guidance = "\n- This is a simple 3-piece lock with obvious interlocking"
        elif self.difficulty == TaskDifficulty.MEDIUM:
            difficulty_guidance = "\n- This 6-piece lock requires understanding of the central structure"
        elif self.difficulty in [TaskDifficulty.HARD, TaskDifficulty.VERY_HARD]:
            difficulty_guidance = "\n- This complex lock may require discovering hidden mechanisms"
            
        task_description = f"""

LUBAN LOCK {mode_text.upper()} PUZZLE

You are presented with a {self.piece_type} Luban lock puzzle.
Your objective is to {"assemble the pieces into a stable interlocked structure" if self.assembly_mode else "disassemble the puzzle by separating all pieces"}.

Task Details:
- Difficulty Level: {self.difficulty.value}
- Number of Pieces: {self.num_pieces}
- Puzzle Type: {self.piece_type}
- Mode: {mode_text}

Luban locks are traditional Chinese wooden puzzles that require pieces to be fitted together
in a specific interlocking pattern. Success depends on understanding the 3D geometry
and finding the correct sequence of moves."""

        return base_prompt + task_specifics + mode_prompt + difficulty_guidance + task_description
        
    def _get_initial_instruction(self) -> str:
        """Get initial instruction for starting the Luban lock task."""
        if self.assembly_mode:
            return """Please start by carefully observing all the puzzle pieces from multiple angles.
Look for grooves, slots, notches, and complementary shapes that indicate how pieces might fit together.

Assembly steps:
1. Observe each piece carefully from all angles
2. Identify potential interlocking features
3. Try fitting pieces together systematically
4. Look for the central structure or key piece
5. Build the puzzle step by step
6. Check the solution when you think it's complete"""
        else:
            return """Please start by observing the assembled puzzle from multiple angles.
Look for pieces that seem to have more freedom of movement or could be key pieces.

Disassembly steps:
1. Observe the assembled structure from all angles
2. Identify which piece might be removable first
3. Try gentle movements to find pieces with play
4. Remove pieces in the correct sequence
5. Avoid forcing - if it doesn't move easily, try another piece
6. Continue until all pieces are separated"""
        
    def _calculate_optimal_steps(self) -> int:
        """Calculate optimal steps for Luban lock task."""
        base_steps = {
            3: 8,   # 3-piece: observe + 3 piece moves + check
            6: 15,  # 6-piece: more complex assembly/disassembly
            15: 25  # 15-piece: very complex multi-step process
        }
        
        optimal = base_steps.get(self.num_pieces, 15)
        
        # Disassembly is usually faster than assembly
        if not self.assembly_mode:
            optimal = int(optimal * 0.8)
            
        return optimal
        
    def get_expected_success_rate(self) -> float:
        """Get expected success rate for this task configuration."""
        base_rates = {
            TaskDifficulty.VERY_EASY: 0.85,
            TaskDifficulty.EASY: 0.75,
            TaskDifficulty.MEDIUM: 0.60,
            TaskDifficulty.HARD: 0.40,
            TaskDifficulty.VERY_HARD: 0.25
        }
        
        base_rate = base_rates.get(self.difficulty, 0.60)
        
        # Assembly is harder than disassembly
        if self.assembly_mode:
            base_rate *= 0.8
        
        return max(0.1, min(1.0, base_rate))
        
    def get_task_metrics(self) -> Dict[str, Any]:
        """Get task-specific metrics for evaluation."""
        return {
            "num_pieces": self.num_pieces,
            "piece_type": self.piece_type,
            "assembly_mode": self.assembly_mode,
            "expected_success_rate": self.get_expected_success_rate(),
            "optimal_steps": self.optimal_steps,
            "difficulty_multiplier": self.get_difficulty_multiplier()
        }
    
    def _evaluate_success(self, final_state, trajectory) -> str:
        """Return success criteria description for LLM judge evaluation."""
        mode_text = "assembly" if self.assembly_mode else "disassembly"
        
        base_criteria = f"""
LUBAN LOCK {mode_text.upper()} SUCCESS CRITERIA:

Task: {self.piece_type} Luban lock {mode_text}
Difficulty: {self.difficulty.value}

SUCCESS REQUIREMENTS:
"""
        
        if self.assembly_mode:
            criteria = base_criteria + """
1. STRUCTURAL INTEGRITY: All pieces must be properly interlocked
   - No loose or disconnected pieces
   - Pieces should form a cohesive 3D structure
   - The assembly should be stable and not fall apart

2. GEOMETRIC CORRECTNESS: Pieces must be in correct positions
   - Each piece should occupy its designated position in the lock
   - No overlapping or improper positioning
   - The overall shape should match the target Luban lock design

3. MECHANICAL FUNCTION: The lock mechanism should work
   - Pieces should be locked in place by interlocking geometry
   - The structure should not easily come apart with gentle handling
   - Key pieces should be properly secured

4. COMPLETION: All available pieces should be used
   - No pieces left unused (unless they are extras)
   - The structure should appear complete and finished

EVALUATION INSTRUCTIONS:
- Examine the final state image carefully
- Look for proper interlocking between pieces
- Check if the structure appears stable and complete
- Consider the complexity appropriate for the difficulty level
- Success = stable, complete, properly interlocked Luban lock structure
"""
        else:
            criteria = base_criteria + """
1. COMPLETE SEPARATION: All pieces must be separated
   - No pieces should remain connected or interlocked
   - Each piece should be individual and free
   - Pieces should be clearly separated in space

2. STRUCTURAL PRESERVATION: Pieces should be undamaged
   - No broken or damaged pieces from forced separation
   - Original piece geometry should be maintained
   - No pieces should be stuck or jammed

3. METHODICAL DISASSEMBLY: Evidence of proper technique
   - Pieces should be removed in logical sequence
   - No signs of forcing or improper manipulation
   - The disassembly should appear controlled

4. SPATIAL ORGANIZATION: Pieces should be clearly distinguishable
   - Separated pieces should not be overlapping
   - Each piece should be visible and identifiable
   - Clear evidence that the lock has been completely disassembled

EVALUATION INSTRUCTIONS:
- Examine the final state image carefully
- Count the separated pieces and compare to expected number
- Look for any remaining connections between pieces  
- Check if pieces appear undamaged from the disassembly process
- Success = all pieces cleanly separated without damage
"""
        
        return criteria
