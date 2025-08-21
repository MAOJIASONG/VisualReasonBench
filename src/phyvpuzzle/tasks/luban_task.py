"""
Luban lock task implementation for 3D puzzle assembly.
"""

from typing import Dict, Any
from ..core.base import BaseEnvironment, TaskType, TaskDifficulty
from .base_task import PuzzleTask


class LubanTask(PuzzleTask):
    """Task for Luban lock (wooden burr puzzle) assembly/disassembly."""
    
    def __init__(self, difficulty: TaskDifficulty, config: Dict[str, Any]):
        super().__init__(TaskType.LUBAN_LOCK, difficulty, config)
        
        # Luban-specific configuration
        self.num_pieces = self._get_num_pieces_for_difficulty()
        self.assembly_mode = config.get("assembly_mode", True)  # True: assembly, False: disassembly
        self.piece_type = config.get("piece_type", "3-piece")  # 3-piece, 6-piece, 15-piece
        
    def _get_num_pieces_for_difficulty(self) -> int:
        """Get number of pieces based on difficulty."""
        piece_counts = {
            TaskDifficulty.VERY_EASY: 3,
            TaskDifficulty.EASY: 3,
            TaskDifficulty.MEDIUM: 6,
            TaskDifficulty.HARD: 15,
            TaskDifficulty.VERY_HARD: 15
        }
        return piece_counts.get(self.difficulty, 6)
        
    def _configure_environment(self, environment: BaseEnvironment) -> None:
        """Configure environment for Luban lock task."""
        # Set appropriate camera positions for 3D puzzle viewing
        if hasattr(environment, 'multi_view_renderer'):
            # Set camera to capture the 3D structure well
            environment.multi_view_renderer.set_camera_config(
                "perspective", [1.2, -1.2, 1.0], [0, 0, 0.5]
            )
            
    def _load_task_objects(self, environment: BaseEnvironment) -> None:
        """Load Luban lock pieces into the environment."""
        # This would load the appropriate URDF files for the Luban pieces
        # Based on the piece_type (3-piece, 6-piece, 15-piece)
        pass
        
    def _set_initial_state(self, environment: BaseEnvironment) -> None:
        """Set initial state for Luban lock task."""
        if self.assembly_mode:
            # Start with pieces separated for assembly task
            self._separate_pieces(environment)
        else:
            # Start with assembled puzzle for disassembly task
            self._assemble_pieces(environment)
            
    def _separate_pieces(self, environment: BaseEnvironment) -> None:
        """Separate Luban pieces for assembly task."""
        # Place pieces in different positions around the workspace
        pass
        
    def _assemble_pieces(self, environment: BaseEnvironment) -> None:
        """Pre-assemble pieces for disassembly task."""
        # Place pieces in assembled configuration
        pass
        
    def _get_base_system_prompt(self) -> str:
        """Get base system prompt for Luban lock tasks."""
        return """You are an AI agent controlling a physics simulation to solve Luban lock puzzles.
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
6. Success is measured by proper interlocking and stability of the assembly"""
        
    def _get_task_specific_prompt(self) -> str:
        """Get Luban lock specific additions to system prompt."""
        mode_text = "assembly" if self.assembly_mode else "disassembly"
        
        prompt = f"""
TASK SPECIFICS:
- Puzzle type: {self.piece_type} Luban lock
- Mode: {mode_text}
- Number of pieces: {self.num_pieces}
- Difficulty: {self.difficulty.value}

"""
        
        if self.assembly_mode:
            prompt += """ASSEMBLY MODE:
- The pieces are initially separated
- Your goal is to assemble them into the interlocked configuration
- Look for grooves, slots, and complementary shapes
- Pieces typically slide together in specific sequences"""
        else:
            prompt += """DISASSEMBLY MODE:
- The puzzle starts in assembled form
- Your goal is to separate the pieces without forcing
- Find the key piece that can be removed first
- Often there's a specific sequence for disassembly"""
            
        # Add difficulty-specific guidance
        if self.difficulty == TaskDifficulty.VERY_EASY:
            prompt += "\n- This is a simple 3-piece lock with obvious interlocking"
        elif self.difficulty == TaskDifficulty.MEDIUM:
            prompt += "\n- This 6-piece lock requires understanding of the central structure"
        elif self.difficulty in [TaskDifficulty.HARD, TaskDifficulty.VERY_HARD]:
            prompt += "\n- This complex lock may require discovering hidden mechanisms"
            
        return prompt
        
    def _get_task_description(self) -> str:
        """Get description of the specific Luban lock task."""
        mode_text = "assembly" if self.assembly_mode else "disassembly"
        
        return f"""LUBAN LOCK {mode_text.upper()} PUZZLE

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
