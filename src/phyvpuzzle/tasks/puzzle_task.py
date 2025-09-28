"""
Puzzle task implementation for jigsaw puzzle assembly.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from phyvpuzzle.core.base import TaskDifficulty, TaskResult, EvaluationResult
from phyvpuzzle.core import BaseEnvironment, register_task, register_task_config, TaskConfig
from phyvpuzzle.tasks.base_task import PhysicsTask
from phyvpuzzle.environment.puzzle_env import PuzzleEnvironment


@register_task_config("puzzle_assembly")
@dataclass
class PuzzleAssemblyTaskConfig(TaskConfig):
    """Configuration for puzzle assembly task."""
    num_pieces: int = 7
    puzzle_size: tuple = (3, 3, 3)  # Grid dimensions (rows, cols)
    piece_size: float = 0.08
    completion_threshold: float = 0.8  # 80% pieces correctly placed = success
    allow_auto_solve: bool = False  # Whether to allow auto-solve tool
    ruled_evaluation: bool = False  # Whether to use rule-based evaluation
    container_based: bool = False  # Support for container-based puzzles
    
    def __post_init__(self):
        super().__post_init__()
        # For container-based puzzles, don't override num_pieces
        # For traditional puzzles, ensure num_pieces matches grid size
        if not getattr(self, 'container_based', False):
            if self.num_pieces != self.puzzle_size[0] * self.puzzle_size[1]:
                self.num_pieces = self.puzzle_size[0] * self.puzzle_size[1]


@register_task("puzzle_assembly")
class PuzzleAssemblyTask(PhysicsTask):
    """Task for jigsaw puzzle assembly."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.completion_threshold = getattr(config, 'completion_threshold', 0.8)
        
    def _calculate_optimal_steps(self) -> int:
        """Calculate optimal steps for puzzle assembly task."""
        # Optimal strategy: observe -> place each piece -> verify completion
        # Base steps: 1 (observe) + num_pieces (place) + 1 (verify)
        base_steps = 1 + self.config.num_pieces + 1
        
        if self.config.difficulty == TaskDifficulty.EASY:
            return base_steps
        elif self.config.difficulty == TaskDifficulty.MEDIUM:
            return base_steps + 2  # May need some adjustments
        elif self.config.difficulty == TaskDifficulty.HARD:
            return base_steps + 4  # More complex positioning needed
        else:
            return base_steps
            
    def _configure_environment(self) -> None:
        """Configure environment for puzzle assembly task."""
        # Ensure we have a puzzle environment
        if not isinstance(self.environment, PuzzleEnvironment):
            raise ValueError("PuzzleAssemblyTask requires PuzzleEnvironment")
        
        # Set puzzle configuration based on task config
        self.environment.config.num_pieces = self.config.num_pieces
        self.environment.config.puzzle_size = self.config.puzzle_size
        self.environment.config.piece_size = self.config.piece_size
        
        # Setup the puzzle environment
        self.environment._setup_task_environment()
        
    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        """Evaluate task success based on puzzle completion."""
        for task_result in task_results:
            if task_result.error_message:
                # Task failed due to error
                task_result.success = False
                continue
                
            # Check final state for puzzle completion
            if not task_result.trajectory:
                task_result.success = False
                continue
                
            # Get the final observation
            final_observation = task_result.trajectory[-1][1]  # (action, observation)
            final_state = final_observation.state
            
            # Check completion percentage from metadata
            completion_pct = final_state.metadata.get('completion_percentage', 0) / 100.0
            correctly_placed = final_state.metadata.get('correctly_placed', 0)
            total_pieces = final_state.metadata.get('total_pieces', self.config.num_pieces)
            
            # Success if completion percentage meets threshold
            success = completion_pct >= self.completion_threshold
            task_result.success = success
            
            # Add detailed metrics to metadata
            task_result.metadata.update({
                'completion_percentage': completion_pct * 100,
                'correctly_placed_pieces': correctly_placed,
                'total_pieces': total_pieces,
                'completion_threshold': self.completion_threshold * 100,
                'connected_pairs': final_state.metadata.get('connected_pairs', 0),
                'task_specific_success': success
            })
            
        return task_results
        
    def _get_initial_system_prompt(self) -> str:
        """Get system prompt for puzzle assembly task."""
        if getattr(self.config, 'container_based', False):
            # 3D Container-based puzzle
            return f"""You are an AI agent controlling a physics simulation to solve a 3D cube stacking puzzle.
Your goal is to place {self.config.num_pieces} complex-shaped puzzle pieces into a container to form a complete 3×3×3 cube.
You can observe the environment through images showing the current state from multiple viewpoints.

PUZZLE UNDERSTANDING:
- This is a 3D SPATIAL PUZZLE, not a flat jigsaw puzzle
- {self.config.num_pieces} puzzle pieces (each is a complex 3D shape made of multiple small cubes)
- 1 container that must hold all pieces perfectly
- Target: All pieces must fit inside the container to form a solid 3×3×3 cube (27 small cubes total)

IMPORTANT GUIDELINES:
1. Examine each piece's 3D shape and understand its spatial properties
2. Consider how pieces interlock, stack, and fit within the container
3. Use 'get_piece_info' to understand each piece's complex geometry
4. Place pieces into the container using 'place_piece'
5. Use 'align_pieces' to ensure pieces fit together without gaps or overlaps
6. Success requires ALL pieces to fit perfectly in the container (≥{self.completion_threshold*100}%)
7. Think in 3D - pieces can stack vertically and interlock in complex ways
8. Work systematically to fill the container completely

AVAILABLE TOOLS:
- place_piece(piece_name, precise=True): Place a piece into the container (e.g., piece_name="piece_1")
- get_piece_info(piece_name): Get detailed 3D information about a specific piece
- align_pieces(piece1, piece2, direction): Align two pieces for better fit in container

PIECE NAMING: Pieces are named "piece_1", "piece_2", ..., "piece_{self.config.num_pieces}"

CRITICAL: Each piece is a unique 3D shape that occupies multiple cube positions. Success requires spatial reasoning!
"""
        else:
            # Traditional 2D puzzle
            return f"""You are an AI agent controlling a physics simulation to solve jigsaw puzzle assembly tasks.
Your goal is to correctly place puzzle pieces in their target positions to complete the puzzle.
You can observe the environment through images showing the current state of the puzzle from multiple viewpoints.
You may use the available tools to interact with the puzzle pieces as needed.

IMPORTANT GUIDELINES:
1. Always observe the initial setup carefully before taking action
2. Identify each puzzle piece by its color and position
3. Use the 'get_piece_info' tool to understand piece positions and targets
4. Place pieces one by one using the 'place_piece' tool
5. You can align adjacent pieces using the 'align_pieces' tool
6. Success is measured by the percentage of pieces correctly placed (≥{self.completion_threshold*100}%)
7. Get different viewpoints if needed to assess the situation
8. Be systematic - work from corners and edges inward for best results

TASK SPECIFICS:
- Number of pieces: {self.config.num_pieces}
- Puzzle grid size: {self.config.puzzle_size[0]}x{self.config.puzzle_size[1]}
- Piece size: {self.config.piece_size}m
- Success threshold: {self.completion_threshold*100}% pieces correctly placed
- Difficulty: {self.config.difficulty.value}

AVAILABLE TOOLS:
- place_piece(piece_name, precise=True): Place a piece at its target position (e.g., piece_name="piece_1")
- align_pieces(piece1, piece2, direction): Align two pieces relative to each other
- get_piece_info(piece_name): Get detailed information about a specific piece (e.g., piece_name="piece_1")
- shuffle_pieces(scatter_radius): Scatter pieces randomly (use carefully)
{'- auto_solve(step_by_step): Automatically solve puzzle (if allowed)' if getattr(self.config, 'allow_auto_solve', False) else ''}

PIECE NAMING: Pieces are named "piece_1", "piece_2", "piece_3", ..., "piece_9"

Remember: Take your time to understand the puzzle layout before acting!
"""

    def _get_initial_instruction(self) -> str:
        """Get initial instruction for the puzzle task."""
        if getattr(self.config, 'container_based', False):
            # 3D Container-based puzzle
            return f"""Welcome to the 3D Cube Stacking Puzzle Challenge!

Your task is to place {self.config.num_pieces} complex-shaped puzzle pieces into a container to form a complete 3×3×3 cube.

Current Setup:
- {self.config.num_pieces} puzzle pieces are scattered around the workspace
- 1 container is positioned at the target area
- Each piece is a unique 3D shape made of multiple small cubes
- You need to place at least {self.completion_threshold*100}% of pieces correctly in the container to succeed

Puzzle Understanding:
- This is NOT a flat jigsaw puzzle - it's a 3D spatial puzzle
- Each piece has a complex 3D geometry that interlocks with others
- The goal is to fit all pieces perfectly inside the container
- The final result should be a solid 3×3×3 cube with no gaps or overlaps

Strategy Tips:
1. Examine each piece's 3D shape using get_piece_info
2. Consider how pieces might stack, interlock, and fit within the container
3. Start with pieces that have distinctive shapes or clear orientations
4. Work systematically to fill the container completely
5. Use spatial reasoning to understand how pieces relate to each other in 3D space

Begin by observing the current state and understanding each piece's geometry!
"""
        else:
            # Traditional 2D puzzle
            return f"""Welcome to the Jigsaw Puzzle Assembly Challenge!

Your task is to assemble a {self.config.puzzle_size[0]}x{self.config.puzzle_size[1]} jigsaw puzzle with {self.config.num_pieces} pieces.

Current Setup:
- Puzzle pieces are scattered around the workspace
- Each piece has a unique color for identification
- Target assembly area is marked in the center
- You need to place at least {self.completion_threshold*100}% of pieces correctly to succeed

Strategy Tips:
1. Start by getting information about available pieces
2. Look for corner and edge pieces first
3. Use piece colors and positions to identify the correct placement
4. Work systematically from one area to another
5. Use alignment tools to ensure pieces fit together properly

Begin by observing the current state and identifying the puzzle pieces!
"""
