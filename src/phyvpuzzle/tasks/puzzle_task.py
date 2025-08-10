"""
Puzzle Task Implementation

This module implements puzzle tasks such as Kongming Lock and other 3D puzzles.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .base_task import BaseTask, TaskConfiguration, TaskType


class PuzzleTask(BaseTask):
    """Implementation of puzzle tasks."""
    
    def __init__(self, config: TaskConfiguration):
        super().__init__(config)
        self.puzzle_type = config.parameters.get("puzzle_type", "kongming_lock")
        self.target_configuration = None
        self.pieces = []
        self.solution_steps = []
        
    def setup_task(self, environment) -> bool:
        """Setup the puzzle task in the environment."""
        try:
            self.environment = environment
            
            if self.puzzle_type == "kongming_lock":
                return self._setup_kongming_lock()
            elif self.puzzle_type == "block_sorting":
                return self._setup_block_sorting()
            else:
                # Default simple puzzle
                return self._setup_simple_puzzle()
                
        except Exception as e:
            print(f"Error setting up puzzle task: {e}")
            return False
    
    def _setup_kongming_lock(self) -> bool:
        """Setup Kongming Lock puzzle."""
        # Create interlocking pieces
        positions = [
            (0.0, 0.0, 0.5),
            (0.1, 0.0, 0.5),
            (0.0, 0.1, 0.5),
            (-0.1, 0.0, 0.5),
            (0.0, -0.1, 0.5),
            (0.0, 0.0, 0.6)
        ]
        
        colors = [
            (1, 0, 0, 1),  # Red
            (0, 1, 0, 1),  # Green
            (0, 0, 1, 1),  # Blue
            (1, 1, 0, 1),  # Yellow
            (1, 0, 1, 1),  # Magenta
            (0, 1, 1, 1)   # Cyan
        ]
        
        for i in range(6):
            piece_name = f"lock_piece_{i}"
            self.environment.create_primitive_object(
                object_name=piece_name,
                shape_type="box",
                size=(0.02, 0.08, 0.02),
                position=positions[i],
                color=colors[i],
                mass=0.1
            )
            self.pieces.append(piece_name)
            self.current_objects[piece_name] = {
                "type": "lock_piece",
                "index": i,
                "color": colors[i]
            }
        
        # Define target configuration (assembled lock)
        self.target_configuration = {
            "lock_piece_0": (0.0, 0.0, 0.5),
            "lock_piece_1": (0.04, 0.0, 0.5),
            "lock_piece_2": (0.0, 0.04, 0.5),
            "lock_piece_3": (-0.04, 0.0, 0.5),
            "lock_piece_4": (0.0, -0.04, 0.5),
            "lock_piece_5": (0.0, 0.0, 0.54)
        }
        
        # Define optimal solution steps
        self.solution_steps = [
            "pick lock_piece_1",
            "place lock_piece_1 on lock_piece_0",
            "pick lock_piece_2",
            "place lock_piece_2 on lock_piece_0",
            "pick lock_piece_3",
            "place lock_piece_3 on lock_piece_0",
            "pick lock_piece_4",
            "place lock_piece_4 on lock_piece_0",
            "pick lock_piece_5",
            "place lock_piece_5 on top"
        ]
        
        return True
    
    def _setup_block_sorting(self) -> bool:
        """Setup block sorting puzzle."""
        # Create colored blocks to sort
        colors = [
            (1, 0, 0, 1),  # Red
            (0, 1, 0, 1),  # Green
            (0, 0, 1, 1)   # Blue
        ]
        
        # Random initial positions
        import random
        positions = [(x * 0.15 - 0.15, 0, 0.5) for x in range(3)]
        random.shuffle(positions)
        
        for i, (color, pos) in enumerate(zip(colors, positions)):
            block_name = f"block_{color[0]}_{color[1]}_{color[2]}"
            self.environment.create_primitive_object(
                object_name=block_name,
                shape_type="box",
                size=(0.04, 0.04, 0.04),
                position=pos,
                color=color,
                mass=0.1
            )
            self.pieces.append(block_name)
            self.current_objects[block_name] = {
                "type": "colored_block",
                "color": color
            }
        
        # Target: Sort blocks by color (red, green, blue from left to right)
        self.target_configuration = {
            "block_1_0_0": (-0.15, 0, 0.5),  # Red left
            "block_0_1_0": (0, 0, 0.5),       # Green center
            "block_0_0_1": (0.15, 0, 0.5)     # Blue right
        }
        
        return True
    
    def _setup_simple_puzzle(self) -> bool:
        """Setup a simple stacking puzzle."""
        # Create three blocks to stack
        for i in range(3):
            block_name = f"block_{i}"
            self.environment.create_primitive_object(
                object_name=block_name,
                shape_type="box",
                size=(0.05 - i*0.01, 0.05 - i*0.01, 0.03),
                position=(i * 0.1 - 0.1, 0, 0.5),
                color=(1 - i*0.3, i*0.3, 0, 1),
                mass=0.1
            )
            self.pieces.append(block_name)
            self.current_objects[block_name] = {
                "type": "stackable_block",
                "size": 3 - i  # Larger blocks have bigger size value
            }
        
        # Target: Stack from largest to smallest
        self.target_configuration = "stacked_pyramid"
        
        self.solution_steps = [
            "pick block_2",
            "place block_2 on table center",
            "pick block_1", 
            "stack block_1 on block_2",
            "pick block_0",
            "stack block_0 on block_1"
        ]
        
        return True
    
    def get_task_description(self) -> str:
        """Get natural language description of the task."""
        if self.puzzle_type == "kongming_lock":
            return "Assemble the Kongming Lock by interlocking all six colored pieces together."
        elif self.puzzle_type == "block_sorting":
            return "Sort the colored blocks from left to right in order: red, green, blue."
        else:
            return "Stack the blocks from largest to smallest to form a pyramid."
    
    def check_completion(self) -> bool:
        """Check if the puzzle is completed."""
        if not self.environment:
            return False
        
        state = self.environment.get_state()
        objects = state.get("objects", {})
        
        if self.puzzle_type == "kongming_lock":
            # Check if all pieces are in target positions (with tolerance)
            for piece, target_pos in self.target_configuration.items():
                if piece not in objects:
                    return False
                current_pos = objects[piece]["position"]
                distance = np.linalg.norm(
                    np.array(current_pos) - np.array(target_pos)
                )
                if distance > 0.05:  # 5cm tolerance
                    return False
            return True
            
        elif self.puzzle_type == "block_sorting":
            # Check if blocks are sorted by color
            for block, target_pos in self.target_configuration.items():
                if block not in objects:
                    return False
                current_pos = objects[block]["position"]
                # Check x-position mainly (y and z can vary slightly)
                if abs(current_pos[0] - target_pos[0]) > 0.05:
                    return False
            return True
            
        else:
            # Simple stacking puzzle - check if stacked
            if len(self.pieces) < 3:
                return False
            
            # Check if blocks are stacked (each block above the previous)
            block_positions = []
            for piece in self.pieces:
                if piece in objects:
                    block_positions.append(objects[piece]["position"])
            
            if len(block_positions) < 3:
                return False
            
            # Sort by z-height
            block_positions.sort(key=lambda p: p[2])
            
            # Check if properly stacked (increasing height, aligned x/y)
            for i in range(1, len(block_positions)):
                height_diff = block_positions[i][2] - block_positions[i-1][2]
                x_diff = abs(block_positions[i][0] - block_positions[i-1][0])
                y_diff = abs(block_positions[i][1] - block_positions[i-1][1])
                
                if height_diff < 0.02 or height_diff > 0.1:  # Improper stacking
                    return False
                if x_diff > 0.02 or y_diff > 0.02:  # Not aligned
                    return False
            
            return True
    
    def evaluate_state(self) -> float:
        """Evaluate the current state and return a score."""
        if not self.environment:
            return 0.0
        
        state = self.environment.get_state()
        objects = state.get("objects", {})
        
        if self.check_completion():
            return 1.0
        
        score = 0.0
        
        if self.puzzle_type == "kongming_lock":
            # Score based on how many pieces are in correct positions
            correct_pieces = 0
            for piece, target_pos in self.target_configuration.items():
                if piece in objects:
                    current_pos = objects[piece]["position"]
                    distance = np.linalg.norm(
                        np.array(current_pos) - np.array(target_pos)
                    )
                    if distance < 0.05:
                        correct_pieces += 1
                    else:
                        # Partial score based on distance
                        score += max(0, 1 - distance) / len(self.target_configuration)
            
            score += correct_pieces / len(self.target_configuration) * 0.5
            
        elif self.puzzle_type == "block_sorting":
            # Score based on correct ordering
            for block, target_pos in self.target_configuration.items():
                if block in objects:
                    current_pos = objects[block]["position"]
                    x_distance = abs(current_pos[0] - target_pos[0])
                    if x_distance < 0.05:
                        score += 1.0 / len(self.target_configuration)
                    else:
                        score += max(0, 1 - x_distance) / len(self.target_configuration) * 0.5
        
        else:
            # Stacking puzzle - score based on stacking progress
            if len(objects) >= 2:
                # Check if any two blocks are stacked
                positions = [objects[p]["position"] for p in self.pieces if p in objects]
                positions.sort(key=lambda p: p[2])
                
                for i in range(1, len(positions)):
                    height_diff = positions[i][2] - positions[i-1][2]
                    if 0.02 < height_diff < 0.1:
                        score += 0.3
                
                # Bonus for proper alignment
                if len(positions) >= 2:
                    x_aligned = all(abs(p[0] - positions[0][0]) < 0.02 for p in positions)
                    y_aligned = all(abs(p[1] - positions[0][1]) < 0.02 for p in positions)
                    if x_aligned and y_aligned:
                        score += 0.2
        
        return min(1.0, score)
    
    def get_optimal_solution(self) -> List[str]:
        """Get the optimal solution sequence."""
        return self.solution_steps
    
    def reset_task(self) -> None:
        """Reset the task to initial state."""
        self.state.step_count = 0
        self.state.is_completed = False
        self.state.is_failed = False
        self.state.current_score = 0.0
        
        if self.environment:
            self.environment.reset()
            self.setup_task(self.environment)
    
    def get_task_specific_context(self) -> Dict[str, Any]:
        """Get task-specific context information."""
        context = {
            "puzzle_type": self.puzzle_type,
            "num_pieces": len(self.pieces),
            "pieces": self.pieces
        }
        
        if self.puzzle_type == "kongming_lock":
            context["assembly_required"] = True
            context["interlocking"] = True
        elif self.puzzle_type == "block_sorting":
            context["sorting_order"] = "red, green, blue"
            context["arrangement"] = "horizontal"
        else:
            context["stacking_order"] = "largest to smallest"
            context["formation"] = "pyramid"
        
        return context