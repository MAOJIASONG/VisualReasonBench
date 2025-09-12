"""
Puzzle environment implementation for physics puzzle tasks.

This module provides a complete jigsaw puzzle environment with
physics simulation capabilities for piece placement and assembly tasks.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass
from pathlib import Path

# Conditional pybullet import - will use mock from base_environment if not available
from phyvpuzzle.environment.base_env import PhysicsEnvironment, p
from phyvpuzzle.core import register_environment_config, register_environment, EnvironmentConfig, BaseEnvironment
from phyvpuzzle.core.base import State, ObjectInfo


@register_environment_config("puzzle")
@dataclass
class PuzzleConfig(EnvironmentConfig):
    """Configuration for jigsaw puzzle environment."""
    num_pieces: int = 9
    puzzle_size: Tuple[int, int] = (3, 3)  # Grid dimensions
    piece_size: float = 0.08
    piece_thickness: float = 0.01
    completion_tolerance: float = 0.05
    target_area_center: Tuple[float, float, float] = (0, 0, 0.41)
    scatter_radius: float = 0.3
    
    def __post_init__(self):
        super().__post_init__()
        # Ensure num_pieces matches grid size
        if self.num_pieces != self.puzzle_size[0] * self.puzzle_size[1]:
            self.num_pieces = self.puzzle_size[0] * self.puzzle_size[1]


@register_environment("puzzle")
class PuzzleEnvironment(PhysicsEnvironment):
    """Physics environment for jigsaw puzzle assembly."""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        
        self.puzzle_pieces: List[ObjectInfo] = []
        self.correctly_placed_pieces: Set[str] = set()
        self.target_positions: Dict[str, Tuple[float, float, float]] = {}
        self.piece_connections: Set[Tuple[str, str]] = set()
        
    def _setup_task_environment(self) -> None:
        """Setup puzzle-specific environment with pieces."""
        self._create_puzzle_pieces()
        self._define_target_positions()
        self._scatter_pieces()
        
    def _create_puzzle_pieces(self) -> None:
        """Create puzzle pieces in the environment."""
        rows, cols = self.config.puzzle_size
        piece_colors = self._generate_piece_colors()
        
        for i in range(self.config.num_pieces):
            row = i // cols
            col = i % cols
            
            piece_name = f"puzzle_piece_{row}_{col}"
            
            # Create piece with unique color
            piece_id = self.create_primitive_object(
                object_name=piece_name,
                shape_type="box",
                size=[self.config.piece_size/2, self.config.piece_size/2, self.config.piece_thickness/2],
                position=(0, 0, 0.5),  # Temporary position
                color=piece_colors[i],
                mass=0.05
            )
            
            # Add to puzzle_pieces list and set properties
            for obj in self.objects:
                if obj.name == piece_name:
                    obj.properties.update({
                        "grid_position": (row, col),
                        "piece_id": i,
                        "is_placed": False,
                        "target_neighbors": self._get_neighbors(row, col)
                    })
                    self.puzzle_pieces.append(obj)
                    break
    
    def _generate_piece_colors(self) -> List[Tuple[float, float, float, float]]:
        """Generate distinct colors for puzzle pieces."""
        colors = []
        for i in range(self.config.num_pieces):
            # Create distinct colors using HSV color space
            hue = (i * 360 / self.config.num_pieces) / 360.0
            colors.append(self._hsv_to_rgba(hue, 0.8, 0.9))
        return colors
    
    def _hsv_to_rgba(self, h: float, s: float, v: float) -> Tuple[float, float, float, float]:
        """Convert HSV to RGBA."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b, 1.0)
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get adjacent grid positions for a piece."""
        rows, cols = self.config.puzzle_size
        neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
        
        return neighbors
    
    def _define_target_positions(self) -> None:
        """Define target positions for each puzzle piece."""
        rows, cols = self.config.puzzle_size
        center = self.config.target_area_center
        
        # Calculate starting position for grid
        start_x = center[0] - (cols - 1) * self.config.piece_size / 2
        start_y = center[1] - (rows - 1) * self.config.piece_size / 2
        
        for piece in self.puzzle_pieces:
            row, col = piece.properties["grid_position"]
            
            target_x = start_x + col * self.config.piece_size
            target_y = start_y + row * self.config.piece_size
            target_z = center[2]
            
            self.target_positions[piece.name] = (target_x, target_y, target_z)
    
    def _scatter_pieces(self) -> None:
        """Scatter puzzle pieces randomly around the workspace."""
        center = self.config.target_area_center
        
        for piece in self.puzzle_pieces:
            # Random position within scatter radius
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.2, self.config.scatter_radius)
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2] + 0.05  # Slightly above target area
            
            p.resetBasePositionAndOrientation(
                piece.object_id,
                [x, y, z],
                piece.orientation
            )
            
            piece.position = (x, y, z)
    
    def _get_current_state(self) -> State:
        """Get current puzzle state."""
        self._update_piece_analysis()
        
        # Check if puzzle is completed
        completed = len(self.correctly_placed_pieces) >= self.config.num_pieces * 0.8
        
        return State(
            step=self.step_count,
            objects=self.objects,
            completed=completed,
            metadata={
                "correctly_placed": len(self.correctly_placed_pieces),
                "total_pieces": len(self.puzzle_pieces),
                "completion_percentage": len(self.correctly_placed_pieces) / len(self.puzzle_pieces) * 100,
                "connected_pairs": len(self.piece_connections),
                "puzzle_size": self.config.puzzle_size
            }
        )
    
    def _update_piece_analysis(self) -> None:
        """Analyze current piece positions and connections."""
        self.correctly_placed_pieces.clear()
        self.piece_connections.clear()
        
        # Check each piece's position
        for piece in self.puzzle_pieces:
            current_pos, _ = p.getBasePositionAndOrientation(piece.object_id)
            target_pos = self.target_positions[piece.name]
            
            # Check if piece is correctly placed
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            if distance <= self.config.completion_tolerance:
                self.correctly_placed_pieces.add(piece.name)
                piece.properties["is_placed"] = True
            else:
                piece.properties["is_placed"] = False
        
        # Check connections between adjacent pieces
        for i, piece1 in enumerate(self.puzzle_pieces):
            if piece1.name not in self.correctly_placed_pieces:
                continue
                
            pos1, _ = p.getBasePositionAndOrientation(piece1.object_id)
            neighbors = piece1.properties["target_neighbors"]
            
            for neighbor_pos in neighbors:
                neighbor_piece = next(
                    (p for p in self.puzzle_pieces 
                     if p.properties["grid_position"] == neighbor_pos), 
                    None
                )
                
                if neighbor_piece and neighbor_piece.name in self.correctly_placed_pieces:
                    # These pieces are connected
                    connection = tuple(sorted([piece1.name, neighbor_piece.name]))
                    self.piece_connections.add(connection)
    
    def _get_state_description(self) -> str:
        """Get textual description of puzzle state."""
        self._update_piece_analysis()
        
        placed_count = len(self.correctly_placed_pieces)
        total_count = len(self.puzzle_pieces)
        completion_pct = (placed_count / total_count) * 100
        
        desc = f"Puzzle Environment - Step {self.step_count}: "
        desc += f"{placed_count}/{total_count} pieces correctly placed ({completion_pct:.1f}% complete). "
        desc += f"Connected piece pairs: {len(self.piece_connections)}. "
        
        if completion_pct >= 80:
            desc += "Puzzle nearly complete!"
        elif completion_pct >= 50:
            desc += "Good progress on puzzle assembly."
        elif placed_count == 0:
            desc += "No pieces placed correctly yet."
        else:
            desc += "Continue placing pieces in correct positions."
            
        return desc
    
    def _get_task_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get puzzle-specific tool schemas."""
        def build_schema(name: str, desc: str, properties: Dict[str, Any], required: List[str]):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        
        return [
            build_schema(
                "place_piece",
                "Place a puzzle piece at its target position",
                {
                    "piece_name": {"type": "string", "description": "Name of puzzle piece to place"},
                    "precise": {"type": "boolean", "description": "Use precise placement", "default": True}
                },
                ["piece_name"]
            ),
            build_schema(
                "align_pieces",
                "Align two puzzle pieces next to each other",
                {
                    "piece1": {"type": "string", "description": "Name of first piece"},
                    "piece2": {"type": "string", "description": "Name of second piece"},
                    "direction": {
                        "type": "string",
                        "enum": ["above", "below", "left", "right"],
                        "description": "Direction to align second piece relative to first"
                    }
                },
                ["piece1", "piece2", "direction"]
            ),
            build_schema(
                "auto_solve",
                "Automatically solve the puzzle by placing all pieces",
                {
                    "step_by_step": {"type": "boolean", "description": "Place pieces one by one", "default": False}
                },
                []
            ),
            build_schema(
                "shuffle_pieces",
                "Scatter all puzzle pieces randomly",
                {
                    "scatter_radius": {"type": "number", "description": "Radius to scatter pieces", "default": 0.3}
                },
                []
            ),
            build_schema(
                "get_piece_info",
                "Get information about a specific puzzle piece",
                {
                    "piece_name": {"type": "string", "description": "Name of puzzle piece"}
                },
                ["piece_name"]
            )
        ]

    @BaseEnvironment.register_tool("place_piece")
    def _tool_place_piece(self, piece_name: str, precise: bool = True) -> Dict[str, Any]:
        """Place a puzzle piece at its target position."""
        piece_obj = self.get_object_by_name(piece_name)
        
        if not piece_obj:
            return {"status": "error", "message": f"Puzzle piece '{piece_name}' not found"}
        
        if piece_name not in self.target_positions:
            return {"status": "error", "message": f"No target position defined for '{piece_name}'"}
        
        target_pos = self.target_positions[piece_name]
        
        if precise:
            # Place exactly at target
            new_pos = target_pos
        else:
            # Place near target with small random offset
            offset = np.random.uniform(-0.02, 0.02, 3)
            new_pos = tuple(np.array(target_pos) + offset)
        
        p.resetBasePositionAndOrientation(
            piece_obj.object_id,
            new_pos,
            piece_obj.orientation
        )
        
        piece_obj.position = new_pos
        
        return {"status": "success", "message": f"Placed '{piece_name}' at target position"}

    @BaseEnvironment.register_tool("align_pieces")
    def _tool_align_pieces(self, piece1: str, piece2: str, direction: str) -> Dict[str, Any]:
        """Align two puzzle pieces next to each other."""
        obj1 = self.get_object_by_name(piece1)
        obj2 = self.get_object_by_name(piece2)
        
        if not obj1:
            return {"status": "error", "message": f"Piece '{piece1}' not found"}
        if not obj2:
            return {"status": "error", "message": f"Piece '{piece2}' not found"}
        
        pos1, _ = p.getBasePositionAndOrientation(obj1.object_id)
        piece_size = self.config.piece_size
        
        # Calculate alignment offset
        direction_offsets = {
            "right": [piece_size, 0, 0],
            "left": [-piece_size, 0, 0],
            "above": [0, piece_size, 0],
            "below": [0, -piece_size, 0]
        }
        
        if direction not in direction_offsets:
            return {"status": "error", "message": f"Invalid direction: {direction}"}
        
        offset = direction_offsets[direction]
        new_pos = [pos1[0] + offset[0], pos1[1] + offset[1], pos1[2] + offset[2]]
        
        p.resetBasePositionAndOrientation(
            obj2.object_id,
            new_pos,
            obj2.orientation
        )
        
        obj2.position = tuple(new_pos)
        
        return {"status": "success", "message": f"Aligned '{piece2}' {direction} of '{piece1}'"}

    @BaseEnvironment.register_tool("auto_solve")
    def _tool_auto_solve(self, step_by_step: bool = False) -> Dict[str, Any]:
        """Automatically solve the puzzle by placing all pieces."""
        if not self.puzzle_pieces:
            return {"status": "error", "message": "No puzzle pieces available"}
        
        placed_count = 0
        for piece in self.puzzle_pieces:
            if piece.name in self.target_positions:
                target_pos = self.target_positions[piece.name]
                
                p.resetBasePositionAndOrientation(
                    piece.object_id,
                    target_pos,
                    piece.orientation
                )
                
                piece.position = target_pos
                placed_count += 1
                
                if step_by_step:
                    # In step-by-step mode, only place one piece per call
                    break
        
        message = f"Auto-solved: placed {placed_count} pieces"
        if step_by_step and placed_count == 1:
            message += " (step-by-step mode)"
        
        return {"status": "success", "message": message}

    @BaseEnvironment.register_tool("shuffle_pieces")
    def _tool_shuffle_pieces(self, scatter_radius: float = 0.3) -> Dict[str, Any]:
        """Scatter all puzzle pieces randomly."""
        center = self.config.target_area_center
        
        for piece in self.puzzle_pieces:
            # Random position within scatter radius
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.1, scatter_radius)
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2] + 0.05
            
            p.resetBasePositionAndOrientation(
                piece.object_id,
                [x, y, z],
                piece.orientation
            )
            
            piece.position = (x, y, z)
        
        return {"status": "success", "message": f"Shuffled {len(self.puzzle_pieces)} pieces"}

    @BaseEnvironment.register_tool("get_piece_info")
    def _tool_get_piece_info(self, piece_name: str) -> Dict[str, Any]:
        """Get information about a specific puzzle piece."""
        piece_obj = self.get_object_by_name(piece_name)
        
        if not piece_obj:
            return {"status": "error", "message": f"Piece '{piece_name}' not found"}
        
        current_pos, _ = p.getBasePositionAndOrientation(piece_obj.object_id)
        target_pos = self.target_positions.get(piece_name, None)
        
        distance_to_target = None
        if target_pos:
            distance_to_target = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        info = {
            "piece_name": piece_name,
            "grid_position": piece_obj.properties.get("grid_position"),
            "current_position": current_pos,
            "target_position": target_pos,
            "distance_to_target": distance_to_target,
            "is_placed": piece_obj.properties.get("is_placed", False),
            "neighbors": piece_obj.properties.get("target_neighbors", [])
        }
        
        return {"status": "success", "message": "Piece information retrieved", "info": info}
