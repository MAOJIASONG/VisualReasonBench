"""
Puzzle environment implementation for physics puzzle tasks.

This module provides a complete jigsaw puzzle environment with
physics simulation capabilities for piece placement and assembly tasks.
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Set, Optional
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
    num_pieces: int = 7  # Default: 7 pieces + 1 container
    puzzle_size: Tuple[int, int, int] = (3, 3, 3)  # Target cube dimensions
    piece_size: float = 0.08
    piece_thickness: float = 0.01
    completion_tolerance: float = 0.05
    target_area_center: Tuple[float, float, float] = (0, 0, 0.41)
    scatter_radius: float = 0.3
    is_3d: bool = True  # Enable 3D stacking puzzle
    container_based: bool = True  # Pieces go into container
    
    def __post_init__(self):
        super().__post_init__()
        # For container-based puzzles, num_pieces is fixed (7 pieces + 1 container)
        # For traditional puzzles, ensure num_pieces matches grid size
        if not self.container_based:
            if self.is_3d and len(self.puzzle_size) == 3:
                # 3D cube: x * y * z
                expected_pieces = self.puzzle_size[0] * self.puzzle_size[1] * self.puzzle_size[2]
            else:
                # 2D grid: x * y (backward compatibility)
                expected_pieces = self.puzzle_size[0] * self.puzzle_size[1]
                
            if self.num_pieces != expected_pieces:
                self.num_pieces = expected_pieces


@register_environment("puzzle")
class PuzzleEnvironment(PhysicsEnvironment):
    """Physics environment for jigsaw puzzle assembly."""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        
        self.puzzle_pieces: List[ObjectInfo] = []
        self.container: ObjectInfo = None  # The container object
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
        self._load_puzzle_models()
        
    def _load_puzzle_models(self) -> None:
        """Load puzzle URDF models."""
        puzzle_base_path = os.path.join(self.config.urdf_local_path, "3x3-stacking-puzzle")
        
        if not os.path.exists(puzzle_base_path):
            print(f"Warning: 3x3 puzzle models not found at {puzzle_base_path}")
            print("Creating simple puzzle pieces instead")
            self._create_simple_puzzle_pieces()
            return
            
        # Load available puzzle piece URDF files
        available_pieces = []
        container_path = None
        
        if os.path.exists(puzzle_base_path):
            for item in os.listdir(puzzle_base_path):
                piece_dir = os.path.join(puzzle_base_path, item)
                urdf_path = os.path.join(piece_dir, "urdf", f"{item}.urdf")
                if os.path.isdir(piece_dir) and os.path.exists(urdf_path):
                    # obj_8 is typically the container
                    if item == "obj_8" and self.config.container_based:
                        container_path = urdf_path
                    else:
                        available_pieces.append(urdf_path)
                
        if not available_pieces:
            print("No puzzle URDF files found, creating simple puzzle pieces")
            self._create_simple_puzzle_pieces()
            return
        
        # Load container first if container-based puzzle
        if self.config.container_based and container_path:
            self._load_container(container_path)
            
        # Select puzzle pieces to use (7 for container-based, or num_pieces)
        if self.config.container_based:
            num_to_load = min(7, len(available_pieces))  # 7 pieces max for container puzzle
        else:
            num_to_load = min(self.config.num_pieces, len(available_pieces))
        selected_pieces = available_pieces[:num_to_load]
        
        print(f"Loading {len(selected_pieces)} puzzle pieces from URDF files")
        
        if self.config.is_3d and len(self.config.puzzle_size) == 3:
            rows, cols, layers = self.config.puzzle_size
        else:
            rows, cols = self.config.puzzle_size[:2]
            layers = 1
        
        # Load puzzle pieces from URDF files
        for i, piece_path in enumerate(selected_pieces):
            if self.config.is_3d:
                # 3D grid position calculation
                layer = i // (rows * cols)
                remaining = i % (rows * cols)
                row = remaining // cols
                col = remaining % cols
                grid_pos_3d = (row, col, layer)
                grid_pos_2d = (row, col)  # For backward compatibility
            else:
                # 2D grid position calculation
                row = i // cols
                col = i % cols
                grid_pos_2d = (row, col)
                grid_pos_3d = (row, col, 0)
                
            piece_name = f"piece_{i+1}"  # Simple naming: piece_1, piece_2, etc.
            
            # Stagger temporary spawn positions to prevent immediate collisions
            temp_x = (i - (num_to_load - 1) / 2) * (self.config.piece_size * 2.0)
            # Use table height + some clearance to ensure pieces are above table
            table_height = self.config.table_position[2] if hasattr(self.config, 'table_position') else 0.4
            temp_z = table_height + 0.1 + 0.01 * i  # Well above table surface
            
            # Calculate 3D neighbors for stacking
            neighbors_3d = self._get_neighbors_3d(row, col, layer) if self.config.is_3d else []
            neighbors_2d = self._get_neighbors(row, col)
            
            self.add_object(
                object_name=piece_name,
                urdf_path=piece_path,
                position=(temp_x, 0, temp_z),
                orientation=(0, 0, 0, 1),
                object_type='puzzle_piece',
                properties={
                    'grid_position': grid_pos_2d,  # Backward compatibility
                    'grid_position_3d': grid_pos_3d,  # 3D position
                    'piece_id': i,
                    'is_placed': False,
                    'target_neighbors': neighbors_2d,
                    'target_neighbors_3d': neighbors_3d
                }
            )
        
        # Update puzzle_pieces list
        self.puzzle_pieces = [obj for obj in self.objects if obj.object_type == "puzzle_piece"]
        
        # If we have fewer pieces than needed, fill with simple pieces
        if len(self.puzzle_pieces) < self.config.num_pieces:
            self._create_additional_simple_pieces(len(self.puzzle_pieces))
        
        # Fix: Make all puzzle pieces kinematic initially to prevent unwanted physics interactions
        for piece in self.puzzle_pieces:
            p.changeDynamics(
                piece.object_id, 
                -1, 
                mass=0,  # Make kinematic (no gravity/physics)
                linearDamping=0.9,
                angularDamping=0.9
            )
    
    def _load_container(self, container_path: str) -> None:
        """Load the container object."""
        print(f"Loading container from {container_path}")
        
        # Position container at target center
        container_pos = self.config.target_area_center
        
        # Load container object
        container_id = p.loadURDF(
            container_path,
            basePosition=container_pos,
            baseOrientation=(0, 0, 0, 1)
        )
        
        # Make container static (kinematic) - it won't move even when collided
        p.changeDynamics(
            container_id, 
            -1, 
            mass=0,  # Set mass to 0 to make it kinematic (static)
            lateralFriction=1.0,
            spinningFriction=0.002,
            rollingFriction=0.002,
            linearDamping=1.0,  # High damping to prevent any movement
            angularDamping=1.0
        )
        
        # Create ObjectInfo for container
        container_obj = ObjectInfo(
            object_id=container_id,
            name="container",
            position=container_pos,
            orientation=(0, 0, 0, 1),
            object_type='container',
            properties={
                'is_container': True,
                'capacity': 7,  # Can hold 7 pieces
                'pieces_inside': [],
                'is_static': True  # Mark as static for reference
            }
        )
        
        self.objects.append(container_obj)
        self.container = container_obj
        
        print(f"Container loaded and fixed at position {container_pos}")
    
    def _ensure_container_fixed(self) -> None:
        """Ensure container remains fixed at its initial position."""
        if self.container and self.container.object_id is not None:
            try:
                # Get current position
                current_pos, current_orient = p.getBasePositionAndOrientation(self.container.object_id)
                target_pos = self.config.target_area_center
                target_orient = (0, 0, 0, 1)
                
                # If position has changed, reset it
                if not np.allclose(current_pos, target_pos, atol=1e-6):
                    p.resetBasePositionAndOrientation(
                        self.container.object_id,
                        target_pos,
                        target_orient
                    )
                    # Update our object info
                    self.container.position = target_pos
                    self.container.orientation = target_orient
            except Exception as e:
                # Object might have been removed, skip the check
                print(f"Warning: Could not check container position: {e}")
                pass
            
    def _create_simple_puzzle_pieces(self) -> None:
        """Create simple box-shaped puzzle pieces as fallback."""
        rows, cols = self.config.puzzle_size
        piece_colors = self._generate_piece_colors()
        
        for i in range(self.config.num_pieces):
            row = i // cols
            col = i % cols
            
            piece_name = f"piece_{i+1}"  # Simple naming: piece_1, piece_2, etc.
            
            # Create piece with unique color
            table_height = self.config.table_position[2] if hasattr(self.config, 'table_position') else 0.4
            piece_id = self.create_primitive_object(
                object_name=piece_name,
                shape_type="box",
                size=[self.config.piece_size/2, self.config.piece_size/2, self.config.piece_thickness/2],
                position=(0, 0, table_height + 0.1),  # Above table surface
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
                    
    def _create_additional_simple_pieces(self, start_index: int) -> None:
        """Create additional simple pieces if not enough URDF models are available."""
        if self.config.is_3d and len(self.config.puzzle_size) == 3:
            rows, cols, layers = self.config.puzzle_size
        else:
            rows, cols = self.config.puzzle_size[:2]
            layers = 1
            
        piece_colors = self._generate_piece_colors()
        
        for i in range(start_index, self.config.num_pieces):
            if self.config.is_3d:
                # 3D grid position calculation
                layer = i // (rows * cols)
                remaining = i % (rows * cols)
                row = remaining // cols
                col = remaining % cols
                grid_pos_3d = (row, col, layer)
                grid_pos_2d = (row, col)
            else:
                # 2D grid position calculation
                row = i // cols
                col = i % cols
                grid_pos_2d = (row, col)
                grid_pos_3d = (row, col, 0)
            
            piece_name = f"piece_{i+1}"  # Simple naming: piece_1, piece_2, etc.
            
            # Create piece with unique color
            table_height = self.config.table_position[2] if hasattr(self.config, 'table_position') else 0.4
            piece_id = self.create_primitive_object(
                object_name=piece_name,
                shape_type="box",
                size=[self.config.piece_size/2, self.config.piece_size/2, self.config.piece_thickness/2],
                position=(0, 0, table_height + 0.1),  # Above table surface
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
        num_pieces = self.config.num_pieces
        if num_pieces <= 0:
            num_pieces = 9  # Default fallback
            
        for i in range(num_pieces):
            # Create distinct colors using HSV color space
            hue = (i * 360 / num_pieces) / 360.0
            colors.append(self._hsv_to_rgba(hue, 0.8, 0.9))
        return colors
    
    def _hsv_to_rgba(self, h: float, s: float, v: float) -> Tuple[float, float, float, float]:
        """Convert HSV to RGBA."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b, 1.0)
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get adjacent grid positions for a piece (2D)."""
        rows, cols = self.config.puzzle_size[:2]
        neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
        
        return neighbors
    
    def _get_neighbors_3d(self, row: int, col: int, layer: int) -> List[Tuple[int, int, int]]:
        """Get neighboring positions for a 3D grid position."""
        neighbors = []
        if not self.config.is_3d or len(self.config.puzzle_size) != 3:
            return neighbors
            
        rows, cols, layers = self.config.puzzle_size
        
        # Check 6-connected neighbors (up, down, left, right, above, below)
        for dr, dc, dl in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nr, nc, nl = row + dr, col + dc, layer + dl
            if 0 <= nr < rows and 0 <= nc < cols and 0 <= nl < layers:
                neighbors.append((nr, nc, nl))
        
        return neighbors
    
    def _define_target_positions(self) -> None:
        """Define target positions for each puzzle piece."""
        if self.config.is_3d and len(self.config.puzzle_size) == 3:
            # 3D cube layout
            rows, cols, layers = self.config.puzzle_size
        else:
            # 2D grid layout (backward compatibility)
            rows, cols = self.config.puzzle_size[:2]
            layers = 1
            
        center = self.config.target_area_center
        
        # Calculate starting position for grid
        start_x = center[0] - (cols - 1) * self.config.piece_size / 2
        start_y = center[1] - (rows - 1) * self.config.piece_size / 2
        start_z = center[2] - (layers - 1) * self.config.piece_size / 2
        
        for piece in self.puzzle_pieces:
            if self.config.is_3d and "grid_position_3d" in piece.properties:
                # 3D position (row, col, layer)
                row, col, layer = piece.properties["grid_position_3d"]
                target_z = start_z + layer * self.config.piece_size
            else:
                # 2D position (backward compatibility)
                row, col = piece.properties.get("grid_position", (0, 0))
                target_z = center[2]
            
            target_x = start_x + col * self.config.piece_size
            target_y = start_y + row * self.config.piece_size
            
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
            # Ensure pieces are well above table surface
            table_height = self.config.table_position[2] if hasattr(self.config, 'table_position') else 0.4
            z = max(center[2], table_height + 0.05)  # At least 5cm above table
            
            p.resetBasePositionAndOrientation(
                piece.object_id,
                [x, y, z],
                piece.orientation
            )
            
            piece.position = (x, y, z)
    
    def _get_current_state(self, metadata: Optional[Dict[str, Any]] = None) -> State:
        """Get current puzzle state."""
        # Ensure container remains fixed
        self._ensure_container_fixed()
        
        self._update_piece_analysis()
        
        # Check if puzzle is completed
        completed = len(self.correctly_placed_pieces) >= self.config.num_pieces * 0.8
        
        return State(
            step=self.step_count,
            objects=self.objects,
            time_stamp=time.time(),
            metadata={
                "correctly_placed": len(self.correctly_placed_pieces),
                "total_pieces": len(self.puzzle_pieces),
                "completion_percentage": (len(self.correctly_placed_pieces) / len(self.puzzle_pieces) * 100) if self.puzzle_pieces else 0,
                "connected_pairs": len(self.piece_connections),
                "puzzle_size": self.config.puzzle_size,
                "completed": completed,
                **(metadata or {}),
            }
        )
    
    def _update_piece_analysis(self) -> None:
        """Analyze current piece positions and connections."""
        self.correctly_placed_pieces.clear()
        self.piece_connections.clear()
        
        if self.config.container_based and self.container:
            # Container-based puzzle: check if pieces are in container
            container_pos = self.container.position
            container_radius = 0.1  # Approximate container size
            
            for piece in self.puzzle_pieces:
                current_pos, _ = p.getBasePositionAndOrientation(piece.object_id)
                
                # Check if piece is inside container (within container bounds)
                distance_to_container = np.linalg.norm(
                    np.array(current_pos[:2]) - np.array(container_pos[:2])  # Only check x,y distance
                )
                
                # Consider piece correctly placed if it's in container or marked as placed
                if (distance_to_container <= container_radius or 
                    piece.properties.get('in_container', False) or
                    piece.properties.get('is_placed', False)):
                    self.correctly_placed_pieces.add(piece.name)
                    piece.properties["is_placed"] = True
                else:
                    piece.properties["is_placed"] = False
            
            # For container puzzles, connections are based on pieces being in container together
            pieces_in_container = [p for p in self.puzzle_pieces if p.name in self.correctly_placed_pieces]
            for i, piece1 in enumerate(pieces_in_container):
                for j, piece2 in enumerate(pieces_in_container[i+1:], i+1):
                    # All pieces in container are considered connected
                    connection = tuple(sorted([piece1.name, piece2.name]))
                    self.piece_connections.add(connection)
                    
        else:
            # Traditional puzzle: check target positions
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
                neighbors = piece1.properties.get("target_neighbors", [])
                
                for neighbor_pos in neighbors:
                    neighbor_piece = next(
                        (p for p in self.puzzle_pieces 
                         if p.properties.get("grid_position") == neighbor_pos), 
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
        completion_pct = (placed_count / total_count) * 100 if total_count > 0 else 0
        
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
        """Place a puzzle piece at its target position or into the container."""
        piece_obj = next((obj for obj in self.objects if obj.name == piece_name), None)
        
        if not piece_obj:
            return {"status": "error", "message": f"Puzzle piece '{piece_name}' not found"}
        
        if self.config.container_based and self.container:
            # Container-based puzzle: place piece into container
            container_pos = self.container.position
            
            # Calculate position inside container
            # For now, place pieces at container center with slight vertical offset
            pieces_in_container = len([p for p in self.puzzle_pieces if p.properties.get('is_placed', False)])
            vertical_offset = 0.02 * pieces_in_container  # Stack pieces slightly
            
            if precise:
                # Place at calculated container position
                new_pos = (
                    container_pos[0],
                    container_pos[1], 
                    container_pos[2] + vertical_offset
                )
            else:
                # Add small random offset within container bounds
                offset_x = np.random.uniform(-0.03, 0.03)
                offset_y = np.random.uniform(-0.03, 0.03)
                new_pos = (
                    container_pos[0] + offset_x,
                    container_pos[1] + offset_y,
                    container_pos[2] + vertical_offset
                )
            
            # Mark piece as placed in container
            piece_obj.properties['is_placed'] = True
            piece_obj.properties['in_container'] = True
            
            # Update container's pieces list
            if 'pieces_inside' in self.container.properties:
                if piece_name not in self.container.properties['pieces_inside']:
                    self.container.properties['pieces_inside'].append(piece_name)
                    
            message = f"Placed '{piece_name}' into the container"
            
        else:
            # Traditional puzzle: place at target position
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
                
            message = f"Placed '{piece_name}' at target position"
        
        # Fix: Set object to be kinematic (non-dynamic) to prevent physics drift
        p.changeDynamics(
            piece_obj.object_id, 
            -1, 
            mass=0,  # Make kinematic (no gravity/physics)
            linearDamping=0.9,  # Add damping to prevent drift
            angularDamping=0.9
        )
        
        p.resetBasePositionAndOrientation(
            piece_obj.object_id,
            new_pos,
            piece_obj.orientation
        )
        
        piece_obj.position = new_pos
        
        return {"status": "success", "message": message}

    @BaseEnvironment.register_tool("align_pieces")
    def _tool_align_pieces(self, piece1: str, piece2: str, direction: str) -> Dict[str, Any]:
        """Align two puzzle pieces next to each other."""
        obj1 = next((obj for obj in self.objects if obj.name == piece1), None)
        obj2 = next((obj for obj in self.objects if obj.name == piece2), None)
        
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
        piece_obj = next((obj for obj in self.objects if obj.name == piece_name), None)
        
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
