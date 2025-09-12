"""
Lego environment implementation for physics puzzle tasks.

This module provides a complete LEGO brick building environment with
physics simulation capabilities for construction and assembly tasks.
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


@register_environment_config("lego")
@dataclass
class LegoConfig(EnvironmentConfig):
    """Configuration for LEGO building environment."""
    num_bricks: int = 8
    brick_types: List[str] = None
    build_area_size: Tuple[float, float] = (1.0, 1.0)
    target_structure: str = "tower"
    connection_tolerance: float = 0.02
    table_position: Tuple[float, float, float] = (0, 0, 0.4)
    
    def __post_init__(self):
        super().__post_init__()
        if self.brick_types is None:
            self.brick_types = ["2x2", "2x4", "1x2", "1x4"]


@register_environment("lego")
class LegoEnvironment(PhysicsEnvironment):
    """Physics environment for LEGO building puzzles."""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        
        self.lego_bricks: List[ObjectInfo] = []
        self.connected_bricks: Set[Tuple[str, str]] = set()
        self.target_height: float = 0.0
        self.current_structure_height: float = 0.0
        
    def _setup_task_environment(self) -> None:
        """Setup LEGO-specific environment with bricks."""
        self._create_lego_bricks()
        self._set_target_structure()
        
    def _create_lego_bricks(self) -> None:
        """Create LEGO bricks in the environment."""
        brick_configs = {
            "2x2": {"size": [0.04, 0.04, 0.02], "color": [1, 0, 0, 1]},  # Red 2x2
            "2x4": {"size": [0.04, 0.08, 0.02], "color": [0, 0, 1, 1]},  # Blue 2x4
            "1x2": {"size": [0.02, 0.04, 0.02], "color": [0, 1, 0, 1]},  # Green 1x2
            "1x4": {"size": [0.02, 0.08, 0.02], "color": [1, 1, 0, 1]},  # Yellow 1x4
        }
        
        # Place bricks randomly around the table
        table_center = self.config.table_position
        
        for i in range(self.config.num_bricks):
            brick_type = self.config.brick_types[i % len(self.config.brick_types)]
            brick_config = brick_configs[brick_type]
            
            # Random position around table
            x = table_center[0] + np.random.uniform(-0.3, 0.3)
            y = table_center[1] + np.random.uniform(-0.3, 0.3)
            z = table_center[2] + 0.1  # Above table
            
            brick_name = f"lego_{brick_type}_{i}"
            
            brick_id = self.create_primitive_object(
                object_name=brick_name,
                shape_type="box",
                size=brick_config["size"],
                position=(x, y, z),
                color=brick_config["color"],
                mass=0.1
            )
            
            # Add to lego_bricks list
            for obj in self.objects:
                if obj.name == brick_name:
                    obj.properties.update({
                        "brick_type": brick_type,
                        "index": i,
                        "can_connect": True
                    })
                    self.lego_bricks.append(obj)
                    break
    
    def _set_target_structure(self) -> None:
        """Set target structure parameters."""
        if self.config.target_structure == "tower":
            self.target_height = len(self.config.brick_types) * 0.02 * 2  # Rough estimate
        elif self.config.target_structure == "wall":
            self.target_height = 0.04  # 2 bricks high
        else:
            self.target_height = 0.06  # Default moderate height
            
    def _get_current_state(self) -> State:
        """Get current LEGO building state."""
        self._update_structure_analysis()
        
        # Check if building task is completed based on structure analysis
        completed = self._evaluate_build_completion()
        
        return State(
            step=self.step_count,
            objects=self.objects,
            completed=completed,
            metadata={
                "connected_bricks": len(self.connected_bricks),
                "total_bricks": len(self.lego_bricks),
                "current_height": self.current_structure_height,
                "target_height": self.target_height,
                "target_structure": self.config.target_structure
            }
        )
    
    def _update_structure_analysis(self) -> None:
        """Analyze current structure and connections."""
        self.connected_bricks.clear()
        max_height = 0.0
        
        # Simple connection detection based on proximity
        for i, brick1 in enumerate(self.lego_bricks):
            pos1, _ = p.getBasePositionAndOrientation(brick1.object_id)
            max_height = max(max_height, pos1[2])
            
            for j, brick2 in enumerate(self.lego_bricks):
                if i >= j:
                    continue
                    
                pos2, _ = p.getBasePositionAndOrientation(brick2.object_id)
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                
                # If bricks are close enough, consider them connected
                if distance < self.config.connection_tolerance + 0.1:
                    self.connected_bricks.add((brick1.name, brick2.name))
        
        self.current_structure_height = max_height - self.config.table_position[2]
    
    def _evaluate_build_completion(self) -> bool:
        """Evaluate if the building task is completed."""
        # Simple completion criteria - could be made more sophisticated
        height_achieved = self.current_structure_height >= self.target_height * 0.8
        connections_made = len(self.connected_bricks) >= len(self.lego_bricks) // 2
        
        return height_achieved and connections_made
    
    def _get_state_description(self) -> str:
        """Get textual description of LEGO building state."""
        self._update_structure_analysis()
        
        desc = f"LEGO Environment - Step {self.step_count}: "
        desc += f"Building {self.config.target_structure} with {len(self.lego_bricks)} bricks. "
        desc += f"Current structure height: {self.current_structure_height:.3f}m, "
        desc += f"Target height: {self.target_height:.3f}m. "
        desc += f"Connected brick pairs: {len(self.connected_bricks)}. "
        
        if self._evaluate_build_completion():
            desc += "Building task completed successfully!"
        else:
            desc += "Continue building to reach target structure."
            
        return desc
    
    def _get_task_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get LEGO-specific tool schemas."""
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
                "stack_brick",
                "Stack one LEGO brick on top of another",
                {
                    "bottom_brick": {"type": "string", "description": "Name of bottom brick"},
                    "top_brick": {"type": "string", "description": "Name of brick to place on top"},
                    "offset": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Horizontal offset [x, y] for placement",
                        "default": [0, 0]
                    }
                },
                ["bottom_brick", "top_brick"]
            ),
            build_schema(
                "connect_bricks",
                "Connect two LEGO bricks side by side",
                {
                    "brick1": {"type": "string", "description": "Name of first brick"},
                    "brick2": {"type": "string", "description": "Name of second brick"},
                    "direction": {
                        "type": "string", 
                        "enum": ["left", "right", "front", "back"],
                        "description": "Direction to connect bricks"
                    }
                },
                ["brick1", "brick2", "direction"]
            ),
            build_schema(
                "build_tower",
                "Automatically build a tower with available bricks",
                {
                    "base_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Base position [x, y, z] for tower",
                        "default": [0, 0, 0.42]
                    }
                },
                []
            ),
            build_schema(
                "disassemble",
                "Take apart connected bricks",
                {
                    "brick_name": {"type": "string", "description": "Name of brick to remove from structure"}
                },
                ["brick_name"]
            )
        ]

    @BaseEnvironment.register_tool("stack_brick")
    def _tool_stack_brick(self, bottom_brick: str, top_brick: str, offset: List[float] = None) -> Dict[str, Any]:
        """Stack one LEGO brick on top of another."""
        if offset is None:
            offset = [0, 0]
        
        bottom_obj = self.get_object_by_name(bottom_brick)
        top_obj = self.get_object_by_name(top_brick)
        
        if not bottom_obj:
            return {"status": "error", "message": f"Bottom brick '{bottom_brick}' not found"}
        if not top_obj:
            return {"status": "error", "message": f"Top brick '{top_brick}' not found"}
        
        # Get bottom brick position
        bottom_pos, _ = p.getBasePositionAndOrientation(bottom_obj.object_id)
        
        # Calculate top position (stack on top with offset)
        brick_height = 0.02  # Standard LEGO brick height
        new_pos = [
            bottom_pos[0] + offset[0],
            bottom_pos[1] + offset[1],
            bottom_pos[2] + brick_height + 0.01  # Small gap for physics
        ]
        
        # Move top brick to new position
        p.resetBasePositionAndOrientation(
            top_obj.object_id,
            new_pos,
            top_obj.orientation
        )
        
        # Update object info
        top_obj.position = tuple(new_pos)
        
        return {"status": "success", "message": f"Stacked '{top_brick}' on '{bottom_brick}'"}

    @BaseEnvironment.register_tool("connect_bricks")
    def _tool_connect_bricks(self, brick1: str, brick2: str, direction: str) -> Dict[str, Any]:
        """Connect two LEGO bricks side by side."""
        obj1 = self.get_object_by_name(brick1)
        obj2 = self.get_object_by_name(brick2)
        
        if not obj1:
            return {"status": "error", "message": f"Brick '{brick1}' not found"}
        if not obj2:
            return {"status": "error", "message": f"Brick '{brick2}' not found"}
        
        # Get first brick position
        pos1, _ = p.getBasePositionAndOrientation(obj1.object_id)
        
        # Calculate second brick position based on direction
        brick_size = 0.04  # Approximate brick width
        
        direction_offsets = {
            "right": [brick_size, 0, 0],
            "left": [-brick_size, 0, 0],
            "front": [0, brick_size, 0],
            "back": [0, -brick_size, 0]
        }
        
        if direction not in direction_offsets:
            return {"status": "error", "message": f"Invalid direction: {direction}"}
        
        offset = direction_offsets[direction]
        new_pos = [pos1[0] + offset[0], pos1[1] + offset[1], pos1[2] + offset[2]]
        
        # Move second brick to new position
        p.resetBasePositionAndOrientation(
            obj2.object_id,
            new_pos,
            obj2.orientation
        )
        
        # Update object info
        obj2.position = tuple(new_pos)
        
        return {"status": "success", "message": f"Connected '{brick2}' to '{brick1}' on {direction} side"}

    @BaseEnvironment.register_tool("build_tower")
    def _tool_build_tower(self, base_position: List[float] = None) -> Dict[str, Any]:
        """Automatically build a tower with available bricks."""
        if base_position is None:
            base_position = [0, 0, 0.42]
        
        if not self.lego_bricks:
            return {"status": "error", "message": "No LEGO bricks available"}
        
        # Sort bricks by size (larger first for better stability)
        sorted_bricks = sorted(self.lego_bricks, key=lambda b: b.properties.get("brick_type", ""), reverse=True)
        
        brick_height = 0.02
        current_z = base_position[2]
        
        for i, brick in enumerate(sorted_bricks):
            new_pos = [base_position[0], base_position[1], current_z + i * (brick_height + 0.001)]
            
            p.resetBasePositionAndOrientation(
                brick.object_id,
                new_pos,
                brick.orientation
            )
            
            brick.position = tuple(new_pos)
        
        return {
            "status": "success", 
            "message": f"Built tower with {len(sorted_bricks)} bricks at {base_position}"
        }

    @BaseEnvironment.register_tool("disassemble")
    def _tool_disassemble(self, brick_name: str) -> Dict[str, Any]:
        """Take apart a brick from the structure."""
        brick_obj = self.get_object_by_name(brick_name)
        
        if not brick_obj:
            return {"status": "error", "message": f"Brick '{brick_name}' not found"}
        
        # Move brick to a random position away from structure
        table_pos = self.config.table_position
        new_pos = [
            table_pos[0] + np.random.uniform(-0.4, 0.4),
            table_pos[1] + np.random.uniform(-0.4, 0.4),
            table_pos[2] + 0.1
        ]
        
        p.resetBasePositionAndOrientation(
            brick_obj.object_id,
            new_pos,
            brick_obj.orientation
        )
        
        brick_obj.position = tuple(new_pos)
        
        return {"status": "success", "message": f"Disassembled and moved '{brick_name}'"}
