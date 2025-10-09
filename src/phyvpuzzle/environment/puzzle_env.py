"""
Puzzle environment implementation for physics puzzle tasks.

This module provides a complete jigsaw puzzle environment with
physics simulation capabilities for piece placement and assembly tasks.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Conditional pybullet import - will use mock from base_environment if not available
from phyvpuzzle.environment.base_env import PhysicsEnvironment, p
from phyvpuzzle.core import register_environment_config, register_environment, EnvironmentConfig, BaseEnvironment
from phyvpuzzle.core.base import State, ObjectInfo


@register_environment_config("puzzle")
@dataclass
class PuzzleConfig(EnvironmentConfig):
    """Configuration for puzzle setup."""
    pass


@register_environment("puzzle")
class PuzzleEnvironment(PhysicsEnvironment):
    """Physics environment for jigsaw puzzle assembly."""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
    
    def _get_current_state(self, metadata: Optional[Dict[str, Any]] = None) -> State:
        """Get current puzzle state."""
        
        return State(
            step=self.step_count,
            objects=self.objects,
            time_stamp=time.time(),
            metadata={
                **(metadata or {}),
            }
        )
    
    def _get_state_description(self) -> str:
        """Get textual description of puzzle state with detailed object information."""
        
        desc = ""
        
        # 1. Tool call result (if any)
        metadata = self.current_state.metadata
        if metadata.get("tool_call") and metadata.get("tool_result"):
            tool_call = metadata['tool_call']
            action_type = tool_call.get('action_type', 'unknown action type')
            parameters = tool_call.get('parameters', 'unknown parameters')
            desc += f"Tool Call - Action Type: {action_type}, Parameters: {parameters}\n"
            tool_result = metadata['tool_result']
            status = tool_result.get('status', 'unknown status')
            message = tool_result.get('message', 'Unknown error')
            desc += f"Tool Call Result - Status: {status}, Message: {message}.\n"
        
        # 2. Detailed object information for the model
        desc += "\n=== OBJECTS IN SCENE ===\n"
        
        movable_objs = []
        container_obj = None
        
        for obj_info in self.objects:
            if obj_info.properties.get('is_container', False):
                container_obj = obj_info
            else:
                movable_objs.append(obj_info)
        
        # Container info first (if exists)
        if container_obj:
            desc += f"\nCONTAINER (object_id={container_obj.object_id}):\n"
            desc += f"  Name: {container_obj.name}\n"
            try:
                pos, _ = p.getBasePositionAndOrientation(container_obj.object_id)
                desc += f"  Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}\n"
                
                aabb_min, aabb_max = p.getAABB(container_obj.object_id)
                width = aabb_max[0] - aabb_min[0]
                depth = aabb_max[1] - aabb_min[1]
                height = aabb_max[2] - aabb_min[2]
                desc += f"  Size: width={width:.4f}m, depth={depth:.4f}m, height={height:.4f}m\n"
            except Exception as e:
                desc += f"  (Position/Size info unavailable: {e})\n"
        
        # Movable objects info
        desc += f"\nMOVABLE OBJECTS ({len(movable_objs)} pieces):\n"
        
        for idx, obj_info in enumerate(movable_objs, 1):
            obj_id = obj_info.object_id
            desc += f"\n[{idx}] Object ID: {obj_id}\n"
            desc += f"    Name: {obj_info.name}\n"
            
            # Position
            try:
                pos, orn = p.getBasePositionAndOrientation(obj_id)
                desc += f"    Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}\n"
            except Exception as e:
                desc += f"    Position: (unavailable: {e})\n"
            
            # Bounding Box (Size)
            try:
                aabb_min, aabb_max = p.getAABB(obj_id)
                width = aabb_max[0] - aabb_min[0]
                depth = aabb_max[1] - aabb_min[1]
                height = aabb_max[2] - aabb_min[2]
                desc += f"    Size: width={width:.4f}m, depth={depth:.4f}m, height={height:.4f}m\n"
            except Exception as e:
                desc += f"    Size: (unavailable: {e})\n"
            
            # Color
            try:
                visual_shapes = p.getVisualShapeData(obj_id)
                if visual_shapes:
                    rgba = visual_shapes[0][7]
                    r = int(rgba[0] * 255)
                    g = int(rgba[1] * 255)
                    b = int(rgba[2] * 255)
                    desc += f"    Color: RGB({r}, {g}, {b}) #{r:02x}{g:02x}{b:02x}\n"
                else:
                    desc += f"    Color: (unknown)\n"
            except Exception:
                desc += f"    Color: (unavailable)\n"
            
            # Status flags
            is_placed = obj_info.properties.get('is_placed', False)
            in_container = obj_info.properties.get('in_container', False)
            if is_placed or in_container:
                status_flags = []
                if is_placed:
                    status_flags.append("placed")
                if in_container:
                    status_flags.append("in_container")
                desc += f"    Status: {', '.join(status_flags)}\n"
        
        desc += "\n=== END OBJECTS ===\n"
            
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
                "place_into_container",
                "Place a puzzle piece into the container at a specified offset position relative to the container's center. This is a convenient alternative to move_object when placing pieces into the container. The piece will be positioned relative to the container's coordinate system. CONSTRAINTS: Offsets must be small to keep pieces inside the container. Use offset_z to stack pieces vertically.",
                {
                    "object_id": {"type": "integer", "description": "Unique object_id (integer) of the puzzle piece to place. Get this from the observation state. Do NOT use the container's object_id."},
                    "offset_x": {"type": "number", "description": "X offset from container center in meters. Positive = right, negative = left. RANGE: -0.1 to 0.1 (must keep piece inside container).", "default": 0.0},
                    "offset_y": {"type": "number", "description": "Y offset from container center in meters. Positive = forward, negative = backward. RANGE: -0.1 to 0.1 (must keep piece inside container).", "default": 0.0},
                    "offset_z": {"type": "number", "description": "Z offset from container bottom in meters. Use this for vertical stacking. RANGE: 0.0 to 0.15. Start from 0.0 for bottom layer, increase for higher layers.", "default": 0.0},
                },
                ["object_id"]
            )
        ]

    @BaseEnvironment.register_tool("place_into_container")
    def _tool_place_into_container(self, object_id: int, offset_x: float = 0.0, offset_y: float = 0.0, offset_z: float = 0.0) -> Dict[str, Any]:
        """Place a puzzle piece into the container at specified offset."""
        piece_obj = self.get_object_by_id(object_id)
        
        if not piece_obj:
            return {"status": "error", "message": f"Puzzle piece with object_id {object_id} not found"}
        
        # Find container
        container = None
        for obj in self.objects:
            if obj.properties.get('is_container', False):
                container = obj
                break
        
        if not container:
            return {"status": "error", "message": "No container found in the environment"}
        
        # Calculate position inside container with offsets
        container_pos = container.position
        new_pos = (
            container_pos[0] + offset_x,
            container_pos[1] + offset_y,
            container_pos[2] + offset_z
        )
        
        # Clamp offsets to safe range
        max_offset = 0.05
        if abs(offset_x) > max_offset or abs(offset_y) > max_offset or offset_z < 0 or offset_z > 0.1:
            return {"status": "error", "message": f"Offsets out of valid range. X,Y: [-{max_offset}, {max_offset}], Z: [0, 0.1]"}
        
        # Mark piece as placed in container
        piece_obj.properties['is_placed'] = True
        piece_obj.properties['in_container'] = True
        
        # Place the piece
        p.resetBasePositionAndOrientation(
            piece_obj.object_id,
            new_pos,
            piece_obj.orientation
        )
        
        piece_obj.position = new_pos
        
        return {"status": "success", "message": f"Placed object_id {object_id} into container at offset ({offset_x:.3f}, {offset_y:.3f}, {offset_z:.3f})"}
