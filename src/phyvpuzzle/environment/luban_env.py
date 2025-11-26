"""
Luban Lock environment implementation.
"""
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from phyvpuzzle.core import (BaseEnvironment, EnvironmentConfig,
                             register_environment, register_environment_config)
from phyvpuzzle.core.base import Action, BaseEnvironment, ObjectInfo, State
from phyvpuzzle.environment.base_env import PhysicsEnvironment, p


@register_environment_config("luban")
@dataclass
class LubanConfig(EnvironmentConfig):
    """Configuration for Luban Lock setup."""
    render_width: int = 512
    render_height: int = 512
    # Quantized steps configuration
    move_unit_step: float = 0.01  # 1 step = 1 cm
    rotate_unit_step: float = 5.0 # 1 step = 5 degrees

@register_environment("luban")
class LubanEnvironment(PhysicsEnvironment):
    """
    Physics environment for Luban Lock.
    Implements 'RuntimeAssembly' logic: Objects are fixed via constraints to an anchor.
    Movement involves detaching, moving kinematically, and re-attaching.
    """
    
    def __init__(self, config: LubanConfig):
        super().__init__(config)
        self.luban_config = config

    def _get_current_state(self, metadata: Optional[Dict[str, Any]] = None) -> State:
        """Get current puzzle state."""
        # Sync Python objects with PyBullet physics state
        if self.objects:
            for obj in self.objects:
                try:
                    pos, orn = p.getBasePositionAndOrientation(obj.object_id)
                    obj.position = pos
                    obj.orientation = orn
                except Exception:
                    pass

        return State(
            step=self.step_count,
            objects=self.objects,
            time_stamp=time.time(),
            metadata={**(metadata or {})}
        )
    
    def _get_state_description(self) -> str:
        """
        Get textual description. Matches user's 'get_object_mapping' logic.
        """
        lines = ["🧩 OBJECT MAPPING (Complete object information - updated this step):"]
        lines.append("=" * 80)
        
        # 1. Last Action Metadata
        metadata = self.current_state.metadata
        if metadata.get("tool_call") and metadata.get("tool_result"):
            tool_call = metadata['tool_call']
            tool_result = metadata['tool_result']
            action_str = f"{tool_call.get('action_type')}({tool_call.get('parameters')})"
            result_str = tool_result.get('message', '')
            lines.append(f"🔄 LAST ACTION: {action_str}")
            lines.append(f"   Result: {result_str}")
            lines.append("-" * 40)

        # 2. Object List
        non_container_count = 0
        for obj_info in self.objects:
            if obj_info.properties.get('is_container', False):
                continue
            
            non_container_count += 1
            obj_id = obj_info.object_id
            pos = obj_info.position
            
            # Extract Color directly from PyBullet
            color_str = "unknown"
            try:
                # visual_shapes: list of (id, link, type, dim, file, scale, rgbaColor)
                vis_data = p.getVisualShapeData(obj_id)
                if vis_data:
                    rgba = vis_data[0][7]
                    r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
                    color_str = f"RGB=({r}, {g}, {b})"
            except Exception: pass
            
            # Extract Size (AABB)
            size_str = "unknown"
            try:
                min_, max_ = p.getAABB(obj_id)
                dims = [max_[i] - min_[i] for i in range(3)]
                size_str = f"({dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f})"
            except Exception: pass
            
            lines.append(f"🧩 Object #{non_container_count} (object_id: {obj_id}):")
            lines.append(f"   - color: {color_str}")
            lines.append(f"   - position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            lines.append(f"   - size: {size_str}") 
            lines.append("")
        
        lines.append("=" * 80)
        lines.append(f"Total movable objects: {non_container_count}")
        lines.append("\n💡 INSTRUCTIONS:")
        lines.append(f"   - Movement is quantized: 1 unit = {self.luban_config.move_unit_step} meters.")
        lines.append(f"   - Rotation is quantized: 1 unit = {self.luban_config.rotate_unit_step} degrees.")
        lines.append("   - Use positive/negative integers to control direction.")
        
        return "\n".join(lines)

    def _get_task_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """Define the manipulation tool."""
        unit_m = self.luban_config.move_unit_step
        unit_deg = self.luban_config.rotate_unit_step
        return [
            {
                "type": "function",
                "function": {
                    "name": "manipulate_piece",
                    "description": f"Move/Rotate a piece using quantized steps. \n1 Move Step = {unit_m}m (1cm). \n1 Rot Step = {unit_deg} deg.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_id": {"type": "integer", "description": "Target Object ID."},
                            "move_x_steps": {"type": "integer", "default": 0},
                            "move_y_steps": {"type": "integer", "default": 0},
                            "move_z_steps": {"type": "integer", "default": 0},
                            "rotate_axis": {"type": "string", "enum": ["x", "y", "z", "none"], "default": "none"},
                            "rotate_steps": {"type": "integer", "default": 0}
                        },
                        "required": ["object_id"]
                    }
                }
            }
        ]

    @BaseEnvironment.register_tool("manipulate_piece")
    def _tool_manipulate_piece(self, object_id: int, 
                               move_x_steps: int = 0, move_y_steps: int = 0, move_z_steps: int = 0,
                               rotate_axis: str = "none", rotate_steps: int = 0) -> Dict[str, Any]:
        """
        Tool: RuntimeAssembly style manipulation.
        Detach Constraint -> Interpolate Move (with physics step) -> Reattach Constraint.
        """
        obj_info = self.get_object_by_id(object_id)
        if not obj_info:
            return {"status": "error", "message": f"Object {object_id} not found."}

        # 1. Config
        move_unit = self.luban_config.move_unit_step
        rot_unit = math.radians(self.luban_config.rotate_unit_step)
        
        # 2. Retrieve RuntimeAssembly Constraint Info
        anchor_id = obj_info.properties.get("anchor_id")
        constraint_id = obj_info.properties.get("constraint_id")
        
        # Safety check: If objects created without anchor (fallback), handle gracefully
        has_constraint = (anchor_id is not None and constraint_id is not None)

        # 3. DETACH (Remove fixed constraint)
        if has_constraint:
            p.removeConstraint(constraint_id)
        
        # 4. MOVEMENT LOOP (Interpolation)
        # We determine loop count by max steps to ensure smoothness
        total_loops = max(abs(move_x_steps), abs(move_y_steps), abs(move_z_steps), abs(rotate_steps))
        if total_loops == 0:
            # Re-attach immediately if no movement
            if has_constraint:
                self._attach_to_anchor(obj_info, anchor_id)
            return {"status": "success", "message": "No movement requested."}

        dx = (move_x_steps * move_unit) / total_loops
        dy = (move_y_steps * move_unit) / total_loops
        dz = (move_z_steps * move_unit) / total_loops
        
        d_rot = 0.0
        if rotate_axis in ["x", "y", "z"]:
            d_rot = (rotate_steps * rot_unit) / total_loops

        current_pos, current_orn = p.getBasePositionAndOrientation(object_id)
        current_pos = list(current_pos)
        
        # Temporarily set dynamics for movement (stable but movable)
        p.changeDynamics(object_id, -1, mass=1.0, linearDamping=0.0, angularDamping=0.0)

        for _ in range(total_loops):
            # Update Kinematics
            current_pos[0] += dx
            current_pos[1] += dy
            current_pos[2] += dz
            
            if d_rot != 0:
                delta_quat = p.getQuaternionFromEuler([
                    d_rot if rotate_axis=="x" else 0,
                    d_rot if rotate_axis=="y" else 0,
                    d_rot if rotate_axis=="z" else 0
                ])
                _, current_orn = p.multiplyTransforms([0,0,0], delta_quat, [0,0,0], current_orn)

            # Apply
            p.resetBasePositionAndOrientation(object_id, current_pos, current_orn)
            p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
            
            # Step physics to update AABBs/Contacts, preventing clipping artifacts
            p.stepSimulation()

        # 5. RE-ATTACH (Create new fixed constraint)
        # Update internal info
        obj_info.position = tuple(current_pos)
        obj_info.orientation = tuple(current_orn)
        
        if has_constraint:
            # Restore high damping for static stability
            p.changeDynamics(object_id, -1, mass=1.0, linearDamping=1.0, angularDamping=1.0)
            self._attach_to_anchor(obj_info, anchor_id)
        else:
            # Fallback for primitives: just freeze mass
            p.changeDynamics(object_id, -1, mass=0.0)

        p.stepSimulation() # Settle

        msg = (f"Moved ID {object_id}: X={move_x_steps}, Y={move_y_steps}, Z={move_z_steps} steps. "
               f"Rotated {rotate_steps} steps ({rotate_axis}).")
        return {"status": "success", "message": msg}

    def _attach_to_anchor(self, obj_info: ObjectInfo, anchor_id: int):
        """Helper to create fixed constraint."""
        uid = obj_info.object_id
        pos, orn = p.getBasePositionAndOrientation(uid)
        
        cid = p.createConstraint(
            parentBodyUniqueId=anchor_id,
            parentLinkIndex=-1,
            childBodyUniqueId=uid,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=pos,
            parentFrameOrientation=orn,
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0, 1]
        )
        # Update the constraint ID in the object info so we can remove it next time
        obj_info.properties["constraint_id"] = cid