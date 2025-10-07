"""
Domino environment implementation for physics puzzle tasks.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Conditional pybullet import - will use mock from base_environment if not available
from phyvpuzzle.environment.base_env import PhysicsEnvironment, p
from phyvpuzzle.core import register_environment_config, register_environment, EnvironmentConfig, BaseEnvironment
from phyvpuzzle.core.base import State, ObjectInfo

@register_environment_config("domino")
@dataclass
class DominoConfig(EnvironmentConfig):
    """Configuration for domino setup."""
    pass


@register_environment("domino")
class DominoEnvironment(PhysicsEnvironment):
    """Physics environment for domino puzzles."""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)

        self.dominoes: List[ObjectInfo] = []
      
    def _get_current_state(self, metadata: Optional[Dict[str, Any]] = None) -> State:
        """Get current domino state."""
        
        return State(
            step=self.step_count,
            objects=self.objects,
            time_stamp=time.time(),
            metadata={
                "total_dominoes": len(self.dominoes),
                **(metadata or {}),
            }
        )
        
    def _get_state_description(self) -> str:
        """Get textual description of domino state used for prompt"""
        
        # desc = f"Domino Environment - Step {self.step_count}:\n"
        desc = ""
        
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
            
        return desc
    
    def _get_task_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get domino-specific tool schemas."""
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
                "push_domino",
                "Apply a push force to a specific domino to start the chain reaction.",
                {
                    "object_id": {"type": "integer", "description": "Unique object_id (integer) of the domino to push"},
                    "force": {"type": "number", "description": "Push force magnitude (recommended: 3-10)", "default": 5.0},
                    "direction_x": {"type": "number", "description": "X component of push direction", "default": 1.0},
                    "direction_y": {"type": "number", "description": "Y component of push direction", "default": 0.0},
                    "direction_z": {"type": "number", "description": "Z component of push direction (usually 0)", "default": 0.0},
                },
                ["object_id"]
            ),
            build_schema(
                "get_domino_info",
                "Get information about a specific domino (position, orientation, status).",
                {
                    "object_id": {"type": "integer", "description": "Unique object_id (integer) of the domino"}
                },
                ["object_id"]
            )
        ]


    @BaseEnvironment.register_tool("push_domino")
    def _tool_push_domino(self, object_id: int, force: float = 5.0, direction_x: float = 1.0, direction_y: float = 0.0, direction_z: float = 0.0) -> Dict[str, Any]:
        """Push a specific domino with given force and direction."""
        target_domino = self.get_object_by_id(object_id)
        
        if not target_domino:
            return {"status": "error", "message": f"Domino with object_id {object_id} not found"}

        # Normalize direction vector
        direction_np = np.array([direction_x, direction_y, direction_z])
        if np.linalg.norm(direction_np) > 0:
            direction_np = direction_np / np.linalg.norm(direction_np)
        else:
            direction_np = np.array([1, 0, 0])
            
        force_vector = direction_np * force
        
        # Apply force to the domino
        p.applyExternalForce(
            target_domino.object_id, -1,
            force_vector.tolist(),
            target_domino.position,
            p.WORLD_FRAME
        )
        
        return {"status": "success", "message": f"Pushed domino object_id {object_id} with force {force:.1f} in direction ({direction_x:.2f}, {direction_y:.2f}, {direction_z:.2f})"}

    @BaseEnvironment.register_tool("get_domino_info")
    def _tool_get_domino_info(self, object_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific domino."""
        domino = self.get_object_by_id(object_id)
        
        if not domino:
            return {"status": "error", "message": f"Domino with object_id {object_id} not found"}
        
        # Get current position and orientation from PyBullet
        current_pos, current_orn = p.getBasePositionAndOrientation(domino.object_id)
        
        # Get velocity to check if it's moving
        lin_vel, ang_vel = p.getBaseVelocity(domino.object_id)
        
        # Check if domino has fallen (based on orientation)
        euler = p.getEulerFromQuaternion(current_orn)
        is_fallen = abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5  # Fallen if tilted more than ~30 degrees
        
        info = {
            "object_id": object_id,
            "name": domino.name,
            "current_position": list(current_pos),
            "current_orientation": list(current_orn),
            "euler_angles": list(euler),
            "linear_velocity": list(lin_vel),
            "angular_velocity": list(ang_vel),
            "is_fallen": is_fallen,
            "is_moving": np.linalg.norm(lin_vel) > 0.01 or np.linalg.norm(ang_vel) > 0.01,
        }
        
        return {"status": "success", "message": "Domino information retrieved", "info": info}
          