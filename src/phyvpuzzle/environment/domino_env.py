"""
Domino environment implementation for physics puzzle tasks.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Conditional pybullet import - will use mock from base_environment if not available
from phyvpuzzle.environment.base_env import PhysicsEnvironment, p
from phyvpuzzle.core import register_environment_config, register_environment, EnvironmentConfig, BaseEnvironment
from phyvpuzzle.core.base import State, ObjectInfo

@register_environment_config("domino")
@dataclass
class DominoConfig(EnvironmentConfig):
    """Configuration for domino setup."""
    initial_push_force: float = 5.0
    table_position: Tuple[float, float, float] = (0, 0, 0.4)


@register_environment("domino")
class DominoEnvironment(PhysicsEnvironment):
    """Physics environment for domino falling puzzles."""
    
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
                "push_specific_domino",
                "Push a specific domino by name",
                {
                    "domino_id": {"type": "string", "description": "Name of domino to push"},
                    "force": {"type": "number", "description": "Push force", "default": 5.0},
                    "direction": {
                        "type": "array",
                        "items": {"type": "number"}, 
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Push direction [x, y, z]",
                        "default": [1, 0, 0]
                    }
                },
                ["domino_id"]
            ),
            build_schema(
                "reset_dominoes",
                "Reset all dominoes to their initial upright positions",
                {},
                []
            )
        ]


    @BaseEnvironment.register_tool("push_specific_domino")
    def _tool_push_specific_domino(self, domino_id: str, force: float = 5.0, direction: List[float] = None) -> Dict[str, Any]:
        """Push a specific domino by its name (e.g., 'domino_1')."""
        if direction is None:
            direction = [1, 0, 0]

        target_domino = next((d for d in self.dominoes if d.name == domino_id), None)
        
        if not target_domino:
            return {"status": "error", "message": f"Domino '{domino_id}' not found"}

        direction_np = np.array(direction)
        if np.linalg.norm(direction_np) > 0:
            direction_np = direction_np / np.linalg.norm(direction_np)
        else:
            direction_np = np.array([1, 0, 0])
            
        force_vector = direction_np * force
        p.applyExternalForce(
            target_domino.object_id, -1,
            force_vector.tolist(),
            target_domino.position,
            p.WORLD_FRAME
        )
        
        return {"status": "success", "message": f"Pushed domino '{domino_id}' with force {force}"}

    @BaseEnvironment.register_tool("reset_dominoes")
    def _tool_reset_dominoes(self) -> Dict[str, Any]:
        """Reset all dominoes to their initial positions and states."""
        self._arrange_dominoes()
        self.fell_dominoes.clear()
        self.push_applied = False
        return {"status": "success", "message": "All dominoes have been reset"}
          