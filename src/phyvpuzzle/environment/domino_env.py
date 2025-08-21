"""
Domino environment implementation for physics puzzle tasks.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Conditional pybullet import - will use mock from base_environment if not available
from .base_env import PhysicsEnvironment, p
from ..core.base import Action, State, Observation, TaskDifficulty


@dataclass
class DominoConfig:
    """Configuration for domino setup."""
    num_dominoes: int = 5
    domino_spacing: float = 0.08  # Distance between dominoes
    domino_height: float = 0.05
    initial_push_force: float = 5.0
    arrangement_pattern: str = "line"  # line, curve, zigzag, circle
    table_position: Tuple[float, float, float] = (0, 0, 0.4)
    
    @classmethod
    def from_difficulty(cls, difficulty: TaskDifficulty) -> "DominoConfig":
        """Create config based on difficulty level."""
        if difficulty == TaskDifficulty.VERY_EASY:
            return cls(num_dominoes=3, arrangement_pattern="line")
        elif difficulty == TaskDifficulty.EASY:
            return cls(num_dominoes=5, arrangement_pattern="line")
        elif difficulty == TaskDifficulty.MEDIUM:
            return cls(num_dominoes=10, arrangement_pattern="curve")
        elif difficulty == TaskDifficulty.HARD:
            return cls(num_dominoes=15, arrangement_pattern="zigzag")
        elif difficulty == TaskDifficulty.VERY_HARD:
            return cls(num_dominoes=21, arrangement_pattern="circle")
        else:
            return cls()


class DominoEnvironment(PhysicsEnvironment):
    """Physics environment for domino falling puzzles."""
    
    def __init__(self, config: Dict[str, Any], domino_config: Optional[DominoConfig] = None):
        # Set up domino-specific config
        self.domino_config = domino_config or DominoConfig()
        self.dominoes = {}
        self.initial_positions = {}
        self.fell_dominoes = set()
        self.push_applied = False
        
        super().__init__(config)
        
    def _setup_task_environment(self) -> None:
        """Setup domino-specific environment."""
        self._load_domino_models()
        self._arrange_dominoes()
        
    def _load_domino_models(self) -> None:
        """Load domino URDF models."""
        domino_base_path = os.path.join(self.urdf_base_path, "domino")
        
        if not os.path.exists(domino_base_path):
            print(f"Warning: Domino models not found at {domino_base_path}")
            print("Creating simple domino shapes instead")
            self._create_simple_dominoes()
            return
            
        # Load available domino URDF files
        available_dominoes = []
        for i in range(1, 22):  # Domino_1 to Domino_21
            domino_name = f"Domino_{i}"
            domino_path = os.path.join(domino_base_path, domino_name, "urdf", f"{domino_name}.urdf")
            
            if os.path.exists(domino_path):
                available_dominoes.append(domino_path)
                
        if not available_dominoes:
            print("No domino URDF files found, creating simple dominoes")
            self._create_simple_dominoes()
            return
            
        # Select dominoes to use
        num_to_load = min(self.domino_config.num_dominoes, len(available_dominoes))
        selected_dominoes = available_dominoes[:num_to_load]
        
        print(f"Loading {num_to_load} dominoes from URDF files")
        
        for i, domino_path in enumerate(selected_dominoes):
            domino_id = f"domino_{i+1}"
            
            try:
                # Load domino at temporary position (will be moved later)
                obj_id = p.loadURDF(domino_path, basePosition=[0, 0, 0.5])
                
                self.objects[domino_id] = type('obj', (), {
                    'object_id': obj_id,
                    'name': domino_id,
                    'position': [0, 0, 0.5],
                    'orientation': [0, 0, 0, 1],
                    'object_type': 'domino',
                    'properties': {'index': i}
                })()
                
                self.dominoes[domino_id] = obj_id
                
                print(f"Loaded {domino_id} from {domino_path}")
                
            except Exception as e:
                print(f"Error loading domino {domino_path}: {e}")
                continue
                
        # If we didn't load enough dominoes, create simple ones for the rest
        if len(self.dominoes) < self.domino_config.num_dominoes:
            remaining = self.domino_config.num_dominoes - len(self.dominoes)
            print(f"Creating {remaining} simple dominoes to reach target count")
            self._create_simple_dominoes(start_index=len(self.dominoes))
            
    def _create_simple_dominoes(self, start_index: int = 0) -> None:
        """Create simple domino shapes using primitive objects."""
        domino_width = 0.02
        domino_length = 0.04
        domino_height = self.domino_config.domino_height
        
        num_to_create = self.domino_config.num_dominoes - start_index
        
        for i in range(num_to_create):
            domino_id = f"domino_{start_index + i + 1}"
            
            # Create collision shape
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[domino_width/2, domino_length/2, domino_height/2]
            )
            
            # Create visual shape
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[domino_width/2, domino_length/2, domino_height/2],
                rgbaColor=[0.8, 0.4, 0.2, 1.0]  # Brown color
            )
            
            # Create domino body
            obj_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0, 0, 0.5]  # Temporary position
            )
            
            self.objects[domino_id] = type('obj', (), {
                'object_id': obj_id,
                'name': domino_id,
                'position': [0, 0, 0.5],
                'orientation': [0, 0, 0, 1],
                'object_type': 'domino',
                'properties': {'index': start_index + i}
            })()
            
            self.dominoes[domino_id] = obj_id
            
    def _arrange_dominoes(self) -> None:
        """Arrange dominoes according to the specified pattern."""
        positions = self._calculate_domino_positions()
        
        for i, (domino_id, obj_id) in enumerate(self.dominoes.items()):
            if i < len(positions):
                pos, orient = positions[i]
                
                # Set domino position and orientation
                p.resetBasePositionAndOrientation(obj_id, pos, orient)
                
                # Update object info
                self.objects[domino_id].position = pos
                self.objects[domino_id].orientation = orient
                
                # Store initial position for reset
                self.initial_positions[domino_id] = (pos, orient)
                
        print(f"Arranged {len(positions)} dominoes in {self.domino_config.arrangement_pattern} pattern")
        
    def _calculate_domino_positions(self) -> List[Tuple[List[float], List[float]]]:
        """Calculate positions and orientations for dominoes based on arrangement pattern."""
        positions = []
        spacing = self.domino_config.domino_spacing
        table_height = self.domino_config.table_position[2] + 0.02  # Slightly above table
        
        if self.domino_config.arrangement_pattern == "line":
            # Simple line arrangement
            for i in range(len(self.dominoes)):
                x = i * spacing - (len(self.dominoes) - 1) * spacing / 2
                pos = [x, 0, table_height + self.domino_config.domino_height/2]
                orient = [0, 0, 0, 1]  # No rotation
                positions.append((pos, orient))
                
        elif self.domino_config.arrangement_pattern == "curve":
            # Curved arrangement (arc)
            import math
            radius = len(self.dominoes) * spacing / (2 * math.pi) * 2
            
            for i in range(len(self.dominoes)):
                angle = i * math.pi / len(self.dominoes)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos = [x, y, table_height + self.domino_config.domino_height/2]
                
                # Orient domino to face the center
                yaw = angle + math.pi/2
                orient = p.getQuaternionFromEuler([0, 0, yaw])
                positions.append((pos, orient))
                
        elif self.domino_config.arrangement_pattern == "zigzag":
            # Zigzag pattern
            import math
            
            for i in range(len(self.dominoes)):
                x = i * spacing - (len(self.dominoes) - 1) * spacing / 2
                y = 0.1 * math.sin(i * math.pi / 3)  # Zigzag with amplitude 0.1
                pos = [x, y, table_height + self.domino_config.domino_height/2]
                
                # Slight rotation based on zigzag direction
                yaw = math.sin(i * math.pi / 3) * 0.2
                orient = p.getQuaternionFromEuler([0, 0, yaw])
                positions.append((pos, orient))
                
        elif self.domino_config.arrangement_pattern == "circle":
            # Circular arrangement
            import math
            radius = len(self.dominoes) * spacing / (2 * math.pi)
            
            for i in range(len(self.dominoes)):
                angle = i * 2 * math.pi / len(self.dominoes)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos = [x, y, table_height + self.domino_config.domino_height/2]
                
                # Orient domino tangent to circle
                yaw = angle + math.pi/2
                orient = p.getQuaternionFromEuler([0, 0, yaw])
                positions.append((pos, orient))
                
        return positions
        
    def _execute_task_specific_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle domino-specific tool calls."""
        if tool_name == "push_domino":
            success, message = self._push_first_domino(arguments)
            return {"status": "success" if success else "error", "message": message}
        elif tool_name == "push_specific_domino":
            success, message = self._push_specific_domino(arguments)
            return {"status": "success" if success else "error", "message": message}
        elif tool_name == "reset_dominoes":
            success, message = self._reset_dominoes()
            return {"status": "success" if success else "error", "message": message}
        else:
            return super()._execute_task_specific_tool(tool_name, arguments)
            
    def _push_first_domino(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Push the first domino to start the chain reaction."""
        if not self.dominoes:
            return False, "No dominoes available to push"
            
        if self.push_applied:
            return False, "First domino has already been pushed"
            
        # Get first domino
        first_domino_id = f"domino_1"
        if first_domino_id not in self.dominoes:
            first_domino_id = list(self.dominoes.keys())[0]
            
        domino_id = self.dominoes[first_domino_id]
        
        # Get push parameters
        force = params.get("force", self.domino_config.initial_push_force)
        direction = params.get("direction", [1, 0, 0])  # Default: push in +X direction
        
        # Normalize direction
        direction = np.array(direction)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1, 0, 0])
            
        # Apply force to first domino
        pos = self.objects[first_domino_id].position
        force_vector = [force * direction[0], force * direction[1], force * direction[2]]
        
        p.applyExternalForce(
            domino_id, -1,  # Apply to base link
            force_vector,
            pos,
            p.WORLD_FRAME
        )
        
        self.push_applied = True
        
        return True, f"Pushed first domino with force {force}"
        
    def _push_specific_domino(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Push a specific domino."""
        domino_name = params.get("object_id") or params.get("domino_id")
        
        if not domino_name or domino_name not in self.dominoes:
            return False, f"Domino {domino_name} not found"
            
        domino_id = self.dominoes[domino_name]
        
        # Get push parameters
        force = params.get("force", 5.0)
        direction = params.get("direction", [1, 0, 0])
        
        # Normalize direction
        direction = np.array(direction)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1, 0, 0])
            
        # Apply force
        pos = self.objects[domino_name].position
        force_vector = [force * direction[0], force * direction[1], force * direction[2]]
        
        p.applyExternalForce(
            domino_id, -1,
            force_vector,
            pos,
            p.WORLD_FRAME
        )
        
        return True, f"Pushed {domino_name} with force {force}"
        
    def _reset_dominoes(self) -> Tuple[bool, str]:
        """Reset all dominoes to initial positions."""
        for domino_name, (pos, orient) in self.initial_positions.items():
            if domino_name in self.dominoes:
                domino_id = self.dominoes[domino_name]
                p.resetBasePositionAndOrientation(domino_id, pos, orient)
                
        self.fell_dominoes.clear()
        self.push_applied = False
        
        return True, "All dominoes reset to initial positions"
        
    def _observe_dominoes(self, angle: float = 0) -> Tuple[bool, str]:
        """Observe dominoes from a specific angle."""
        import math
        
        # Calculate new camera position
        radius = 1.5
        height = 0.8
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        self.multi_view_renderer.set_camera_config(
            "perspective", [x, y, height], [0, 0, 0.4]
        )
        
        return True, f"Observing dominoes from angle {angle:.2f} radians"
        
    def _check_domino_solution(self) -> Tuple[bool, str]:
        """Check if dominoes have fallen correctly."""
        fallen_count = self._count_fallen_dominoes()
        total_count = len(self.dominoes)
        
        success = fallen_count >= total_count * 0.8  # 80% must fall for success
        
        if success:
            return True, f"Success! {fallen_count}/{total_count} dominoes fell"
        else:
            return False, f"Only {fallen_count}/{total_count} dominoes fell"
            
    def _count_fallen_dominoes(self) -> int:
        """Count how many dominoes have fallen."""
        fallen_count = 0
        
        for domino_name, domino_id in self.dominoes.items():
            pos, orient = p.getBasePositionAndOrientation(domino_id)
            
            # Convert quaternion to euler angles
            euler = p.getEulerFromQuaternion(orient)
            
            # Check if domino is tilted significantly (fallen)
            tilt_threshold = 0.5  # radians (about 28 degrees)
            
            if abs(euler[0]) > tilt_threshold or abs(euler[1]) > tilt_threshold:
                fallen_count += 1
                self.fell_dominoes.add(domino_name)
                
        return fallen_count
        
    def _get_current_state(self) -> State:
        """Get current domino state."""
        fallen_count = self._count_fallen_dominoes()
        total_count = len(self.dominoes)
        
        # Collect object states
        objects = {}
        for domino_name, domino_id in self.dominoes.items():
            pos, orient = p.getBasePositionAndOrientation(domino_id)
            euler = p.getEulerFromQuaternion(orient)
            
            objects[domino_name] = {
                "position": pos,
                "orientation": orient,
                "euler": euler,
                "fallen": domino_name in self.fell_dominoes
            }
            
        # Determine success
        success_ratio = fallen_count / total_count if total_count > 0 else 0
        success = success_ratio >= 0.8
        completed = self.push_applied and (success or fallen_count == 0)
        
        return State(
            step=self.step_count,
            objects=objects,
            completed=completed,
            success=success,
            metadata={
                "fallen_dominoes": fallen_count,
                "total_dominoes": total_count,
                "success_ratio": success_ratio,
                "push_applied": self.push_applied
            }
        )
        
    def _get_state_description(self) -> str:
        """Get textual description of domino state."""
        fallen_count = self._count_fallen_dominoes()
        total_count = len(self.dominoes)
        
        desc = f"Domino Environment - Step {self.step_count}: "
        desc += f"{fallen_count}/{total_count} dominoes have fallen. "
        
        if not self.push_applied:
            desc += "No push has been applied yet. "
        
        if fallen_count == total_count:
            desc += "All dominoes have fallen - Success!"
        elif fallen_count > 0:
            desc += "Chain reaction in progress..."
        else:
            desc += "Dominoes are still standing."
            
        return desc
        
    def _evaluate_success(self) -> bool:
        """Evaluate if domino puzzle is solved."""
        fallen_count = self._count_fallen_dominoes()
        total_count = len(self.dominoes)
        
        return (fallen_count / total_count) >= 0.8 if total_count > 0 else False
        
    def get_available_actions(self) -> List[str]:
        """Get domino-specific available actions."""
        actions = super().get_available_actions()
        
        domino_actions = [
            "push_domino", "push_specific_domino", "reset_dominoes"
        ]
        
        return actions + domino_actions
        
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
                "Push the first domino to start chain reaction",
                {
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
                []
            ),
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
