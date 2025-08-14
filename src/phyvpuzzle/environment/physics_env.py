"""
Physics Environment Module

This module provides the 3D physics environment using PyBullet for 
physical visual reasoning tasks.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data
import os
import time
from dataclasses import dataclass
import inspect
from abc import ABC, abstractmethod
from ..core.translator import EnvironmentCommand
from ..core.action_descriptor import ActionDescriptor, ParsedAction
from ..core.vllm_processor import VLLMProcessor, DecisionParser


@dataclass
class CameraConfig:
    """Camera configuration for environment rendering."""
    position: Tuple[float, float, float] = (0.0, -1.0, 1.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 60.0
    aspect_ratio: float = 1.0
    near_plane: float = 0.1
    far_plane: float = 10.0
    image_width: int = 512
    image_height: int = 512


@dataclass
class RobotConfig:
    """Robot configuration."""
    urdf_path: str
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    joint_limits: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.joint_limits is None:
            self.joint_limits = {}


@dataclass
class ObjectInfo:
    """Information about an object in the environment."""
    object_id: int
    name: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    object_type: str
    properties: Dict[str, Any]


@dataclass
class TaskDefinition:
    """Definition of a benchmark task."""
    task_id: str
    name: str
    description: str
    initial_setup: Dict[str, Any]
    goal_conditions: Dict[str, Any]
    max_steps: int = 50
    time_limit: float = 300.0
    difficulty_level: int = 1
    task_type: str = "manipulation"


@dataclass
class ExecutionStep:
    """A single step in task execution."""
    step_id: int
    vlm_input: Dict[str, Any]
    vlm_response: str
    parsed_action: Optional[ParsedAction]
    environment_command: Optional[EnvironmentCommand]
    execution_result: bool
    environment_state: Dict[str, Any]
    feedback: str
    timestamp: float


@dataclass
class BenchmarkResult:
    """Result of benchmark execution."""
    task_id: str
    success: bool
    total_steps: int
    execution_time: float
    efficiency_score: float
    steps_history: List[ExecutionStep]
    error_message: Optional[str] = None
    final_state: Optional[Dict[str, Any]] = None


class PhysicsEnvironment(ABC):
    """Abstract base class for physics environments."""
    
    def __init__(self, gui: bool = False):
        self.gui = gui
        self.client_id = None
        self.objects = {}
        self.robot_id = None
        self.camera_config = CameraConfig()
        self.timestep = 0
        self.max_steps = 1000
        
    @abstractmethod
    def setup_environment(self) -> None:
        """Setup the physics environment."""
        pass
    
    @abstractmethod
    def execute_command(self, command: EnvironmentCommand) -> bool:
        """Execute a command in the environment."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        pass
    
    @abstractmethod
    def render(self) -> Image.Image:
        """Render the environment and return an image."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the environment."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the environment."""
        pass


class PyBulletEnvironment(PhysicsEnvironment):
    """PyBullet-based physics environment."""
    
    def __init__(self, gui: bool = False, gravity: float = -9.81):
        super().__init__(gui)
        self.gravity = gravity
        self.plane_id = None
        self.table_id = None
        self.robot_config = None
        self.held_objects = []
        
    def setup_environment(self) -> None:
        """Setup PyBullet environment."""
        # Connect to PyBullet
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        
        # Set gravity
        p.setGravity(0, 0, self.gravity)
        
        # Add search path for URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load table
        self.table_id = p.loadURDF("table/table.urdf", 
                                  basePosition=[0, 0, 0],
                                  globalScaling=1.0)
        
        # Set timestep
        p.setTimeStep(1./240.)
        
        # Enable real-time simulation
        p.setRealTimeSimulation(0)
        
        print(f"PyBullet environment setup complete. Client ID: {self.client_id}")
    
    def load_robot(self, robot_config: RobotConfig) -> int:
        """Load robot into the environment."""
        self.robot_config = robot_config
        
        if not os.path.exists(robot_config.urdf_path):
            # Use default robot if file doesn't exist
            robot_config.urdf_path = "kuka_iiwa/model.urdf"
        
        self.robot_id = p.loadURDF(
            robot_config.urdf_path,
            basePosition=robot_config.base_position,
            baseOrientation=robot_config.base_orientation
        )
        
        # Get robot info
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Robot loaded with {num_joints} joints")
        
        return self.robot_id
    
    def add_object(self, object_name: str, urdf_path: str, 
                  position: Tuple[float, float, float],
                  orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
                  object_type: str = "block",
                  properties: Dict[str, Any] = None) -> int:
        """Add object to the environment."""
        if properties is None:
            properties = {}
            
        object_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=orientation
        )
        
        self.objects[object_name] = ObjectInfo(
            object_id=object_id,
            name=object_name,
            position=position,
            orientation=orientation,
            object_type=object_type,
            properties=properties
        )
        
        return object_id
    
    def create_primitive_object(self, object_name: str, 
                               shape_type: str = "box",
                               size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
                               position: Tuple[float, float, float] = (0, 0, 0.5),
                               color: Tuple[float, float, float, float] = (1, 0, 0, 1),
                               mass: float = 1.0) -> int:
        """Create primitive object (box, sphere, cylinder)."""
        
        # Create collision shape
        if shape_type == "box":
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX, 
                halfExtents=size
            )
        elif shape_type == "sphere":
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE, 
                radius=size[0]
            )
        elif shape_type == "cylinder":
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, 
                radius=size[0], 
                height=size[1]
            )
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            collision_shape,
            rgbaColor=color
        )
        
        # Create multi-body
        object_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        # Store object info
        self.objects[object_name] = ObjectInfo(
            object_id=object_id,
            name=object_name,
            position=position,
            orientation=(0, 0, 0, 1),
            object_type=shape_type,
            properties={"mass": mass, "color": color, "size": size}
        )
        
        return object_id
    
    def execute_command(self, command: EnvironmentCommand) -> bool:
        """Execute a command in the environment."""
        try:
            if command.command_type == "pick_object":
                return self._pick_object(command.parameters)
            elif command.command_type == "place_object":
                return self._place_object(command.parameters)
            elif command.command_type == "move_object":
                return self._move_object(command.parameters)
            elif command.command_type == "rotate_object":
                return self._rotate_object(command.parameters)
            elif command.command_type == "push_object":
                return self._push_object(command.parameters)
            elif command.command_type == "pull_object":
                return self._pull_object(command.parameters)
            elif command.command_type == "stack_object":
                return self._stack_object(command.parameters)
            elif command.command_type == "unstack_object":
                return self._unstack_object(command.parameters)
            elif command.command_type == "wait":
                return self._wait(command.parameters)
            elif command.command_type == "task_complete":
                return True
            else:
                print(f"Unknown command type: {command.command_type}")
                return False
        except Exception as e:
            print(f"Error executing command {command.command_type}: {e}")
            return False
    
    def _pick_object(self, params: Dict[str, Any]) -> bool:
        """Pick up an object."""
        object_name = params.get("object_id")
        if object_name not in self.objects:
            return False
        
        obj_info = self.objects[object_name]
        
        # Simple pick implementation: add to held objects
        if object_name not in self.held_objects:
            self.held_objects.append(object_name)
        
        # In a real implementation, this would involve robot arm control
        print(f"Picked up object: {object_name}")
        return True
    
    def _place_object(self, params: Dict[str, Any]) -> bool:
        """Place an object."""
        object_name = params.get("object_id")
        target_position = params.get("target_position")
        
        if object_name not in self.objects:
            return False
        
        obj_info = self.objects[object_name]
        
        # Move object to target position
        if target_position:
            p.resetBasePositionAndOrientation(
                obj_info.object_id,
                target_position,
                obj_info.orientation
            )
            obj_info.position = target_position
        
        # Remove from held objects
        if object_name in self.held_objects:
            self.held_objects.remove(object_name)
        
        print(f"Placed object: {object_name} at {target_position}")
        return True
    
    def _move_object(self, params: Dict[str, Any]) -> bool:
        """Move an object."""
        object_name = params.get("object_id")
        target_position = params.get("target_position")
        
        if object_name not in self.objects or not target_position:
            return False
        
        obj_info = self.objects[object_name]
        
        # Smoothly move object to target position
        current_pos = obj_info.position
        steps = 10
        
        for i in range(steps):
            alpha = (i + 1) / steps
            new_pos = (
                current_pos[0] + alpha * (target_position[0] - current_pos[0]),
                current_pos[1] + alpha * (target_position[1] - current_pos[1]),
                current_pos[2] + alpha * (target_position[2] - current_pos[2])
            )
            
            p.resetBasePositionAndOrientation(
                obj_info.object_id,
                new_pos,
                obj_info.orientation
            )
            
            p.stepSimulation()
            time.sleep(0.01)
        
        obj_info.position = target_position
        print(f"Moved object: {object_name} to {target_position}")
        return True
    
    def _rotate_object(self, params: Dict[str, Any]) -> bool:
        """Rotate an object."""
        object_name = params.get("object_id")
        target_orientation = params.get("target_orientation")
        
        if object_name not in self.objects or not target_orientation:
            return False
        
        obj_info = self.objects[object_name]
        
        p.resetBasePositionAndOrientation(
            obj_info.object_id,
            obj_info.position,
            target_orientation
        )
        
        obj_info.orientation = target_orientation
        print(f"Rotated object: {object_name}")
        return True
    
    def _push_object(self, params: Dict[str, Any]) -> bool:
        """Push an object."""
        object_name = params.get("object_id")
        push_force = params.get("push_force", 10.0)
        push_direction = params.get("push_direction", (1, 0, 0))
        
        if object_name not in self.objects:
            return False
        
        obj_info = self.objects[object_name]
        
        # Apply force to object
        p.applyExternalForce(
            obj_info.object_id,
            -1,  # Link index (-1 for base)
            [push_force * push_direction[0],
             push_force * push_direction[1],
             push_force * push_direction[2]],
            obj_info.position,
            p.WORLD_FRAME
        )
        
        # Step simulation
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.01)
        
        print(f"Pushed object: {object_name}")
        return True

    # --- Atomic tool functions exposed to VLM tool calls ---
    def pick(self, object_id: str) -> Dict[str, Any]:
        """
        Pick an object by name into a virtual gripper.

        Args:
            object_id: Registered object name in environment

        Returns:
            Dict with success flag and held objects list
        """
        ok = self._pick_object({"object_id": object_id})
        return {"success": ok, "held_objects": list(self.held_objects)}

    def place(self, object_id: str, position: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Place a previously picked object at a target 3D position.

        Args:
            object_id: Object name
            position: Target world position (x, y, z)

        Returns:
            Dict with success flag and new position
        """
        ok = self._place_object({"object_id": object_id, "target_position": position})
        pos = None
        if object_id in self.objects:
            pos = self.objects[object_id].position
        return {"success": ok, "position": pos}

    def push(self, object_id: str, force: float = 10.0, direction: Tuple[float, float, float] = (1, 0, 0)) -> Dict[str, Any]:
        """
        Apply a pushing force to an object.

        Args:
            object_id: Object name
            force: Magnitude of the push force (Newton)
            direction: Unit direction vector (x, y, z)

        Returns:
            Dict with success flag
        """
        ok = self._push_object({
            "object_id": object_id,
            "push_force": float(force),
            "push_direction": tuple(direction),
        })
        return {"success": ok}

    def move(self, object_id: str, position: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Move an object to a specific 3D position.

        Args:
            object_id: Object name
            position: Target world position (x, y, z)

        Returns:
            Dict with success flag and new position
        """
        ok = self._move_object({"object_id": object_id, "target_position": position})
        pos = None
        if object_id in self.objects:
            pos = self.objects[object_id].position
        return {"success": ok, "position": pos}

    def observe(self, angle: float = 0.0) -> Dict[str, Any]:
        """
        Observe the scene possibly from a rotated camera angle.

        Args:
            angle: Yaw rotation (radians) to rotate camera around z-axis

        Returns:
            Dict with success flag and a note
        """
        # Simple camera yaw around target
        cx, cy, cz = self.camera_config.position
        tx, ty, tz = self.camera_config.target
        # Very basic rotation around z, small radius change ignored for simplicity
        # This keeps function side-effect small while enabling a different view
        import math
        r = math.hypot(cx - tx, cy - ty)
        base_angle = math.atan2(cy - ty, cx - tx)
        new_angle = base_angle + angle
        self.camera_config.position = (tx + r * math.cos(new_angle), ty + r * math.sin(new_angle), cz)
        return {"success": True, "angle": angle}

    # --- Tool schema export ---
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Generate JSON schema for exposed tool functions to pass into VLM.
        """
        def build_schema(func, name: str, desc: str, properties: Dict[str, Any], required: List[str]):
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
                self.pick,
                "pick",
                "Pick an object by name into a virtual gripper.",
                {
                    "object_id": {"type": "string", "description": "Registered object name"},
                },
                ["object_id"],
            ),
            build_schema(
                self.place,
                "place",
                "Place a previously picked object at a target 3D position.",
                {
                    "object_id": {"type": "string"},
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                },
                ["object_id", "position"],
            ),
            build_schema(
                self.push,
                "push",
                "Apply a pushing force to an object.",
                {
                    "object_id": {"type": "string"},
                    "force": {"type": "number", "default": 10.0},
                    "direction": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "default": [1, 0, 0],
                    },
                },
                ["object_id"],
            ),
            build_schema(
                self.move,
                "move",
                "Move an object to a specific 3D position.",
                {
                    "object_id": {"type": "string"},
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                },
                ["object_id", "position"],
            ),
            build_schema(
                self.observe,
                "observe",
                "Observe the scene from a rotated camera angle.",
                {
                    "angle": {"type": "number", "default": 0.0},
                },
                [],
            ),
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Dynamically call a tool function by name with arguments.
        """
        if not hasattr(self, name):
            raise AttributeError(f"Unknown tool: {name}")
        func = getattr(self, name)
        if not callable(func):
            raise AttributeError(f"Attribute is not callable: {name}")
        return func(**arguments)
    
    def _pull_object(self, params: Dict[str, Any]) -> bool:
        """Pull an object."""
        # Similar to push but in opposite direction
        params["push_direction"] = tuple(-x for x in params.get("pull_direction", (-1, 0, 0)))
        params["push_force"] = params.get("pull_force", 10.0)
        return self._push_object(params)
    
    def _stack_object(self, params: Dict[str, Any]) -> bool:
        """Stack one object on another."""
        object_name = params.get("object_id")
        target_object_name = params.get("target_object_id")
        
        if (object_name not in self.objects or 
            target_object_name not in self.objects):
            return False
        
        target_obj = self.objects[target_object_name]
        
        # Calculate stacking position
        stack_position = (
            target_obj.position[0],
            target_obj.position[1],
            target_obj.position[2] + 0.1  # Stack height offset
        )
        
        # Move object to stack position
        return self._move_object({
            "object_id": object_name,
            "target_position": stack_position
        })
    
    def _unstack_object(self, params: Dict[str, Any]) -> bool:
        """Unstack an object."""
        object_name = params.get("object_id")
        
        if object_name not in self.objects:
            return False
        
        obj_info = self.objects[object_name]
        
        # Move object away from stack
        new_position = (
            obj_info.position[0] + 0.2,
            obj_info.position[1],
            obj_info.position[2]
        )
        
        return self._move_object({
            "object_id": object_name,
            "target_position": new_position
        })
    
    def _wait(self, params: Dict[str, Any]) -> bool:
        """Wait for specified duration."""
        duration = params.get("duration", 1.0)
        
        for _ in range(int(duration * 240)):  # 240 Hz simulation
            p.stepSimulation()
            time.sleep(1./240.)
        
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        state = {
            "timestep": self.timestep,
            "objects": {},
            "robot": {
                "id": self.robot_id,
                "position": (0, 0, 0),  # Would get from robot
                "held_objects": self.held_objects
            }
        }
        
        # Update object positions
        for name, obj_info in self.objects.items():
            pos, orn = p.getBasePositionAndOrientation(obj_info.object_id)
            state["objects"][name] = {
                "position": pos,
                "orientation": orn,
                "type": obj_info.object_type,
                "properties": obj_info.properties
            }
        
        return state
    
    def render(self) -> Image.Image:
        """Render the environment and return an image."""
        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_config.position,
            cameraTargetPosition=self.camera_config.target,
            cameraUpVector=self.camera_config.up_vector
        )
        
        # Compute projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_config.fov,
            aspect=self.camera_config.aspect_ratio,
            nearPlane=self.camera_config.near_plane,
            farPlane=self.camera_config.far_plane
        )
        
        # Capture image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_config.image_width,
            height=self.camera_config.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to PIL Image
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        rgb_array = rgb_array.reshape(height, width, 3)
        
        return Image.fromarray(rgb_array)
    
    def set_camera_config(self, camera_config: CameraConfig) -> None:
        """Set camera configuration."""
        self.camera_config = camera_config
    
    def step(self) -> None:
        """Step the physics simulation."""
        p.stepSimulation()
        self.timestep += 1
    
    def reset(self) -> None:
        """Reset the environment."""
        # Remove all objects except plane and table
        for obj_info in self.objects.values():
            p.removeBody(obj_info.object_id)
        
        self.objects.clear()
        self.held_objects.clear()
        self.timestep = 0
        
        print("Environment reset")
    
    def close(self) -> None:
        """Close the environment."""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None
        
        print("Environment closed")


def create_environment(env_type: str = "pybullet", **kwargs) -> PhysicsEnvironment:
    """
    Factory function to create physics environment.
    
    Args:
        env_type: Type of environment ("pybullet")
        **kwargs: Additional arguments for environment
        
    Returns:
        PhysicsEnvironment instance
    """
    if env_type == "pybullet":
        return PyBulletEnvironment(**kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}") 