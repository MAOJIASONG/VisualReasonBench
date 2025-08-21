"""
Unified physics environment implementation using PyBullet.

This module provides the complete physics environment with PyBullet integration,
serving as the base class for all specific puzzle environments.
"""

import os
import time
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

# Conditional pybullet import
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    # Mock pybullet for testing
    class MockPyBullet:
        GUI = "GUI"
        DIRECT = "DIRECT"
        GEOM_BOX = 0
        GEOM_SPHERE = 1
        GEOM_CYLINDER = 2
        WORLD_FRAME = 0
        ER_BULLET_HARDWARE_OPENGL = 0
        
        @staticmethod
        def connect(mode): return 0
        @staticmethod
        def setGravity(x, y, z): pass
        @staticmethod
        def setTimeStep(dt): pass
        @staticmethod
        def setRealTimeSimulation(enable): pass
        @staticmethod
        def setAdditionalSearchPath(path): pass
        @staticmethod
        def loadURDF(path, **kwargs): return 1
        @staticmethod
        def stepSimulation(): pass
        @staticmethod
        def disconnect(client): pass
        @staticmethod
        def resetSimulation(): pass
        @staticmethod
        def createCollisionShape(*args, **kwargs): return 0
        @staticmethod
        def createVisualShape(*args, **kwargs): return 0
        @staticmethod
        def createMultiBody(*args, **kwargs): return 1
        @staticmethod
        def resetBasePositionAndOrientation(obj, pos, orn): pass
        @staticmethod
        def getBasePositionAndOrientation(obj): return ([0,0,0], [0,0,0,1])
        @staticmethod
        def removeBody(obj): pass
        @staticmethod
        def applyExternalForce(*args, **kwargs): pass
        @staticmethod
        def computeViewMatrix(*args, **kwargs): return []
        @staticmethod
        def computeProjectionMatrixFOV(*args, **kwargs): return []
        @staticmethod
        def getCameraImage(*args, **kwargs): return (512, 512, np.zeros((512, 512, 3), dtype=np.uint8), None, None)
        @staticmethod
        def getEulerFromQuaternion(q): return [0, 0, 0]
        @staticmethod
        def getQuaternionFromEuler(euler): return [0, 0, 0, 1]
        @staticmethod
        def getBaseVelocity(obj): return ([0,0,0], [0,0,0])
        @staticmethod
        def getContactPoints(*args, **kwargs): return []
        @staticmethod
        def getNumJoints(obj): return 0
    
    p = MockPyBullet()
    
    class MockPyBulletData:
        @staticmethod
        def getDataPath(): return ""
    
    pybullet_data = MockPyBulletData()

from ..core.base import BaseEnvironment, Action, State, Observation


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
class ObjectInfo:
    """Information about an object in the environment."""
    object_id: int
    name: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    object_type: str
    properties: Dict[str, Any]

# Conditional multi-view renderer import
try:
    from ..utils.multi_view_renderer import MultiViewRenderer
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    class MockMultiViewRenderer:
        def __init__(self, *args, **kwargs): pass
        def render_multi_view(self): return Image.new("RGB", (512, 512), "gray")
        def render_single_view(self, name): return Image.new("RGB", (512, 512), "gray")
        def set_camera_config(self, *args, **kwargs): pass
    MultiViewRenderer = MockMultiViewRenderer


class PhysicsEnvironment(BaseEnvironment):
    """Unified PyBullet physics environment for all puzzle tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # PyBullet configuration
        self.gui = config.get("gui", False)
        self.gravity = config.get("gravity", -9.81)
        self.time_step = config.get("time_step", 1.0/240.0)
        self.urdf_base_path = config.get("urdf_base_path", "")
        
        # Rendering configuration
        self.render_width = config.get("render_width", 512)
        self.render_height = config.get("render_height", 512)
        self.multi_view = config.get("multi_view", True)
        
        # PyBullet state
        self.client_id = None
        self.plane_id = None
        self.table_id = None
        self.objects = {}
        self.held_objects = []
        self.max_steps = config.get("max_steps", 100)
        
        # Camera configuration
        self.camera_config = CameraConfig(
            image_width=self.render_width,
            image_height=self.render_height
        )
        
        # Multi-view renderer
        self.multi_view_renderer = MultiViewRenderer(
            width=self.render_width,
            height=self.render_height
        )
        
        # Initialize environment
        self._initialize_pybullet()
        
    def _initialize_pybullet(self) -> None:
        """Initialize PyBullet physics simulation."""
        # Connect to PyBullet
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
            
        # Set gravity
        p.setGravity(0, 0, self.gravity)
        
        # Set time step
        p.setTimeStep(self.time_step)
        
        # Disable real-time simulation
        p.setRealTimeSimulation(0)
        
        # Add search paths
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if os.path.exists(self.urdf_base_path):
            p.setAdditionalSearchPath(self.urdf_base_path)
            
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Optionally load a table
        if self.config.get("load_table", True):
            try:
                self.table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
            except:
                # Create simple table if default doesn't exist
                self._create_simple_table()
                
        print(f"PyBullet initialized. Client ID: {self.client_id}")
        
    def _create_simple_table(self) -> None:
        """Create a simple table using primitive shapes."""
        # Table dimensions
        table_width = 1.5
        table_length = 1.5 
        table_height = 0.02
        table_z = 0.4
        
        # Create table collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[table_width/2, table_length/2, table_height/2]
        )
        
        # Create table visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[table_width/2, table_length/2, table_height/2],
            rgbaColor=[0.6, 0.4, 0.2, 1.0]  # Brown color
        )
        
        # Create table body
        self.table_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, table_z]
        )
        
    def reset(self) -> Observation:
        """Reset environment to initial state."""
        # Remove all dynamic objects
        for obj_name, obj_info in list(self.objects.items()):
            try:
                p.removeBody(obj_info.object_id)
            except:
                pass
                
        self.objects.clear()
        self.step_count = 0
        
        # Reset physics simulation
        p.resetSimulation()
        
        # Reinitialize basic environment
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.time_step)
        p.setRealTimeSimulation(0)
        
        # Reload ground and table
        self.plane_id = p.loadURDF("plane.urdf")
        if self.config.get("load_table", True):
            try:
                self.table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
            except:
                self._create_simple_table()
        
        # Setup task-specific environment
        self._setup_task_environment()
        
        # Create initial state
        self.current_state = self._get_current_state()
        
        # Create initial observation
        return self._create_observation()
        
    def step(self, action: Action) -> Tuple[Observation, str, bool]:
        """Execute action and return new observation."""
        self.step_count += 1
        
        # Execute action
        success, feedback = self._execute_action(action)
        
        # Step physics simulation
        for _ in range(10):  # Multiple steps for stability
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)
        
        # Update state
        self.current_state = self._get_current_state()
        
        # Check if done
        done = self._is_done()
        
        # Create observation
        observation = self._create_observation()
        
        return observation, feedback, done
        
    def render(self, multi_view: bool = True) -> Union[Image.Image, Dict[str, Image.Image]]:
        """Render the current environment state."""
        if multi_view and self.multi_view:
            return self.multi_view_renderer.render_multi_view()
        else:
            return self.multi_view_renderer.render_single_view("perspective")
    
    def get_available_actions(self) -> List[str]:
        """Get list of available action types."""
        return [
            "pick", "place", "move", "push", "pull", "rotate",
            "observe", "wait", "check_solution"
        ]
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for tool functions."""
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

        # Base tool schemas available to all environments
        base_schemas = [
            build_schema(
                "pick",
                "Pick an object by name into a virtual gripper.",
                {
                    "object_id": {"type": "string", "description": "Name of object to pick"},
                },
                ["object_id"],
            ),
            build_schema(
                "place",
                "Place a previously picked object at a target 3D position.",
                {
                    "object_id": {"type": "string", "description": "Name of object to place"},
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Target position [x, y, z]"
                    },
                },
                ["object_id", "position"],
            ),
            build_schema(
                "move",
                "Move an object to a specific 3D position.",
                {
                    "object_id": {"type": "string", "description": "Name of object to move"},
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Target position [x, y, z]"
                    },
                },
                ["object_id", "position"],
            ),
            build_schema(
                "push",
                "Apply a pushing force to an object.",
                {
                    "object_id": {"type": "string", "description": "Name of object to push"},
                    "force": {"type": "number", "default": 10.0, "description": "Push force magnitude"},
                    "direction": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "default": [1, 0, 0],
                        "description": "Push direction [x, y, z]"
                    },
                },
                ["object_id"],
            ),
            build_schema(
                "rotate",
                "Rotate an object to a specific orientation.",
                {
                    "object_id": {"type": "string", "description": "Name of object to rotate"},
                    "axis": {"type": "string", "enum": ["x", "y", "z"], "description": "Rotation axis"},
                    "angle": {"type": "number", "description": "Rotation angle in radians"},
                },
                ["object_id", "axis", "angle"],
            ),
            build_schema(
                "observe",
                "Observe the scene from a rotated camera angle.",
                {
                    "angle": {"type": "number", "default": 0.0, "description": "Camera rotation angle in radians"},
                },
                [],
            ),
            build_schema(
                "wait",
                "Wait for a specified duration to let physics settle.",
                {
                    "duration": {"type": "number", "default": 1.0, "description": "Wait duration in seconds"},
                },
                [],
            ),
            build_schema(
                "check_solution",
                "Check if the puzzle is solved.",
                {},
                [],
            ),
        ]
        
        # Allow subclasses to add their own tool schemas
        task_specific_schemas = self._get_task_specific_tool_schemas()
        
        return base_schemas + task_specific_schemas
    
    def _get_task_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """Override this method in subclasses to add task-specific tools."""
        return []
        
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call from VLM."""
        try:
            # Base tool implementations
            if tool_name == "pick":
                return self._tool_pick(arguments.get("object_id"))
            elif tool_name == "place":
                return self._tool_place(arguments.get("object_id"), arguments.get("position"))
            elif tool_name == "move":
                return self._tool_move(arguments.get("object_id"), arguments.get("position"))
            elif tool_name == "push":
                return self._tool_push(
                    arguments.get("object_id"),
                    arguments.get("force", 10.0),
                    arguments.get("direction", [1, 0, 0])
                )
            elif tool_name == "rotate":
                return self._tool_rotate(
                    arguments.get("object_id"),
                    arguments.get("axis", "z"),
                    arguments.get("angle", 0.0)
                )
            elif tool_name == "observe":
                return self._tool_observe(arguments.get("angle", 0.0))
            elif tool_name == "wait":
                return self._tool_wait(arguments.get("duration", 1.0))
            elif tool_name == "check_solution":
                return self._tool_check_solution()
            else:
                # Try task-specific tool implementations
                return self._execute_task_specific_tool(tool_name, arguments)
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _execute_task_specific_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method in subclasses to handle task-specific tools."""
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    def is_success(self) -> bool:
        """Check if task completed successfully."""
        return self.current_state.success if self.current_state else False
    
    def close(self) -> None:
        """Clean up environment resources."""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None
            print("PyBullet environment closed")
    
    # Abstract methods to be implemented by subclasses
    def _setup_task_environment(self) -> None:
        """Setup task-specific objects and constraints."""
        pass
        
    def _execute_action(self, action: Action) -> Tuple[bool, str]:
        """Execute specific action in environment."""
        return True, "Action executed"
        
    def _get_current_state(self) -> State:
        """Get current environment state."""
        return State(
            step=self.step_count,
            objects={},
            completed=False,
            success=False
        )
        
    def _create_observation(self) -> Observation:
        """Create observation for VLM."""
        # Render images
        if self.multi_view:
            main_image = self.multi_view_renderer.render_multi_view()
            multi_view_images = {
                "front": self.multi_view_renderer.render_single_view("front"),
                "side": self.multi_view_renderer.render_single_view("side"),
                "top": self.multi_view_renderer.render_single_view("top")
            }
        else:
            main_image = self.multi_view_renderer.render_single_view("perspective")
            multi_view_images = None
            
        return Observation(
            image=main_image,
            state=self.current_state,
            description=self._get_state_description(),
            available_actions=self.get_available_actions(),
            multi_view_images=multi_view_images
        )
        
    def _get_state_description(self) -> str:
        """Get textual description of current state."""
        return f"Step {self.step_count}: Environment contains {len(self.objects)} objects."
        
    def _is_done(self) -> bool:
        """Check if episode is done."""
        return (self.step_count >= self.max_steps or 
                (self.current_state and self.current_state.completed))
    
    # Enhanced tool implementations
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
        if shape_type == "box":
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=size,
                rgbaColor=color
            )
        elif shape_type == "sphere":
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size[0],
                rgbaColor=color
            )
        elif shape_type == "cylinder":
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=size[0],
                length=size[1],
                rgbaColor=color
            )
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
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

    # Tool implementations
    def _tool_pick(self, object_id: str) -> Dict[str, Any]:
        """Pick tool implementation."""
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}
        
        # Add to held objects
        if object_id not in self.held_objects:
            self.held_objects.append(object_id)
        
        return {"status": "success", "message": f"Picked up {object_id}", "held_objects": self.held_objects}
        
    def _tool_place(self, object_id: str, position: List[float]) -> Dict[str, Any]:
        """Place tool implementation."""
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}
            
        if not position or len(position) != 3:
            return {"status": "error", "message": "Invalid position"}
            
        obj_info = self.objects[object_id]
        p.resetBasePositionAndOrientation(obj_info.object_id, position, obj_info.orientation)
        
        # Update object info
        obj_info.position = tuple(position)
        
        # Remove from held objects
        if object_id in self.held_objects:
            self.held_objects.remove(object_id)
        
        return {"status": "success", "message": f"Placed {object_id} at {position}"}
    
    def _tool_move(self, object_id: str, position: List[float]) -> Dict[str, Any]:
        """Move tool implementation."""
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}
            
        if not position or len(position) != 3:
            return {"status": "error", "message": "Invalid position"}
            
        obj_info = self.objects[object_id]
        
        # Smoothly move object to target position
        current_pos = obj_info.position
        steps = 10
        
        for i in range(steps):
            alpha = (i + 1) / steps
            new_pos = (
                current_pos[0] + alpha * (position[0] - current_pos[0]),
                current_pos[1] + alpha * (position[1] - current_pos[1]),
                current_pos[2] + alpha * (position[2] - current_pos[2])
            )
            
            p.resetBasePositionAndOrientation(
                obj_info.object_id,
                new_pos,
                obj_info.orientation
            )
            
            p.stepSimulation()
            time.sleep(0.01)
        
        # Update object info
        obj_info.position = tuple(position)
        
        return {"status": "success", "message": f"Moved {object_id} to {position}"}
        
    def _tool_push(self, object_id: str, force: float, direction: List[float]) -> Dict[str, Any]:
        """Push tool implementation."""
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}
            
        obj_info = self.objects[object_id]
        
        # Normalize direction
        direction = np.array(direction)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1, 0, 0])
        
        # Apply force
        p.applyExternalForce(
            obj_info.object_id, -1,
            [force * direction[0], force * direction[1], force * direction[2]],
            obj_info.position, p.WORLD_FRAME
        )
        
        return {"status": "success", "message": f"Pushed {object_id} with force {force}"}
    
    def _tool_rotate(self, object_id: str, axis: str, angle: float) -> Dict[str, Any]:
        """Rotate tool implementation."""
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}
            
        obj_info = self.objects[object_id]
        
        # Get current orientation
        current_orient = obj_info.orientation
        current_euler = p.getEulerFromQuaternion(current_orient)
        
        # Apply rotation
        if axis.lower() == "x":
            new_euler = [current_euler[0] + angle, current_euler[1], current_euler[2]]
        elif axis.lower() == "y":
            new_euler = [current_euler[0], current_euler[1] + angle, current_euler[2]]
        elif axis.lower() == "z":
            new_euler = [current_euler[0], current_euler[1], current_euler[2] + angle]
        else:
            return {"status": "error", "message": f"Invalid axis: {axis}"}
        
        new_orientation = p.getQuaternionFromEuler(new_euler)
        
        # Apply new orientation
        p.resetBasePositionAndOrientation(
            obj_info.object_id,
            obj_info.position,
            new_orientation
        )
        
        # Update object info
        obj_info.orientation = new_orientation
        
        return {"status": "success", "message": f"Rotated {object_id} {angle} radians around {axis}-axis"}
        
    def _tool_observe(self, angle: float) -> Dict[str, Any]:
        """Observe tool implementation."""
        # Rotate camera view
        import math
        radius = 1.5
        height = 1.0
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        self.multi_view_renderer.set_camera_config("perspective", [x, y, height], [0, 0, 0.3])
        
        return {"status": "success", "message": f"Observing from angle {angle:.2f} radians"}
    
    def _tool_wait(self, duration: float) -> Dict[str, Any]:
        """Wait tool implementation."""
        steps = int(duration * 240)  # 240 Hz simulation
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1./240.)
        
        return {"status": "success", "message": f"Waited for {duration} seconds"}
        
    def _tool_check_solution(self) -> Dict[str, Any]:
        """Check solution tool implementation."""
        success = self._evaluate_success()
        
        return {
            "status": "success",
            "solved": success,
            "message": "Solution checked"
        }
        
    def _evaluate_success(self) -> bool:
        """Evaluate if puzzle is solved. Override in subclasses."""
        return False
    
    def get_object_state(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Get current state of an object."""
        if object_name not in self.objects:
            return None
            
        obj_info = self.objects[object_name]
        pos, orn = p.getBasePositionAndOrientation(obj_info.object_id)
        lin_vel, ang_vel = p.getBaseVelocity(obj_info.object_id)
        
        return {
            "position": pos,
            "orientation": orn,
            "linear_velocity": lin_vel,
            "angular_velocity": ang_vel,
            "euler_angles": p.getEulerFromQuaternion(orn),
            "object_type": obj_info.object_type,
            "properties": obj_info.properties
        }
    
    def get_all_objects_state(self) -> Dict[str, Dict[str, Any]]:
        """Get current state of all objects."""
        return {name: self.get_object_state(name) for name in self.objects.keys()}
    
    def get_contact_points(self, obj1_name: str, obj2_name: str = None) -> List[Dict[str, Any]]:
        """Get contact points between objects."""
        if obj1_name not in self.objects:
            return []
            
        obj1_id = self.objects[obj1_name].object_id
        
        if obj2_name is None:
            contacts = p.getContactPoints(obj1_id)
        else:
            if obj2_name not in self.objects:
                return []
            obj2_id = self.objects[obj2_name].object_id
            contacts = p.getContactPoints(obj1_id, obj2_id)
        
        contact_list = []
        for contact in contacts:
            contact_list.append({
                "position_on_a": contact[5],
                "position_on_b": contact[6],
                "normal_on_b": contact[7],
                "distance": contact[8],
                "normal_force": contact[9]
            })
            
        return contact_list
