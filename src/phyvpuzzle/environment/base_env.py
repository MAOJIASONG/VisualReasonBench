"""
Unified physics environment implementation using PyBullet.

This module provides the complete physics environment with PyBullet integration,
serving as the base class for all specific puzzle environments.
"""

from pathlib import Path
import time
import numpy as np
from abc import abstractmethod
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Any
from phyvpuzzle.utils.multi_view_renderer import MultiViewRenderer
from phyvpuzzle.core import BaseEnvironment, EnvironmentConfig
from phyvpuzzle.core.base import Action, State, Observation
from phyvpuzzle.core.base import ObjectInfo
# Removed unused import to avoid circular dependency
# from phyvpuzzle.tasks.base_task import PhysicsTask


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

class PhysicsEnvironment(BaseEnvironment):
    """Unified PyBullet physics environment for all puzzle tasks."""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)

        self.step_count: int = 0
        self.current_state: Optional[State] = None
        
        self.client_id = None
        self.plane_id = None
        self.table_id = None
        self.objects: List[ObjectInfo] = []
        self.held_object_ids: List[int] = []
        
        # Multi-view renderer
        self.multi_view_renderer = MultiViewRenderer(
            width=self.config.render_width,
            height=self.config.render_height
        )

        # Connect to PyBullet
        if self.config.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

    def reset(self) -> Observation:
        """Reset environment to initial state."""
        # Step 1: Empty the environment
        self.objects = []
        self.held_object_ids = []
        self.step_count = 0
        self.current_state = None
        self.plane_id = None
        self.table_id = None

        # Step 2: Initialize environment
        self._initialize_pybullet()
        # Update state
        self.current_state = self._get_current_state()
        
        # Step 3: Create initial observation
        return self._create_observation()
        
    def _initialize_pybullet(self) -> None:
        """Initialize PyBullet physics simulation."""

        # Reset simulation
        p.resetSimulation(physicsClientId=self.client_id)  
          
        p.setGravity(0, 0, self.config.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.config.time_step, physicsClientId=self.client_id)
        p.setRealTimeSimulation(1 if self.config.real_time else 0, physicsClientId=self.client_id)
        
        # Add search paths using pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load ground plane as basis
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, spinningFriction=0.002, rollingFriction=0.002, linearDamping=0.02, angularDamping=0.02)

        
        # Optionally load a table
        if self.config.load_table:
            try:
                self.table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0], globalScaling=1.0, physicsClientId=self.client_id)
                p.changeDynamics(self.table_id, -1, lateralFriction=1.0, spinningFriction=0.002, rollingFriction=0.002, linearDamping=0.02, angularDamping=0.02)
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
    
    def _create_observation(self) -> Observation:
        """Create observation for VLM."""
        # Render images
        main_image = self.render()
            
        return Observation(
            image=main_image,
            state=self.current_state,
            description=self._get_state_description(),
        )
   
    def step(self, action: Action) -> Observation:
        """Execute action and return new observation."""
        self.step_count += 1
        
        # Execute action
        tool_result: Dict[str, Any] = self.execute_tool_call(action)
        settle_used = self._wait_until_stable()
        tool_result["settle_steps"] = settle_used
        
        # Update state
        self.current_state = self._get_current_state(
            metadata={
                "tool_call": action.to_dict(),
                "tool_result": tool_result,
            }
        )
        
        # Create observation
        observation = self._create_observation()

        return observation
    
    def _wait_until_stable(self, target_body_ids: Optional[List[int]] = None) -> int:
        """
        Advance the simulation until it stabilizes or reaches the upper limit of steps.
        Return the actual number of steps advanced.
        """
        # By default, all dynamic objects are stabilized (excluding static objects such as planes and tables).
        if target_body_ids is None:
            dyn_ids = []
            for obj in self.objects:
                dyn_ids.append(obj.object_id)
            target_body_ids = dyn_ids

        used = 0
        for _ in range(self.config.max_settle_steps):
            p.stepSimulation()
            used += 1
            all_ok = True
            for bid in target_body_ids:
                lin, ang = p.getBaseVelocity(bid)
                if any(abs(v) >= self.config.lin_vel_tol for v in lin) or any(abs(w) >= self.config.ang_vel_tol for w in ang):
                    all_ok = False
                    break
            if all_ok:
                break
        return used
        
    def render(self, multi_view: Optional[bool] = None) -> Union[Image.Image, Dict[str, Image.Image]]:
        """Render the current environment state."""
        if multi_view is None:
            multi_view = self.config.multi_view
        
        if multi_view:
            return self.multi_view_renderer.render_multi_view()
        else:
            return self.multi_view_renderer.render_single_view("perspective")

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
                ["object_id", "force", "direction"],
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
                ["angle"],
            ),
            build_schema(
                "finish",
                "Finish the task.",
                {},
                [],
            ),
        ]
        
        # Allow subclasses to add their own tool schemas
        task_specific_schemas = self._get_task_specific_tool_schemas()
        
        return base_schemas + task_specific_schemas
    
    def get_object_by_id(self, object_id: int) -> Optional[ObjectInfo]:
        """Get object by name from the objects list."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    # Tool implementations
    @BaseEnvironment.register_tool("pick")
    def _tool_pick(self, object_id: int) -> Dict[str, Any]:
        """Pick tool implementation."""
        obj_info = self.get_object_by_id(object_id)
        if not obj_info:
            return {"status": "error", "message": f"Object {object_id} not found"}
        
        # Add to held objects
        if object_id not in self.held_object_ids:
            self.held_object_ids.append(object_id)
        
        return {"status": "success", "message": f"Picked up {object_id}", "held_object_ids": self.held_object_ids}

    @BaseEnvironment.register_tool("place")    
    def _tool_place(self, object_id: int, position: List[float]) -> Dict[str, Any]:
        """Place tool implementation."""
        obj_info = self.get_object_by_id(object_id)
        if not obj_info:
            return {"status": "error", "message": f"Object {object_id} not found"}
            
        if not position or len(position) != 3:
            return {"status": "error", "message": "Invalid position"}
            
        p.resetBasePositionAndOrientation(obj_info.object_id, position, obj_info.orientation)
        
        # Update object info
        obj_info.position = tuple(position)
        
        # Remove from held objects
        try:
            self.held_object_ids.remove(object_id)
        except ValueError:
            pass
        
        return {"status": "success", "message": f"Placed {object_id} at {position}"}
    
    @BaseEnvironment.register_tool("move")    
    def _tool_move(self, object_id: int, position: List[float]) -> Dict[str, Any]:
        """Move tool implementation."""
        obj_info = self.get_object_by_id(object_id)
        if not obj_info:
            return {"status": "error", "message": f"Object {object_id} not found"}
            
        if not position or len(position) != 3:
            return {"status": "error", "message": "Invalid position"}
        
        # Directly move object to target position
        p.resetBasePositionAndOrientation(
            obj_info.object_id,
            position,
            obj_info.orientation
        )
        
        # Update object info
        obj_info.position = tuple(position)
        
        return {"status": "success", "message": f"Moved {object_id} to {position}"}
        
    @BaseEnvironment.register_tool("push")    
    def _tool_push(self, object_id: int, force: float, direction: List[float]) -> Dict[str, Any]:
        """Push tool implementation."""
        obj_info = self.get_object_by_id(object_id)
        if not obj_info:
            return {"status": "error", "message": f"Object {object_id} not found"}
        
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
    
    @BaseEnvironment.register_tool("rotate")    
    def _tool_rotate(self, object_id: int, axis: str, angle: float) -> Dict[str, Any]:
        """Rotate tool implementation."""
        obj_info = self.get_object_by_id(object_id)
        if not obj_info:
            return {"status": "error", "message": f"Object {object_id} not found"}
        
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
        
    @BaseEnvironment.register_tool("observe")    
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

    @BaseEnvironment.register_tool("finish")    
    def _tool_finish(self) -> Dict[str, Any]:
        """Finish tool implementation."""
        return {"status": "success", "message": "Finished the task"}
       
    def execute_tool_call(self, action: Action) -> Dict[str, Any]:
        """Execute a tool call from VLM."""
        try:
            tool_func = BaseEnvironment.tool_registry[action.action_type]
        except Exception as e:
            return {"status": "error", "message": f"Tool '{action.action_type}' not found: {str(e)}"}

        try:
            return tool_func(self, **action.parameters)
        except Exception as e:
            return {"status": "error", "message": f"Failed to execute tool '{action.action_type}': {str(e)}"}
    
    def close(self) -> None:
        """Clean up environment resources."""
        p.disconnect(self.client_id)
        self.client_id = None
        print("PyBullet environment closed")
    
    # Extra utilisation helper functions
    def add_object(self, object_name: str, urdf_path: str, 
                  position: Tuple[float, float, float],
                  orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
                  object_type: str = "block",
                  properties: Dict[str, Any] = None) -> int:
        """Add object to the environment."""
        if properties is None:
            properties = {}
        try:
            object_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                baseOrientation=orientation
            )

            p.changeDynamics(object_id, -1, lateralFriction=1.0, spinningFriction=0.002, rollingFriction=0.002, linearDamping=0.04, angularDamping=0.04)
            
            obj_info = ObjectInfo(
                object_id=object_id,
                name=object_name,
                position=position,
                orientation=orientation,
                object_type=object_type,
                properties=properties
            )
            self.objects.append(obj_info)
            
            return object_id
        except Exception as e:
            print(f"Error loading object {object_name} from {urdf_path}: {e}")
            return -1
    
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

        p.changeDynamics(object_id, -1, lateralFriction=1.0, spinningFriction=0.002, rollingFriction=0.002, linearDamping=0.04, angularDamping=0.04)
        
        # Store object info
        obj_info = ObjectInfo(
            object_id=object_id,
            name=object_name,
            position=position,
            orientation=(0, 0, 0, 1),
            object_type=shape_type,
            properties={"mass": mass, "color": color, "size": size}
        )
        self.objects.append(obj_info)
        
        return object_id

    @abstractmethod
    def _get_current_state(self, metadata: Optional[Dict[str, Any]] = None) -> State:
        """Get current state of the environment."""
        # 样板代码: 返回一个空的 State 对象，子类应实现具体逻辑
        return State(
            step=self.step_count,
            objects=self.objects,
            time_stamp=time.time(),
            metadata=metadata,
        )
    
    @abstractmethod
    def _get_state_description(self) -> str:
        """Get textual description of current state."""
        return f"Step {self.step_count}: Environment contains {len(self.objects)} objects."
    
    @abstractmethod
    def _get_task_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get task-specific tool schemas."""
        pass