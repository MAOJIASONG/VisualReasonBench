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
        def connect(mode):
            return 0

        @staticmethod
        def setGravity(x, y, z):
            pass

        @staticmethod
        def setTimeStep(dt):
            pass

        @staticmethod
        def setRealTimeSimulation(enable):
            pass

        @staticmethod
        def setAdditionalSearchPath(path):
            pass

        @staticmethod
        def loadURDF(path, **kwargs):
            return 1

        @staticmethod
        def stepSimulation():
            pass

        @staticmethod
        def disconnect(client):
            pass

        @staticmethod
        def resetSimulation():
            pass

        @staticmethod
        def createCollisionShape(*args, **kwargs):
            return 0

        @staticmethod
        def createVisualShape(*args, **kwargs):
            return 0

        @staticmethod
        def createMultiBody(*args, **kwargs):
            return 1

        @staticmethod
        def resetBasePositionAndOrientation(obj, pos, orn):
            pass

        @staticmethod
        def getBasePositionAndOrientation(obj):
            return ([0, 0, 0], [0, 0, 0, 1])

        @staticmethod
        def removeBody(obj):
            pass

        @staticmethod
        def applyExternalForce(*args, **kwargs):
            pass

        @staticmethod
        def computeViewMatrix(*args, **kwargs):
            return []

        @staticmethod
        def computeProjectionMatrixFOV(*args, **kwargs):
            return []

        @staticmethod
        def getCameraImage(*args, **kwargs):
            return (512, 512, np.zeros((512, 512, 3), dtype=np.uint8), None, None)

        @staticmethod
        def getEulerFromQuaternion(q):
            return [0, 0, 0]

        @staticmethod
        def getQuaternionFromEuler(euler):
            return [0, 0, 0, 1]

        @staticmethod
        def getBaseVelocity(obj):
            return ([0, 0, 0], [0, 0, 0])

        @staticmethod
        def getContactPoints(*args, **kwargs):
            return []

        @staticmethod
        def getNumJoints(obj):
            return 0

    p = MockPyBullet()

    class MockPyBulletData:
        @staticmethod
        def getDataPath():
            return ""

    pybullet_data = MockPyBulletData()

from ..core.base import BaseEnvironment, Action, State, Observation

# Note: Global automatic holding (SecondHandManager) has been removed. Tools
# operate explicitly; placement can apply a local temporary hold on the target.


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
        def __init__(self, *args, **kwargs):
            pass

        def render_multi_view(self):
            return Image.new("RGB", (512, 512), "gray")

        def render_single_view(self, name):
            return Image.new("RGB", (512, 512), "gray")

        def set_camera_config(self, *args, **kwargs):
            pass

    MultiViewRenderer = MockMultiViewRenderer


class PhysicsEnvironment(BaseEnvironment):
    """Unified PyBullet physics environment for all puzzle tasks."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # PyBullet configuration
        self.gui = config.get("gui", False)
        self.gravity = config.get("gravity", -9.81)
        self.time_step = config.get("time_step", 1.0 / 240.0)
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

        # Constraint management for pick/place operations
        self.active_constraints = {}  # {object_name: constraint_id}
        self.picked_objects = set()  # Track which objects are currently picked

        # No global automatic holding; some tools may apply local stabilization

        # Camera configuration
        self.camera_config = CameraConfig(
            image_width=self.render_width, image_height=self.render_height
        )

        # Multi-view renderer
        self.multi_view_renderer = MultiViewRenderer(
            width=self.render_width, height=self.render_height
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
            halfExtents=[table_width / 2, table_length / 2, table_height / 2],
        )

        # Create table visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[table_width / 2, table_length / 2, table_height / 2],
            rgbaColor=[0.6, 0.4, 0.2, 1.0],  # Brown color
        )

        # Create table body
        self.table_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, table_z],
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

        # Reinitialize basic environment with gravity temporarily disabled
        # to prevent initial settling before constraints/placements are applied.
        p.setTimeStep(self.time_step)
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, 0)

        # Reload ground and table
        self.plane_id = p.loadURDF("plane.urdf")
        if self.config.get("load_table", True):
            try:
                self.table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
            except:
                self._create_simple_table()

        # Setup task-specific environment (load pieces, create constraints, etc.)
        self._setup_task_environment()

        # Enable configured gravity after objects/constraints are in place
        p.setGravity(0, 0, self.gravity)

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

    def render(
        self, multi_view: bool = True
    ) -> Union[Image.Image, Dict[str, Image.Image]]:
        """Render the current environment state."""
        if multi_view and self.multi_view:
            return self.multi_view_renderer.render_multi_view()
        else:
            return self.multi_view_renderer.render_single_view("perspective")

    def get_available_actions(self) -> List[str]:
        """Get list of available action types."""
        return [
            "pick",
            "release",
            "move",
            "place",
            "push",
            "pull",
            "rotate",
            "observe",
            "wait",
            "check_solution",
        ]

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for tool functions."""

        def build_schema(
            name: str, desc: str, properties: Dict[str, Any], required: List[str]
        ):
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
                "Pick an object by name into a virtual gripper. Optionally reset orientation to original state.",
                {
                    "object_id": {
                        "type": "string",
                        "description": "Name of object to pick",
                    },
                    "reset_orientation": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, reset object orientation to [0,0,0,1] (original/intended orientation before physics affected it)",
                    },
                },
                ["object_id"],
            ),
            build_schema(
                "release",
                "Release a previously picked object (removes constraint).",
                {
                    "object_id": {
                        "type": "string",
                        "description": "Name of object to release",
                    },
                },
                ["object_id"],
            ),
            build_schema(
                "move",
                "Move a picked object to a specific 3D position using constraints.",
                {
                    "object_id": {
                        "type": "string",
                        "description": "Name of picked object to move",
                    },
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Target position [x, y, z]",
                    },
                },
                ["object_id", "position"],
            ),
            build_schema(
                "place",
                "Place a picked object on top of a target object's top surface with optional stabilization.",
                {
                    "object_id": {"type": "string", "description": "Picked object to place"},
                    "target_id": {"type": "string", "description": "Target object to place onto"},
                    "offset_xy": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "default": [0.0, 0.0],
                        "description": "Lateral offset on target top plane [dx, dy]",
                    },
                    "offset_local": {
                        "type": "boolean",
                        "default": True,
                        "description": "Interpret offset_xy in target local frame",
                    },
                    "align_orientation": {
                        "type": "string",
                        "enum": ["keep", "align_target", "snap_90"],
                        "default": "align_target",
                        "description": "Yaw alignment policy",
                    },
                    "clearance": {
                        "type": "number",
                        "default": 0.001,
                        "description": "Vertical clearance above target (m)",
                    },
                    "release_after": {
                        "type": "boolean",
                        "default": False,
                        "description": "Release after placement (free-physics only)",
                    },
                    "stabilize_target": {
                        "type": "boolean",
                        "default": True,
                        "description": "Temporarily hold target to world during placement",
                    },
                    "hold_max_force": {
                        "type": "number",
                        "default": 300.0,
                        "description": "Max force for temporary target hold",
                    },
                    "hold_erp": {
                        "type": "number",
                        "default": 0.4,
                        "description": "ERP for temporary target hold",
                    },
                },
                ["object_id", "target_id"],
            ),
            build_schema(
                "push",
                "Apply a pushing force to an object.",
                {
                    "object_id": {
                        "type": "string",
                        "description": "Name of object to push",
                    },
                    "force": {
                        "type": "number",
                        "default": 10.0,
                        "description": "Push force magnitude",
                    },
                    "direction": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "default": [1, 0, 0],
                        "description": "Push direction [x, y, z]",
                    },
                },
                ["object_id"],
            ),
            build_schema(
                "rotate",
                "Rotate an object to a specific orientation.",
                {
                    "object_id": {
                        "type": "string",
                        "description": "Name of object to rotate",
                    },
                    "axis": {
                        "type": "string",
                        "enum": ["x", "y", "z"],
                        "description": "Rotation axis",
                    },
                    "angle": {
                        "type": "number",
                        "description": "Rotation angle in radians",
                    },
                },
                ["object_id", "axis", "angle"],
            ),
            build_schema(
                "observe",
                "Observe the scene from a rotated camera angle.",
                {
                    "angle": {
                        "type": "number",
                        "default": 0.0,
                        "description": "Camera rotation angle in radians",
                    },
                },
                [],
            ),
            build_schema(
                "wait",
                "Wait for a specified duration to let physics settle.",
                {
                    "duration": {
                        "type": "number",
                        "default": 1.0,
                        "description": "Wait duration in seconds",
                    },
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

    def execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        disable_second_hand: bool = False,
    ) -> Dict[str, Any]:
        """Execute a tool call from VLM.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            disable_second_hand: Deprecated and ignored.
        """
        try:
            # Base tool implementations
            if tool_name == "pick":
                return self._tool_pick(
                    arguments.get("object_id"),
                    arguments.get("reset_orientation", False),
                )
            elif tool_name == "release":
                return self._tool_release(arguments.get("object_id"))
            elif tool_name == "move":
                return self._tool_move(arguments.get("object_id"), arguments.get("position"))
            elif tool_name == "place":
                return self._tool_place(
                    arguments.get("object_id"),
                    arguments.get("target_id"),
                    arguments.get("offset_xy", [0.0, 0.0]),
                    arguments.get("offset_local", True),
                    arguments.get("align_orientation", "align_target"),
                    arguments.get("clearance", 0.001),
                    arguments.get("release_after", False),
                    arguments.get("stabilize_target", True),
                    arguments.get("hold_max_force", 300.0),
                    arguments.get("hold_erp", 0.4),
                )
            elif tool_name == "push":
                return self._tool_push(
                    arguments.get("object_id"),
                    arguments.get("force", 10.0),
                    arguments.get("direction", [1, 0, 0]),
                )
            elif tool_name == "rotate":
                return self._tool_rotate(
                    arguments.get("object_id"), arguments.get("axis", "z"), arguments.get("angle", 0.0)
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

    def _execute_task_specific_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        return State(step=self.step_count, objects={}, completed=False, success=False)

    def _create_observation(self) -> Observation:
        """Create observation for VLM."""
        # Render images
        if self.multi_view:
            main_image = self.multi_view_renderer.render_multi_view()
            multi_view_images = {
                "front": self.multi_view_renderer.render_single_view("front"),
                "side": self.multi_view_renderer.render_single_view("side"),
                "top": self.multi_view_renderer.render_single_view("top"),
            }
        else:
            main_image = self.multi_view_renderer.render_single_view("perspective")
            multi_view_images = None

        return Observation(
            image=main_image,
            state=self.current_state,
            description=self._get_state_description(),
            available_actions=self.get_available_actions(),
            multi_view_images=multi_view_images,
        )

    def _get_state_description(self) -> str:
        """Get textual description of current state."""
        return (
            f"Step {self.step_count}: Environment contains {len(self.objects)} objects."
        )

    def _is_done(self) -> bool:
        """Check if episode is done."""
        return self.step_count >= self.max_steps or (
            self.current_state and self.current_state.completed
        )

    # Enhanced tool implementations
    def add_object(
        self,
        object_name: str,
        urdf_path: str,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        object_type: str = "block",
        properties: Dict[str, Any] = None,
    ) -> int:
        """Add object to the environment."""
        if properties is None:
            properties = {}

        object_id = p.loadURDF(
            urdf_path, basePosition=position, baseOrientation=orientation
        )

        self.objects[object_name] = ObjectInfo(
            object_id=object_id,
            name=object_name,
            position=position,
            orientation=orientation,
            object_type=object_type,
            properties=properties,
        )

        return object_id

    def create_primitive_object(
        self,
        object_name: str,
        shape_type: str = "box",
        size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        position: Tuple[float, float, float] = (0, 0, 0.5),
        color: Tuple[float, float, float, float] = (1, 0, 0, 1),
        mass: float = 1.0,
    ) -> int:
        """Create primitive object (box, sphere, cylinder)."""

        # Create collision shape
        if shape_type == "box":
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        elif shape_type == "sphere":
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=size[0])
        elif shape_type == "cylinder":
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=size[0], height=size[1]
            )
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        # Create visual shape
        if shape_type == "box":
            visual_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=size, rgbaColor=color
            )
        elif shape_type == "sphere":
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE, radius=size[0], rgbaColor=color
            )
        elif shape_type == "cylinder":
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER, radius=size[0], length=size[1], rgbaColor=color
            )
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        # Create multi-body
        object_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )

        # Store object info
        self.objects[object_name] = ObjectInfo(
            object_id=object_id,
            name=object_name,
            position=position,
            orientation=(0, 0, 0, 1),
            object_type=shape_type,
            properties={"mass": mass, "color": color, "size": size},
        )

        return object_id

    # Tool implementations
    def _tool_pick(
        self, object_id: str, reset_orientation: bool = False
    ) -> Dict[str, Any]:
        """Pick tool implementation using constraints.

        Args:
            object_id: Name of object to pick
            reset_orientation: If True, reset object orientation to [0,0,0,1] (original/intended orientation)
        """
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}

        # Implement single-pick constraint with auto-swap
        auto_swapped = False
        if object_id in self.picked_objects:
            return {
                "status": "error",
                "message": f"Object {object_id} is already picked",
            }
        elif self.picked_objects:
            # Auto-release currently picked object before picking new one
            old_object = next(iter(self.picked_objects))
            release_result = self._tool_release(old_object)
            auto_swapped = True
            if release_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"Failed to auto-release {old_object} before picking {object_id}: {release_result.get('message')}",
                }

        obj_info = self.objects[object_id]

        # Get current position and orientation
        current_pos, current_orn = p.getBasePositionAndOrientation(
            obj_info.object_id, physicsClientId=self.client_id
        )

        # Set target orientation based on reset_orientation parameter
        if reset_orientation:
            # Reset to identity quaternion [x, y, z, w] = [0, 0, 0, 1] (no rotation)
            target_orientation = [0, 0, 0, 1]
            orientation_msg = " with orientation reset to [0,0,0,1]"
        else:
            # Keep current orientation
            target_orientation = current_orn
            orientation_msg = ""

        # Create constraint to "pick" the object (similar to your luban script)
        constraint_id = p.createConstraint(
            parentBodyUniqueId=obj_info.object_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,  # World
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 1],  # Ignored for JOINT_FIXED
            parentFramePosition=[0, 0, 0],  # Grab at center
            childFramePosition=current_pos,
            childFrameOrientation=target_orientation,
            physicsClientId=self.client_id,
        )

        # Track the constraint
        self.active_constraints[object_id] = constraint_id
        self.picked_objects.add(object_id)

        # Create success message indicating auto-swap if it occurred
        success_msg = (
            f"Picked up {object_id} using constraint {constraint_id}{orientation_msg}"
        )
        if auto_swapped:
            success_msg = f"Auto-released {old_object}, then picked up {object_id} using constraint {constraint_id}{orientation_msg}"

        return {
            "status": "success",
            "message": success_msg,
            "picked_objects": list(self.picked_objects),
            "constraint_id": constraint_id,
            "auto_swapped": auto_swapped,
            "reset_orientation": reset_orientation,
        }

    def get_currently_picked_object(self) -> Optional[str]:
        """Get the currently picked object, or None if nothing is picked."""
        return next(iter(self.picked_objects)) if self.picked_objects else None

    def _tool_release(self, object_id: str) -> Dict[str, Any]:
        """Release tool implementation - removes constraint."""
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}

        if object_id not in self.picked_objects:
            return {"status": "error", "message": f"Object {object_id} is not picked"}

        # Remove the constraint
        constraint_id = self.active_constraints.get(object_id)
        if constraint_id is not None:
            p.removeConstraint(constraint_id, physicsClientId=self.client_id)
            del self.active_constraints[object_id]

        self.picked_objects.remove(object_id)

        return {
            "status": "success",
            "message": f"Released {object_id} (constraint {constraint_id})",
            "picked_objects": list(self.picked_objects),
        }

    def _tool_place(
        self,
        object_id: str,
        target_id: str,
        offset_xy: List[float] = None,
        offset_local: bool = True,
        align_orientation: str = "align_target",
        clearance: float = 0.001,
        release_after: bool = False,
        stabilize_target: bool = True,
        hold_max_force: float = 300.0,
        hold_erp: float = 0.4,
    ) -> Dict[str, Any]:
        """Place a picked object on top of a target object's top surface.

        Applies a temporary world-fixed hold on the target during placement when
        stabilize_target is true. No auto-release; uses release_after flag.
        """
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}
        if not target_id or target_id not in self.objects:
            return {"status": "error", "message": f"Target {target_id} not found"}
        if object_id not in self.picked_objects:
            return {
                "status": "error",
                "message": f"Object {object_id} must be picked before placing",
            }

        if offset_xy is None:
            offset_xy = [0.0, 0.0]

        obj_info = self.objects[object_id]
        tgt_info = self.objects[target_id]

        # Get target top (AABB max.z) and XY center
        t_min, t_max = p.getAABB(tgt_info.object_id)
        top_z = t_max[2]
        t_xy = [(t_min[0] + t_max[0]) / 2.0, (t_min[1] + t_max[1]) / 2.0]

        # Object AABB and origin-offset correction
        o_min, o_max = p.getAABB(obj_info.object_id)
        half_h = (o_max[2] - o_min[2]) / 2.0
        o_center = [
            (o_min[0] + o_max[0]) / 2.0,
            (o_min[1] + o_max[1]) / 2.0,
            (o_min[2] + o_max[2]) / 2.0,
        ]
        o_pos, o_orn = p.getBasePositionAndOrientation(obj_info.object_id)
        delta_xy = [o_pos[0] - o_center[0], o_pos[1] - o_center[1]]
        delta_z = o_pos[2] - o_center[2]

        # Target yaw for local offset and orientation alignment
        t_pos, t_orn = p.getBasePositionAndOrientation(tgt_info.object_id)
        t_euler = p.getEulerFromQuaternion(t_orn)
        t_yaw = t_euler[2]

        # Compute world offset
        dx, dy = float(offset_xy[0]), float(offset_xy[1])
        if offset_local:
            import math
            cos_y = math.cos(t_yaw)
            sin_y = math.sin(t_yaw)
            wx = dx * cos_y - dy * sin_y
            wy = dx * sin_y + dy * cos_y
        else:
            wx, wy = dx, dy

        place_xy_world = [t_xy[0] + wx, t_xy[1] + wy]
        pivot_xy = [place_xy_world[0] + delta_xy[0], place_xy_world[1] + delta_xy[1]]
        pivot_z = top_z + half_h + float(clearance) + delta_z

        # Orientation policy (yaw only)
        import math

        def set_yaw(base_quat, yaw):
            cur_euler = p.getEulerFromQuaternion(base_quat, physicsClientId=self.client_id)
            new_euler = [cur_euler[0], cur_euler[1], yaw]
            return p.getQuaternionFromEuler(new_euler, physicsClientId=self.client_id)

        if align_orientation == "keep":
            new_orn = o_orn
        elif align_orientation == "align_target":
            new_orn = set_yaw(o_orn, t_yaw)
        elif align_orientation == "snap_90":
            cur_euler = p.getEulerFromQuaternion(o_orn, physicsClientId=self.client_id)
            rel = cur_euler[2] - t_yaw
            snapped = round(rel / (math.pi / 2)) * (math.pi / 2) + t_yaw
            new_orn = set_yaw(o_orn, snapped)
        else:
            return {"status": "error", "message": f"Invalid align_orientation: {align_orientation}"}

        # Optional target stabilization
        temp_hold_cid = None
        try:
            if stabilize_target:
                temp_hold_cid = p.createConstraint(
                    parentBodyUniqueId=tgt_info.object_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=-1,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=t_pos,
                    parentFrameOrientation=[0, 0, 0, 1],
                    childFrameOrientation=t_orn,
                    physicsClientId=self.client_id,
                )
                p.changeConstraint(
                    temp_hold_cid,
                    maxForce=float(hold_max_force),
                    erp=float(hold_erp),
                    physicsClientId=self.client_id,
                )

            # Move picked object via its constraint
            constraint_id = self.active_constraints.get(object_id)
            if constraint_id is None:
                return {
                    "status": "error",
                    "message": f"No pick constraint found for {object_id}",
                }

            p.changeConstraint(
                userConstraintUniqueId=constraint_id,
                jointChildPivot=[pivot_xy[0], pivot_xy[1], pivot_z],
                jointChildFrameOrientation=new_orn,
                maxForce=100000,
                physicsClientId=self.client_id,
            )

            # Simple penetration guard and settle
            def penetration_sum():
                contacts = p.getContactPoints(physicsClientId=self.client_id)
                return sum(abs(c[8]) for c in contacts if c[8] < 0)

            prev_pen = penetration_sum()
            for _ in range(10):
                p.stepSimulation(physicsClientId=self.client_id)
                if self.gui:
                    time.sleep(0.01)
            cur_pen = penetration_sum()
            if cur_pen > prev_pen + 1e-6:
                pivot_z += max(0.0005, float(clearance) * 0.5)
                p.changeConstraint(
                    userConstraintUniqueId=constraint_id,
                    jointChildPivot=[pivot_xy[0], pivot_xy[1], pivot_z],
                    jointChildFrameOrientation=new_orn,
                    maxForce=100000,
                    physicsClientId=self.client_id,
                )
                for _ in range(10):
                    p.stepSimulation(physicsClientId=self.client_id)
                    if self.gui:
                        time.sleep(0.01)

        finally:
            if temp_hold_cid is not None:
                try:
                    p.removeConstraint(temp_hold_cid, physicsClientId=self.client_id)
                except Exception:
                    pass

        # Optionally release after placement
        released = False
        if release_after:
            rel = self._tool_release(object_id)
            released = rel.get("status") == "success"

        # Update cached pose
        obj_info.position = (pivot_xy[0], pivot_xy[1], pivot_z)
        obj_info.orientation = tuple(new_orn)

        msg = f"Placed {object_id} on top of {target_id}"
        if released:
            msg += " and released"
        return {
            "status": "success",
            "message": msg,
            "final_position": obj_info.position,
            "final_orientation": obj_info.orientation,
        }

    def _tool_move(self, object_id: str, position: List[float]) -> Dict[str, Any]:
        """Move tool implementation."""
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}

        if not position or len(position) != 3:
            return {"status": "error", "message": "Invalid position"}

        return self._tool_move_simple(object_id, position)

    def _tool_move_simple(
        self, object_id: str, position: List[float]
    ) -> Dict[str, Any]:
        """Constraint-based move implementation."""
        obj_info = self.objects[object_id]

        # Check if object is picked (has constraint)
        if object_id not in self.picked_objects:
            return {
                "status": "error",
                "message": f"Object {object_id} must be picked first",
            }

        constraint_id = self.active_constraints.get(object_id)
        if constraint_id is None:
            return {
                "status": "error",
                "message": f"No constraint found for {object_id}",
            }

        # Use changeConstraint to move the picked object (like your luban script)
        p.changeConstraint(
            userConstraintUniqueId=constraint_id,
            jointChildPivot=position,
            maxForce=100000,  # Strong force to ensure movement
            physicsClientId=self.client_id,
        )

        # Let physics settle
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)
            time.sleep(0.01)

        # Update object info
        obj_info.position = tuple(position)

        # In free-physics mode, keep the pick active; VLM should call release explicitly
        return {"status": "success", "message": f"Moved {object_id} to {position}"}

    # Removed legacy _tool_move_with_holding

    def _tool_push(
        self, object_id: str, force: float, direction: List[float]
    ) -> Dict[str, Any]:
        """Push tool implementation."""
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}

        return self._tool_push_simple(object_id, force, direction)

    def _tool_push_simple(
        self, object_id: str, force: float, direction: List[float]
    ) -> Dict[str, Any]:
        """Simple push without automatic holding (original implementation)."""
        obj_info = self.objects[object_id]

        # Normalize direction
        direction = np.array(direction)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1, 0, 0])

        # Apply force
        p.applyExternalForce(
            obj_info.object_id,
            -1,
            [force * direction[0], force * direction[1], force * direction[2]],
            obj_info.position,
            p.WORLD_FRAME,
        )

        return {
            "status": "success",
            "message": f"Pushed {object_id} with force {force}",
        }

    # Removed legacy automatic holding for push

    def _tool_rotate(self, object_id: str, axis: str, angle: float) -> Dict[str, Any]:
        """Rotate tool implementation."""
        if not object_id or object_id not in self.objects:
            return {"status": "error", "message": f"Object {object_id} not found"}
        return self._tool_rotate_simple(object_id, axis, angle)

    def _tool_rotate_simple(
        self, object_id: str, axis: str, angle: float
    ) -> Dict[str, Any]:
        """Constraint-based rotate implementation."""
        obj_info = self.objects[object_id]

        # Check if object is picked (has constraint)
        if object_id not in self.picked_objects:
            return {
                "status": "error",
                "message": f"Object {object_id} must be picked first",
            }

        constraint_id = self.active_constraints.get(object_id)
        if constraint_id is None:
            return {
                "status": "error",
                "message": f"No constraint found for {object_id}",
            }

        # Get current orientation
        current_orient = obj_info.orientation
        current_euler = p.getEulerFromQuaternion(
            current_orient, physicsClientId=self.client_id
        )

        # Apply rotation
        if axis.lower() == "x":
            new_euler = [current_euler[0] + angle, current_euler[1], current_euler[2]]
        elif axis.lower() == "y":
            new_euler = [current_euler[0], current_euler[1] + angle, current_euler[2]]
        elif axis.lower() == "z":
            new_euler = [current_euler[0], current_euler[1], current_euler[2] + angle]
        else:
            return {"status": "error", "message": f"Invalid axis: {axis}"}

        new_orientation = p.getQuaternionFromEuler(
            new_euler, physicsClientId=self.client_id
        )

        # Use changeConstraint to apply new orientation
        p.changeConstraint(
            userConstraintUniqueId=constraint_id,
            jointChildPivot=obj_info.position,  # Keep same position
            jointChildFrameOrientation=new_orientation,  # Change orientation
            maxForce=100000,
            physicsClientId=self.client_id,
        )

        # Let physics settle
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)
            time.sleep(0.01)

        # Update object info
        obj_info.orientation = new_orientation

        # In free-physics mode, keep the pick active; VLM should call release explicitly
        return {
            "status": "success",
            "message": f"Rotated {object_id} {angle} radians around {axis}-axis",
        }

    # Removed legacy automatic holding for rotate

    def _tool_observe(self, angle: float) -> Dict[str, Any]:
        """Observe tool implementation."""
        # Rotate camera view
        import math

        radius = 1.5
        height = 1.0
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        self.multi_view_renderer.set_camera_config(
            "perspective", [x, y, height], [0, 0, 0.3]
        )

        return {
            "status": "success",
            "message": f"Observing from angle {angle:.2f} radians",
        }

    def _tool_wait(self, duration: float) -> Dict[str, Any]:
        """Wait tool implementation."""
        steps = int(duration * 240)  # 240 Hz simulation
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

        return {"status": "success", "message": f"Waited for {duration} seconds"}

    def _tool_check_solution(self) -> Dict[str, Any]:
        """Check solution tool implementation."""
        success = self._evaluate_success()

        return {"status": "success", "solved": success, "message": "Solution checked"}

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
            "properties": obj_info.properties,
        }

    def get_all_objects_state(self) -> Dict[str, Dict[str, Any]]:
        """Get current state of all objects."""
        return {name: self.get_object_state(name) for name in self.objects.keys()}

    def get_contact_points(
        self, obj1_name: str, obj2_name: str = None
    ) -> List[Dict[str, Any]]:
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
            contact_list.append(
                {
                    "position_on_a": contact[5],
                    "position_on_b": contact[6],
                    "normal_on_b": contact[7],
                    "distance": contact[8],
                    "normal_force": contact[9],
                }
            )

        return contact_list

    # Legacy holding state helpers removed
