"""
PhyVPuzzle Environment

This module provides specialized physics puzzle environments for VLM evaluation.
Supports Luban lock and Pagoda puzzle tasks with complex manipulation requirements.
"""

import os
import shutil
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

from .physics_env import (
    PhysicsEnvironment, 
    CameraConfig, 
    ObjectInfo, 
    TaskDefinition,
    ExecutionStep,
    BenchmarkResult
)
from ..core.translator import EnvironmentCommand
from ..core.action_descriptor import ParsedAction


class PuzzleType(Enum):
    """Types of physics puzzles."""
    LUBAN_LOCK = "luban_lock"
    PAGODA = "pagoda"


@dataclass
class PhyVPuzzleConfig:
    """Configuration for PhyVPuzzle environments."""
    puzzle_type: PuzzleType
    urdf_base_path: str
    meshes_path: str
    initial_camera_config: CameraConfig
    max_steps: int = 100
    time_limit: float = 600.0
    success_criteria: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.success_criteria is None:
            self.success_criteria = {}


@dataclass
class PuzzleObjectConfig:
    """Configuration for a puzzle object."""
    object_name: str
    object_id: int
    urdf_path: str
    initial_position: Tuple[float, float, float]
    initial_orientation: Tuple[float, float, float, float]
    object_type: str
    properties: Dict[str, Any]
    movable: bool = True
    collision_meshes: List[str] = None
    visual_mesh: str = None


class PhyVPuzzleEnvironment(PhysicsEnvironment):
    """
    Specialized physics environment for complex puzzle manipulation tasks.
    Supports VLM-based reasoning with observation-action-feedback loops.
    """
    
    def __init__(self, config: PhyVPuzzleConfig, gui: bool = False):
        super().__init__(gui)
        self.config = config
        self.puzzle_objects = {}
        self.puzzle_state = "initial"
        self.step_count = 0
        self.start_time = None
        self.task_completed = False
        self.last_action_success = True
        self.interaction_history = []
        
        # Initialize camera with puzzle-specific config
        self.camera_config = config.initial_camera_config
        
    def setup_environment(self) -> None:
        """Setup PyBullet environment for puzzle simulation."""
        # Connect to PyBullet
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        
        # Set physics parameters
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1./240.)
        p.setRealTimeSimulation(0)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Setup puzzle-specific environment
        if self.config.puzzle_type == PuzzleType.LUBAN_LOCK:
            self._setup_luban_lock()
        elif self.config.puzzle_type == PuzzleType.PAGODA:
            self._setup_pagoda()
        
        self.start_time = time.time()
        print(f"PhyVPuzzle environment setup complete: {self.config.puzzle_type.value}")
    
    def _setup_luban_lock(self) -> None:
        """Setup Luban lock puzzle environment."""
        # Parse URDF to get object configurations
        urdf_path = os.path.join(self.config.urdf_base_path, "luban-simple-prismatic/base_link/urdf/base_link.urdf")
        object_configs = self._parse_luban_urdf(urdf_path)
        
        # Create individual URDF files and load objects
        temp_dir = "./temp_phyvpuzzle_urdfs"
        os.makedirs(temp_dir, exist_ok=True)
        
        for obj_name, config in object_configs.items():
            try:
                # Create individual URDF
                urdf_path = self._create_luban_object_urdf(obj_name, config, temp_dir)
                
                # Load object
                obj_id = p.loadURDF(
                    urdf_path,
                    basePosition=config['position'],
                    baseOrientation=p.getQuaternionFromEuler(config['orientation']),
                    useFixedBase=True,  # Start with fixed base for initial setup
                    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
                )
                
                self.puzzle_objects[obj_name] = PuzzleObjectConfig(
                    object_name=obj_name,
                    object_id=obj_id,
                    urdf_path=urdf_path,
                    initial_position=config['position'],
                    initial_orientation=p.getQuaternionFromEuler(config['orientation']),
                    object_type="luban_piece",
                    properties={"obj_num": config['obj_num']},
                    movable=True
                )
                
            except Exception as e:
                print(f"Error loading {obj_name}: {e}")
        
        # Set camera for Luban lock
        self.camera_config = CameraConfig(
            position=(-1.5, 2.2, 1.8),
            target=(0, 0, 0.5),
            fov=60.0,
            image_width=512,
            image_height=512
        )
    
    def _setup_pagoda(self) -> None:
        """Setup Pagoda puzzle environment."""
        urdf_path = os.path.join(
            self.config.urdf_base_path, 
            "lego-pagoda-setup-v2/urdf/test-pagoda-urdf-auto-3.urdf"
        )
        
        try:
            # Load pagoda URDF
            self.pagoda_id = p.loadURDF(
                urdf_path,
                useFixedBase=False,
                flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
            )
            
            # Get joint information for movable pole
            num_joints = p.getNumJoints(self.pagoda_id)
            self.joint_indices = {}
            
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.pagoda_id, i)
                joint_name = joint_info[1].decode("utf-8")
                
                if "Pole.001_link" in joint_name:
                    self.joint_indices[joint_name] = i
            
            # Store object info
            self.puzzle_objects["pagoda_base"] = PuzzleObjectConfig(
                object_name="pagoda_base",
                object_id=self.pagoda_id,
                urdf_path=urdf_path,
                initial_position=(0, 0, 0),
                initial_orientation=(0, 0, 0, 1),
                object_type="pagoda",
                properties={"joint_indices": self.joint_indices},
                movable=False
            )
            
        except Exception as e:
            print(f"Error loading pagoda: {e}")
        
        # Set camera for Pagoda
        self.camera_config = CameraConfig(
            position=(2.0, -2.0, 2.0),
            target=(0, 0, 1.0),
            fov=60.0,
            image_width=512,
            image_height=512
        )
    
    def _parse_luban_urdf(self, urdf_path: str) -> Dict[str, Dict]:
        """Parse Luban URDF to extract object configurations."""
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        objects = {}
        
        for joint in root.findall('.//joint'):
            joint_name = joint.get('name')
            if joint_name and joint_name.startswith('obj_') and joint_name.endswith('_joint'):
                obj_num = int(joint_name.split('_')[1])
                obj_name = f"obj_{obj_num}"
                
                origin = joint.find('origin')
                xyz = [float(x) for x in origin.get('xyz').split()]
                rpy = [float(x) for x in origin.get('rpy').split()]
                
                child_link = joint.find('child').get('link')
                
                objects[obj_name] = {
                    'position': xyz,
                    'orientation': rpy,
                    'link_name': child_link,
                    'obj_num': obj_num
                }
        
        return objects
    
    def _create_luban_object_urdf(self, obj_name: str, config: Dict, output_dir: str) -> str:
        """Create individual URDF for Luban object."""
        obj_num = config['obj_num']
        
        # Get mesh files (simplified - would need full mesh mapping)
        visual_mesh = f"luban-lock-sliding.stl" if obj_num == 1 else f"luban-lock-sliding.{obj_num-1:03d}.stl"
        
        # Create simplified URDF content
        urdf_content = f'''<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="{obj_name}_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{self.config.meshes_path}/{visual_mesh}" scale="0.05 0.05 0.05"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>'''
        
        urdf_path = os.path.join(output_dir, f"{obj_name}.urdf")
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
        
        return urdf_path
    
    def execute_command(self, command: EnvironmentCommand) -> bool:
        """Execute a command in the puzzle environment."""
        try:
            self.step_count += 1
            
            if command.command_type == "move_piece":
                return self._move_puzzle_piece(command.parameters)
            elif command.command_type == "rotate_piece":
                return self._rotate_puzzle_piece(command.parameters)
            elif command.command_type == "slide_piece":
                return self._slide_puzzle_piece(command.parameters)
            elif command.command_type == "lift_piece":
                return self._lift_puzzle_piece(command.parameters)
            elif command.command_type == "insert_piece":
                return self._insert_puzzle_piece(command.parameters)
            elif command.command_type == "remove_piece":
                return self._remove_puzzle_piece(command.parameters)
            elif command.command_type == "check_solution":
                return self._check_puzzle_solution()
            elif command.command_type == "reset_puzzle":
                return self._reset_puzzle_state()
            else:
                # Fallback to base class commands
                return super().execute_command(command)
                
        except Exception as e:
            print(f"Error executing puzzle command {command.command_type}: {e}")
            self.last_action_success = False
            return False
    
    def _move_puzzle_piece(self, params: Dict[str, Any]) -> bool:
        """Move a puzzle piece to a target position."""
        piece_name = params.get("piece_id")
        target_position = params.get("target_position")
        
        if piece_name not in self.puzzle_objects or not target_position:
            return False
        
        piece = self.puzzle_objects[piece_name]
        
        if not piece.movable:
            return False
        
        # Smooth movement animation
        current_pos, current_orn = p.getBasePositionAndOrientation(piece.object_id)
        steps = 20
        
        for i in range(steps):
            alpha = (i + 1) / steps
            new_pos = [
                current_pos[j] + alpha * (target_position[j] - current_pos[j])
                for j in range(3)
            ]
            
            p.resetBasePositionAndOrientation(piece.object_id, new_pos, current_orn)
            p.stepSimulation()
            time.sleep(0.01)
        
        self.last_action_success = True
        return True
    
    def _rotate_puzzle_piece(self, params: Dict[str, Any]) -> bool:
        """Rotate a puzzle piece."""
        piece_name = params.get("piece_id")
        rotation_axis = params.get("rotation_axis", "z")
        rotation_angle = params.get("rotation_angle", 90.0)
        
        if piece_name not in self.puzzle_objects:
            return False
        
        piece = self.puzzle_objects[piece_name]
        
        if not piece.movable:
            return False
        
        current_pos, current_orn = p.getBasePositionAndOrientation(piece.object_id)
        current_euler = p.getEulerFromQuaternion(current_orn)
        
        # Apply rotation
        new_euler = list(current_euler)
        if rotation_axis == "x":
            new_euler[0] += np.radians(rotation_angle)
        elif rotation_axis == "y":
            new_euler[1] += np.radians(rotation_angle)
        elif rotation_axis == "z":
            new_euler[2] += np.radians(rotation_angle)
        
        new_orn = p.getQuaternionFromEuler(new_euler)
        p.resetBasePositionAndOrientation(piece.object_id, current_pos, new_orn)
        
        # Step simulation
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.01)
        
        self.last_action_success = True
        return True
    
    def _slide_puzzle_piece(self, params: Dict[str, Any]) -> bool:
        """Slide a puzzle piece along a direction."""
        piece_name = params.get("piece_id")
        slide_direction = params.get("slide_direction")
        slide_distance = params.get("slide_distance", 0.1)
        
        if piece_name not in self.puzzle_objects or not slide_direction:
            return False
        
        piece = self.puzzle_objects[piece_name]
        current_pos, current_orn = p.getBasePositionAndOrientation(piece.object_id)
        
        # Calculate target position
        target_pos = [
            current_pos[i] + slide_direction[i] * slide_distance
            for i in range(3)
        ]
        
        return self._move_puzzle_piece({
            "piece_id": piece_name,
            "target_position": target_pos
        })
    
    def _lift_puzzle_piece(self, params: Dict[str, Any]) -> bool:
        """Lift a puzzle piece vertically."""
        return self._slide_puzzle_piece({
            **params,
            "slide_direction": [0, 0, 1],
            "slide_distance": params.get("lift_height", 0.05)
        })
    
    def _insert_puzzle_piece(self, params: Dict[str, Any]) -> bool:
        """Insert a puzzle piece into a slot or position."""
        # This would implement complex insertion logic
        piece_name = params.get("piece_id")
        target_slot = params.get("target_slot")
        
        if piece_name not in self.puzzle_objects:
            return False
        
        # Simplified insertion - just move to slot position
        slot_position = params.get("slot_position")
        if slot_position:
            return self._move_puzzle_piece({
                "piece_id": piece_name,
                "target_position": slot_position
            })
        
        return False
    
    def _remove_puzzle_piece(self, params: Dict[str, Any]) -> bool:
        """Remove a puzzle piece from its current position."""
        piece_name = params.get("piece_id")
        removal_direction = params.get("removal_direction", [0, 0, 1])
        removal_distance = params.get("removal_distance", 0.2)
        
        return self._slide_puzzle_piece({
            "piece_id": piece_name,
            "slide_direction": removal_direction,
            "slide_distance": removal_distance
        })
    
    def _check_puzzle_solution(self) -> bool:
        """Check if the puzzle is solved."""
        if self.config.puzzle_type == PuzzleType.LUBAN_LOCK:
            return self._check_luban_solution()
        elif self.config.puzzle_type == PuzzleType.PAGODA:
            return self._check_pagoda_solution()
        return False
    
    def _check_luban_solution(self) -> bool:
        """Check if Luban lock is solved (pieces properly interlocked)."""
        # Simplified check - would need complex geometric validation
        collision_count = 0
        piece_ids = [obj.object_id for obj in self.puzzle_objects.values()]
        
        for i in range(len(piece_ids)):
            for j in range(i + 1, len(piece_ids)):
                contacts = p.getContactPoints(piece_ids[i], piece_ids[j])
                if contacts:
                    collision_count += 1
        
        # If most pieces are in contact, consider it solved
        expected_contacts = len(piece_ids) - 1
        success = collision_count >= expected_contacts * 0.7
        
        if success:
            self.task_completed = True
            self.puzzle_state = "solved"
        
        return success
    
    def _check_pagoda_solution(self) -> bool:
        """Check if Pagoda puzzle is solved (pieces in correct configuration)."""
        # Simplified check for pagoda stability/configuration
        if self.pagoda_id is None:
            return False
        
        # Check if all pieces are stable (not moving)
        linear_vel, angular_vel = p.getBaseVelocity(self.pagoda_id)
        is_stable = (np.linalg.norm(linear_vel) < 0.01 and 
                    np.linalg.norm(angular_vel) < 0.01)
        
        if is_stable:
            self.task_completed = True
            self.puzzle_state = "solved"
        
        return is_stable
    
    def _reset_puzzle_state(self) -> bool:
        """Reset puzzle to initial state."""
        for obj_name, piece in self.puzzle_objects.items():
            p.resetBasePositionAndOrientation(
                piece.object_id,
                piece.initial_position,
                piece.initial_orientation
            )
            p.resetBaseVelocity(piece.object_id, [0, 0, 0], [0, 0, 0])
        
        self.puzzle_state = "initial"
        self.task_completed = False
        self.step_count = 0
        self.start_time = time.time()
        
        # Let physics settle
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)
        
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get current puzzle environment state."""
        state = {
            "timestep": self.timestep,
            "step_count": self.step_count,
            "puzzle_type": self.config.puzzle_type.value,
            "puzzle_state": self.puzzle_state,
            "task_completed": self.task_completed,
            "last_action_success": self.last_action_success,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
            "pieces": {},
            "success_criteria": self.config.success_criteria
        }
        
        # Update piece positions and states
        for name, piece in self.puzzle_objects.items():
            pos, orn = p.getBasePositionAndOrientation(piece.object_id)
            lin_vel, ang_vel = p.getBaseVelocity(piece.object_id)
            
            state["pieces"][name] = {
                "position": pos,
                "orientation": orn,
                "linear_velocity": lin_vel,
                "angular_velocity": ang_vel,
                "type": piece.object_type,
                "movable": piece.movable,
                "properties": piece.properties
            }
        
        return state
    
    def render(self) -> Image.Image:
        """Render the puzzle environment."""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_config.position,
            cameraTargetPosition=self.camera_config.target,
            cameraUpVector=self.camera_config.up_vector
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_config.fov,
            aspect=self.camera_config.aspect_ratio,
            nearPlane=self.camera_config.near_plane,
            farPlane=self.camera_config.far_plane
        )
        
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_config.image_width,
            height=self.camera_config.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]
        rgb_array = rgb_array.reshape(height, width, 3)
        
        return Image.fromarray(rgb_array)
    
    def get_observation_for_vlm(self) -> Dict[str, Any]:
        """Get formatted observation for VLM processing."""
        state = self.get_state()
        image = self.render()
        
        # Create text description of current state
        description = f"Puzzle Type: {state['puzzle_type']}\n"
        description += f"Current State: {state['puzzle_state']}\n"
        description += f"Step: {state['step_count']}\n"
        description += f"Task Completed: {state['task_completed']}\n"
        description += f"Last Action Success: {state['last_action_success']}\n\n"
        
        description += "Piece Positions:\n"
        for name, piece_info in state["pieces"].items():
            pos = piece_info["position"]
            description += f"- {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n"
        
        return {
            "image": image,
            "state_description": description,
            "raw_state": state,
            "available_actions": self._get_available_actions()
        }
    
    def _get_available_actions(self) -> List[str]:
        """Get list of available actions for current state."""
        actions = [
            "move_piece",
            "rotate_piece", 
            "slide_piece",
            "lift_piece",
            "insert_piece",
            "remove_piece",
            "check_solution",
            "reset_puzzle"
        ]
        
        if self.config.puzzle_type == PuzzleType.PAGODA:
            actions.extend([
                "move_pole_x",
                "move_pole_y", 
                "move_pole_z"
            ])
        
        return actions
    
    def is_task_complete(self) -> bool:
        """Check if the puzzle task is complete."""
        if self.task_completed:
            return True
        
        # Check time limit
        if self.start_time and (time.time() - self.start_time) > self.config.time_limit:
            return True
        
        # Check step limit
        if self.step_count >= self.config.max_steps:
            return True
        
        return False
    
    def get_success_status(self) -> Tuple[bool, str]:
        """Get success status and reason."""
        if self.task_completed:
            return True, "Puzzle solved successfully"
        elif self.start_time and (time.time() - self.start_time) > self.config.time_limit:
            return False, "Time limit exceeded"
        elif self.step_count >= self.config.max_steps:
            return False, "Step limit exceeded"
        else:
            return False, "Task in progress"
    
    def reset(self) -> None:
        """Reset the puzzle environment."""
        self._reset_puzzle_state()
        self.interaction_history.clear()
        print("Puzzle environment reset")
    
    def close(self) -> None:
        """Close the environment and clean up."""
        # Clean up temporary URDF files
        temp_dir = "./temp_phyvpuzzle_urdfs"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None
        
        print("PhyVPuzzle environment closed")


def create_luban_lock_environment(urdf_base_path: str, gui: bool = False) -> PhyVPuzzleEnvironment:
    """Create a Luban lock puzzle environment."""
    config = PhyVPuzzleConfig(
        puzzle_type=PuzzleType.LUBAN_LOCK,
        urdf_base_path=urdf_base_path,
        meshes_path=os.path.join(urdf_base_path, "luban-simple-prismatic/base_link/meshes/stl"),
        initial_camera_config=CameraConfig(
            position=(-1.5, 2.2, 1.8),
            target=(0, 0, 0.5),
            fov=60.0,
            image_width=512,
            image_height=512
        ),
        max_steps=100,
        time_limit=600.0,
        success_criteria={"interlocked_pieces": 8}
    )
    
    return PhyVPuzzleEnvironment(config, gui)


def create_pagoda_environment(urdf_base_path: str, gui: bool = False) -> PhyVPuzzleEnvironment:
    """Create a Pagoda puzzle environment."""
    config = PhyVPuzzleConfig(
        puzzle_type=PuzzleType.PAGODA,
        urdf_base_path=urdf_base_path,
        meshes_path=os.path.join(urdf_base_path, "lego-pagoda-setup-v2/meshes/stl"),
        initial_camera_config=CameraConfig(
            position=(2.0, -2.0, 2.0),
            target=(0, 0, 1.0),
            fov=60.0,
            image_width=512,
            image_height=512
        ),
        max_steps=50,
        time_limit=300.0,
        success_criteria={"stability_threshold": 0.01}
    )
    
    return PhyVPuzzleEnvironment(config, gui)