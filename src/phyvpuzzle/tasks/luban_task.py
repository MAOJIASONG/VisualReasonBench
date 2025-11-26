"""
Luban Lock disassembly task implementation.
"""
import os
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from phyvpuzzle.core import register_task, register_task_config, TaskConfig
from phyvpuzzle.core.base import ObjectInfo, TaskResult
from phyvpuzzle.tasks.base_task import PhysicsTask
from phyvpuzzle.environment.base_env import p

from phyvpuzzle.environment.luban_env import LubanEnvironment

@register_task_config("luban_disassembly")
@dataclass
class LubanTaskConfig(TaskConfig):
    urdf_root: str = "src/phyvpuzzle/environment/phobos_models/luban-3-piece"
    target_displacement_threshold: float = 0.08
    ruled_evaluation: bool = True

@register_task("luban_disassembly")
class LubanDisassemblyTask(PhysicsTask):
    """
    Task: Disassemble a Luban Lock.
    Loads pieces using RuntimeAssembly logic (Fixed Constraints).
    """

    def __init__(self, config: LubanTaskConfig):
        super().__init__(config)
        self.target_piece_id = -1
        self.initial_positions = {}
        # Fallback colors if URDF has none
        self.piece_colors = [
            (0.8, 0.1, 0.1, 1.0), (0.1, 0.8, 0.1, 1.0), (0.1, 0.1, 0.8, 1.0),
            (0.9, 0.8, 0.1, 1.0), (0.8, 0.4, 0.0, 1.0), (0.6, 0.0, 0.8, 1.0)
        ]

    def _calculate_optimal_steps(self) -> int:
        return 1

    def _configure_environment(self) -> None:
        """Load Luban pieces using RuntimeAssembly logic."""
        if not isinstance(self.environment, LubanEnvironment):
            raise ValueError("LubanTask requires LubanEnvironment")
        
        # 1. Path Resolution
        puzzle_base_path = self.config.urdf_root
        if not os.path.exists(puzzle_base_path):
             puzzle_base_path = os.path.join(os.getcwd(), self.config.urdf_root)
        
        available_urdfs = []
        if os.path.exists(puzzle_base_path):
            for item in sorted(os.listdir(puzzle_base_path)):
                piece_dir = os.path.join(puzzle_base_path, item)
                if os.path.isdir(piece_dir):
                    urdf_path = os.path.join(piece_dir, "urdf", f"{item}.urdf")
                    if os.path.exists(urdf_path):
                        available_urdfs.append(urdf_path)
        
        if not available_urdfs:
            print(f"⚠️ Warning: No URDFs at {puzzle_base_path}. Using fallback.")
            self._create_fallback_primitives()
            return

        # 2. Create Anchor (RuntimeAssembly Base)
        # Invisible static body to weld pieces to
        anchor_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=[0,0,0])

        # 3. Load Pieces
        center_pos = [0, 0, 0.5] 
        print(f"Loading {len(available_urdfs)} pieces...")
        
        for i, urdf_path in enumerate(available_urdfs):
            # 3.1 Load (Dynamic initially)
            obj_id = p.loadURDF(
                urdf_path,
                basePosition=center_pos,
                baseOrientation=[0, 0, 0, 1],
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                useFixedBase=False
            )
            
            # 3.2 Color Check (Priority: Existing -> Custom)
            vis_data = p.getVisualShapeData(obj_id)
            has_color = False
            if vis_data:
                # Check alpha channel > 0
                if vis_data[0][7][3] > 0.0: has_color = True
            
            if not has_color:
                color = self.piece_colors[i % len(self.piece_colors)]
                p.changeVisualShape(obj_id, -1, rgbaColor=color)

            # 3.3 Constraint Setup (RuntimeAssembly)
            # Stabilize dynamics first
            p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])
            p.changeDynamics(obj_id, -1, linearDamping=1.0, angularDamping=1.0)
            
            # Create Fixed Constraint
            curr_pos, curr_orn = p.getBasePositionAndOrientation(obj_id)
            cid = p.createConstraint(
                parentBodyUniqueId=anchor_id, parentLinkIndex=-1,
                childBodyUniqueId=obj_id, childLinkIndex=-1,
                jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
                parentFramePosition=curr_pos, parentFrameOrientation=curr_orn,
                childFramePosition=[0,0,0], childFrameOrientation=[0,0,0,1]
            )

            # 3.4 Register
            # Store constraints in properties for Env tool usage
            obj_info = ObjectInfo(
                object_id=obj_id,
                name=f"Luban_Piece_{i+1}",
                position=tuple(center_pos),
                orientation=(0,0,0,1),
                object_type="luban_piece",
                properties={
                    "index": i, "is_container": False,
                    "anchor_id": anchor_id,
                    "constraint_id": cid
                }
            )
            self.environment.objects.append(obj_info)
            self.initial_positions[obj_id] = center_pos
            
            if i == 0: self.target_piece_id = obj_id

        p.stepSimulation()
        print(f"✅ Loaded. Target ID: {self.target_piece_id}")

    def _create_fallback_primitives(self):
        """RuntimeAssembly primitives."""
        anchor_id = p.createMultiBody(baseMass=0, basePosition=[0,0,0])
        center_pos = [0, 0, 0.5]
        
        for i in range(3):
            pos = [center_pos[0] + i*0.01, center_pos[1], center_pos[2]]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.1, 0.02])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.1, 0.02], rgbaColor=self.piece_colors[i])
            
            # Use mass 1.0 but constrain it immediately
            obj_id = p.createMultiBody(1.0, col, vis, pos)
            
            cid = p.createConstraint(
                anchor_id, -1, obj_id, -1, p.JOINT_FIXED, [0,0,0],
                pos, [0,0,0,1], [0,0,0], [0,0,0,1]
            )
            
            self.environment.objects.append(ObjectInfo(
                obj_id, f"Prim_{i}", tuple(pos), (0,0,0,1), "luban_piece",
                {"anchor_id": anchor_id, "constraint_id": cid, "is_container": False}
            ))
            self.initial_positions[obj_id] = pos
            if i == 0: self.target_piece_id = obj_id

    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        if not task_results: return []
        latest = task_results[-1]
        
        if self.target_piece_id == -1:
            latest.success = False; return task_results

        try:
            curr, _ = p.getBasePositionAndOrientation(self.target_piece_id)
            start = self.initial_positions[self.target_piece_id]
            dist = np.linalg.norm(np.array(curr) - np.array(start))
        except:
            latest.success = False; return task_results

        thresh = self.config.target_displacement_threshold
        success = dist > thresh
        latest.success = success
        latest.feedback = f"Displacement: {dist:.4f}m (Goal > {thresh}m). {'✅' if success else '❌'}"
        return task_results

    def _get_initial_system_prompt(self) -> str:
        return """You are a spatial reasoning expert solving a "Luban Lock".
OBJECTS:
- Pieces are mechanically interlocked and held STATIC.
- You must use the tool to move them.

TOOL:
`manipulate_piece(object_id, move_x_steps, ...)`
- 1 Move Step = 1 cm.
- 1 Rotation Step = 5 degrees.
- Physics simulation runs during movement to prevent clipping.

GOAL:
Identify the KEY piece (unblocked) and slide it out.
"""

    def _get_initial_instruction(self) -> str:
        if self.target_piece_id != -1:
            return "The puzzle is LOCKED. Find the Key piece and slide it out by 10-15 steps (10-15cm) along its free axis."
        return "Error."