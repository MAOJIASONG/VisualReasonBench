
"""
unified_puzzle_env.py

A unified PyBullet environment for:
  1) Grid Stacking / Packing with discrete placement & rotation.
  2) Kongming/lock-style puzzles with discrete sliding primitives and a small searcher.

Key ideas:
- Discrete, numerically-stable action space (snap to grid, 90° rotations, 1-cell slides).
- Environment guarantees geometric precision (collision checks, rollback).
- Optional search (BFS/Beam) for disassembly/assembly sequences.

Install:
  pip install pybullet numpy

Usage:
  from unified_puzzle_env import UnifiedPuzzleEnv, PieceSpec
  env = UnifiedPuzzleEnv(gui=True)
  env.build_stacking_scene(...)
  env.ROTATE90(...); env.PLACE_AT(...); env.SLIDE_STEP(...)

  # For locks:
  env.build_lock_scene(lock="burr3")  # a minimal 3-piece sliding lock demo
  env.search_disassembly(max_depth=20, beam=40)
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pybullet as p
import pybullet_data

# ------------------------- Constants -------------------------

CELL = 0.04       # grid cell size (m)
CLEAR = 0.0008    # small clearance to avoid sticking
H = CELL
LAYER_MAX_DEFAULT = 3

AXIS_V = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}
ROT_K_RAD = {1: math.pi/2, 2: math.pi, 3: 3*math.pi/2}

# ------------------------- Utilities -------------------------

def unit(v):
    v = np.array(v, float)
    n = np.linalg.norm(v) + 1e-9
    return v / n

def quat_about(axis, angle):
    a = unit(axis)
    s = math.sin(angle/2)
    return (a[0]*s, a[1]*s, a[2]*s, math.cos(angle/2))

def make_compound_from_voxels(
    voxels: List[Tuple[int,int,int]], rgba=(0.7,0.7,0.7,1), mass: float = 0.2
) -> int:
    """Create a compound rigid body from integer voxel coords."""
    half = [CELL/2 - CLEAR]*3
    col_types = []
    vis_types = []
    col_pos = []
    col_orn = []
    vis_pos = []
    vis_orn = []

    for (x, y, z) in voxels:
        col_types.append(p.GEOM_BOX)
        vis_types.append(p.GEOM_BOX)
        pos = (x*CELL, y*CELL, z*CELL)
        col_pos.append(pos); vis_pos.append(pos)
        col_orn.append((0,0,0,1)); vis_orn.append((0,0,0,1))

    col_id = p.createCollisionShapeArray(
        shapeTypes=col_types, halfExtents=[half]*len(col_types),
        collisionFramePositions=col_pos, collisionFrameOrientations=col_orn
    )
    vis_id = p.createVisualShapeArray(
        shapeTypes=vis_types, halfExtents=[half]*len(vis_types),
        rgbaColors=[rgba]*len(vis_types),
        visualFramePositions=vis_pos, visualFrameOrientations=vis_orn
    )
    bid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col_id,
                            baseVisualShapeIndex=vis_id,
                            basePosition=(0,0,0), baseOrientation=(0,0,0,1))
    return bid

# ------------------------- Spec -------------------------

@dataclass
class PieceSpec:
    name: str
    voxels: List[Tuple[int,int,int]]
    rgba: Tuple[float,float,float,float] = (0.7,0.7,0.7,1.0)
    mass: float = 0.2

# ------------------------- Environment -------------------------

class UnifiedPuzzleEnv:
    def __init__(self, gui: bool = False):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        self.gui = gui

        # scene
        p.loadURDF("plane.urdf")
        if gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=40, cameraPitch=-35,
                                         cameraTargetPosition=[0,0,0.1])
        self.pieces: Dict[str, int] = {}
        self.cells: Dict[Tuple[int,int,int], Tuple[float,float,float]] = {}
        self.inner_w = None
        self.floor = None

    # ---- shared helpers ----
    def _make_wall(self, pos, half, color=(0.2,0.2,0.2,1)):
        p.createMultiBody(baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half)),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=color),
            basePosition=pos)

    def _spawn(self, bid: int, pos=(0,0,0.2), orn=(0,0,0,1)):
        p.resetBasePositionAndOrientation(bid, pos, orn)

    # -------------------- Stacking Scene --------------------

    def build_stacking_scene(self, grid_size: int = 3, layers: int = LAYER_MAX_DEFAULT):
        """Create a walled container of grid_size x grid_size x layers, and compute all snap cells."""
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        p.loadURDF("plane.urdf")
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=40, cameraPitch=-35,
                                         cameraTargetPosition=[0,0,0.12])
        self.pieces.clear()
        self.cells.clear()

        inner_w = grid_size*CELL
        wall_t = 0.01
        height = layers*CELL + 0.02
        floor_z = H/2

        # floor
        self.floor = p.createMultiBody(baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[inner_w/2, inner_w/2, H/2]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[inner_w/2, inner_w/2, H/2],
                                                     rgbaColor=(0.92,0.92,0.95,1)),
            basePosition=(0,0,floor_z))

        # walls
        hw = inner_w/2 + wall_t
        self._make_wall(( hw, 0, height/2), (wall_t/2, inner_w/2+wall_t, height/2))
        self._make_wall((-hw, 0, height/2), (wall_t/2, inner_w/2+wall_t, height/2))
        self._make_wall((0,  hw, height/2), (inner_w/2+wall_t, wall_t/2, height/2))
        self._make_wall((0, -hw, height/2), (inner_w/2+wall_t, wall_t/2, height/2))

        # cells
        base = - (grid_size-1)*CELL/2
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(layers):
                    x = base + i*CELL
                    y = base + j*CELL
                    z = floor_z + k*CELL + (CELL/2)
                    self.cells[(i,j,k)] = (x,y,z)

        self.inner_w = inner_w

    # -------------------- Lock Scene (minimal demo) --------------------

    def build_lock_scene(self, lock: str = "burr3"):
        """A small 3-piece burr-like lock where one key must slide to free others.
        This is a *minimal* example for demonstrating sliding actions.
        """
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        p.loadURDF("plane.urdf")
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=40, cameraPitch=-35,
                                         cameraTargetPosition=[0,0,0.12])
        self.pieces.clear()
        self.cells.clear()
        self.inner_w = None

        # Build three long bars (3x1x1 voxels) arranged orthogonally like a mini burr
        bars = {
            "A": PieceSpec("A", [(0,0,0),(1,0,0),(2,0,0)], (0.9,0.3,0.2,1), 0.3),
            "B": PieceSpec("B", [(0,0,0),(0,1,0),(0,2,0)], (0.2,0.6,0.9,1), 0.3),
            "C": PieceSpec("C", [(0,0,0),(0,0,1),(0,0,2)], (0.3,0.8,0.3,1), 0.3),
        }
        # create bodies
        for name,spec in bars.items():
            bid = make_compound_from_voxels(spec.voxels, rgba=spec.rgba, mass=spec.mass)
            self.pieces[name] = bid

        # place them intersecting at origin (with small offsets to avoid initial collisions)
        self._spawn(self.pieces["A"], (-CELL, -CLEAR, CELL*1.5))
        self._spawn(self.pieces["B"], (-CLEAR, -CELL, CELL*1.5))
        self._spawn(self.pieces["C"], (-CLEAR, -CLEAR, CELL*1.5 + CLEAR))

        # a surrounding "frame" with soft walls to limit motion (optional)
        s = CELL*3 + 0.02
        half = (s/2, s/2, H/4)
        p.createMultiBody(baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half)),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=(0.92,0.92,0.95,0.7)),
            basePosition=(0,0,half[2]))

    # -------------------- Discrete Actions --------------------

    def ROTATE90(self, name: str, axis: str, k: int = 1):
        assert name in self.pieces and axis in AXIS_V and k in (1,2,3)
        bid = self.pieces[name]
        pos, orn = p.getBasePositionAndOrientation(bid)
        dq = quat_about(AXIS_V[axis], ROT_K_RAD[k])
        new_orn = p.getQuaternionSlerp(orn, p.multiplyTransforms([0,0,0], dq, [0,0,0], orn)[1], 1.0)
        p.resetBasePositionAndOrientation(bid, pos, new_orn)
        p.stepSimulation()
        if self.gui: time.sleep(0.03)
        return {"ok": True}

    def PLACE_AT(self, name: str, cell: Tuple[int,int], layer: int):
        """Snap to (i,j,layer) and drop vertically until contact. Revert on penetration."""
        assert name in self.pieces
        if not self.cells:
            return {"ok": False, "reason": "no_grid_scene"}

        bid = self.pieces[name]
        i,j = cell; k = layer
        if (i,j,k) not in self.cells:
            return {"ok": False, "reason": "cell_out_of_range"}

        target_xy = self.cells[(i,j,k)][:2]
        pos, orn = p.getBasePositionAndOrientation(bid)
        above = (target_xy[0], target_xy[1], 0.35)
        p.resetBasePositionAndOrientation(bid, above, orn)
        p.stepSimulation()

        step = 0.003
        z_target = self.cells[(i,j,k)][2]
        z = above[2]
        ok = False
        last_free_pos = above

        for _ in range(400):
            z = max(z - step, z_target)
            p.resetBasePositionAndOrientation(bid, (target_xy[0], target_xy[1], z), orn)
            p.stepSimulation()
            if self.gui: time.sleep(1/240)

            cps = p.getContactPoints(bodyA=bid)
            interpenetrate = any(cp[8] < -1e-5 for cp in cps)
            if interpenetrate:
                p.resetBasePositionAndOrientation(bid, last_free_pos, orn)
                p.stepSimulation()
                return {"ok": False, "reason": "collision"}

            if len(cps) > 0 and z <= z_target + 1e-4:
                ok = True
                break

            last_free_pos = (target_xy[0], target_xy[1], z)
            if z <= z_target + 1e-6:
                ok = True
                break

        return {"ok": ok}

    def SLIDE_STEP(self, name: str, axis: str, steps: int = 1, tiny: float = 0.002):
        """Slide exactly |steps| * CELL along ±axes. Stops early on blocking penetration."""
        assert name in self.pieces and axis in ("+X","-X","+Y","-Y","+Z","-Z")
        bid = self.pieces[name]
        sign = 1 if axis[0] == "+" else -1
        base_axis = AXIS_V[axis[1]]
        dir_vec = base_axis * sign

        total = CELL * abs(steps)
        moved = 0.0
        pos, orn = p.getBasePositionAndOrientation(bid)
        while moved + tiny <= total + 1e-9:
            new_pos = (pos[0] + dir_vec[0]*tiny, pos[1] + dir_vec[1]*tiny, pos[2] + dir_vec[2]*tiny)
            p.resetBasePositionAndOrientation(bid, new_pos, orn)
            p.stepSimulation()
            if self.gui: time.sleep(1/240)

            cps = p.getContactPoints(bodyA=bid)
            if any(cp[8] < -5e-4 for cp in cps):
                break

            pos = new_pos
            moved += tiny

        snapped = int(round(moved / CELL))
        return {"ok": snapped == abs(steps), "cells_moved": snapped}

    # -------------------- Queries for LLM --------------------

    def get_free_axes(self, name: str) -> List[str]:
        """Probe which axes allow a +1 or -1 slide. Non-destructive (revert pose)."""
        assert name in self.pieces
        bid = self.pieces[name]
        pos0, orn0 = p.getBasePositionAndOrientation(bid)
        free = []
        for axis in ["+X","-X","+Y","-Y","+Z","-Z"]:
            r = self.SLIDE_STEP(name, axis, steps=1)
            p.resetBasePositionAndOrientation(bid, pos0, orn0)
            p.stepSimulation()
            if r.get("ok"):
                free.append(axis)
        return free

    def get_contact_graph(self) -> Dict[str, List[str]]:
        """Very simple contact graph: which pieces are touching which."""
        ids_to_name = {bid: n for n,bid in self.pieces.items()}
        graph = {n: [] for n in self.pieces.keys()}
        for a_name, a_bid in self.pieces.items():
            for b_name, b_bid in self.pieces.items():
                if a_name >= b_name: 
                    continue
                cps = p.getContactPoints(bodyA=a_bid, bodyB=b_bid)
                if len(cps) > 0:
                    graph[a_name].append(b_name)
                    graph[b_name].append(a_name)
        return graph

    # -------------------- Simple Disassembly Search --------------------

    def state_hash(self) -> Tuple:
        """Quantize positions/orientations to grid for hashing (not exact)."""
        shp = []
        for name in sorted(self.pieces.keys()):
            bid = self.pieces[name]
            pos, orn = p.getBasePositionAndOrientation(bid)
            qpos = tuple(int(round(c / (CELL/2))) for c in pos)  # half-cell quant
            # quantize yaw only (for sliding locks)
            yaw = p.getEulerFromQuaternion(orn)[2]
            qyaw = int(round(yaw / (math.pi/2))) % 4
            shp.append((name, qpos, qyaw))
        return tuple(shp)

    def is_piece_freed(self, name: str, bound=0.12) -> bool:
        """Heuristic free detection: if piece moved outside a small bound around origin."""
        bid = self.pieces[name]
        pos, _ = p.getBasePositionAndOrientation(bid)
        return any(abs(c) > bound for c in pos)

    def clone_poses(self):
        poses = {}
        for n,bid in self.pieces.items():
            poses[n] = p.getBasePositionAndOrientation(bid)
        return poses

    def restore_poses(self, poses):
        for n,(pos,orn) in poses.items():
            p.resetBasePositionAndOrientation(self.pieces[n], pos, orn)
        p.stepSimulation()

    def successors(self) -> List[Tuple[str, Dict, str]]:
        """Generate all local slide actions for current state (both directions, each piece)."""
        out = []
        for n in self.pieces.keys():
            for axis in ["+X","-X","+Y","-Y","+Z","-Z"]:
                out.append(("SLIDE_STEP", {"name": n, "axis": axis, "steps": 1}, f"{n}:{axis}"))
        return out

    def apply(self, op: str, args: Dict) -> Dict:
        if op == "SLIDE_STEP":
            return self.SLIDE_STEP(**args)
        elif op == "ROTATE90":
            return self.ROTATE90(**args)
        elif op == "PLACE_AT":
            return self.PLACE_AT(**args)
        else:
            raise ValueError(op)

    def search_disassembly(self, max_depth=20, beam=40, verbose=True) -> Optional[List[Tuple[str, Dict]]]:
        """Beam search for a sequence that frees any piece (Kongming-style)."""
        Node = Tuple[Tuple, List[Tuple[str,Dict]], Dict[str,Tuple]]  # (hash, actions, poses)

        start_poses = self.clone_poses()
        start = (self.state_hash(), [], start_poses)
        frontier: List[Node] = [start]
        seen = {start[0]}

        for depth in range(max_depth):
            cand: List[Node] = []
            for h, actions, poses in frontier:
                self.restore_poses(poses)
                # goal check
                for n in self.pieces.keys():
                    if self.is_piece_freed(n):
                        if verbose: print("Freed piece:", n, "at depth", len(actions))
                        return actions
                # expand
                for op,args,tag in self.successors():
                    cur = self.clone_poses()
                    r = self.apply(op, args)
                    if not r.get("ok"):
                        self.restore_poses(cur)
                        continue
                    h2 = self.state_hash()
                    if h2 in seen:
                        self.restore_poses(cur)
                        continue
                    seen.add(h2)
                    cand.append((h2, actions+[(op,args)], self.clone_poses()))
                    # revert to parent to try next successor
                    self.restore_poses(cur)

            # beam select by a simple heuristic: max distance sum from origin (encourage moving out)
            def score(node: Node) -> float:
                _, _, poses = node
                s = 0.0
                for (pos,_) in poses.values():
                    s += sum(abs(c) for c in pos)
                return s
            cand.sort(key=score, reverse=True)
            frontier = cand[:beam]
            if verbose: print(f"depth {depth+1}: expanded {len(cand)} -> beam {len(frontier)}")
            if not frontier: break

        self.restore_poses(start_poses)
        return None
