
"""
grid_snap_demo.py

Discrete-action puzzle interface:
- A 3x3 placement grid inside a "box" (container).
- Pieces are voxel compounds (sets of unit cubes).
- Allowed actions:
  * ROTATE90(piece, axis, k)   # k in {1,2,3} -> 90/180/270 deg
  * PLACE_AT(piece, cell=(i,j), layer=k)  # snaps, then drops vertically until contact
  * SLIDE_STEP(piece, axis, steps=+/-1)   # for Kongming-like sliding puzzles (optional)

Run:
  pip install pybullet numpy
  python grid_snap_demo.py --gui
"""
import argparse
import math
import time
from typing import List, Tuple, Dict

import numpy as np
import pybullet as p
import pybullet_data

CELL = 0.04       # grid cell size (m)
CLEAR = 0.0008    # small clearance
H = CELL          # unit cube height
LAYER_MAX = 3

AXIS = {
    "X": np.array([1.0,0,0]),
    "Y": np.array([0,1.0,0]),
    "Z": np.array([0,0,1.0]),
}
ROT_K = {1: math.pi/2, 2: math.pi, 3: 3*math.pi/2}

def unit(v):
    v = np.array(v, float)
    n = np.linalg.norm(v) + 1e-9
    return v/n

def quat_about(axis, angle):
    a = unit(axis)
    s = math.sin(angle/2)
    return (a[0]*s, a[1]*s, a[2]*s, math.cos(angle/2))

def compound_from_voxels(voxels: List[Tuple[int,int,int]], rgba=(0.2,0.8,0.2,1), mass=0.1):
    half = [CELL/2 - CLEAR]*3
    col_shapes = []
    vis_shapes = []
    col_pos = []
    col_orn = []
    vis_pos = []
    vis_orn = []
    for (x,y,z) in voxels:
        col_shapes.append(p.GEOM_BOX)
        vis_shapes.append(p.GEOM_BOX)
        pos = (x*CELL, y*CELL, z*CELL)
        col_pos.append(pos); vis_pos.append(pos)
        col_orn.append((0,0,0,1)); vis_orn.append((0,0,0,1))

    col_id = p.createCollisionShapeArray(shapeTypes=col_shapes, halfExtents=[half]*len(col_shapes),
                                         collisionFramePositions=col_pos, collisionFrameOrientations=col_orn)
    vis_id = p.createVisualShapeArray(shapeTypes=vis_shapes, halfExtents=[half]*len(vis_shapes),
                                      rgbaColors=[rgba]*len(vis_shapes),
                                      visualFramePositions=vis_pos, visualFrameOrientations=vis_orn)

    bid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                            basePosition=(0,0,0), baseOrientation=(0,0,0,1))
    return bid

class GridSnapEnv:
    def __init__(self, gui=False):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        self.gui = gui
        self._build_scene()

    def _build_scene(self):
        p.loadURDF("plane.urdf")
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=45, cameraPitch=-35,
                                         cameraTargetPosition=[0.0,0.0,0.1])

        inner_w = 3*CELL
        wall_t = 0.01
        height = LAYER_MAX*CELL + 0.02
        floor_z = H/2

        self.floor = p.createMultiBody(baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[inner_w/2, inner_w/2, H/2]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[inner_w/2, inner_w/2, H/2],
                                                     rgbaColor=(0.9,0.9,0.9,1)),
            basePosition=(0,0,floor_z))

        hw = inner_w/2 + wall_t
        self._make_wall(( hw, 0, height/2), (wall_t/2, inner_w/2+wall_t, height/2))
        self._make_wall((-hw, 0, height/2), (wall_t/2, inner_w/2+wall_t, height/2))
        self._make_wall((0,  hw, height/2), (inner_w/2+wall_t, wall_t/2, height/2))
        self._make_wall((0, -hw, height/2), (inner_w/2+wall_t, wall_t/2, height/2))

        self.cells = {}
        base_x = -CELL + CELL/2
        base_y = -CELL + CELL/2
        for i in range(3):
            for j in range(3):
                for k in range(LAYER_MAX):
                    x = base_x + i*CELL
                    y = base_y + j*CELL
                    z = floor_z + k*CELL + (CELL/2)
                    self.cells[(i,j,k)] = (x,y,z)

        self.pieces: Dict[str,int] = {}
        self.pieces["L1"] = compound_from_voxels([(0,0,0),(1,0,0),(2,0,0),(2,1,0)], rgba=(0.9,0.3,0.2,1))
        self.pieces["T1"] = compound_from_voxels([(0,0,0),(1,0,0),(2,0,0),(1,1,0)], rgba=(0.2,0.5,0.9,1))
        self.pieces["D1"] = compound_from_voxels([(0,0,0),(1,0,0)], rgba=(0.2,0.8,0.3,1))

        self._spawn_above(self.pieces["L1"], (-0.25,-0.1,0.20))
        self._spawn_above(self.pieces["T1"], (0.25,0.0,0.24))
        self._spawn_above(self.pieces["D1"], (0.0,0.2,0.28))

    def _make_wall(self, pos, half):
        p.createMultiBody(baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half)),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=(0.2,0.2,0.2,1)),
            basePosition=pos)

    def _spawn_above(self, bid, pos):
        p.resetBasePositionAndOrientation(bid, pos, (0,0,0,1))

    def ROTATE90(self, name: str, axis: str, k: int = 1):
        assert name in self.pieces and axis in AXIS and k in (1,2,3)
        bid = self.pieces[name]
        pos, orn = p.getBasePositionAndOrientation(bid)
        dq = quat_about(AXIS[axis], ROT_K[k])
        new_orn = p.getQuaternionSlerp(orn, p.multiplyTransforms([0,0,0], dq, [0,0,0], orn)[1], 1.0)
        p.resetBasePositionAndOrientation(bid, pos, new_orn)
        p.stepSimulation()
        if self.gui: time.sleep(0.05)
        return {"ok": True}

    def PLACE_AT(self, name: str, cell: Tuple[int,int], layer: int):
        assert name in self.pieces
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

    def SLIDE_STEP(self, name: str, axis: str, steps: int):
        assert name in self.pieces and axis in ("+X","-X","+Y","-Y","+Z","-Z")
        bid = self.pieces[name]
        sign = 1 if axis[0] == "+" else -1
        base_axis = AXIS[axis[1]]
        dir_vec = base_axis * sign

        total = CELL * abs(steps)
        tiny = 0.002
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

def demo(gui=False):
    env = GridSnapEnv(gui=gui)
    plan = [
        ("ROTATE90", {"name":"T1", "axis":"Z", "k":1}),
        ("PLACE_AT", {"name":"L1", "cell":(0,0), "layer":0}),
        ("PLACE_AT", {"name":"T1", "cell":(2,1), "layer":0}),
        ("PLACE_AT", {"name":"D1", "cell":(1,0), "layer":1}),
        ("SLIDE_STEP", {"name":"D1", "axis":"+Y", "steps":1}),
    ]

    for i,(op,args) in enumerate(plan,1):
        print(f"\n[{i}/{len(plan)}] {op} {args}")
        if op == "ROTATE90":
            print(env.ROTATE90(**args))
        elif op == "PLACE_AT":
            print(env.PLACE_AT(**args))
        elif op == "SLIDE_STEP":
            print(env.SLIDE_STEP(**args))

    print("\nDone. Close the window to exit.")
    if gui:
        while True:
            p.stepSimulation()
            time.sleep(1/240)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    demo(gui=args.gui)
