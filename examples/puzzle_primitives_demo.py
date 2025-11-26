
"""
puzzle_primitives_demo.py

A minimal PyBullet demo that shows how to control puzzle pieces using
event/constraint-based high-level primitives instead of precise numbers.

Run:
  pip install pybullet numpy
  python puzzle_primitives_demo.py --gui   # GUI mode
  python puzzle_primitives_demo.py         # headless (DIRECT) mode

What it shows:
- MoveUntilEvent: move along an axis until a contact or until "free_out"
- RotateUntilAligned: rotate around an axis in fixed angle increments until an alignment checker passes
- A simple "executor" that takes an action plan (what an LLM would produce) and runs it

This is not the exact puzzle in your image; it's a tiny toy scene with 2-3 blocks
to illustrate the interface and the primitives. You can replace the geometries with
your own URDFs or convex shapes.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

EPS = 1e-8

AXIS_MAP = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + EPS)

def quat_mult(q1, q2):
    """Quaternion multiply (x, y, z, w) format as in PyBullet."""
    return p.getQuaternionFromEuler(
        p.getEulerFromQuaternion(q1) + np.array(p.getEulerFromQuaternion(q2))
    )

def axis_angle_quat(axis: np.ndarray, angle: float):
    """Build quaternion for rotation around axis by angle (radians)."""
    ax = unit(axis)
    # pybullet uses (x,y,z,w)
    s = math.sin(angle / 2.0)
    return (ax[0]*s, ax[1]*s, ax[2]*s, math.cos(angle/2.0))

def rotate_quat(q, axis, angle):
    """Right-multiply quaternion q by rotation R(axis, angle)."""
    dq = axis_angle_quat(axis, angle)
    return p.getQuaternionSlerp(q, p.multiplyTransforms([0,0,0], dq, [0,0,0], q)[1], 1.0)

def world_vec_from_local(body_id: int, local_vec: np.ndarray) -> np.ndarray:
    _, orn = p.getBasePositionAndOrientation(body_id)
    R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    return R @ local_vec

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    un = unit(u); vn = unit(v)
    dot = float(np.clip(np.dot(un, vn), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def set_pose(body_id: int, pos, orn):
    p.resetBasePositionAndOrientation(body_id, pos, orn)

def get_contacts_with(body_id: int) -> List:
    return p.getContactPoints(bodyA=body_id)

@dataclass
class MoveResult:
    event: str
    traveled: float
    pos: Tuple[float, float, float]
    orn: Tuple[float, float, float, float]

def move_until_event(
    body_id: int,
    axis: str,
    max_travel_m: float = 0.4,
    step: float = 0.002,
    stop_on_contact: bool = True,
    stop_on_free: bool = False,
    clearance_threshold_m: float = 0.002,
    max_steps: int = 2000,
    visualize: bool = True,
) -> MoveResult:
    d = unit(AXIS_MAP[axis])
    pos, orn = p.getBasePositionAndOrientation(body_id)
    traveled = 0.0
    last_event = "none"

    for _ in range(max_steps):
        new_pos = (pos[0] + d[0]*step, pos[1] + d[1]*step, pos[2] + d[2]*step)
        set_pose(body_id, new_pos, orn)
        p.stepSimulation()
        if visualize:
            time.sleep(1/240)

        pos = new_pos
        traveled += step

        contacts = get_contacts_with(body_id)

        if stop_on_contact and len(contacts) > 0:
            last_event = "contact"
            break

        if stop_on_free:
            # naive "free_out": no contacts for a few consecutive steps
            # (for demo we check just now; you can add hysteresis / ray checks)
            if len(contacts) == 0 and traveled > 0.02:
                last_event = "free_out"
                break

        if traveled >= max_travel_m:
            last_event = "max_reached"
            break

    return MoveResult(last_event, traveled, pos, orn)

@dataclass
class RotateResult:
    event: str
    pos: Tuple[float, float, float]
    orn: Tuple[float, float, float, float]

def rotate_until_aligned(
    body_id: int,
    axis: str,
    angle_grid_deg: int = 15,
    max_turn_deg: int = 180,
    alignment_checker: Optional[Callable[[Tuple[float, float, float, float]], bool]] = None,
    visualize: bool = True,
) -> RotateResult:
    step = math.radians(angle_grid_deg)
    max_step = int(max_turn_deg / angle_grid_deg + 0.5)
    a = AXIS_MAP[axis]
    pos, orn = p.getBasePositionAndOrientation(body_id)

    for _ in range(max_step):
        new_orn = rotate_quat(orn, a, step)
        set_pose(body_id, pos, new_orn)
        p.stepSimulation()
        if visualize:
            time.sleep(1/240)

        orn = new_orn
        if alignment_checker is not None and alignment_checker(orn):
            return RotateResult("aligned", pos, orn)

    return RotateResult("max_turn_reached", pos, orn)

# --------- Simple Scene Builders ----------

def create_box(half_extents, mass=0.0, rgba=(1, 0, 0, 1), pos=(0,0,0), orn=(0,0,0,1)):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    bid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                            basePosition=pos, baseOrientation=orn)
    return bid

def setup_world(gui: bool = False):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # plane.urdf
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    plane = p.loadURDF("plane.urdf")

    # camera for GUI
    if gui:
        p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=45, cameraPitch=-35,
                                     cameraTargetPosition=[0.0, 0.0, 0.1])
    return cid, plane

# ---------- Example Alignment Checker ---------

def mk_face_alignment_checker(src_body: int, tgt_body: int, 
                              src_local_normal=(0,0,1), tgt_local_normal=(0,0,1),
                              tol_deg=5.0):
    src_n = np.array(src_local_normal, float)
    tgt_n = np.array(tgt_local_normal, float)

    def checker(curr_orn):
        # use current orientation of src_body (already set before calling)
        # compute world normals
        sN = world_vec_from_local(src_body, src_n)
        tN = world_vec_from_local(tgt_body, tgt_n)
        ang = angle_between(sN, tN)
        # print("ang:", ang)
        return ang <= tol_deg
    return checker

# ---------- Action Executor (what LLM would call) ---------

def execute_action(action: Dict, visualize=True):
    prim = action["primitive"]
    piece = action["object_id"]
    params = action.get("params", {})
    result = None

    if prim == "MoveUntilEvent":
        result = move_until_event(
            piece,
            axis=params.get("axis", "+X"),
            max_travel_m=float(params.get("max_travel_m", 0.5)),
            step=float(params.get("step", 0.002)),
            stop_on_contact=bool(params.get("stop_on_contact", True)),
            stop_on_free=bool(params.get("stop_on_free", False)),
            visualize=visualize,
        )
        print(f"[MoveUntilEvent] -> {result.event}, traveled={result.traveled:.3f} m")
        return result.__dict__

    elif prim == "RotateUntilEvent":
        # optional alignment to another body
        ref = action.get("ref_object_id", None)
        axis = params.get("axis", "+Z")
        angle_grid = int(params.get("angle_grid_deg", 15))
        max_turn_deg = int(params.get("max_turn_deg", 180))

        checker = None
        if ref is not None and params.get("align_faces", False):
            # by default, align +Z of piece to +Z of ref
            checker = mk_face_alignment_checker(piece, ref, (0,0,1), (0,0,1),
                                                tol_deg=float(params.get("tolerance_deg", 5)))

        result = rotate_until_aligned(
            piece,
            axis=axis,
            angle_grid_deg=angle_grid,
            max_turn_deg=max_turn_deg,
            alignment_checker=checker,
            visualize=visualize,
        )
        print(f"[RotateUntilEvent] -> {result.event}")
        return result.__dict__

    else:
        raise ValueError(f"Unknown primitive: {prim}")

# ---------- Demo Scenario ---------

def demo(gui=False):
    setup_world(gui=gui)
    # Ground guide block (fixed, orange)
    base = create_box((0.25, 0.05, 0.025), mass=0.0, rgba=(1, 0.5, 0, 1),
                      pos=(0.0, 0.0, 0.025))

    # Movable "green" block to slide & rotate
    mover = create_box((0.07, 0.04, 0.04), mass=0.2, rgba=(0.2, 0.8, 0.2, 1),
                       pos=(-0.20, -0.02, 0.04))

    # A second fixed block (red) to cause contact
    obstacle = create_box((0.07, 0.04, 0.04), mass=0.0, rgba=(0.9, 0.2, 0.2, 1),
                          pos=(0.05, -0.02, 0.04))

    print("Bodies: base(orange)=", base, " mover(green)=", mover, " obstacle(red)=", obstacle)

    # Example "LLM" action plan (coarse, no precise numbers)
    plan = [
        # 1) Move mover along +X until first contact (with red obstacle)
        {"primitive": "MoveUntilEvent",
         "object_id": mover,
         "params": {"axis": "+X", "max_travel_m": 0.20, "step": 0.003, "stop_on_contact": True}},
        # 2) Rotate around +Z in 15° increments until +Z face aligns with base's +Z (trivial here)
        {"primitive": "RotateUntilEvent",
         "object_id": mover,
         "ref_object_id": base,
         "params": {"axis": "+Z", "angle_grid_deg": 60, "max_turn_deg": 180, "align_faces": True, "tolerance_deg": 5}},
        # 3) Try to slide out along -Y until "free_out"
        {"primitive": "MoveUntilEvent",
         "object_id": mover,
         "params": {"axis": "-Y", "max_travel_m": 0.5, "step": 0.003, "stop_on_contact": False, "stop_on_free": True}},
    ]

    for i, act in enumerate(plan, 1):
        import time
        time.sleep(5)
        print(f"\n=== Execute action {i}/{len(plan)}: {act['primitive']} ===")
        execute_action(act, visualize=gui)

    print("\nDemo done. Close the window or press Ctrl+C to exit.")
    if gui:
        while True:
            p.stepSimulation()
            time.sleep(1/60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="use PyBullet GUI")
    args = parser.parse_args()
    demo(gui=args.gui)
