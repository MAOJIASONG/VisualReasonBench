
"""
examples_unified.py

Two runnable examples using UnifiedPuzzleEnv:
  1) Grid stacking: place a few voxel pieces into a 3x3x2 container.
  2) Kongming-style lock: tiny 3-bar burr; search a disassembly sequence with beam search.

Run:
  pip install pybullet numpy
  python examples_unified.py --gui
"""
import argparse
import time
from unified_puzzle_env import UnifiedPuzzleEnv, PieceSpec, CELL

def stacking_demo(env: UnifiedPuzzleEnv):
    env.build_stacking_scene(grid_size=3, layers=2)

    # define 3 pieces (voxel coords in their local frame)
    specs = [
        PieceSpec("L1", [(0,0,0),(1,0,0),(2,0,0),(2,1,0)], (0.9,0.3,0.2,1), 0.2),
        PieceSpec("T1", [(0,0,0),(1,0,0),(2,0,0),(1,1,0)], (0.2,0.5,0.9,1), 0.2),
        PieceSpec("D1", [(0,0,0),(1,0,0)], (0.2,0.8,0.3,1), 0.2),
    ]

    # spawn pieces above the box
    for i,s in enumerate(specs):
        bid = env.pieces[s.name] = env.pieces.get(s.name, 0) or 0
        from unified_puzzle_env import make_compound_from_voxels
        env.pieces[s.name] = make_compound_from_voxels(s.voxels, rgba=s.rgba, mass=s.mass)
        env._spawn(env.pieces[s.name], pos=(-0.25 + i*0.25, 0.2, 0.25))

    # a "plan" you would normally get from an LLM:
    plan = [
        ("ROTATE90", {"name":"T1","axis":"Z","k":1}),
        ("PLACE_AT", {"name":"L1","cell":(0,0), "layer":0}),
        ("PLACE_AT", {"name":"T1","cell":(2,1), "layer":0}),
        ("PLACE_AT", {"name":"D1","cell":(1,0), "layer":1}),
    ]

    for op,args in plan:
        print(op, args, "->", env.apply(op, args))
        time.sleep(0.2)

def lock_demo(env: UnifiedPuzzleEnv):
    env.build_lock_scene(lock="burr3")

    # quick probe for free axes
    for name in env.pieces.keys():
        print("free axes of", name, ":", env.get_free_axes(name))

    seq = env.search_disassembly(max_depth=20, beam=60, verbose=True)
    if seq is None:
        print("No sequence found within limits.")
    else:
        print("Found sequence:")
        for i,(op,args) in enumerate(seq,1):
            print(i, op, args)
            env.apply(op,args)
            time.sleep(0.15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env = UnifiedPuzzleEnv(gui=args.gui)
    print("\n=== STACKING DEMO ===")
    stacking_demo(env)
    print("\n=== LOCK DEMO ===")
    lock_demo(env)

    print("\nDemos finished.")
    if args.gui:
        while True:
            import pybullet as p, time
            p.stepSimulation(); time.sleep(1/240)
