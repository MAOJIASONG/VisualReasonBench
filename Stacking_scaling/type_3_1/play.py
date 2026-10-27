"""Auto-play a single Domino Chain Reaction 3D level.

Usage:
  python -m type_3_1.play [seed]

Renders every key step (initial, after each placement, pre-push, after-push,
submit) to `type_3_1/images/seed_<N>/`.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

from .generator import generate_level
from .levels import CATEGORY
from .game_core import init_state
from .loader import save_level_to_json
from .mechanics import step_place, step_push, submit, submit_placement
from .solver import solve
from .visualizer_plotly import render_state_pair


def play_seed(seed: int = 1, out_root: str = "type_3_1/images") -> None:
    print(f"[play] seed={seed}")
    r = generate_level(seed=seed, tier="medium",
                       **CATEGORY["C_steps"]["medium"])
    if r is None:
        print("[play] generation failed")
        return
    spec, meta = r
    out_dir = os.path.join(out_root, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    save_level_to_json(spec, os.path.join(out_dir, "level.json"), meta=meta)

    st = init_state(spec)
    render_state_pair(st, out_dir, "00_initial",
                      title=f"seed {seed}  initial")

    plan = solve(spec)
    if plan is None:
        print("[play] no plan")
        return

    step_idx = 1
    for act in plan:
        kind = act[0]
        if kind == "place":
            _, pt, pos = act
            r = step_place(st, pt, pos)
            print(f"  place {pt.value}@{pos}: {r.code.value}")
            render_state_pair(st, out_dir, f"{step_idx:02d}_place",
                               title=f"seed {seed}  place {pt.value}@{pos}")
            step_idx += 1
        elif kind == "submit_placement":
            r = submit_placement(st)
            print(f"  submit_placement: {r.code.value}")
            render_state_pair(st, out_dir, f"{step_idx:02d}_pre_push",
                               title=f"seed {seed}  ready to push")
            step_idx += 1
        elif kind == "push":
            _, pos, d = act
            r = step_push(st, pos, d)
            print(f"  push@{pos} {d}: {r.code.value}  "
                  f"toppled={len(r.detail.get('toppled', []))}")
            render_state_pair(st, out_dir, f"{step_idx:02d}_post_push",
                               title=f"seed {seed}  after push {d}")
            step_idx += 1
        elif kind == "submit":
            r = submit(st)
            print(f"  submit: {r.code.value}")
            render_state_pair(st, out_dir, f"99_submit",
                               title=f"seed {seed}  submit -> {r.code.value}")

    print(f"[play] done -> {out_dir}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t0 = time.time()
    play_seed(seed)
    print(f"[play] elapsed {time.time() - t0:.1f}s")
