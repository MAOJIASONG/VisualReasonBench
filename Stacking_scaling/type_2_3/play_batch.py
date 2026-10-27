"""PipeFit 3D — 10-puzzle batch across easy/medium/hard tiers."""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from type_2_3.levels import CATEGORY, find_level
    from type_2_3.solver import solve
    from type_2_3.game_core import init_state
    from type_2_3.mechanics import submit, step_place, step_remove
    from type_2_3.visualizer_plotly import render_state_pair
    from type_2_3.loader import save_level_to_json
else:
    from .levels import CATEGORY, find_level
    from .solver import solve
    from .game_core import init_state
    from .mechanics import submit, step_place, step_remove
    from .visualizer_plotly import render_state_pair
    from .loader import save_level_to_json


PUZZLES: List[Tuple[str, str, str]] = [
    ("01_easy",   "A_route",     "easy"),
    ("02_easy",   "B_obstacles", "easy"),
    ("03_easy",   "D_tall",      "easy"),
    ("04_medium", "A_route",     "medium"),
    ("05_medium", "B_obstacles", "medium"),
    ("06_medium", "C_branch",    "medium"),
    ("07_medium", "D_tall",      "medium"),
    ("08_hard",   "B_obstacles", "hard"),
    ("09_hard",   "C_branch",    "hard"),
    ("10_hard",   "D_tall",      "hard"),
]


def _render_episode(label: str, spec, meta, plan, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_level_to_json(spec, os.path.join(out_dir, "level.json"), meta=meta)

    state = init_state(spec)
    render_state_pair(state, out_dir, "00_initial",
                      title=f"{label}  initial")

    for i, p in enumerate(plan, 1):
        r = step_place(state, p.type, p.pos, p.orientation)
        assert r.code.value == "OK"
        title = f"{label}  step {i}/{len(plan)}  place {p.type.value} at {p.pos}"
        prefix = f"{i:02d}_{p.type.value}_{p.pos[0]}_{p.pos[1]}_{p.pos[2]}"
        render_state_pair(state, out_dir, prefix, title=title)

    res = submit(state)
    render_state_pair(state, out_dir, "98_submit",
                      title=f"{label}  submit -> {res.code.value}")

    # Counterfactual: remove a support, show the violation
    sup = next((p for p in plan if p.type.value in ("S1", "S2")), None)
    if sup is not None:
        cf = init_state(spec)
        for pl in plan:
            step_place(cf, pl.type, pl.pos, pl.orientation)
        step_remove(cf, sup.pos)
        cf_res = submit(cf)
        prefix = f"99_cf_remove_{sup.type.value}_{sup.pos[0]}_{sup.pos[1]}_{sup.pos[2]}"
        render_state_pair(cf, out_dir, prefix,
                          title=f"{label}  CF remove {sup.type.value}@{sup.pos} -> {cf_res.code.value}")

    with open(os.path.join(out_dir, "solution.txt"), "w") as f:
        f.write(f"{label}\n")
        f.write(f"grid_size: {spec.grid_size}\n")
        f.write(f"E: {spec.entrances}\n")
        f.write(f"X: {spec.exits}\n")
        f.write(f"obs: {spec.obstacles}\n")
        f.write(f"inventory: { {pt.value: n for pt, n in spec.inventory.items() if n > 0} }\n")
        f.write(f"meta: {meta}\n\n")
        f.write(f"plan ({len(plan)} pieces):\n")
        for i, p in enumerate(plan, 1):
            f.write(f"  step {i}: {p.type.value} at {p.pos}\n")


def main():
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "images_batch")
    if os.path.exists(out_root):
        for entry in os.listdir(out_root):
            full = os.path.join(out_root, entry)
            if os.path.isdir(full):
                for f in os.listdir(full):
                    os.remove(os.path.join(full, f))
                os.rmdir(full)
            else:
                os.remove(full)
    os.makedirs(out_root, exist_ok=True)

    lines = ["# PipeFit 3D — 10-puzzle batch playthrough (rules v2)\n",
             "| # | label | category | grid | seed | plan | pipes | supports | solutions | staircase wins | inventory |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    t0 = time.time()
    seen: set = set()
    for idx, (label, cat, tier) in enumerate(PUZZLES, 1):
        params = CATEGORY[cat][tier]
        print(f"\n=== puzzle {idx:2d}: {label}  {cat} ===")
        found = find_level(params, tier, range(1000 * idx, 1000 * idx + 300), seen)
        if found is None:
            print("  !! failed")
            lines.append(f"| {idx} | {label} | {cat} | {params['grid_size']} | -- | FAILED | | | | | |")
            continue
        seed, spec, meta = found
        plan = solve(spec)
        out_dir = os.path.join(out_root, f"level_{label}_seed{seed}")
        _render_episode(label, spec, meta, plan, out_dir)
        d = meta["difficulty"]
        inv = {pt.value: n for pt, n in spec.inventory.items() if n > 0}
        lines.append(f"| {idx} | {label} | {cat} | {params['grid_size']} | {seed} | {len(plan)} "
                     f"| {meta['n_pipes']} | {meta['n_supports']} | {d['n_solutions']} "
                     f"| {d['staircase_ok']} | {inv} |")
        print(f"  seed={seed}  plan={len(plan)}  {d} -> {out_dir}/")

    lines.append(f"\nelapsed: {time.time() - t0:.1f}s")
    with open(os.path.join(out_root, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nall processed -> {out_root}")


if __name__ == "__main__":
    main()
