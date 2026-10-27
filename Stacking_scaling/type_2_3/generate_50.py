"""
PipeFit 3D — 50 puzzles (rules v2).

Categories (10 each = 3 easy + 4 medium + 3 hard):
  A_route      one entrance, one exit, open box
  B_obstacles  obstacles block the obvious staircase route
  C_branch     two exits: the network needs a T junction
  D_tall       high exits: support columns dominate
  E_mixed      two exits, tall box, obstacles

Tiers are measured by exhaustive search (difficulty.TIERS): number of
solutions, minimum pipes, and whether the staircase shortcut works.  Every
slot scans its own seed range and layouts are de-duplicated; the output
folder is wiped first so no stale puzzles remain.

level.json's meta and solution.txt contain the answer: keep both out of
model inputs (models should only see the rendered image).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from type_2_3.game_core import init_state
    from type_2_3.levels import CATEGORY, find_level
    from type_2_3.loader import save_level_to_json
    from type_2_3.mechanics import ends_of, step_place, submit
    from type_2_3.solver import solve
    from type_2_3.visualizer_plotly import render_state_pair
else:
    from .game_core import init_state
    from .levels import CATEGORY, find_level
    from .loader import save_level_to_json
    from .mechanics import ends_of, step_place, submit
    from .solver import solve
    from .visualizer_plotly import render_state_pair


def _make_plan() -> List[Tuple[str, str, str]]:
    plan, idx = [], 1
    for cat in CATEGORY:
        for tier, count in (("easy", 3), ("medium", 4), ("hard", 3)):
            for _ in range(count):
                plan.append((f"P{idx:02d}_{cat}_{tier}", cat, tier))
                idx += 1
    return plan


def _render_minimal(label: str, spec, meta, plan, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_level_to_json(spec, os.path.join(out_dir, "level.json"), meta=meta)
    state = init_state(spec)
    render_state_pair(state, out_dir, "00_initial", title=f"{label}  initial")
    for p in plan:
        r = step_place(state, p.type, p.pos, p.orientation)
        if r.code.value != "OK":
            print(f"  !! replay failed at {p}: {r.message}")
            return
    res = submit(state)
    render_state_pair(state, out_dir, "99_submit",
                      title=f"{label}  submit -> {res.code.value}")
    with open(os.path.join(out_dir, "solution.txt"), "w") as f:
        f.write(f"{label}\n")
        f.write(f"grid: {spec.grid_size}\n")
        f.write(f"entrances: {spec.entrances}\nexits: {spec.exits}\n")
        f.write(f"obstacles: {spec.obstacles}\n")
        f.write(f"inventory: { {pt.value: n for pt, n in spec.inventory.items() if n} }\n")
        f.write(f"difficulty: {meta['difficulty']}\n\n")
        f.write(f"plan ({len(plan)} pieces):\n")
        for i, p in enumerate(plan, 1):
            ends = ends_of(p.type, p.orientation)
            f.write(f"  {i:02d}: {p.type.value} at {p.pos}"
                    + (f" ends={','.join(ends)}" if ends else "") + "\n")


def main():
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "puzzles_50")
    os.makedirs(out_root, exist_ok=True)
    for entry in os.listdir(out_root):          # no stale puzzles
        full = os.path.join(out_root, entry)
        if os.path.isdir(full):
            for f in os.listdir(full):
                os.remove(os.path.join(full, f))
            os.rmdir(full)

    rows: List[Dict] = []
    seen: set = set()
    t_all = time.time()
    for i, (label, cat, tier) in enumerate(_make_plan(), 1):
        params = CATEGORY[cat][tier]
        print(f"=== [{i:2d}/50] {label}")
        found = find_level(params, tier, range(10000 + 500 * i, 10000 + 500 * i + 300), seen)
        if found is None:
            print("  !! no level found")
            rows.append({"label": label, "cat": cat, "diff": tier, "seed": None,
                         "grid": list(params["grid_size"])})
            continue
        seed, spec, meta = found
        plan = solve(spec)
        out_dir = os.path.join(out_root, f"{label}_seed{seed}")
        _render_minimal(label, spec, meta, plan, out_dir)
        d = meta["difficulty"]
        print(f"  seed={seed}  plan={len(plan)}  {d}")
        rows.append({"label": label, "cat": cat, "diff": tier, "seed": seed,
                     "grid": list(params["grid_size"]), "plan_length": len(plan),
                     "pipes": meta["n_pipes"], "supports": meta["n_supports"],
                     "inventory": {pt.value: n for pt, n in spec.inventory.items() if n},
                     "out_dir": os.path.basename(out_dir), **d})

    lines = ["# PipeFit 3D — 50 puzzles (rules v2)\n",
             "| # | label | grid | seed | plan | pipes | supports | solutions "
             "| staircase wins | inventory |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        if r.get("seed") is None:
            lines.append(f"| {i} | {r['label']} | {tuple(r['grid'])} | FAILED | | | | | | |")
            continue
        lines.append(f"| {i} | {r['label']} | {tuple(r['grid'])} | {r['seed']} | "
                     f"{r['plan_length']} | {r['pipes']} | {r['supports']} | "
                     f"{r['n_solutions']} | {r['staircase_ok']} | {r['inventory']} |")
    lines += ["", f"elapsed total: {time.time() - t_all:.1f}s"]
    with open(os.path.join(out_root, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[generate_50] done in {time.time() - t_all:.1f}s -> {out_root}")


if __name__ == "__main__":
    main()
