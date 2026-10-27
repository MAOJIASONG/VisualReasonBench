"""
Domino Chain Reaction 3D — 50 puzzles (rules v2).

Categories (10 each = 3 easy + 4 medium + 3 hard):
  A_line   — straight chain
  B_decoy  — chain plus decoy lines (gap -> dominoes -> red piece)
  C_steps  — height changes on grey STEP blocks
  D_split  — a splitter sends the wave both ways
  E_full   — steps + splitter + decoys

Tiers are measured, not assumed (difficulty.TIERS): every level has exactly
one winning (placement, direction), uses all its patches, and the medium /
hard tiers defeat the "fill the obvious holes" shortcut.  Layouts are
de-duplicated up to rotation, reflection and translation.

Per puzzle we render (initial, submit) x front and save level.json +
solution.txt.  level.json's meta and solution.txt contain the answer: keep
both out of model inputs (models should only see the rendered image).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from type_3_1.game_core import init_state
    from type_3_1.levels import CATEGORY, find_level
    from type_3_1.loader import save_level_to_json
    from type_3_1.mechanics import step_place, step_push, submit, submit_placement
    from type_3_1.solver import solve
    from type_3_1.visualizer_plotly import render_state_pair
else:
    from .game_core import init_state
    from .levels import CATEGORY, find_level
    from .loader import save_level_to_json
    from .mechanics import step_place, step_push, submit, submit_placement
    from .solver import solve
    from .visualizer_plotly import render_state_pair


def _make_plan() -> List[Tuple[str, str, str]]:
    plan = []
    idx = 1
    for cat in CATEGORY:
        for tier, count in (("easy", 3), ("medium", 4), ("hard", 3)):
            for _ in range(count):
                plan.append((f"P{idx:02d}_{cat}_{tier}", cat, tier))
                idx += 1
    return plan


def _render_one(label: str, spec, meta, plan, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_level_to_json(spec, os.path.join(out_dir, "level.json"), meta=meta)
    st = init_state(spec)
    render_state_pair(st, out_dir, "00_initial",
                      title=f"{label}  initial", views=("front",))
    for act in plan:
        kind = act[0]
        if kind == "place":
            step_place(st, act[1], act[2])
        elif kind == "submit_placement":
            submit_placement(st)
        elif kind == "push":
            step_push(st, act[1], act[2])
        elif kind == "submit":
            res = submit(st)
            render_state_pair(st, out_dir, "99_submit",
                              title=f"{label}  submit -> {res.code.value}",
                              views=("front",))
    with open(os.path.join(out_dir, "solution.txt"), "w") as f:
        f.write(f"{label}\n")
        f.write(f"grid: {spec.grid_size}\n")
        f.write(f"fixed: {[(pt.value, pos) for pt, pos in spec.fixed_pieces]}\n")
        f.write(f"inventory: { {pt.value: n for pt, n in spec.inventory.items() if n} }\n")
        f.write(f"patches: {meta['patch_pieces']}\n")
        f.write(f"push:    START {meta['origin']} dir={meta['origin_dir']}\n")
        f.write(f"difficulty: {meta['difficulty']}\n")


def main():
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "puzzles_50")
    os.makedirs(out_root, exist_ok=True)
    for entry in os.listdir(out_root):          # no stale puzzles
        full = os.path.join(out_root, entry)
        if os.path.isdir(full):
            for f in os.listdir(full):
                os.remove(os.path.join(full, f))
            os.rmdir(full)

    plan = _make_plan()
    rows: List[Dict] = []
    seen: set = set()
    t_all = time.time()
    for i, (label, cat, tier) in enumerate(plan, 1):
        kwargs = CATEGORY[cat][tier]
        print(f"=== [{i:2d}/{len(plan)}] {label}")
        found = find_level(kwargs, tier, range(10000 + 500 * i, 10000 + 500 * i + 400), seen)
        if found is None:
            print("  !! no level found")
            rows.append({"label": label, "cat": cat, "diff": tier, "seed": None,
                         "grid": list(kwargs["grid_size"])})
            continue
        seed, spec, meta = found
        out_dir = os.path.join(out_root, f"{label}_seed{seed}")
        _render_one(label, spec, meta, solve(spec), out_dir)
        d = meta["difficulty"]
        print(f"  seed={seed}  {d}")
        rows.append({"label": label, "cat": cat, "diff": tier, "seed": seed,
                     "grid": list(kwargs["grid_size"]),
                     "fixed_count": meta["fixed_count"],
                     "patch_count": meta["patch_count"],
                     "out_dir": os.path.basename(out_dir), **d})

    lines = [
        "# Domino Chain Reaction 3D — 50 puzzles (rules v2)\n",
        "Categories: A_line, B_decoy, C_steps, D_split, E_full "
        "(3 easy + 4 medium + 3 hard each).  Every level has exactly one "
        "winning placement + push direction (exhaustive search).\n",
        "| # | label | grid | seed | fixed | patches | decoy gaps | P(random) "
        "| P(gap-fill) | P(wave-follow) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        if r.get("seed") is None:
            lines.append(f"| {i} | {r['label']} | {tuple(r['grid'])} | FAILED | | | | | | |")
            continue
        lines.append(
            f"| {i} | {r['label']} | {tuple(r['grid'])} | {r['seed']} | "
            f"{r['fixed_count']} | {r['patch_count']} | {r['decoy_gaps']} | "
            f"{r['p_random']:.3f} | {r['p_gapfill']:.2f} | {r['p_wavefill']:.2f} |")
    lines += ["", f"elapsed total: {time.time() - t_all:.1f}s"]
    with open(os.path.join(out_root, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[generate_50] done in {time.time() - t_all:.1f}s -> {out_root}")


if __name__ == "__main__":
    main()
