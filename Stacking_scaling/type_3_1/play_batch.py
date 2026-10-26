"""Generate a 10-level mini batch (easy / medium / hard).

Usage:
  python -m type_3_1.play_batch

Writes to `type_3_1/images_batch/`.  For each level, saves initial +
post-push + submit frames (front only, to save time; ~6 images per level).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

from .generator import generate_level, level_has_features
from .game_core import init_state
from .loader import save_level_to_json
from .mechanics import step_place, step_push, submit, submit_placement
from .solver import solve
from .visualizer_plotly import render_state_pair


# (label, kwargs)
LEVELS: List[Tuple[str, Dict]] = [
    # Easy (3): short line, no ramps/splitters, 0-1 P
    ("E1_easy_basic",   dict(grid_size=(4,3,3), target_skeleton_len=4,
                              n_patches=1, n_protected=0, allow_ramp=False)),
    ("E2_easy_prot",    dict(grid_size=(4,3,3), target_skeleton_len=4,
                              n_patches=1, n_protected=1, allow_ramp=False)),
    ("E3_easy_two",     dict(grid_size=(5,3,3), target_skeleton_len=5,
                              n_patches=2, n_protected=1, allow_ramp=False)),

    # Medium (4): with ramps, 1-2 patches, 1 P
    ("M1_med_ramp",     dict(grid_size=(5,4,4), target_skeleton_len=6,
                              n_patches=2, n_protected=1, allow_ramp=True)),
    ("M2_med_ramp2",    dict(grid_size=(5,4,4), target_skeleton_len=6,
                              n_patches=2, n_protected=2, allow_ramp=True)),
    ("M3_med_long",     dict(grid_size=(6,4,4), target_skeleton_len=7,
                              n_patches=3, n_protected=1, allow_ramp=True)),
    ("M4_med_wide",     dict(grid_size=(5,5,4), target_skeleton_len=6,
                              n_patches=2, n_protected=2, allow_ramp=True)),

    # Hard (3): splitters + ramps + several protected
    ("H1_hard_split",   dict(grid_size=(6,5,5), target_skeleton_len=8,
                              n_patches=3, n_protected=2, allow_ramp=True,
                              allow_split=True)),
    ("H2_hard_split2",  dict(grid_size=(6,5,5), target_skeleton_len=9,
                              n_patches=4, n_protected=2, allow_ramp=True,
                              allow_split=True)),
    ("H3_hard_big",     dict(grid_size=(7,5,5), target_skeleton_len=10,
                              n_patches=4, n_protected=3, allow_ramp=True,
                              allow_split=True)),
]


def _find_seed(kwargs: Dict, max_seed: int = 200) -> int:
    for strict in (True, False):
        for s in range(max_seed):
            r = generate_level(seed=s, **kwargs)
            if r is not None and level_has_features(*r, strict=strict, **kwargs):
                return s
    return -1


def _render_level(label: str, spec, meta, plan, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_level_to_json(spec, os.path.join(out_dir, "level.json"), meta=meta)

    st = init_state(spec)
    render_state_pair(st, out_dir, "00_initial", title=f"{label}  initial",
                       views=("front",))

    for act in plan:
        kind = act[0]
        if kind == "place":
            step_place(st, act[1], act[2])
        elif kind == "submit_placement":
            submit_placement(st)
            render_state_pair(st, out_dir, "01_ready",
                               title=f"{label}  ready to push",
                               views=("front",))
        elif kind == "push":
            step_push(st, act[1], act[2])
        elif kind == "submit":
            res = submit(st)
            render_state_pair(st, out_dir, "99_submit",
                               title=f"{label}  submit -> {res.code.value}",
                               views=("front",))


def main():
    out_root = "type_3_1/images_batch"
    os.makedirs(out_root, exist_ok=True)
    summary: List[Dict] = []
    t0 = time.time()

    for i, (label, kwargs) in enumerate(LEVELS, 1):
        print(f"[{i:2d}/{len(LEVELS)}] {label}  {kwargs}")
        seed = _find_seed(kwargs)
        if seed < 0:
            print("  !! no seed found")
            summary.append({"label": label, "seed": None,
                             "grid": kwargs["grid_size"]})
            continue
        r = generate_level(seed=seed, **kwargs)
        spec, meta = r
        plan = solve(spec)
        out_dir = os.path.join(out_root, f"{label}_seed{seed}")
        _render_level(label, spec, meta, plan, out_dir)
        print(f"  seed={seed}  skeleton={meta['skeleton_len']}  "
              f"patches={meta['patch_count']}  protected={meta['n_protected']}")
        summary.append({
            "label": label, "seed": seed,
            "grid": kwargs["grid_size"],
            "skeleton_len": meta["skeleton_len"],
            "fixed_count": meta["fixed_count"],
            "patch_count": meta["patch_count"],
            "n_protected": meta["n_protected"],
            "allow_ramp": kwargs.get("allow_ramp", False),
            "allow_split": kwargs.get("allow_split", False),
        })

    # SUMMARY.md
    lines = [
        "# Domino Chain Reaction 3D — 10-level batch\n",
        "| # | label | seed | grid | skeleton | fixed | patch | P | ramp | split |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(summary, 1):
        if r.get("seed") is None:
            lines.append(f"| {i} | {r['label']} | — | {r['grid']} | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {i} | {r['label']} | {r['seed']} | {r['grid']} | "
                f"{r['skeleton_len']} | {r['fixed_count']} | {r['patch_count']} | "
                f"{r['n_protected']} | {r['allow_ramp']} | {r['allow_split']} |")
    lines.append("")
    lines.append(f"elapsed: {time.time() - t0:.1f}s")
    with open(os.path.join(out_root, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n[play_batch] done in {time.time() - t0:.1f}s -> {out_root}")


if __name__ == "__main__":
    main()
