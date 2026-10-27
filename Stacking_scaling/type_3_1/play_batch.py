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

from .levels import CATEGORY, find_level
from .game_core import init_state
from .loader import save_level_to_json
from .mechanics import step_place, step_push, submit, submit_placement
from .solver import solve
from .visualizer_plotly import render_state_pair


# (label, category, tier): one level each, spanning easy -> hard.
LEVELS: List[Tuple[str, str, str]] = [
    ("E1_easy_line",    "A_line",  "easy"),
    ("E2_easy_decoy",   "B_decoy", "easy"),
    ("E3_easy_split",   "D_split", "easy"),
    ("M1_med_decoy",    "B_decoy", "medium"),
    ("M2_med_steps",    "C_steps", "medium"),
    ("M3_med_split",    "D_split", "medium"),
    ("M4_med_full",     "E_full",  "medium"),
    ("H1_hard_steps",   "C_steps", "hard"),
    ("H2_hard_split",   "D_split", "hard"),
    ("H3_hard_full",    "E_full",  "hard"),
]


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
    for entry in os.listdir(out_root):          # no stale levels
        full = os.path.join(out_root, entry)
        if os.path.isdir(full):
            for f in os.listdir(full):
                os.remove(os.path.join(full, f))
            os.rmdir(full)
    summary: List[Dict] = []
    seen: set = set()
    t0 = time.time()

    for i, (label, cat, tier) in enumerate(LEVELS, 1):
        kwargs = CATEGORY[cat][tier]
        print(f"[{i:2d}/{len(LEVELS)}] {label}  {cat}/{tier}")
        found = find_level(kwargs, tier, range(1000 * i, 1000 * i + 400), seen)
        if found is None:
            print("  !! no level found")
            summary.append({"label": label, "seed": None, "grid": kwargs["grid_size"]})
            continue
        seed, spec, meta = found
        out_dir = os.path.join(out_root, f"{label}_seed{seed}")
        _render_level(label, spec, meta, solve(spec), out_dir)
        d = meta["difficulty"]
        print(f"  seed={seed}  patches={meta['patch_count']}  {d}")
        summary.append({"label": label, "seed": seed, "grid": kwargs["grid_size"],
                        "cat": cat, "tier": tier, "fixed": meta["fixed_count"],
                        "patches": meta["patch_count"], **d})

    lines = [
        "# Domino Chain Reaction 3D — 10-level batch (rules v2)\n",
        "| # | label | category | tier | seed | grid | fixed | patches | decoy gaps "
        "| solutions | P(random) | P(gap-fill) | P(wave-follow) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(summary, 1):
        if r.get("seed") is None:
            lines.append(f"| {i} | {r['label']} | | | — | {r['grid']} | | | | | | | |")
            continue
        lines.append(
            f"| {i} | {r['label']} | {r['cat']} | {r['tier']} | {r['seed']} | "
            f"{tuple(r['grid'])} | {r['fixed']} | {r['patches']} | {r['decoy_gaps']} "
            f"| {r['n_solutions']} | {r['p_random']:.3f} | {r['p_gapfill']:.2f} "
            f"| {r['p_wavefill']:.2f} |")
    lines += ["",
              "P(random) = random legal placement + random push direction; "
              "P(gap-fill) = fill the holes nearest START and push at its "
              "neighbour; P(wave-follow) = follow the wave, filling whichever "
              "hole it stops at (see difficulty.py).",
              "", f"elapsed: {time.time() - t0:.1f}s"]
    with open(os.path.join(out_root, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\n[play_batch] done in {time.time() - t0:.1f}s -> {out_root}")


if __name__ == "__main__":
    main()
