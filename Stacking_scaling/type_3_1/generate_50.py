"""
Domino Chain Reaction 3D — 50 diverse puzzles.

Categories (10 each):
  A_straight  — basic line, no ramps / splitters / protected
  B_protect   — basic line WITH protected near the path
  C_ramp      — ramps enabled, no splitters
  D_split     — splitters enabled (multi-goal), no ramps
  E_full      — ramps + splitters + protected (full difficulty)

Per puzzle we render only (initial, submit) × front (2 PNGs).  We also
save level.json + solution.txt so a consumer can replay step-by-step.
"""

from __future__ import annotations

import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from type_3_1.generator import generate_level, level_has_features
    from type_3_1.game_core import init_state
    from type_3_1.loader import save_level_to_json
    from type_3_1.mechanics import step_place, step_push, submit, submit_placement
    from type_3_1.solver import solve
    from type_3_1.visualizer_plotly import render_state_pair
else:
    from .generator import generate_level, level_has_features
    from .game_core import init_state
    from .loader import save_level_to_json
    from .mechanics import step_place, step_push, submit, submit_placement
    from .solver import solve
    from .visualizer_plotly import render_state_pair


# difficulty triple: (grid, skeleton_len, n_patches)
TIERS: Dict[str, List[Tuple[Tuple[int, int, int], int, int]]] = {
    "easy":   [((4, 3, 3), 4, 1),
                ((5, 3, 3), 5, 1),
                ((5, 3, 3), 4, 2)],
    "medium": [((5, 4, 4), 6, 2),
                ((6, 4, 4), 7, 3),
                ((5, 4, 4), 6, 3)],
    "hard":   [((6, 5, 5), 8, 3),
                ((6, 5, 5), 9, 4),
                ((7, 5, 5), 10, 4)],
}


def _make_plan() -> List[Tuple[str, str, str, dict]]:
    """Return (label, category, difficulty, kwargs) × 50."""
    plan: List[Tuple[str, str, str, dict]] = []
    idx = 1

    def add_category(cat: str, per_tier_kwargs_fn):
        nonlocal idx
        # 3 easy + 4 medium + 3 hard = 10 per category
        for tier, count in [("easy", 3), ("medium", 4), ("hard", 3)]:
            for i in range(count):
                gs, skel, pat = TIERS[tier][i % len(TIERS[tier])]
                kwargs = per_tier_kwargs_fn(gs, skel, pat, tier)
                label = f"P{idx:02d}_{cat}_{tier}"
                plan.append((label, cat, tier, kwargs))
                idx += 1

    # A_straight: plain dominos only
    add_category("A_straight", lambda gs, skel, pat, tier: dict(
        grid_size=gs, target_skeleton_len=skel, n_patches=pat,
        n_protected=0, allow_ramp=False, allow_split=False))

    # B_protect: dominos + 1-2 protected
    add_category("B_protect", lambda gs, skel, pat, tier: dict(
        grid_size=gs, target_skeleton_len=skel, n_patches=pat,
        n_protected=(1 if tier == "easy" else 2),
        allow_ramp=False, allow_split=False))

    # C_ramp: ramps allowed
    add_category("C_ramp", lambda gs, skel, pat, tier: dict(
        grid_size=gs, target_skeleton_len=skel, n_patches=pat,
        n_protected=(1 if tier != "hard" else 2),
        allow_ramp=True, allow_split=False))

    # D_split: splitters allowed
    add_category("D_split", lambda gs, skel, pat, tier: dict(
        grid_size=gs, target_skeleton_len=skel, n_patches=pat,
        n_protected=(1 if tier != "hard" else 2),
        allow_ramp=False, allow_split=True))

    # E_full: ramps + splitters + 2-3 protected
    add_category("E_full", lambda gs, skel, pat, tier: dict(
        grid_size=gs, target_skeleton_len=skel, n_patches=pat,
        n_protected=(1 if tier == "easy" else (2 if tier == "medium" else 3)),
        allow_ramp=True, allow_split=True))

    return plan


def _find_seed(kwargs: dict, used: set, max_seed: int = 300
               ) -> Optional[int]:
    """First seed that generates a level with the category's features and
    has not already been used for an identical config (several slots in a
    tier share a config, which previously produced duplicate puzzles).
    Strict feature matching first; relaxed only if no seed qualifies."""
    key = repr(sorted(kwargs.items()))
    for strict in (True, False):
        for s in range(max_seed):
            if (key, s) in used:
                continue
            r = generate_level(seed=s, max_attempts=60, **kwargs)
            if r is not None and level_has_features(*r, strict=strict, **kwargs):
                used.add((key, s))
                return s
    return None


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
    # solution.txt
    with open(os.path.join(out_dir, "solution.txt"), "w") as f:
        f.write(f"{label}\n")
        f.write(f"grid: {spec.grid_size}\n")
        f.write(f"fixed: {[(pt.value, pos) for pt, pos in spec.fixed_pieces]}\n")
        f.write(f"inventory: {{ {', '.join(pt.value+':'+str(n) for pt,n in spec.inventory.items() if n>0)} }}\n")
        f.write(f"push_budget: {spec.push_budget}\n")
        f.write(f"patches: {meta['patch_pieces']}\n")
        f.write(f"pushes:  {meta['pushes']}\n")
        f.write(f"origin:  {meta['origin']} dir={meta['origin_dir']}\n")
        f.write(f"skeleton_len: {meta['skeleton_len']}  fixed={meta['fixed_count']}  patches={meta['patch_count']}\n")
        f.write("\nplan:\n")
        for i, act in enumerate(plan, 1):
            f.write(f"  {i:02d}: {act}\n")


def main():
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "puzzles_50")
    os.makedirs(out_root, exist_ok=True)

    plan = _make_plan()
    print(f"[generate_50] plan = {len(plan)} puzzles")
    cats = sorted({c for _, c, _, _ in plan})
    for c in cats:
        n = sum(1 for _, cat, _, _ in plan if cat == c)
        print(f"  {c}: {n}")

    summary_rows: List[dict] = []
    used_seeds: set = set()
    t_all = time.time()
    for i, (label, cat, diff, kwargs) in enumerate(plan, 1):
        t0 = time.time()
        print(f"\n=== [{i:2d}/{len(plan)}] {label}  cat={cat}  diff={diff}")
        seed = _find_seed(kwargs, used_seeds)
        if seed is None:
            print("  !! no seed found")
            summary_rows.append({"label": label, "cat": cat, "diff": diff,
                                  "seed": None, "grid": kwargs["grid_size"]})
            continue
        r = generate_level(seed=seed, **kwargs)
        spec, meta = r
        solver_plan = solve(spec)
        out_dir = os.path.join(out_root, f"{label}_seed{seed}")
        _render_one(label, spec, meta, solver_plan, out_dir)
        dt = time.time() - t0
        print(f"  seed={seed}  skel={meta['skeleton_len']}  "
              f"fixed={meta['fixed_count']}  patch={meta['patch_count']}  "
              f"P={meta['n_protected']}  ({dt:.1f}s)")
        summary_rows.append({
            "label": label, "cat": cat, "diff": diff, "seed": seed,
            "grid": list(kwargs["grid_size"]),
            "skeleton_len": meta["skeleton_len"],
            "fixed_count": meta["fixed_count"],
            "patch_count": meta["patch_count"],
            "n_protected": meta["n_protected"],
            "allow_ramp": kwargs.get("allow_ramp", False),
            "allow_split": kwargs.get("allow_split", False),
            "origin": meta["origin"],
            "origin_dir": meta["origin_dir"],
            "inventory": {pt.value: n for pt, n in spec.inventory.items() if n > 0},
            "out_dir": os.path.basename(out_dir),
        })

    # SUMMARY.md
    lines = [
        "# Domino Chain Reaction 3D — 50-puzzle diverse batch\n",
        "Categories:",
        "- **A_straight** — basic line, no ramps / splitters / protected (10)",
        "- **B_protect**  — basic line WITH protected dominos (10)",
        "- **C_ramp**     — ramps enabled, no splitters (10)",
        "- **D_split**    — splitters enabled (multi-goal), no ramps (10)",
        "- **E_full**     — ramps + splitters + protected, full difficulty (10)",
        "",
        "Total: 50 puzzles across 5 categories × 3 difficulty tiers (easy/medium/hard).",
        "",
        "| # | label | cat | diff | grid | seed | skel | fixed | patch | P | ramp | split | inventory |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(summary_rows, 1):
        if r.get("seed") is None:
            lines.append(f"| {i} | {r['label']} | {r['cat']} | {r['diff']} | "
                          f"{r['grid']} | — | — | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {i} | {r['label']} | {r['cat']} | {r['diff']} | "
                f"{r['grid']} | {r['seed']} | {r['skeleton_len']} | "
                f"{r['fixed_count']} | {r['patch_count']} | "
                f"{r['n_protected']} | {r['allow_ramp']} | {r['allow_split']} | "
                f"{r['inventory']} |")
    lines.append("")
    lines.append(f"elapsed total: {time.time() - t_all:.1f}s")

    with open(os.path.join(out_root, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    print(f"\n[generate_50] done in {time.time() - t_all:.1f}s -> {out_root}")
    print("[generate_50] see SUMMARY.md and index.json")


if __name__ == "__main__":
    main()
