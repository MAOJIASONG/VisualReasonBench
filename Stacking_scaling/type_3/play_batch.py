"""
FluidNet 3D - Batch playthrough: 10 concept puzzles spanning easy → hard.

For each puzzle:
  1. Generate the level with given parameters and seed.
  2. Solve it with the BFS solver (`solver.solve`).
  3. Render the initial state + every state along the solver's plan +
     a counterfactual frame from opening one dangerous valve.

Output layout:
  type_3/images_batch/
    level_01_easy_seed5/
      00_initial_front.png
      00_initial_back.png
      01_open_<x>_<y>_<z>_front.png
      ...
      99_counterfactual_<x>_<y>_<z>_front.png
      level.json
      solution.txt
    ...
    SUMMARY.md
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chain_search import pick_levels, tier_table  # noqa: E402

if __package__ in (None, ""):
    from type_3.game_core import ActionCode
    from type_3.concepts import generate
    from type_3.loader import create_game_state, save_level_to_json
    from type_3.propagation import step_open_valve
    from type_3.solver import order_by_flow, solve
    from type_3.visualizer_plotly import render_state_pair
else:
    from .game_core import ActionCode
    from .concepts import generate
    from .loader import create_game_state, save_level_to_json
    from .propagation import step_open_valve
    from .solver import order_by_flow, solve
    from .visualizer_plotly import render_state_pair


# ---------------------------------------------------------------------------
# 10 puzzle specs spanning easy / medium / hard
# Each entry: (label, params, candidate_seeds_to_try)
# We try seeds in order until one yields a valid level meeting the criteria.
# ---------------------------------------------------------------------------

PUZZLES: List[Tuple[str, str, str]] = [
    # (label, concept, tier) -- one idea per level, see concepts.py.  Each
    # slot searches its own seed range so no two slots share a level.
    ("01_easy",   "trace",     "easy"),
    ("02_easy",   "series",    "easy"),
    ("03_easy",   "two_pumps", "easy"),
    ("04_medium", "trace",     "medium"),
    ("05_medium", "backdoor",  "medium"),
    ("06_medium", "two_pumps", "medium"),
    ("07_medium", "mixed",     "medium"),
    ("08_hard",   "series",    "hard"),
    ("09_hard",   "backdoor",  "hard"),
    ("10_hard",   "mixed",     "hard"),
]


def _make(slot: Tuple[str, str, str], seed: int):
    return generate(slot[1], slot[2], seed, max_attempts=300)


def _breaching_counterfactual(spec, plan):
    """(k, valve): after the first k plan steps, opening `valve` visibly
    breaches a protected cell -- the most instructive mistake to show.
    Opening a trap that is still dry would show nothing happening."""
    for k in range(len(plan) + 1):
        for dv in spec.valves:
            if dv in plan[:k]:
                continue
            st = create_game_state(spec)
            ok = True
            for v in plan[:k]:
                if step_open_valve(st, v).code != ActionCode.OK:
                    ok = False
                    break
            if not ok or dv in st.opened_valves:
                continue
            if step_open_valve(st, dv).code in (ActionCode.PROTECTED_BREACH,
                                                ActionCode.WRONG_COLOR_TARGET,
                                                ActionCode.COLOR_CONTAMINATION):
                return k, dv
    return None


def _render_episode(label: str, spec, meta, plan: List[Tuple[int,int,int]],
                    out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    save_level_to_json(spec, os.path.join(out_dir, "level.json"), meta=meta)

    state = create_game_state(spec)
    titles = [f"{label}  initial"]
    states = [state]

    # Render initial state
    render_state_pair(state, out_dir, "00_initial",
                      title=f"{label}  initial")

    # Step through plan
    for i, v in enumerate(plan, 1):
        r = step_open_valve(state, v)
        title = f"{label}  step {i}: open {v} -> {r.code.value}"
        prefix = f"{i:02d}_open_{v[0]}_{v[1]}_{v[2]}"
        render_state_pair(state, out_dir, prefix, title=title)

    # Counterfactual: the earliest point where one wrong valve visibly
    # floods a protected cell.
    cfx = _breaching_counterfactual(spec, plan)
    if cfx is not None:
        k, dv = cfx
        cf = create_game_state(spec)
        for v in plan[:k]:
            step_open_valve(cf, v)
        cf_r = step_open_valve(cf, dv)
        prefix = f"99_counterfactual_{dv[0]}_{dv[1]}_{dv[2]}"
        after = f"after step {k}, " if k else ""
        render_state_pair(cf, out_dir, prefix,
                          title=f"{label}  mistake: {after}open {dv} -> {cf_r.code.value}")

    # Plain-text solution log
    with open(os.path.join(out_dir, "solution.txt"), "w") as f:
        f.write(f"{label}\n")
        f.write(f"insight: {meta.get('insight', '')}\n")
        f.write(f"grid_size: {spec.grid_size}\n")
        f.write(f"sources:   {spec.sources}\n")
        f.write(f"targets:   {spec.targets}\n")
        f.write(f"valves:    {spec.valves}\n")
        f.write(f"protected: {spec.protected}\n")
        f.write(f"meta:      {meta}\n\n")
        f.write(f"BFS plan ({len(plan)} opens):\n")
        for i, v in enumerate(plan, 1):
            f.write(f"  step {i}: open {v}\n")
    return {"plan_len": len(plan)}


def main():
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "images_batch")
    if os.path.exists(out_root):
        # clean previous renders
        for entry in os.listdir(out_root):
            full = os.path.join(out_root, entry)
            if os.path.isdir(full):
                for f in os.listdir(full):
                    os.remove(os.path.join(full, f))
                os.rmdir(full)
            else:
                os.remove(full)
    os.makedirs(out_root, exist_ok=True)

    summary_lines = ["# FluidNet 3D — 10-puzzle batch playthrough (concepts)\n"]
    summary_lines.append("| # | label | concept | grid | seed | valves | plan | traps | P(random) | P(frontier) | P(lookahead) | shortest route wins | score |")
    summary_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows: List[Dict] = []

    t0 = time.time()
    from type_3.generate_50 import layout_key
    picks = pick_levels(PUZZLES, _make, layout_key, base=0, stride=1000, n_seeds=600, need=1)
    for idx, ((label, concept, tier), found) in enumerate(zip(PUZZLES, picks), 1):
        print(f"\n=== puzzle {idx:2d}: {label}  {concept}/{tier} ===")
        if found is not None:
            seed, spec, meta = found
            found = (seed, spec, meta, order_by_flow(spec, solve(spec)))
        if found is None:
            print("  !! no seed produced a valid level")
            summary_lines.append(f"| {idx} | {label} | {concept} | | FAILED | | | | | | | | |")
            continue
        seed, spec, meta, plan = found
        print(f"  seed={seed}  plan={plan}")
        out_dir = os.path.join(out_root, f"level_{label}_{concept}_seed{seed}")
        _render_episode(f"{label} {concept}", spec, meta, plan, out_dir)
        d = meta["difficulty"]
        summary_lines.append(
            f"| {idx} | {label} | {concept} | {spec.grid_size} | {seed} | {d['n_valves']} "
            f"| {len(plan)} | {d['traps']} | {d['p_random']:.2f} | {d['p_frontier']:.2f} "
            f"| {d['p_lookahead']:.2f} | {'yes' if d['shortest_ok'] else 'no'} | {d['score']:.1f} |")
        rows.append({"tier": tier, "seed": seed, **d})
        print(f"  rendered to {out_dir}/")

    summary_lines.append(
        "\nP(random / frontier / lookahead) = exact win probability of a player who "
        "opens random valves / opens a wet valve not next to a red cell / "
        "simulates one valve ahead (difficulty.py).\n")
    summary_lines += tier_table(rows, "tier", [("n_valves", "valves"), ("score", "score")])
    summary_lines.append(f"\nelapsed: {time.time() - t0:.1f}s")
    with open(os.path.join(out_root, "SUMMARY.md"), "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nall puzzles processed -> {out_root}")
    print("see SUMMARY.md for the table")


if __name__ == "__main__":
    main()
