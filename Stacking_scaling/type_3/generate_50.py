"""
FluidNet 3D — 50 concept puzzles.

Concepts (10 each = 3 easy + 4 medium + 3 hard), see concepts.py:
  trace, series, backdoor, two_pumps, mixed

Every level has exactly one minimal winning valve set, no tank is full at
the start, and every valve is either in that set or fatal when added to it.
Layouts are de-duplicated up to rotation, reflection and translation.
Tiers are one ladder for the whole game (concepts.TIER_SCORE): valves
<= 4 / 5-6 / 9-10 and score <= 3 / 4-6 / >= 7 for easy / medium / hard
(hard levels are pruned to their real decisions, concepts.prune_filler).

Per puzzle: 00_initial (front + back), 01_solved (front), 02_mistake (front:
the most tempting losing move), level.json and solution.txt.  level.json's
meta and solution.txt contain the answer: keep both out of model inputs.

    python -m type_3.generate_50
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chain_search import pick_levels, tier_table  # noqa: E402

if __package__ in (None, ""):
    from type_3.concepts import CONCEPTS, generate
    from type_3.difficulty import shortest_plan
    from type_3.game_core import ActionCode, LevelSpec
    from type_3.loader import create_game_state, save_level_to_json
    from type_3.propagation import step_open_valve
    from type_3.solver import order_by_flow, solve
    from type_3.visualizer_plotly import render_state_pair
else:
    from .concepts import CONCEPTS, generate
    from .difficulty import shortest_plan
    from .game_core import ActionCode, LevelSpec
    from .loader import create_game_state, save_level_to_json
    from .propagation import step_open_valve
    from .solver import order_by_flow, solve
    from .visualizer_plotly import render_state_pair

Pos = Tuple[int, int, int]
LOSS = {ActionCode.PROTECTED_BREACH, ActionCode.WRONG_COLOR_TARGET,
        ActionCode.COLOR_CONTAMINATION}


def layout_key(spec: LevelSpec) -> Tuple:
    pts = [(c.type.value, c.color or "", pos) for pos, c in spec.cells.items()]
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for swap in (False, True):
                t = []
                for kind, col, (x, y, z) in pts:
                    a, b = (y, x) if swap else (x, y)
                    t.append((kind, col, sx * a, sy * b, z))
                mx = min(p[2] for p in t)
                my = min(p[3] for p in t)
                key = tuple(sorted((k, c, a - mx, b - my, z) for k, c, a, b, z in t))
                best = key if best is None or key < best else best
    return best


def mistake(spec: LevelSpec, plan: List[Pos]) -> Optional[Tuple[int, Pos, str]]:
    """The most instructive losing move: the obvious fewest-valve route if
    it loses, else the earliest point where one extra valve loses visibly."""
    sp = shortest_plan(spec)
    st = create_game_state(spec)
    for i, v in enumerate(order_by_flow(spec, sp)):
        r = step_open_valve(st, v)
        if r.code in LOSS:
            return i, v, "the fewest-valve route"
    for k in range(len(plan) + 1):
        for v in spec.valves:
            if v in plan[:k]:
                continue
            st = create_game_state(spec)
            for u in plan[:k]:
                step_open_valve(st, u)
            if step_open_valve(st, v).code in LOSS:
                return k, v, "one valve too many"
    return None


def _make(slot: Tuple[str, str, str], seed: int):
    return generate(slot[1], slot[2], seed, max_attempts=300)


def _render(label: str, spec: LevelSpec, meta: Dict, plan: List[Pos],
            out_dir: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    save_level_to_json(spec, os.path.join(out_dir, "level.json"), meta=meta)
    st = create_game_state(spec)
    render_state_pair(st, out_dir, "00_initial", title=f"{label}  initial")
    for v in plan:
        step_open_valve(st, v)
    render_state_pair(st, out_dir, "01_solved",
                      title=f"{label}  solved ({len(plan)} valves)", views=("front",))
    mk = mistake(spec, plan)
    note = None
    if mk is not None:
        k, v, why = mk
        st = create_game_state(spec)
        prefix = (order_by_flow(spec, shortest_plan(spec))[:k]
                  if why.startswith("the fewest") else plan[:k])
        for u in prefix:
            step_open_valve(st, u)
        r = step_open_valve(st, v)
        note = f"{why}: open {v} -> {r.code.value}"
        render_state_pair(st, out_dir, "02_mistake",
                          title=f"{label}  mistake: open {v} -> {r.code.value}",
                          views=("front",))
    with open(os.path.join(out_dir, "solution.txt"), "w") as f:
        f.write(f"{label}\n")
        f.write(f"concept: {meta['concept']}  tier: {meta['tier']}\n")
        f.write(f"insight: {meta['insight']}\n")
        f.write(f"grid: {spec.grid_size}\n")
        f.write(f"pumps: {[(s, spec.cells[s].color) for s in spec.sources]}\n")
        f.write(f"tanks: {[(t, spec.cells[t].color) for t in spec.targets]}\n")
        f.write(f"valves: {spec.valves}\n")
        f.write(f"red cells: {spec.protected}\n")
        f.write(f"solution (open in this order): {plan}\n")
        if note:
            f.write(f"tempting mistake ({note})\n")
        f.write(f"difficulty: {meta['difficulty']}\n")
    return note


def main():
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "puzzles_50")
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)                  # no stale puzzles
    os.makedirs(out_root)

    plan: List[Tuple[str, str, str]] = []
    for concept in CONCEPTS:
        for tier, count in (("easy", 3), ("medium", 4), ("hard", 3)):
            for _ in range(count):
                plan.append((f"P{len(plan) + 1:02d}_{concept}_{tier}", concept, tier))

    rows: List[Dict] = []
    t_all = time.time()
    picks = pick_levels(plan, _make, layout_key, n_seeds=1200)  # series / backdoor hard are rare
    for i, ((label, concept, tier), found) in enumerate(zip(plan, picks), 1):
        print(f"=== [{i:2d}/{len(plan)}] {label}", flush=True)
        if found is None:
            print("  !! no level found")
            rows.append({"label": label, "concept": concept, "tier": tier, "seed": None})
            continue
        seed, spec, meta = found
        sol = order_by_flow(spec, solve(spec))
        out_dir = os.path.join(out_root, f"{label}_seed{seed}")
        note = _render(label, spec, meta, sol, out_dir)
        d = meta["difficulty"]
        print(f"  seed={seed}  {d}", flush=True)
        rows.append({"label": label, "concept": concept, "tier": tier, "seed": seed,
                     "grid": list(spec.grid_size), "out_dir": os.path.basename(out_dir),
                     "mistake": note, **d})

    lines = [
        "# FluidNet 3D — 50 concept puzzles\n",
        "Concepts: trace, series, backdoor, two_pumps, mixed (3 easy + 4 medium + "
        "3 hard each).  Unique minimal valve set; every valve is needed or fatal.\n",
        "| # | label | grid | seed | valves | plan | traps | P(random) | P(frontier) "
        "| P(lookahead) | shortest route wins | score |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        if r.get("seed") is None:
            lines.append(f"| {i} | {r['label']} | | FAILED | | | | | | | | |")
            continue
        lines.append(
            f"| {i} | {r['label']} | {tuple(r['grid'])} | {r['seed']} | {r['n_valves']} | "
            f"{r['min_plan']} | {r['traps']} | {r['p_random']:.3f} | {r['p_frontier']:.2f} | "
            f"{r['p_lookahead']:.2f} | {'yes' if r['shortest_ok'] else 'no'} | {r['score']:.1f} |")
    lines += [
        "",
        "P(random / frontier / lookahead) = exact win probability of a player who opens "
        "random valves / opens a wet valve not next to a red cell / simulates one valve "
        "ahead (difficulty.py).  'shortest route wins' = opening each tank's "
        "fewest-valve route wins.  score = log2 1/P(frontier) + 2 log2 1/P(lookahead) "
        "+ 1 if the shortest route loses.  Tiers are one ladder for the whole game "
        "(concepts.TIER_SCORE); ranges per tier:", "",
        *tier_table(rows, "tier", [("n_valves", "valves"), ("min_plan", "plan"),
                                   ("p_random", "P(random)"), ("p_frontier", "P(frontier)"),
                                   ("score", "score")]),
        "", f"elapsed total: {time.time() - t_all:.1f}s",
    ]
    with open(os.path.join(out_root, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[generate_50] done in {time.time() - t_all:.1f}s -> {out_root}")


if __name__ == "__main__":
    main()
