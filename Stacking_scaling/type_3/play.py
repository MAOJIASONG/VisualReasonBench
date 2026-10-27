"""
FluidNet 3D - Auto-play a level with the BFS solver.

Generates a level, solves it, plays through the plan step by step, renders a
3D PNG for every frame (initial + after each valve open), and also renders a
few *counterfactual* frames that show what would have happened had a
dangerous valve been opened.

Usage:
    python -m type_3.play                         # default seed=70
    python -m type_3.play --seed 70 --out type_3/images
"""

import argparse
import os
import sys
from typing import List, Tuple

from .game_core import ActionCode
from .generator import generate_level
from .loader import create_game_state, save_level_to_json
from .propagation import step_open_valve
from .solver import order_by_flow, solve
from .visualizer_plotly import render_state_pair


def _print_state_ascii(state, turn: int):
    spec = state.spec
    filled = sum(1 for t in spec.targets
                 if state.fluid_cells.get(t) == spec.cells[t].color)
    opened = len(state.opened_valves)
    print(f"  turn {turn}: targets {filled}/{len(spec.targets)}  "
          f"valves open {opened}/{len(spec.valves)}  "
          f"fluid cells {len(state.fluid_cells)}")


def play(seed: int, out_dir: str,
         size: Tuple[int, int, int] = (5, 5, 5),
         n_targets: int = 2, n_valves: int = 6, n_protected: int = 3,
         also_counterfactual: bool = True, tier: str = "medium") -> int:
    print(f"=== FluidNet auto-play  seed={seed} ===")
    result = generate_level(
        grid_size=size,
        n_targets=n_targets,
        n_valves=n_valves,
        n_protected=n_protected,
        seed=seed,
        tier=tier,
    )
    if result is None:
        print("generation failed"); return 1
    spec, meta = result

    print(f"\nlevel")
    print(f"  grid {spec.grid_size}")
    print(f"  sources   {spec.sources}")
    print(f"  targets   {spec.targets}  (colors: {[spec.cells[t].color for t in spec.targets]})")
    print(f"  valves    {spec.valves}")
    print(f"  protected {spec.protected}")
    print(f"  meta      {meta}")

    # Save JSON level for reference
    os.makedirs(out_dir, exist_ok=True)
    save_level_to_json(spec, os.path.join(out_dir, f"level_seed{seed}.json"), meta=meta)

    # Solve with BFS
    print("\nsolving with BFS over valve-subset space ...")
    plan = solve(spec)
    if plan is not None:
        plan = order_by_flow(spec, plan)
    if plan is None:
        print("  no solution (should not happen for generator output)")
        return 2
    print(f"  shortest plan: {len(plan)} opens -> {plan}")

    # Render initial (two angles)
    state = create_game_state(spec)
    _print_state_ascii(state, 0)
    render_state_pair(
        state, out_dir, f"seed{seed}_00_initial",
        title=f"FluidNet seed={seed}  initial state",
    )

    # Play the plan
    print("\nplaying the plan ...")
    for i, v in enumerate(plan, 1):
        r = step_open_valve(state, v)
        _print_state_ascii(state, i)
        print(f"    action: open {v} -> {r.code.value}  "
              f"new_fluid={len(r.new_fluid)}  newly_filled={sorted(r.newly_filled)}")
        render_state_pair(
            state, out_dir,
            f"seed{seed}_{i:02d}_open_{v[0]}_{v[1]}_{v[2]}",
            title=f"FluidNet seed={seed} step {i}: open {v} → {r.code.value}",
        )
        if state.terminal:
            break

    print(f"\nterminal: {state.terminal_code.value}")

    # Counterfactual: the earliest point where a wrong valve visibly floods
    # a protected cell (a still-dry trap would show nothing happening).
    if also_counterfactual:
        from .play_batch import _breaching_counterfactual
        cfx = _breaching_counterfactual(spec, plan)
        if cfx is not None:
            k, dv = cfx
            print(f"\ncounterfactual: after {k} plan steps, open {dv}")
            cf_state = create_game_state(spec)
            for v in plan[:k]:
                step_open_valve(cf_state, v)
            cf_r = step_open_valve(cf_state, dv)
            print(f"    -> {cf_r.code.value}  violation_cell={cf_r.violation_cell}")
            render_state_pair(
                cf_state, out_dir,
                f"seed{seed}_counterfactual_open_{dv[0]}_{dv[1]}_{dv[2]}",
                title=f"Mistake: open {dv} -> {cf_r.code.value}",
            )

    print(f"\nrenders saved to {out_dir}/")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=70)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "images"))
    ap.add_argument("--size", type=int, nargs=3, default=[5, 5, 5])
    ap.add_argument("--targets", type=int, default=2)
    ap.add_argument("--valves", type=int, default=6)
    ap.add_argument("--protected", type=int, default=3)
    ap.add_argument("--tier", choices=["easy", "medium", "hard"], default="medium")
    args = ap.parse_args()

    return play(
        seed=args.seed, out_dir=args.out,
        size=tuple(args.size), n_targets=args.targets,
        n_valves=args.valves, n_protected=args.protected, tier=args.tier,
    )


if __name__ == "__main__":
    sys.exit(main())
