"""PipeFit 3D — single-level demo with solver-replay rendering."""

from __future__ import annotations

import os
import sys
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from type_2_3.generator import generate_level
    from type_2_3.solver import solve
    from type_2_3.game_core import init_state
    from type_2_3.mechanics import submit, step_place
    from type_2_3.visualizer_plotly import render_state_pair
    from type_2_3.loader import save_level_to_json
else:
    from .generator import generate_level
    from .solver import solve
    from .game_core import init_state
    from .mechanics import submit, step_place
    from .visualizer_plotly import render_state_pair
    from .loader import save_level_to_json


def play(seed: int = 1,
         grid_size=(5, 4, 4),
         n_obstacles: int = 1,
         out_dir: Optional[str] = None) -> None:
    r = generate_level(seed=seed, grid_size=grid_size,
                        n_obstacles=n_obstacles, max_attempts=300)
    if r is None:
        print(f"[seed {seed}] generator failed")
        return
    spec, meta = r

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "images", f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    save_level_to_json(spec, os.path.join(out_dir, "level.json"), meta=meta)

    print(f"[seed {seed}] grid={grid_size}  E={spec.entrances[0]}  X={spec.exits[0]}  "
          f"obs={spec.obstacles}  plan_len={meta['plan_length']}")

    state = init_state(spec)
    render_state_pair(state, out_dir, "00_initial",
                      title=f"PipeFit seed={seed}   initial")

    plan = solve(spec)
    assert plan is not None
    for i, p in enumerate(plan, 1):
        r = step_place(state, p.type, p.pos, p.orientation)
        assert r.code.value == "OK", (r, p)
        title = f"PipeFit seed={seed}   step {i}/{len(plan)}  place {p.type.value} at {p.pos}"
        prefix = f"{i:02d}_{p.type.value}_{p.pos[0]}_{p.pos[1]}_{p.pos[2]}"
        render_state_pair(state, out_dir, prefix, title=title)

    res = submit(state)
    render_state_pair(state, out_dir, "98_submit",
                      title=f"PipeFit seed={seed}   submit -> {res.code.value}")

    # Counterfactual: remove one anchor-critical support and show violation
    # (optional: pick the first unanchored pipe cell)
    from .mechanics import unanchored_pipe_cells, step_remove
    cf_state = init_state(spec)
    for p in plan:
        step_place(cf_state, p.type, p.pos, p.orientation)
    # Remove the first support to create a violation
    sup = next((p for p in plan if p.type.value in ("S1", "S2")), None)
    if sup is not None:
        step_remove(cf_state, sup.pos)
        cf_res = submit(cf_state)
        prefix = f"99_counterfactual_remove_{sup.type.value}_{sup.pos[0]}_{sup.pos[1]}_{sup.pos[2]}"
        render_state_pair(cf_state, out_dir, prefix,
                          title=f"PipeFit seed={seed}   CF remove {sup.type.value}@{sup.pos} -> {cf_res.code.value}")

    # Write solution text
    with open(os.path.join(out_dir, "solution.txt"), "w") as f:
        f.write(f"seed: {seed}\n")
        f.write(f"grid_size: {spec.grid_size}\n")
        f.write(f"entrances: {spec.entrances}\n")
        f.write(f"exits:     {spec.exits}\n")
        f.write(f"obstacles: {spec.obstacles}\n")
        f.write(f"inventory: { {pt.value: n for pt, n in spec.inventory.items() if n > 0} }\n")
        f.write(f"meta:      {meta}\n\n")
        f.write(f"BFS plan ({len(plan)} pieces):\n")
        for i, p in enumerate(plan, 1):
            f.write(f"  step {i}: place {p.type.value} at {p.pos} (orient={p.orientation})\n")

    print(f"[seed {seed}] rendered to {out_dir}/")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    play(seed=seed)


if __name__ == "__main__":
    main()
