"""
FluidNet 3D - Minimal usage example.

Generate a level, render initial/final states, open a few valves programmatically.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from type_3.generator import generate_level
from type_3.loader import create_game_state
from type_3.propagation import step_open_valve
from type_3.visualizer_3d import render_state


def main():
    print("=== FluidNet 3D — example ===\n")
    result = generate_level(
        grid_size=(5, 5, 5),
        n_targets=2,
        n_valves=5,
        n_protected=2,
        seed=42,
    )
    assert result is not None, "generation failed; try a different seed"
    spec, meta = result

    print(f"level: grid {spec.grid_size}")
    print(f"  sources:    {spec.sources}")
    print(f"  targets:    {spec.targets}")
    print(f"  valves:     {spec.valves}")
    print(f"  protected:  {spec.protected}")
    print(f"  meta:       {meta}")

    state = create_game_state(spec)

    # Render initial state
    out_dir = "/tmp/fluidnet_example"
    os.makedirs(out_dir, exist_ok=True)
    render_state(state, save_path=os.path.join(out_dir, "00_initial.png"),
                 title="FluidNet — initial")

    # Play a known-safe valve sequence from meta.
    example = [tuple(v) for v in meta["min_safe_example"]]
    print(f"\nplaying safe sequence: {example}")
    for i, v in enumerate(example):
        r = step_open_valve(state, v)
        print(f"  step {i+1}: open {v} -> {r.code.value}")
        render_state(state,
                     save_path=os.path.join(out_dir, f"{i+1:02d}_after_{v}.png"),
                     title=f"after open {v}")
        if state.terminal:
            break

    print(f"\nterminal code: {state.terminal_code.value if state.terminal_code else 'N/A'}")
    print(f"images saved under {out_dir}")


if __name__ == "__main__":
    main()
