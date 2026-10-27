"""
FluidNet 3D - Level Loader / Saver

Levels are stored as JSON.  Schema:

{
    "grid_size": [A, B, C],
    "cells": [
        {"pos": [x, y, z], "type": "pipe"},
        {"pos": [x, y, z], "type": "source",    "color": "blue"},
        {"pos": [x, y, z], "type": "target",    "color": "blue"},
        {"pos": [x, y, z], "type": "valve"},
        {"pos": [x, y, z], "type": "protected"}
    ],
    "meta": { ... optional level metadata ... }
}

EMPTY cells are not stored.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from .game_core import CellType, Vec3, Cell, LevelSpec, GameState


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------

def load_level_from_json(json_path: str) -> Optional[LevelSpec]:
    with open(json_path, "r") as f:
        data = json.load(f)
    grid_size = tuple(data["grid_size"])            # type: ignore[assignment]
    cells: Dict[Tuple[int, int, int], Cell] = {}
    for entry in data["cells"]:
        pos = tuple(entry["pos"])                   # type: ignore[assignment]
        ctype = CellType(entry["type"])
        color = entry.get("color")
        cells[pos] = Cell(pos=Vec3(*pos), type=ctype, color=color)
    return LevelSpec(grid_size=grid_size, cells=cells)


def save_level_to_json(spec: LevelSpec, json_path: str, meta: Optional[dict] = None) -> None:
    data = {
        "grid_size": list(spec.grid_size),
        "cells": [
            {
                "pos": list(cell.pos.to_tuple()),
                "type": cell.type.value,
                **({"color": cell.color} if cell.color is not None else {}),
            }
            for cell in spec.cells.values()
        ],
    }
    if meta is not None:
        data["meta"] = meta
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# GameState factory
# ---------------------------------------------------------------------------

def create_game_state(spec: LevelSpec) -> GameState:
    """Fresh GameState with propagation-from-sources run once (no valves open)."""
    from .propagation import _compute_full_fluid
    state = GameState(spec=spec)
    state.fluid_cells = _compute_full_fluid(state)
    return state


# ---------------------------------------------------------------------------
# Directory browser (mirrors stacking_game/loader.find_all_puzzles)
# ---------------------------------------------------------------------------

def find_all_levels(base_dir: str) -> List[Tuple[str, str]]:
    """Return list of (level_name, json_path) pairs under base_dir."""
    out: List[Tuple[str, str]] = []
    base = Path(base_dir)
    if not base.exists():
        return out
    for f in sorted(base.rglob("*.json")):
        rel = f.relative_to(base).with_suffix("")
        out.append((str(rel), str(f)))
    return out


def list_available_levels(base_dir: str) -> None:
    levels = find_all_levels(base_dir)
    print(f"Found {len(levels)} levels under {base_dir}")
    for name, path in levels:
        print(f"  - {name}")
