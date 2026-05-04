"""JSON load/save for Domino Chain Reaction 3D levels."""

from __future__ import annotations

import json
from typing import Dict, Optional

from .game_core import LevelSpec, PieceType, Vec3


def save_level_to_json(spec: LevelSpec, path: str,
                       meta: Optional[Dict] = None) -> None:
    data = {
        "grid_size": list(spec.grid_size),
        "fixed_pieces": [[pt.value, list(pos)] for pt, pos in spec.fixed_pieces],
        "inventory": {pt.value: n for pt, n in spec.inventory.items() if n > 0},
        "push_budget": spec.push_budget,
        "meta": meta or spec.meta or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_level_from_json(path: str) -> LevelSpec:
    with open(path) as f:
        data = json.load(f)
    fixed = [(PieceType(pt), tuple(pos)) for pt, pos in data.get("fixed_pieces", [])]
    inv = {PieceType(v): n for v, n in data.get("inventory", {}).items()}
    spec = LevelSpec(
        grid_size=tuple(data["grid_size"]),
        fixed_pieces=fixed,
        inventory=inv,
        push_budget=int(data.get("push_budget", 1)),
        meta=data.get("meta", {}),
    )
    return spec
