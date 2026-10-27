"""JSON load/save for PipeFit levels."""

from __future__ import annotations

import json
from typing import Dict, Optional

from .game_core import LevelSpec, PieceType, Vec3


def save_level_to_json(spec: LevelSpec, path: str,
                       meta: Optional[Dict] = None) -> None:
    data = {
        "grid_size": list(spec.grid_size),
        "entrances": [list(p) for p in spec.entrances],
        "exits": [list(p) for p in spec.exits],
        "obstacles": [list(p) for p in spec.obstacles],
        "obstacle_shapes": [[list(c) for c in shape] for shape in spec.obstacle_shapes],
        "inventory": {pt.value: n for pt, n in spec.inventory.items() if n > 0},
        "meta": meta or spec.meta,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_level_from_json(path: str) -> LevelSpec:
    with open(path) as f:
        data = json.load(f)
    inv = {PieceType(v): n for v, n in data.get("inventory", {}).items()}
    spec = LevelSpec(
        grid_size=tuple(data["grid_size"]),
        entrances=[tuple(p) for p in data["entrances"]],
        exits=[tuple(p) for p in data["exits"]],
        obstacles=[tuple(p) for p in data.get("obstacles", [])],
        obstacle_shapes=[[tuple(c) for c in shape]
                         for shape in data.get("obstacle_shapes", [])],
        inventory=inv,
        meta=data.get("meta", {}),
    )
    return spec
