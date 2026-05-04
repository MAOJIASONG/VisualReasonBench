"""
Domino Chain Reaction 3D — Replay validator ("solver").

The generator records the canonical solution in `spec.meta`:
    meta["patch_pieces"] = [("D"|"R", [x,y,z]), ...]
    meta["pushes"]       = [([x,y,z], "+x"|...), ...]

`solve(spec)` returns a plan: a list of Action tuples the mechanics layer
can replay.  Action forms:
    ("place", PieceType, (x,y,z))
    ("submit_placement",)
    ("push", (x,y,z), dir_name)
    ("submit",)

`replay(state, plan)` applies the plan via mechanics and returns a list of
(action, ActionResult).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .game_core import GameState, LevelSpec, PieceType, Vec3
from .mechanics import step_place, step_push, submit, submit_placement


Action = tuple  # ("place", pt, pos) | ("submit_placement",) | ("push", pos, dir) | ("submit",)


def solve(spec: LevelSpec) -> Optional[List[Action]]:
    meta = spec.meta or {}
    patches = meta.get("patch_pieces")
    pushes = meta.get("pushes")
    if patches is None or pushes is None:
        return None
    plan: List[Action] = []
    for pt_str, pos in patches:
        pt = PieceType(pt_str)
        plan.append(("place", pt, tuple(pos)))
    plan.append(("submit_placement",))
    for pos, d in pushes:
        plan.append(("push", tuple(pos), d))
    plan.append(("submit",))
    return plan


def replay(state: GameState, plan: List[Action]):
    log = []
    for act in plan:
        kind = act[0]
        if kind == "place":
            _, pt, pos = act
            r = step_place(state, pt, pos)
        elif kind == "submit_placement":
            r = submit_placement(state)
        elif kind == "push":
            _, pos, d = act
            r = step_push(state, pos, d)
        elif kind == "submit":
            r = submit(state)
        else:
            raise ValueError(f"unknown action {act}")
        log.append((act, r))
    return log
