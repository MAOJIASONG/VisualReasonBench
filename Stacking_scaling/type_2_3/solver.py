"""
PipeFit 3D — solver (rules v2).

`solve(spec)` returns a list of PlacedPiece (supports bottom-up, then
pipes) that wins: the generator's stored plan when present, otherwise the
first network found by the exhaustive search in difficulty.py.  (The v1
solver re-derived a route from a skeleton that was never saved with the
levels, so it could not solve any shipped level.)
"""

from __future__ import annotations

from typing import List, Optional

from .difficulty import Board, plan_of, solutions
from .game_core import GameState, LevelSpec, PieceType, PlacedPiece


def solve(spec: LevelSpec) -> Optional[List[PlacedPiece]]:
    plan = (spec.meta or {}).get("plan")
    if plan:
        return [PlacedPiece(PieceType(t), tuple(pos), o) for t, pos, o in plan]
    sols = solutions(spec, cap=1)
    if not sols:
        return None
    return plan_of(Board(spec), sols[0])


def replay(state: GameState, placements: List[PlacedPiece]):
    from .mechanics import step_place
    return [(p, step_place(state, p.type, p.pos, p.orientation)) for p in placements]
