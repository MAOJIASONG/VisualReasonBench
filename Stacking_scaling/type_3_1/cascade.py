"""
Domino Chain Reaction 3D — cascade simulator (rules v2, physical).

A wave is (falling piece, direction).  The falling piece stands on base
height z and is two cells tall, so when it tips over it sweeps the next
column in its direction at heights z and z+1.  It knocks over every
standing, not-yet-toppled piece whose body occupies one of those two cells.
Consequences, all visible in the image:

  * a domino can hit a piece standing one STEP higher (its upper half meets
    the other's lower half), or one step lower (its lower half meets the
    other's upper half), but not two steps away;
  * a grey STEP block in the swept cells is solid: if nothing standing was
    hit, the wave stops there;
  * an empty column stops the wave (that is the gap a patch must fill).

What a hit piece does:
  D, S, G, P : falls in the same direction and passes the wave on.
               (P falling loses the game; we still finish the cascade so
               the renderer can show the damage.)
  T          : falls, but sends the wave out to both sides (the two
               directions perpendicular to the incoming one), not straight.

The cascade is deterministic; every piece topples at most once.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

from .game_core import (
    DIR_NAME_TO_VEC, GameState, PieceType, PlacedPiece, Vec3,
    in_grid, perp_dirs, piece_is_standing,
)


@dataclass
class CascadeResult:
    toppled: List[Vec3]                         # anchor positions in fall order
    events: List[Dict]                          # per-event log (see below)
    game_over: bool                             # True iff any PROTECTED toppled


def swept_cells(piece: PlacedPiece, dir_name: str) -> List[Vec3]:
    """Cells of the next column that a piece falling in `dir_name` sweeps."""
    dx, dy, _ = DIR_NAME_TO_VEC[dir_name]
    x, y, z = piece.pos
    return [(x + dx, y + dy, z + 1), (x + dx, y + dy, z)]


def hits(state: GameState, piece: PlacedPiece, dir_name: str
         ) -> List[PlacedPiece]:
    """Standing, not-toppled pieces knocked by `piece` falling that way."""
    out: List[PlacedPiece] = []
    for c in swept_cells(piece, dir_name):
        if not in_grid(c, state.spec.grid_size):
            continue
        p = state.piece_at_cell(c)
        if p is None or p in out or p.toppled:
            continue
        if piece_is_standing(p.type):
            out.append(p)
    return out


def simulate_push(state: GameState, origin_anchor: Vec3,
                  dir_name: str) -> CascadeResult:
    """Push the piece at `origin_anchor` and run the whole cascade.

    Mutates `state.pieces` (toppled / fall_dir / fall_order) and appends the
    event log to state.all_cascades.  mechanics.step_push validates that the
    origin is the START domino.
    """
    origin = state.piece_at_cell(origin_anchor)
    assert origin is not None and piece_is_standing(origin.type)
    assert not origin.toppled

    events: List[Dict] = []
    toppled_order: List[Vec3] = []
    game_over = False
    counter = 0

    def topple(piece: PlacedPiece, fall_dir: str) -> None:
        nonlocal counter, game_over
        piece.toppled = True
        piece.fall_dir = fall_dir
        piece.fall_order = counter
        counter += 1
        toppled_order.append(piece.pos)
        if piece.type == PieceType.PROTECTED:
            game_over = True

    def outgoing(piece: PlacedPiece, d: str) -> List[str]:
        return perp_dirs(d) if piece.type == PieceType.SPLITTER else [d]

    topple(origin, dir_name)
    events.append({"kind": "push", "anchor": origin.pos,
                   "type": origin.type.value, "dir": dir_name,
                   "z": origin.pos[2]})
    queue: deque = deque((origin, d) for d in outgoing(origin, dir_name))

    while queue:
        src, d = queue.popleft()
        struck = hits(state, src, d)
        if not struck:
            events.append({"kind": "wave_stop", "from": src.pos, "dir": d})
            continue
        for p in struck:
            if p.toppled:
                continue
            topple(p, d)
            events.append({"kind": "topple", "anchor": p.pos, "from": src.pos,
                           "type": p.type.value, "dir": d, "z": p.pos[2],
                           "protected": p.type == PieceType.PROTECTED,
                           "split": p.type == PieceType.SPLITTER})
            for nd in outgoing(p, d):
                queue.append((p, nd))

    state.last_cascade = events
    state.all_cascades.append(events)
    return CascadeResult(toppled=toppled_order, events=events,
                         game_over=game_over)
