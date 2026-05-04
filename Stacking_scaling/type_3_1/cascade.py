"""
Domino Chain Reaction 3D — Pulse simulator.

A cascade is driven by pulses.  A pulse = (from_anchor, dir, z) where:
  from_anchor = the anchor-cell we're "flying out of" after it toppled
  dir         = horizontal direction ("+x"/"-x"/"+y"/"-y")
  z           = z-level of the pulse (standing piece's BASE-z is what we
                look up to decide whether the pulse hits it)

Cascade is deterministic: every piece topples at most once, and the
set of toppled pieces is independent of pulse processing order.  We
process pulses in FIFO for a stable fall_order that renderers can use
to show the wave expanding outward.

Rules:

  1. target_base = (from_anchor + dir_vec) at height z.
  2. If target_base is out-of-bounds → pulse dies.
  3. Look up the piece at target_base.  If no piece, or the piece is a
     standing piece whose BASE is not at z, or the piece is already
     toppled → pulse dies.  (A standing piece is "at z" iff its
     anchor.z == z.  A ramp is "at z" iff ramp.z == z.)
  4. Dispatch by piece type:

     - DOMINO (D): topple, emit one pulse (target_base, dir, z).
     - GOAL   (G): topple, emit one pulse (target_base, dir, z).
     - PROTECTED (P): topple, raise game-over flag, emit one pulse
       (target_base, dir, z) — we still propagate so the renderer shows
       the full damage.
     - SPLITTER (T): topple, emit two pulses, one per orthogonal dir.
       Straight-through is NOT continued.  (dir_vec and -dir_vec both
       disallowed; the two orthogonal directions get one pulse each.)
     - RAMP (R): activate, emit one pulse (target_base, dir, z+1).

  Exit condition: pulse queue empty.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .game_core import (
    DIR_NAME_TO_VEC, GameState, PieceType, PlacedPiece, Vec3,
    in_grid, perp_dirs,
)


@dataclass
class CascadeResult:
    toppled: List[Vec3]                         # anchor positions in fall order
    events: List[Dict]                          # per-event log (see below)
    game_over: bool                             # True iff any PROTECTED toppled


def _find_target(state: GameState, from_pos: Vec3, dir_name: str,
                 z: int) -> Optional[PlacedPiece]:
    """
    Return the piece whose BASE sits at (from_pos + dir_vec) with base-z == z.
    None if OOB, empty, or the occupying piece's base is at a different z.
    """
    dx, dy, _ = DIR_NAME_TO_VEC[dir_name]
    target_base = (from_pos[0] + dx, from_pos[1] + dy, z)
    if not in_grid(target_base, state.spec.grid_size):
        return None
    piece = state.piece_at_cell(target_base)
    if piece is None:
        return None
    # The pulse "hits" the piece only if the target_base IS the piece's
    # anchor (base cell).  This prevents phantom hits on the upper half of
    # a standing piece, or on a piece that stands at a higher base but
    # extends down through this cell (impossible since bases are at z).
    if piece.pos != target_base:
        return None
    return piece


def simulate_push(state: GameState, origin_anchor: Vec3,
                  dir_name: str) -> CascadeResult:
    """
    Run a cascade from the initial push.  Mutates `state.pieces`: sets
    `toppled`, `fall_dir`, `fall_order` on every hit piece.  Records the
    cascade log into state.last_cascade and appends to all_cascades.
    Returns a CascadeResult for inspection.

    Preconditions (validate in mechanics.step_push before calling):
      - origin is a standing (D/T/G/P) piece, not toppled, and exists.
      - dir_name in {+x,-x,+y,-y}.
    """
    origin = state.piece_at_cell(origin_anchor)
    assert origin is not None, f"no piece at {origin_anchor}"
    assert origin.type in (PieceType.DOMINO, PieceType.SPLITTER,
                            PieceType.GOAL, PieceType.PROTECTED), \
        f"can't push {origin.type}"
    assert not origin.toppled, "origin already toppled"

    events: List[Dict] = []
    toppled_order: List[Vec3] = []
    game_over = False
    fall_counter = 0

    def topple(piece: PlacedPiece, fall_dir: Optional[str]) -> None:
        nonlocal fall_counter
        piece.toppled = True
        piece.fall_dir = fall_dir
        piece.fall_order = fall_counter
        fall_counter += 1
        toppled_order.append(piece.pos)

    # Topple the origin first (the push itself).  It falls in dir_name.
    topple(origin, dir_name)
    events.append({
        "kind": "push",
        "anchor": origin_anchor,
        "type": origin.type.value,
        "dir": dir_name,
        "z": origin.pos[2],
    })
    if origin.type == PieceType.PROTECTED:
        game_over = True

    # Seed pulse queue with the origin's outgoing wave.
    queue: deque = deque()
    if origin.type == PieceType.SPLITTER:
        for pd in perp_dirs(dir_name):
            queue.append((origin.pos, pd, origin.pos[2]))
    elif origin.type == PieceType.RAMP:
        # Ramps can't be origins, but mirror the behavior anyway.
        queue.append((origin.pos, dir_name, origin.pos[2] + 1))
    else:
        # D / G / P: one pulse in the push direction
        queue.append((origin.pos, dir_name, origin.pos[2]))

    while queue:
        from_anchor, d, z = queue.popleft()
        target = _find_target(state, from_anchor, d, z)
        if target is None:
            events.append({"kind": "pulse_die", "from": from_anchor,
                            "dir": d, "z": z, "reason": "empty_or_oob"})
            continue
        if target.toppled:
            events.append({"kind": "pulse_die", "from": from_anchor,
                            "dir": d, "z": z, "reason": "already_toppled",
                            "target": target.pos})
            continue

        # Dispatch by type
        pt = target.type
        if pt == PieceType.DOMINO or pt == PieceType.GOAL:
            topple(target, d)
            events.append({"kind": "topple", "anchor": target.pos,
                            "type": pt.value, "dir": d, "z": z})
            queue.append((target.pos, d, z))
        elif pt == PieceType.PROTECTED:
            topple(target, d)
            events.append({"kind": "topple", "anchor": target.pos,
                            "type": pt.value, "dir": d, "z": z,
                            "protected": True})
            game_over = True
            queue.append((target.pos, d, z))
        elif pt == PieceType.SPLITTER:
            topple(target, d)
            events.append({"kind": "topple", "anchor": target.pos,
                            "type": pt.value, "dir": d, "z": z,
                            "split": True})
            for pd in perp_dirs(d):
                queue.append((target.pos, pd, z))
        elif pt == PieceType.RAMP:
            # Ramp activates; re-emerge at z+1
            topple(target, None)
            events.append({"kind": "ramp", "anchor": target.pos,
                            "dir": d, "z_in": z, "z_out": z + 1})
            queue.append((target.pos, d, z + 1))
        else:
            events.append({"kind": "pulse_die", "from": from_anchor,
                            "dir": d, "z": z,
                            "reason": f"unknown_piece_{pt.value}"})

    result = CascadeResult(
        toppled=toppled_order,
        events=events,
        game_over=game_over,
    )
    state.last_cascade = events
    state.all_cascades.append(events)
    return result
