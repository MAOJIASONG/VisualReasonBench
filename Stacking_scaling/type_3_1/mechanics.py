"""
Domino Chain Reaction 3D — placement + push mechanics (rules v2).

Phase PLACE:
  step_place(state, PieceType.DOMINO, pos) -> ActionResult
  step_remove(state, pos)                  -> ActionResult
  submit_placement(state)                  -> ActionResult  (to PUSH; needs
                                                             empty inventory)
Phase PUSH:
  step_push(state, start_pos, dir_name)    -> ActionResult  (runs cascade;
                                                             START only, once)
  submit(state)                            -> ActionResult  (WIN / LOSE)

Support rule: a domino stands on the ground (z=1) or directly on a grey
STEP block.  Nothing stands on another domino.
"""

from __future__ import annotations

from typing import List, Optional

from .cascade import simulate_push
from .game_core import (
    ActionCode, ActionResult, DIR_NAME_TO_VEC, GameState, Phase, PieceType,
    PlacedPiece, Vec3, in_grid, piece_is_ramp, piece_is_standing,
    piece_occupied_cells, piece_top_cell,
)


# Patches are plain dominoes only.
PATCHABLE_TYPES = (PieceType.DOMINO,)


def _cell_free(state: GameState, cell: Vec3) -> ActionResult:
    if not in_grid(cell, state.spec.grid_size):
        return ActionResult(ActionCode.OUT_OF_BOUNDS, f"{cell} OOB")
    if state.cell_occupied(cell):
        return ActionResult(ActionCode.CELL_OCCUPIED, f"{cell} occupied")
    return ActionResult(ActionCode.OK)


def _has_physical_support(state: GameState, piece_type: PieceType,
                          pos: Vec3) -> bool:
    """True iff the piece stands on the ground or directly on a STEP."""
    x, y, z = pos
    if z == 1:
        return True
    under = state.piece_at_cell((x, y, z - 1))
    return under is not None and under.type == PieceType.BLOCK


# ---------------------------------------------------------------------------
# Phase PLACE
# ---------------------------------------------------------------------------

def step_place(state: GameState, piece_type: PieceType,
               pos: Vec3) -> ActionResult:
    if state.terminal:
        return ActionResult(state.terminal_code or ActionCode.OK, "terminal")
    if state.phase != Phase.PLACE:
        return ActionResult(ActionCode.BAD_PHASE,
                            f"placement only in PLACE phase (current: {state.phase.value})")
    if piece_type not in PATCHABLE_TYPES:
        return ActionResult(ActionCode.INVALID_PIECE,
                            f"{piece_type.value} is not a patch type")
    if state.remaining.get(piece_type, 0) <= 0:
        return ActionResult(ActionCode.INVENTORY_EMPTY,
                            f"no {piece_type.value} left")

    cells = piece_occupied_cells(piece_type, pos)
    for c in cells:
        r = _cell_free(state, c)
        if r.code != ActionCode.OK:
            return r
    if not _has_physical_support(state, piece_type, pos):
        return ActionResult(ActionCode.NO_PHYSICAL_SUPPORT,
                            f"{piece_type.value}@{pos}: no cell under base")

    piece = PlacedPiece(type=piece_type, pos=pos)
    for c in cells:
        state.pieces[c] = piece
    state.remaining[piece_type] -= 1
    state.action_count += 1
    return ActionResult(ActionCode.OK, f"placed {piece_type.value}@{pos}")


def step_remove(state: GameState, pos: Vec3) -> ActionResult:
    if state.terminal:
        return ActionResult(state.terminal_code or ActionCode.OK, "terminal")
    if state.phase != Phase.PLACE:
        return ActionResult(ActionCode.BAD_PHASE, "remove only in PLACE phase")
    piece = state.piece_at_cell(pos)
    if piece is None:
        return ActionResult(ActionCode.NO_PIECE_TO_REMOVE, f"no piece @{pos}")
    if piece.pos in state.fixed_anchors:
        return ActionResult(ActionCode.CANT_REMOVE_FIXED,
                            f"{piece.type.value}@{piece.pos} is fixed")
    # Don't let the player yank a piece that is supporting something above
    above = (piece.pos[0], piece.pos[1], piece.top_cell()[2] + 1)
    if in_grid(above, state.spec.grid_size) and state.piece_at_cell(above) is not None:
        supported = state.piece_at_cell(above)
        if supported is not piece:
            return ActionResult(ActionCode.NO_PHYSICAL_SUPPORT,
                                f"{piece.type.value}@{piece.pos} is holding up "
                                f"{supported.type.value}@{supported.pos}; "
                                "remove that first")
    for c in piece.cells():
        if c in state.pieces and state.pieces[c] is piece:
            del state.pieces[c]
    state.remaining[piece.type] = state.remaining.get(piece.type, 0) + 1
    state.action_count += 1
    return ActionResult(ActionCode.OK, f"removed {piece.type.value}@{piece.pos}")


def submit_placement(state: GameState) -> ActionResult:
    """Transition to PUSH phase.  Requires inventory to be empty."""
    if state.terminal:
        return ActionResult(state.terminal_code or ActionCode.OK, "terminal")
    if state.phase != Phase.PLACE:
        return ActionResult(ActionCode.BAD_PHASE,
                            f"already past PLACE phase (current: {state.phase.value})")
    leftover = {pt.value: n for pt, n in state.remaining.items() if n > 0}
    if leftover:
        return ActionResult(ActionCode.INVENTORY_NOT_EMPTY,
                            f"inventory not empty: {leftover}")
    state.phase = Phase.PUSH
    return ActionResult(ActionCode.OK, "entered PUSH phase")


# ---------------------------------------------------------------------------
# Phase PUSH
# ---------------------------------------------------------------------------

def step_push(state: GameState, anchor_pos: Vec3,
              dir_name: str) -> ActionResult:
    if state.terminal:
        return ActionResult(state.terminal_code or ActionCode.OK, "terminal")
    if state.phase != Phase.PUSH:
        return ActionResult(ActionCode.BAD_PHASE,
                            f"push only in PUSH phase (current: {state.phase.value})")
    if state.push_count >= state.spec.push_budget:
        return ActionResult(ActionCode.BUDGET_EXCEEDED,
                            f"budget {state.spec.push_budget} used up")
    if dir_name not in DIR_NAME_TO_VEC:
        return ActionResult(ActionCode.BAD_DIRECTION,
                            f"bad direction {dir_name!r}; use +x, -x, +y or -y")
    piece = state.piece_at_cell(anchor_pos)
    if piece is None or piece.pos != anchor_pos:
        return ActionResult(ActionCode.NOT_STANDING,
                            f"no piece anchored @{anchor_pos}")
    if piece.type != PieceType.START:
        return ActionResult(ActionCode.NOT_A_DOMINO,
                            f"only the green START domino can be pushed "
                            f"({piece.type.value}@{piece.pos} is not it)")
    if piece.toppled:
        return ActionResult(ActionCode.NOT_STANDING, "START already toppled")
    result = simulate_push(state, anchor_pos, dir_name)
    state.push_count += 1
    state.action_count += 1
    if result.game_over:
        # Protected toppled — game is effectively lost, but we don't mark
        # terminal yet (player might still call submit to see the verdict;
        # this matches PipeFit's behavior).
        return ActionResult(ActionCode.OK,
                            f"cascade: {len(result.toppled)} toppled (PROTECTED hit)",
                            detail={"toppled": result.toppled,
                                    "game_over": True})
    return ActionResult(ActionCode.OK,
                        f"cascade: {len(result.toppled)} toppled",
                        detail={"toppled": result.toppled, "game_over": False})


def submit(state: GameState) -> ActionResult:
    """Evaluate WIN / LOSE and mark terminal."""
    if state.terminal:
        return ActionResult(state.terminal_code or ActionCode.OK, "terminal")
    if state.phase != Phase.PUSH:
        return ActionResult(ActionCode.BAD_PHASE,
                            f"submit only in PUSH phase (current: {state.phase.value})")
    # Count goals/protected
    goal_anchors: List[Vec3] = []
    protected_anchors: List[Vec3] = []
    for p in state.anchors():
        if p.type == PieceType.GOAL:
            goal_anchors.append(p.pos)
        elif p.type == PieceType.PROTECTED:
            protected_anchors.append(p.pos)
    # Any protected toppled → LOSE
    protected_toppled = [a for a in protected_anchors
                         if state.pieces[a].toppled]
    if protected_toppled:
        state.terminal = True
        state.terminal_code = ActionCode.LOSE
        state.phase = Phase.DONE
        return ActionResult(ActionCode.PROTECTED_TOPPLED,
                            f"{len(protected_toppled)} protected toppled",
                            detail={"protected_toppled": protected_toppled})
    # Every goal must be toppled
    goals_standing = [a for a in goal_anchors
                      if not state.pieces[a].toppled]
    if goals_standing:
        state.terminal = True
        state.terminal_code = ActionCode.LOSE
        state.phase = Phase.DONE
        return ActionResult(ActionCode.GOAL_NOT_TOPPLED,
                            f"{len(goals_standing)} goal(s) still standing",
                            detail={"goals_standing": goals_standing})
    state.terminal = True
    state.terminal_code = ActionCode.WIN
    state.phase = Phase.DONE
    return ActionResult(ActionCode.WIN, "all goals toppled, protected intact")
