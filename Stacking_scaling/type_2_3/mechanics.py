"""
PipeFit 3D — mechanics (rules v2).

Actions:
  step_place(state, piece_type, pos, orientation=None, ends=None)
      Put a piece in any free cell.  For pipes give either an orientation
      index or, more readably, its open ends, e.g. ends=("+x", "+z").
  step_remove(state, pos)
      Take a piece back (free undo; it still costs an action).
  submit(state)
      One final check; the game ends WIN or LOSE.

Rules checked at submit (stated the same way in the image HUD):
  1. Water-tight network: every pipe end must meet another pipe end or a
     portal's inward face -- no open ends.  Each entrance / exit takes
     exactly one pipe, from the inside of the box.
  2. Connected: all entrances and exits are joined by one pipe network.
  3. Supported: every pipe and support rests on something solid directly
     below it -- the floor (z=1), a support block, or another pipe that is
     itself supported.

Rules v1 accepted reaching one of two exits, routing through the other
entrance, open pipe ends and a "one unanchored pipe may dangle" exception,
none of which the image explained; see README.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .game_core import (
    ActionCode, ActionResult, DIRS, DIR_OPPOSITE, GameState, ORIENTATIONS,
    PieceType, PlacedPiece, Vec3, connector_dirs, piece_is_pipe,
    piece_is_support, piece_occupied_cells, portal_inward,
)

DIR_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"]


def orientation_from_ends(pt: PieceType, ends: Iterable[str]) -> Optional[int]:
    """Orientation index of a pipe whose open ends are `ends`."""
    try:
        mask = sum(1 << DIR_NAMES.index(e.strip().lower()) for e in ends)
    except ValueError:
        return None
    try:
        return ORIENTATIONS[pt].index(mask)
    except ValueError:
        return None


def ends_of(pt: PieceType, orientation: int) -> List[str]:
    return [DIR_NAMES[d] for d in connector_dirs(pt, orientation)]


def _in_grid(cell: Vec3, grid_size: Vec3) -> bool:
    A, B, C = grid_size
    return 1 <= cell[0] <= A and 1 <= cell[1] <= B and 1 <= cell[2] <= C


def _check_cell_free(state: GameState, cell: Vec3) -> ActionResult:
    if not _in_grid(cell, state.spec.grid_size):
        return ActionResult(ActionCode.OUT_OF_BOUNDS, f"{cell} outside the box")
    if state.is_obstacle(cell):
        return ActionResult(ActionCode.OBSTACLE_HIT, f"{cell} is an obstacle")
    if state.is_portal(cell):
        return ActionResult(ActionCode.ENTRANCE_EXIT_OCCUPIED, f"{cell} is a portal")
    if state.occupied(cell):
        return ActionResult(ActionCode.CELL_OCCUPIED, f"{cell} is occupied")
    return ActionResult(ActionCode.OK)


def step_place(state: GameState, piece_type: PieceType, pos: Vec3,
               orientation: Optional[int] = None,
               ends: Optional[Sequence[str]] = None) -> ActionResult:
    if state.terminal:
        return ActionResult(state.terminal_code or ActionCode.OK, "game over")
    if state.action_count >= state.max_actions:
        return ActionResult(ActionCode.ACTION_LIMIT_REACHED, "action limit reached")
    if state.remaining.get(piece_type, 0) <= 0:
        return ActionResult(ActionCode.INVENTORY_EMPTY, f"no {piece_type.value} left")
    if ends is not None:
        orientation = orientation_from_ends(piece_type, ends)
        if orientation is None:
            return ActionResult(ActionCode.INVALID_ORIENTATION,
                                f"{piece_type.value} cannot have ends {list(ends)}")
    if orientation is None:
        orientation = 0
    if not (0 <= orientation < len(ORIENTATIONS[piece_type])):
        return ActionResult(ActionCode.INVALID_ORIENTATION,
                            f"orientation {orientation} invalid for {piece_type.value}")
    cells = piece_occupied_cells(piece_type, pos)
    for c in cells:
        r = _check_cell_free(state, c)
        if r.code != ActionCode.OK:
            return r
    piece = PlacedPiece(type=piece_type, pos=pos, orientation=orientation)
    for c in cells:
        state.placed[c] = piece
    state.remaining[piece_type] -= 1
    state.action_count += 1
    state.move_history.append(("place", pos))
    return ActionResult(ActionCode.OK, f"placed {piece_type.value} at {pos}")


def step_remove(state: GameState, pos: Vec3) -> ActionResult:
    if state.terminal:
        return ActionResult(state.terminal_code or ActionCode.OK, "game over")
    if state.action_count >= state.max_actions:
        return ActionResult(ActionCode.ACTION_LIMIT_REACHED, "action limit reached")
    piece = state.placed.get(pos)
    if piece is None:
        return ActionResult(ActionCode.NO_PIECE_TO_REMOVE, f"nothing at {pos}")
    for c in piece.cells():
        state.placed.pop(c, None)
    state.remaining[piece.type] = state.remaining.get(piece.type, 0) + 1
    state.action_count += 1
    state.move_history.append(("remove", pos))
    return ActionResult(ActionCode.OK, f"removed {piece.type.value} at {pos}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _pipe_cells(state: GameState) -> Set[Vec3]:
    return {c for c, p in state.placed.items() if piece_is_pipe(p.type)}


def _support_cells(state: GameState) -> Set[Vec3]:
    return {c for c, p in state.placed.items() if piece_is_support(p.type)}


def _ends(state: GameState, cell: Vec3) -> List[int]:
    p = state.placed[cell]
    return connector_dirs(p.type, p.orientation)


def _neighbor(cell: Vec3, d: int) -> Vec3:
    v = DIRS[d][1]
    return (cell[0] + v[0], cell[1] + v[1], cell[2] + v[2])


def leak_ends(state: GameState) -> List[Tuple[Vec3, int]]:
    """Pipe ends that do not meet a matching pipe end or a portal face."""
    pipes = _pipe_cells(state)
    out = []
    for c in pipes:
        for d in _ends(state, c):
            n = _neighbor(c, d)
            back = DIR_OPPOSITE[d]
            if n in pipes and back in _ends(state, n):
                continue
            if state.is_portal(n) and portal_inward(n, state.spec.grid_size) == back:
                continue
            out.append((c, d))
    return out


def portal_links(state: GameState) -> Dict[Vec3, int]:
    """How many pipes each portal is joined to (should be exactly 1)."""
    pipes = _pipe_cells(state)
    links = {}
    for p in list(state.spec.entrances) + list(state.spec.exits):
        inward = portal_inward(p, state.spec.grid_size)
        n = _neighbor(p, inward)
        links[p] = int(n in pipes and DIR_OPPOSITE[inward] in _ends(state, n))
    return links


def pipe_graph_connected(state: GameState) -> bool:
    """All portals (and every pipe) in one network of matched ends."""
    pipes = _pipe_cells(state)
    portals = list(state.spec.entrances) + list(state.spec.exits)
    nodes = set(pipes) | set(portals)
    if not portals:
        return False
    adj: Dict[Vec3, Set[Vec3]] = {n: set() for n in nodes}
    for c in pipes:
        for d in _ends(state, c):
            n = _neighbor(c, d)
            back = DIR_OPPOSITE[d]
            if (n in pipes and back in _ends(state, n)) or (
                    n in portals and portal_inward(n, state.spec.grid_size) == back):
                adj[c].add(n)
                adj[n].add(c)
    seen = {portals[0]}
    todo = [portals[0]]
    while todo:
        for n in adj[todo.pop()]:
            if n not in seen:
                seen.add(n)
                todo.append(n)
    return nodes <= seen


def _resting(state: GameState) -> Set[Vec3]:
    """Pipe / support cells that rest on something solid, bottom-up: the
    floor, or a resting pipe or support directly below."""
    solid = _pipe_cells(state) | _support_cells(state)
    ok: Set[Vec3] = set()
    for c in sorted(solid, key=lambda c: c[2]):
        if c[2] == 1 or (c[0], c[1], c[2] - 1) in ok:
            ok.add(c)
    return ok


def unsupported_pipe_cells(state: GameState) -> List[Vec3]:
    """Pipes with nothing solid directly below."""
    ok = _resting(state)
    return sorted(c for c in _pipe_cells(state) if c not in ok)


def floating_supports(state: GameState) -> List[Vec3]:
    ok = _resting(state)
    return sorted(c for c in _support_cells(state) if c not in ok)


def problem_cells(state: GameState) -> Set[Vec3]:
    """Cells to highlight after a failed submit."""
    cells = set(unsupported_pipe_cells(state)) | set(floating_supports(state))
    cells |= {c for c, _ in leak_ends(state)}
    return cells


def submit(state: GameState) -> ActionResult:
    """Final check; ends the game either way."""
    if state.terminal:
        return ActionResult(state.terminal_code or ActionCode.OK, "game over")
    state.action_count += 1
    pipes = _pipe_cells(state)
    detail: Dict = {}
    if not pipes:
        code = ActionCode.NO_PIPES_PLACED
    else:
        leaks = leak_ends(state)
        links = portal_links(state)
        unsup = unsupported_pipe_cells(state)
        floating = floating_supports(state)
        detail = {"leaks": [(c, DIR_NAMES[d]) for c, d in leaks],
                  "unlinked_portals": [p for p, n in links.items() if n != 1],
                  "unsupported": unsup, "floating_supports": floating}
        if leaks:
            code = ActionCode.LEAK
        elif any(n != 1 for n in links.values()) or not pipe_graph_connected(state):
            code = ActionCode.NOT_CONNECTED
        elif unsup or floating:
            code = ActionCode.UNSUPPORTED
        else:
            code = ActionCode.WIN
    state.terminal = True
    state.terminal_code = code
    return ActionResult(code, code.value, detail=detail)


# Backwards-compatible names used by older scripts / the renderer.
def unanchored_pipe_cells(state: GameState) -> List[Vec3]:
    return unsupported_pipe_cells(state)


def double_unanchored_violations(state: GameState) -> List[Tuple[Vec3, Vec3]]:
    """v1 name; now: problem cells after a FAILED submit, as (c, c) pairs."""
    if state.terminal_code in (None, ActionCode.WIN):
        return []
    return [(c, c) for c in sorted(problem_cells(state))]
