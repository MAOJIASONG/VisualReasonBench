"""
FluidNet 3D - Fluid Propagation & Action Logic

The core T3 primitive: after each valve toggle the fluid state is recomputed
from all sources via multi-source BFS through the currently conductive graph.

Conductive cells are: SOURCE / PIPE / TARGET / open VALVE.
Blocking cells are : EMPTY / PROTECTED / closed VALVE.

Semantics
---------
- Fluid cannot enter a PROTECTED cell.  If a BFS step would reach one, the
  action is a PROTECTED_BREACH failure.
- Fluid flowing from a source of color X reaching a TARGET of color Y != X is
  a WRONG_COLOR_TARGET failure.
- Two different colors meeting on the same conductive cell is a
  COLOR_CONTAMINATION failure.
- On any failure, the state is still updated so the caller/visualizer can
  render the exact violating frontier.
"""

from collections import deque
from typing import Dict, Set, Tuple, Optional

from .game_core import (
    CellType, ActionCode, Vec3, GameState, ActionResult, face_neighbors,
)


# ---------------------------------------------------------------------------
# Pure propagation (no state mutation beyond fluid_cells)
# ---------------------------------------------------------------------------

def _cell_is_conductive(pos: Tuple[int, int, int], state: GameState) -> bool:
    cells = state.spec.cells
    if pos not in cells:
        return False                                    # EMPTY
    c = cells[pos]
    if c.type in (CellType.SOURCE, CellType.PIPE, CellType.TARGET):
        return True
    if c.type == CellType.VALVE:
        return pos in state.opened_valves
    return False                                        # PROTECTED or closed VALVE


def propagate_fluid(state: GameState) -> ActionResult:
    """
    Recompute fluid_cells from scratch, multi-source BFS per color.

    Returns ActionResult with the *proposed* new fluid.  The caller is
    responsible for writing the result back onto the state (this lets the
    caller inspect violations before committing, e.g. for dry-run planning).
    """
    spec = state.spec
    grid_size = spec.grid_size

    # color -> set of cells the BFS has reached
    reached: Dict[str, Set[Tuple[int, int, int]]] = {c: set() for c in spec.colors}
    # cell -> first color that reached it (for contamination detection)
    cell_color: Dict[Tuple[int, int, int], str] = {}

    # Seed BFS from every SOURCE cell.
    queue: deque = deque()
    for s in spec.sources:
        col = spec.cells[s].color
        reached[col].add(s)
        cell_color[s] = col
        queue.append((s, col))

    violation: Optional[Tuple[ActionCode, Tuple[int, int, int]]] = None

    while queue:
        pos, color = queue.popleft()
        for nb in face_neighbors(pos, grid_size):
            nb_cell = spec.cells.get(nb)
            if nb_cell is None:                         # EMPTY
                continue
            # Protected cell: immediate breach
            if nb_cell.type == CellType.PROTECTED:
                violation = (ActionCode.PROTECTED_BREACH, nb)
                continue                                # keep going so visualization shows full frontier
            # Closed valve blocks propagation
            if nb_cell.type == CellType.VALVE and nb not in state.opened_valves:
                continue
            # Target: must match color
            if nb_cell.type == CellType.TARGET and nb_cell.color != color:
                violation = violation or (ActionCode.WRONG_COLOR_TARGET, nb)
                continue
            # Source of another color is itself a fluid origin — don't re-enter.
            if nb_cell.type == CellType.SOURCE and nb_cell.color != color:
                # This direction is occupied by a foreign-color source; treat as blocked.
                continue
            if nb in reached[color]:
                continue                                # already visited with this color
            # Contamination: another color already holds this cell.
            if nb in cell_color and cell_color[nb] != color:
                violation = violation or (ActionCode.COLOR_CONTAMINATION, nb)
                continue
            reached[color].add(nb)
            cell_color[nb] = color
            queue.append((nb, color))

    # Flatten new fluid state
    new_fluid: Dict[Tuple[int, int, int], str] = dict(cell_color)
    prev = set(state.fluid_cells.keys())
    new_cells = set(new_fluid.keys()) - prev
    prev_targets = {t for t in spec.targets
                    if state.fluid_cells.get(t) == spec.cells[t].color}
    now_targets  = {t for t in spec.targets
                    if new_fluid.get(t) == spec.cells[t].color}
    newly_filled = now_targets - prev_targets

    if violation is not None:
        code, vcell = violation
        return ActionResult(
            code=code,
            message=f"{code.value} at {vcell}",
            new_fluid=new_cells,
            newly_filled=newly_filled,
            violation_cell=vcell,
        )

    # No violation
    return ActionResult(
        code=ActionCode.OK,
        message="propagated",
        new_fluid=new_cells,
        newly_filled=newly_filled,
    )


# ---------------------------------------------------------------------------
# Public action: open one valve and run propagation
# ---------------------------------------------------------------------------

def step_open_valve(state: GameState,
                    valve_pos: Tuple[int, int, int]) -> ActionResult:
    """
    Attempt to open `valve_pos` and immediately propagate fluid.

    Mutates `state` on success (including failure terminals — those also count
    as state transitions, but the game is marked terminal).
    """
    if state.terminal:
        return ActionResult(code=state.terminal_code or ActionCode.OK,
                            message="game already terminal")

    spec = state.spec
    cell = spec.cells.get(valve_pos)
    if cell is None or cell.type != CellType.VALVE:
        return ActionResult(code=ActionCode.VALVE_NOT_FOUND,
                            message=f"no valve at {valve_pos}")
    if valve_pos in state.opened_valves:
        return ActionResult(code=ActionCode.VALVE_ALREADY_OPEN,
                            message=f"valve {valve_pos} already open")

    # Commit valve open (irreversible)
    state.opened_valves.add(valve_pos)
    state.move_history.append(valve_pos)

    result = propagate_fluid(state)

    # Write fluid state back
    spec = state.spec
    # Recompute full fluid dict from reached sets
    full_fluid: Dict[Tuple[int, int, int], str] = {}
    # `propagate_fluid` returned only *new* cells; we need the full dict.
    # Cheapest: rerun a minimal merge by re-scanning neighbors via the
    # same BFS.  For clarity, redo full BFS and store cell_color.
    full_fluid = _compute_full_fluid(state)
    state.fluid_cells = full_fluid

    # Terminal detection
    if result.code in (ActionCode.PROTECTED_BREACH,
                       ActionCode.WRONG_COLOR_TARGET,
                       ActionCode.COLOR_CONTAMINATION):
        state.terminal = True
        state.terminal_code = result.code
        state.violation_cell = result.violation_cell
        return result

    # Check WIN
    if not state.missing_targets():
        state.terminal = True
        state.terminal_code = ActionCode.WIN
        return ActionResult(code=ActionCode.WIN,
                            message="all targets filled",
                            new_fluid=result.new_fluid,
                            newly_filled=result.newly_filled)

    # No hidden "deadlock" loss: the game ends only on a visible event
    # (all tanks full, or fluid touching a protected cell).  Ending the game
    # because a still-dry valve *would* leak later made half of all losses
    # look like nothing happened.  _is_deadlocked() remains available for
    # analysis.
    return result


def check_terminal(state: GameState) -> Optional[ActionCode]:
    """Return the terminal code if game is over, else None."""
    return state.terminal_code if state.terminal else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_full_fluid(state: GameState) -> Dict[Tuple[int, int, int], str]:
    """Full multi-source BFS, ignoring violations (same rules as propagate_fluid)."""
    spec = state.spec
    grid_size = spec.grid_size
    cell_color: Dict[Tuple[int, int, int], str] = {}
    queue: deque = deque()
    for s in spec.sources:
        col = spec.cells[s].color
        cell_color[s] = col
        queue.append((s, col))
    while queue:
        pos, color = queue.popleft()
        for nb in face_neighbors(pos, grid_size):
            nb_cell = spec.cells.get(nb)
            if nb_cell is None:
                continue
            if nb_cell.type == CellType.PROTECTED:
                cell_color.setdefault(nb, color)        # record for viz
                continue
            if nb_cell.type == CellType.VALVE and nb not in state.opened_valves:
                continue
            if nb_cell.type == CellType.TARGET and nb_cell.color != color:
                cell_color.setdefault(nb, color)
                continue
            if nb_cell.type == CellType.SOURCE and nb_cell.color != color:
                continue
            if nb in cell_color:
                continue
            cell_color[nb] = color
            queue.append((nb, color))
    return cell_color


def _is_deadlocked(state: GameState) -> bool:
    """
    True iff no continuation of the current opened-valve set can fill the
    remaining missing targets without triggering a violation.

    We brute-force across subsets of the still-CLOSED valves and check
    whether any superset of the currently-opened set wins.  This is correct
    but exponential in the number of closed valves; for the puzzles we
    generate (|valves| <= 10) it stays fast.
    """
    spec = state.spec
    missing = state.missing_targets()
    if not missing:
        return False
    closed = list(state.closed_valves())
    base_open = set(state.opened_valves)
    from itertools import combinations
    for k in range(0, len(closed) + 1):
        for combo in combinations(closed, k):
            opened = base_open | set(combo)
            # Build a temp state and run propagation.
            tmp = GameState(spec=spec, opened_valves=opened)
            fluid = _compute_full_fluid(tmp)
            # Skip subsets that violate.
            if any(p in fluid for p in spec.protected):
                continue
            wrong = False
            for t in spec.targets:
                got = fluid.get(t)
                col = spec.cells[t].color
                if got is not None and got != col:
                    wrong = True
                    break
            if wrong:
                continue
            # All targets filled?
            if all(fluid.get(t) == spec.cells[t].color for t in spec.targets):
                return False
    return True
