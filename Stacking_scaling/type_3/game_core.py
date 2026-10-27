"""
FluidNet 3D - Core Data Structures
核心数据结构和枚举定义

Grid coordinates are 1-based externally (x,y,z in [1..A]×[1..B]×[1..C]) to
mirror the stacking_game convention.  Internally we store cells in dicts keyed
by (x,y,z) int tuples.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CellType(Enum):
    """
    Per-cell semantic type.  A cell can be exactly one of these.

    - EMPTY     : Air.  Fluid cannot enter; no pipe here.
    - PIPE      : Fluid conduit.  Always conducting.  Fluid propagates between
                  face-adjacent PIPE / SOURCE / TARGET / open-VALVE cells.
    - SOURCE    : Infinite fluid origin.  Always filled with its color.
    - TARGET    : Reservoir that must be filled with matching color.
    - PROTECTED : Protected cell.  Fluid MUST NOT enter (instant fail).
    - VALVE     : Gated pipe.  Initially closed (blocks fluid).  Once the
                  agent opens it, it behaves like a PIPE for the rest of
                  the game (irreversible).
    """
    EMPTY     = "empty"
    PIPE      = "pipe"
    SOURCE    = "source"
    TARGET    = "target"
    PROTECTED = "protected"
    VALVE     = "valve"


class ActionCode(Enum):
    """Result code for a single `open_valve` action."""
    OK                   = "OK"                    # Action accepted, game continues
    WIN                  = "Win"                   # All targets filled, no violations
    PROTECTED_BREACH     = "ProtectedBreach"       # Fluid reached a PROTECTED cell
    WRONG_COLOR_TARGET   = "WrongColorTarget"      # Fluid of color X reached TARGET of color Y≠X
    COLOR_CONTAMINATION  = "ColorContamination"    # Two colors met in same PIPE/VALVE cell
    VALVE_NOT_FOUND      = "ValveNotFound"         # No valve at given coords
    VALVE_ALREADY_OPEN   = "ValveAlreadyOpen"      # Valve was already opened
    NO_PROGRESS_DEADLOCK = "NoProgressDeadlock"    # No remaining action can fill more targets


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Vec3:
    """1-based integer 3D coordinate (immutable, hashable)."""
    x: int
    y: int
    z: int

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    def to_key(self) -> str:
        return f"{self.x},{self.y},{self.z}"

    @staticmethod
    def from_tuple(t: Tuple[int, int, int]) -> "Vec3":
        return Vec3(t[0], t[1], t[2])


def face_neighbors(pos: Tuple[int, int, int],
                   grid_size: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """Return the up-to-6 face-adjacent cells that are inside the grid."""
    x, y, z = pos
    A, B, C = grid_size
    cand = [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1),
    ]
    return [p for p in cand if 1 <= p[0] <= A and 1 <= p[1] <= B and 1 <= p[2] <= C]


# ---------------------------------------------------------------------------
# Level & state containers
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One cell of the grid.  Not stored for EMPTY cells (sparse representation)."""
    pos: Vec3
    type: CellType
    color: Optional[str] = None   # Required for SOURCE / TARGET; None otherwise.


@dataclass
class LevelSpec:
    """
    Immutable level definition.

    Only non-EMPTY cells are stored in `cells`.  Quick-index lookups
    (sources/targets/valves/protected) are cached for the solver and CLI.
    """
    grid_size: Tuple[int, int, int]                 # (A, B, C)
    cells: Dict[Tuple[int, int, int], Cell]         # sparse: only non-EMPTY

    # Derived lookups (populated in __post_init__)
    sources:   List[Tuple[int, int, int]] = field(default_factory=list)
    targets:   List[Tuple[int, int, int]] = field(default_factory=list)
    valves:    List[Tuple[int, int, int]] = field(default_factory=list)
    protected: List[Tuple[int, int, int]] = field(default_factory=list)
    colors:    List[str]                  = field(default_factory=list)

    def __post_init__(self):
        if self.sources or self.targets or self.valves or self.protected:
            return  # Already populated (e.g. from loader).
        srcs, tgts, vlvs, prot, cols = [], [], [], [], set()
        for pos, cell in self.cells.items():
            if cell.type == CellType.SOURCE:
                srcs.append(pos); cols.add(cell.color)
            elif cell.type == CellType.TARGET:
                tgts.append(pos); cols.add(cell.color)
            elif cell.type == CellType.VALVE:
                vlvs.append(pos)
            elif cell.type == CellType.PROTECTED:
                prot.append(pos)
        self.sources = sorted(srcs)
        self.targets = sorted(tgts)
        self.valves  = sorted(vlvs)
        self.protected = sorted(prot)
        self.colors  = sorted(cols)


@dataclass
class GameState:
    """Mutable per-session state."""
    spec: LevelSpec
    opened_valves: Set[Tuple[int, int, int]] = field(default_factory=set)
    # cell_pos -> fluid color (updated after every propagation)
    fluid_cells:  Dict[Tuple[int, int, int], str] = field(default_factory=dict)
    move_history: List[Tuple[int, int, int]]      = field(default_factory=list)

    # Terminal flags
    terminal:       bool                 = False
    terminal_code:  Optional[ActionCode] = None
    violation_cell: Optional[Tuple[int, int, int]] = None

    def filled_targets(self) -> Dict[Tuple[int, int, int], str]:
        """Targets that currently hold their matching-color fluid."""
        out = {}
        for t in self.spec.targets:
            tc = self.spec.cells[t].color
            if self.fluid_cells.get(t) == tc:
                out[t] = tc
        return out

    def missing_targets(self) -> List[Tuple[int, int, int]]:
        return [t for t in self.spec.targets
                if self.fluid_cells.get(t) != self.spec.cells[t].color]

    def closed_valves(self) -> List[Tuple[int, int, int]]:
        return [v for v in self.spec.valves if v not in self.opened_valves]


# ---------------------------------------------------------------------------
# Action return value
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Outcome of applying one action to the game state."""
    code:           ActionCode
    message:        str                                     = ""
    new_fluid:      Set[Tuple[int, int, int]]               = field(default_factory=set)
    newly_filled:   Set[Tuple[int, int, int]]               = field(default_factory=set)
    violation_cell: Optional[Tuple[int, int, int]]          = None

    @property
    def success(self) -> bool:
        """True if the action was legal (may still end the game via WIN)."""
        return self.code in (ActionCode.OK, ActionCode.WIN)
