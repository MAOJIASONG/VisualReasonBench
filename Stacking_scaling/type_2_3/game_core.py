"""
PipeFit 3D — Core data structures.

An assembly puzzle that mixes T2 (Spatial Commitment) and T3 (Structural
Propagation).  The player places pipe + support pieces from a bounded
inventory to connect entrance(s) to exit(s).  Placements are reversible
(free undo) but action count is bounded.

Coordinate system: 1-indexed (x, y, z) with z pointing up.  A piece's
`pos` is its anchor cell (for S2 the lower of its two cells).

Connector mask:
  bit 0 = +X, bit 1 = -X, bit 2 = +Y, bit 3 = -Y, bit 4 = +Z, bit 5 = -Z
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


Vec3 = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

DIRS: List[Tuple[int, Vec3]] = [
    (0, (+1, 0, 0)),   # +X
    (1, (-1, 0, 0)),   # -X
    (2, (0, +1, 0)),   # +Y
    (3, (0, -1, 0)),   # -Y
    (4, (0, 0, +1)),   # +Z
    (5, (0, 0, -1)),   # -Z
]
DIR_OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}


def face_neighbors(pos: Vec3, grid_size: Vec3) -> List[Vec3]:
    A, B, C = grid_size
    x, y, z = pos
    out = []
    for _, d in DIRS:
        nx, ny, nz = x + d[0], y + d[1], z + d[2]
        if 1 <= nx <= A and 1 <= ny <= B and 1 <= nz <= C:
            out.append((nx, ny, nz))
    return out


# ---------------------------------------------------------------------------
# Piece catalog
# ---------------------------------------------------------------------------

def portal_inward(pos: Vec3, grid_size: Vec3) -> int:
    """Direction (index into DIRS) from a portal on a side wall into the
    box.  Portals sit on the x=1 / x=A / y=1 / y=B walls; the pipe must
    join them from that side."""
    A, B, C = grid_size
    x, y, z = pos
    if x == 1:
        return 0
    if x == A:
        return 1
    if y == 1:
        return 2
    if y == B:
        return 3
    return 4 if z == 1 else 5


class PieceType(Enum):
    PIPE_I = "I"       # straight, two opposite connectors
    PIPE_L = "L"       # corner, two perpendicular connectors
    PIPE_T = "T"       # fork, two opposite + one perpendicular
    SUPPORT_S1 = "S1"  # 1x1x1 cube
    SUPPORT_S2 = "S2"  # 1x1x2 vertical bar


# All orientations are encoded as a connector mask over the 6 faces.
# (Support pieces have no connectors — they just occupy cells.)

def _mask(*dirs: int) -> int:
    m = 0
    for d in dirs:
        m |= (1 << d)
    return m


# PIPE_I: 3 orientations (X, Y, Z axis aligned)
PIPE_I_ORIENTATIONS: List[int] = [
    _mask(0, 1),  # X-axis
    _mask(2, 3),  # Y-axis
    _mask(4, 5),  # Z-axis
]

# PIPE_L: 12 orientations — pick one of {+X,-X} × one of {+Y,-Y,+Z,-Z}, etc.
def _build_L_orientations() -> List[int]:
    out: List[int] = []
    axes = [(0, 1), (2, 3), (4, 5)]
    for i in range(3):
        for j in range(i + 1, 3):
            a1, a2 = axes[i]
            b1, b2 = axes[j]
            out.append(_mask(a1, b1))
            out.append(_mask(a1, b2))
            out.append(_mask(a2, b1))
            out.append(_mask(a2, b2))
    return out

PIPE_L_ORIENTATIONS: List[int] = _build_L_orientations()  # 12

# PIPE_T: 12 orientations — 3 through-axes × 4 perpendicular branches
def _build_T_orientations() -> List[int]:
    out: List[int] = []
    axes = [(0, 1), (2, 3), (4, 5)]
    for i in range(3):
        through = axes[i]
        for j in range(3):
            if j == i:
                continue
            for branch in axes[j]:
                out.append(_mask(through[0], through[1], branch))
    # Dedup (each T is counted twice because we iterate both perpendicular axes)
    return sorted(set(out))

PIPE_T_ORIENTATIONS: List[int] = _build_T_orientations()  # 12

# Supports: single orientation each
SUPPORT_S1_ORIENTATIONS: List[int] = [0]
SUPPORT_S2_ORIENTATIONS: List[int] = [0]


ORIENTATIONS: Dict[PieceType, List[int]] = {
    PieceType.PIPE_I: PIPE_I_ORIENTATIONS,
    PieceType.PIPE_L: PIPE_L_ORIENTATIONS,
    PieceType.PIPE_T: PIPE_T_ORIENTATIONS,
    PieceType.SUPPORT_S1: SUPPORT_S1_ORIENTATIONS,
    PieceType.SUPPORT_S2: SUPPORT_S2_ORIENTATIONS,
}


def piece_is_pipe(pt: PieceType) -> bool:
    return pt in (PieceType.PIPE_I, PieceType.PIPE_L, PieceType.PIPE_T)


def piece_is_support(pt: PieceType) -> bool:
    return pt in (PieceType.SUPPORT_S1, PieceType.SUPPORT_S2)


def piece_occupied_cells(pt: PieceType, pos: Vec3) -> List[Vec3]:
    """Cells the piece occupies when anchored at `pos`."""
    if pt == PieceType.SUPPORT_S2:
        return [pos, (pos[0], pos[1], pos[2] + 1)]
    return [pos]


def connector_dirs(pt: PieceType, orientation_idx: int) -> List[int]:
    """Return the connector direction indices (into DIRS) for this orientation."""
    if not piece_is_pipe(pt):
        return []
    mask = ORIENTATIONS[pt][orientation_idx]
    return [d for d in range(6) if (mask >> d) & 1]


# ---------------------------------------------------------------------------
# Placed piece + Level spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlacedPiece:
    type: PieceType
    pos: Vec3
    orientation: int         # index into ORIENTATIONS[type]

    def cells(self) -> List[Vec3]:
        return piece_occupied_cells(self.type, self.pos)


@dataclass
class LevelSpec:
    grid_size: Vec3                         # (A, B, C)
    entrances: List[Vec3]                   # 1-2 boundary cells
    exits: List[Vec3]                       # 1-2 boundary cells
    obstacles: List[Vec3] = field(default_factory=list)  # flat list of BLOCKED cells
    # Shapes: each obstacle shape is a list of consecutive cells (1, 2, or 3).
    # The flat `obstacles` list = union of all shapes.  Renderer uses shapes
    # to merge cells into one visual piece.  Mechanics use `obstacles`.
    obstacle_shapes: List[List[Vec3]] = field(default_factory=list)
    inventory: Dict[PieceType, int] = field(default_factory=dict)  # available counts
    # Optional metadata for rendering / solver seed.
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action codes
# ---------------------------------------------------------------------------

class ActionCode(Enum):
    OK = "OK"
    WIN = "WIN"

    OUT_OF_BOUNDS = "OutOfBounds"
    CELL_OCCUPIED = "CellOccupied"
    OBSTACLE_HIT = "ObstacleHit"
    ENTRANCE_EXIT_OCCUPIED = "EntranceExitOccupied"
    INVENTORY_EMPTY = "InventoryEmpty"
    NO_PIECE_TO_REMOVE = "NoPieceToRemove"
    INVALID_ORIENTATION = "InvalidOrientation"
    ACTION_LIMIT_REACHED = "ActionLimitReached"

    # Placement-time physical-support violation
    NO_PHYSICAL_SUPPORT = "NoPhysicalSupport"

    # Submit-time violations
    SUPPORT_DOUBLE_UNANCHORED = "SupportDoubleUnanchored"
    NOT_CONNECTED = "NotConnected"
    NO_PIPES_PLACED = "NoPipesPlaced"
    LEAK = "Leak"                       # a pipe end opens into nothing
    UNSUPPORTED = "Unsupported"         # a pipe / support has nothing below


@dataclass
class ActionResult:
    code: ActionCode
    message: str = ""
    detail: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    spec: LevelSpec
    placed: Dict[Vec3, PlacedPiece] = field(default_factory=dict)  # cell -> piece
    remaining: Dict[PieceType, int] = field(default_factory=dict)
    action_count: int = 0
    max_actions: int = 999
    terminal: bool = False
    terminal_code: Optional[ActionCode] = None
    move_history: List[Tuple[str, Vec3]] = field(default_factory=list)  # ('place'|'remove', pos)

    def clone(self) -> "GameState":
        return GameState(
            spec=self.spec,
            placed=dict(self.placed),
            remaining=dict(self.remaining),
            action_count=self.action_count,
            max_actions=self.max_actions,
            terminal=self.terminal,
            terminal_code=self.terminal_code,
            move_history=list(self.move_history),
        )

    def occupied(self, cell: Vec3) -> bool:
        return cell in self.placed

    def is_entrance(self, cell: Vec3) -> bool:
        return cell in self.spec.entrances

    def is_exit(self, cell: Vec3) -> bool:
        return cell in self.spec.exits

    def is_obstacle(self, cell: Vec3) -> bool:
        return cell in self.spec.obstacles

    def is_portal(self, cell: Vec3) -> bool:
        return self.is_entrance(cell) or self.is_exit(cell)

    def piece_anchors(self) -> List[PlacedPiece]:
        """Each placed piece appears once (keyed by anchor cell)."""
        seen: Set[Vec3] = set()
        out: List[PlacedPiece] = []
        for cell, pc in self.placed.items():
            if pc.pos in seen:
                continue
            seen.add(pc.pos)
            out.append(pc)
        return out


def init_state(spec: LevelSpec, max_actions: Optional[int] = None) -> GameState:
    state = GameState(
        spec=spec,
        placed={},
        remaining=dict(spec.inventory),
        max_actions=max_actions if max_actions is not None else 3 * sum(spec.inventory.values()),
    )
    return state
