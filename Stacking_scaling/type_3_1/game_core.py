"""
Domino Chain Reaction 3D — Core data structures (rules v2).

Two phases:

  PLACE: put your K patch dominoes from the inventory into empty cells.
    A domino stands on the ground (z=1) or on a grey STEP block.  Free undo.

  PUSH: push the green START domino in one of four directions.  It is the
    only piece you may push, and you push it once.

Cascade (v2, physical): a falling domino tips one cell forward and knocks
over every standing piece whose body is in that next column at the faller's
own height or one level higher (a domino is two cells tall).  So waves climb
or descend one step automatically.  A STEP block stops the wave.

Piece catalog:
  S = 1x1x2 green START domino (the only pushable piece)
  D = 1x1x2 domino: falls forward, passes the wave on
  T = 1x1x2 splitter: when hit, sends the wave out to BOTH sides
      (left and right of the incoming direction), not straight on
  G = 1x1x2 gold goal: topple every G to win
  P = 1x1x2 red protected: toppling any P loses
  B = 1x1x1 grey STEP block: holds a piece one level up; never falls

Coordinate system: 1-indexed (x, y, z) with z pointing up.  A standing
piece's pos is its base cell; it also occupies (x, y, z+1).

Rules v1 (any standing piece could be pushed, a pulse only hit pieces
whose base was at exactly its height, ramps lifted the pulse) allowed
winning most levels by pushing the goal itself; see README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


Vec3 = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

# Push directions are horizontal only (dominoes topple in ±x or ±y).
DIRS4: List[Tuple[str, Vec3]] = [
    ("+x", (+1, 0, 0)),
    ("-x", (-1, 0, 0)),
    ("+y", (0, +1, 0)),
    ("-y", (0, -1, 0)),
]

DIR_NAME_TO_VEC: Dict[str, Vec3] = {n: v for n, v in DIRS4}
DIR_VEC_TO_NAME: Dict[Vec3, str] = {v: n for n, v in DIRS4}


def perp_dirs(dir_name: str) -> List[str]:
    """Two horizontal directions orthogonal to `dir_name`."""
    if dir_name in ("+x", "-x"):
        return ["+y", "-y"]
    if dir_name in ("+y", "-y"):
        return ["+x", "-x"]
    raise ValueError(f"bad direction {dir_name}")


def face_neighbors_4(pos: Vec3, grid_size: Vec3) -> List[Vec3]:
    A, B, C = grid_size
    x, y, z = pos
    out = []
    for _, d in DIRS4:
        nx, ny, nz = x + d[0], y + d[1], z + d[2]
        if 1 <= nx <= A and 1 <= ny <= B and 1 <= nz <= C:
            out.append((nx, ny, nz))
    return out


def in_grid(pos: Vec3, grid_size: Vec3) -> bool:
    A, B, C = grid_size
    x, y, z = pos
    return 1 <= x <= A and 1 <= y <= B and 1 <= z <= C


# ---------------------------------------------------------------------------
# Piece catalog
# ---------------------------------------------------------------------------

class PieceType(Enum):
    START = "S"        # 1x1x2 green starter -- the ONLY piece you may push
    DOMINO = "D"       # 1x1x2 basic standing
    SPLITTER = "T"     # 1x1x2 splitter (cross-cap)
    GOAL = "G"         # 1x1x2 gold — must be toppled
    PROTECTED = "P"    # 1x1x2 red — toppling it LOSES
    BLOCK = "B"        # 1x1x1 inert step; raised pieces stand on it, the
                       # wave stops at it and it never falls
    RAMP = "R"         # legacy (rules v1 only); no longer generated


def piece_is_standing(pt: PieceType) -> bool:
    """Standing pieces are tall (1x1x2) dominos."""
    return pt in (PieceType.START, PieceType.DOMINO, PieceType.SPLITTER,
                  PieceType.GOAL, PieceType.PROTECTED)


def piece_is_ramp(pt: PieceType) -> bool:
    return pt == PieceType.RAMP


def piece_occupied_cells(pt: PieceType, pos: Vec3) -> List[Vec3]:
    """Cells occupied by a piece anchored at `pos`."""
    if piece_is_standing(pt):
        return [pos, (pos[0], pos[1], pos[2] + 1)]
    return [pos]  # ramp


def piece_top_cell(pt: PieceType, pos: Vec3) -> Vec3:
    """Top cell of a piece (the one that supports things above)."""
    if piece_is_standing(pt):
        return (pos[0], pos[1], pos[2] + 1)
    return pos


# ---------------------------------------------------------------------------
# Placed piece
# ---------------------------------------------------------------------------

@dataclass
class PlacedPiece:
    type: PieceType
    pos: Vec3
    toppled: bool = False
    # For toppled standing pieces: the horizontal direction they fell in
    # (used by renderer to lay them flat).  For toppled ramps: always None.
    fall_dir: Optional[str] = None
    # Index in which this piece was toppled during the last cascade run (for
    # rendering order / debugging).  None = not yet toppled.
    fall_order: Optional[int] = None

    def cells(self) -> List[Vec3]:
        return piece_occupied_cells(self.type, self.pos)

    def top_cell(self) -> Vec3:
        return piece_top_cell(self.type, self.pos)


# ---------------------------------------------------------------------------
# Level spec
# ---------------------------------------------------------------------------

@dataclass
class LevelSpec:
    grid_size: Vec3
    fixed_pieces: List[Tuple[PieceType, Vec3]] = field(default_factory=list)
    # Inventory of PATCH pieces the player can place (phase PLACE).
    # Only D and R are ever patch types (T / G / P are always fixed).
    inventory: Dict[PieceType, int] = field(default_factory=dict)
    # Push budget (phase PUSH).  Max number of push actions the player may
    # spend; must be >= canonical plan length.
    push_budget: int = 1
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase + action codes
# ---------------------------------------------------------------------------

class Phase(Enum):
    PLACE = "PLACE"
    PUSH = "PUSH"
    DONE = "DONE"


class ActionCode(Enum):
    OK = "OK"
    WIN = "WIN"
    LOSE = "LOSE"

    # Placement-phase
    OUT_OF_BOUNDS = "OutOfBounds"
    CELL_OCCUPIED = "CellOccupied"
    INVENTORY_EMPTY = "InventoryEmpty"
    INVALID_PIECE = "InvalidPiece"
    NO_PHYSICAL_SUPPORT = "NoPhysicalSupport"
    NO_PIECE_TO_REMOVE = "NoPieceToRemove"
    CANT_REMOVE_FIXED = "CantRemoveFixed"

    # Phase control
    BAD_PHASE = "BadPhase"
    INVENTORY_NOT_EMPTY = "InventoryNotEmpty"

    # Push-phase
    NOT_STANDING = "NotStanding"
    NOT_A_DOMINO = "NotADomino"         # tried to push something other than START
    BAD_DIRECTION = "BadDirection"
    BUDGET_EXCEEDED = "BudgetExceeded"

    # Submit failures
    GOAL_NOT_TOPPLED = "GoalNotToppled"
    PROTECTED_TOPPLED = "ProtectedToppled"


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
    # anchor-cell -> PlacedPiece.  (cell -> anchor lookup uses cell_to_anchor)
    pieces: Dict[Vec3, PlacedPiece] = field(default_factory=dict)
    # Which piece-anchors were FIXED at init (not removable).
    fixed_anchors: Set[Vec3] = field(default_factory=set)
    remaining: Dict[PieceType, int] = field(default_factory=dict)
    phase: Phase = Phase.PLACE
    push_count: int = 0
    action_count: int = 0
    max_actions: int = 9999
    terminal: bool = False
    terminal_code: Optional[ActionCode] = None
    # Cascade log for the most recent push: list of dicts
    #   {"pulse_from": (pos, dir, z), "toppled_anchor": pos, "type": "D"|"T"|"R"|"G"|"P"}
    last_cascade: List[Dict] = field(default_factory=list)
    # All cascade logs across all pushes (for rendering history).
    all_cascades: List[List[Dict]] = field(default_factory=list)

    # ---------- helpers ----------

    def clone(self) -> "GameState":
        return GameState(
            spec=self.spec,
            pieces={c: PlacedPiece(p.type, p.pos, p.toppled, p.fall_dir, p.fall_order)
                    for c, p in self.pieces.items()},
            fixed_anchors=set(self.fixed_anchors),
            remaining=dict(self.remaining),
            phase=self.phase,
            push_count=self.push_count,
            action_count=self.action_count,
            max_actions=self.max_actions,
            terminal=self.terminal,
            terminal_code=self.terminal_code,
            last_cascade=list(self.last_cascade),
            all_cascades=[list(c) for c in self.all_cascades],
        )

    def cell_occupied(self, cell: Vec3) -> bool:
        return cell in self.pieces

    def piece_at_cell(self, cell: Vec3) -> Optional[PlacedPiece]:
        return self.pieces.get(cell)

    def anchors(self) -> List[PlacedPiece]:
        """Each placed piece exactly once, keyed by anchor pos."""
        seen: Set[Vec3] = set()
        out: List[PlacedPiece] = []
        for p in self.pieces.values():
            if p.pos in seen:
                continue
            seen.add(p.pos)
            out.append(p)
        return out


def init_state(spec: LevelSpec, max_actions: Optional[int] = None) -> GameState:
    """Create a fresh state with fixed pieces installed and inventory loaded."""
    state = GameState(
        spec=spec,
        pieces={},
        fixed_anchors=set(),
        remaining=dict(spec.inventory),
        max_actions=max_actions if max_actions is not None
                    else 3 * (sum(spec.inventory.values()) + spec.push_budget + 4),
    )
    for pt, pos in spec.fixed_pieces:
        piece = PlacedPiece(type=pt, pos=pos)
        for c in piece_occupied_cells(pt, pos):
            state.pieces[c] = piece
        state.fixed_anchors.add(pos)
    return state
