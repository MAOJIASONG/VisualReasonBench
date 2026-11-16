"""
Exact model of the Stacking game (polycube packing, assets/stacking_game) for
the oracle and the search baselines.

Rules (placement.py / game_core.py of the game):
* the box is A x B x C cells, 1-based (x, y, z), z up;
* `place piece cells`: the cells must be a proper rotation (24, no mirror) of
  the piece, inside the box, on free cells, and supported: one cell at z = 1
  or one cell 6-adjacent to a cell of another placed piece;
* `pickup piece`: removes a placed piece (always allowed);
* won when every cell is filled (then every piece is placed: the piece volumes
  add up to the box volume).

A state is (occupied bitmask, set of unplaced pieces).  Cells are indexed
z-major (z, then y, then x), so the lowest free index is a lowest free cell.

`count(state)` counts the ways to finish the box from a state without picking
anything up (exact cover of the free cells by the unplaced pieces, labelled
pieces).  Supports never block a completion: the box is connected, so the
remaining pieces of any tiling can be placed in an order where each touches
the floor or an already placed piece.  A state is *solvable* when count > 0.

`exposed(state)` is the visible part of a dead end: an unplaced piece fits
nowhere in the free cells, or a free cell cannot be covered by any piece that
still fits.  A dead end that is not exposed looks fine but cannot be finished.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache
from itertools import permutations, product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SS = os.path.abspath(os.path.join(HERE, "..", ".."))
PUZZLE_DIRS = [os.environ.get("STACK_PUZZLE_DIR", ""), os.path.join(SS, "puzzles_full_v9"),
               os.path.join(SS, "..", "..", "CHAIN", "assets", "stacking_game", "puzzles_full_v9")]

Cell = Tuple[int, int, int]


def rotations() -> List[np.ndarray]:
    out = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=int)
            for i, (j, s) in enumerate(zip(perm, signs)):
                m[i, j] = s
            if round(np.linalg.det(m)) == 1:
                out.append(m)
    return out                                   # the 24 proper rotations


ROTS = rotations()


def normalise(cells: Sequence[Cell]) -> Tuple[Cell, ...]:
    mx = min(c[0] for c in cells); my = min(c[1] for c in cells); mz = min(c[2] for c in cells)
    return tuple(sorted((x - mx, y - my, z - mz) for x, y, z in cells))


def puzzle_path(name: str) -> str:
    """'2x3x3_hard_004' -> .../puzzles_full_v9/2x3x3/hard/004/2x3x3_hard_004.json"""
    size, diff, num = name.split("_")
    for root in PUZZLE_DIRS:
        p = os.path.join(root, size, diff, num, f"{name}.json")
        if root and os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def all_level_names() -> List[str]:
    for root in PUZZLE_DIRS:
        if root and os.path.isdir(root):
            return sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(root, "*", "*", "*", "*.json")))
    raise FileNotFoundError("puzzles_full_v9")


class Level:
    def __init__(self, name: str):
        self.name = name
        d = json.load(open(puzzle_path(name)))
        self.data = d
        self.box = tuple(d["box"])
        A, B, C = self.box
        self.A, self.B, self.C = A, B, C
        self.ncell = A * B * C
        self.full = (1 << self.ncell) - 1
        self.pieces = [normalise([tuple(c) for c in p]) for p in d["pieces"]]
        self.n = len(self.pieces)
        self.sizes = [len(p) for p in self.pieces]
        self.all_pieces = (1 << self.n) - 1
        # neighbour masks per cell
        self.nbr = [0] * self.ncell
        for x, y, z in product(range(1, A + 1), range(1, B + 1), range(1, C + 1)):
            m = 0
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                q = (x + dx, y + dy, z + dz)
                if 1 <= q[0] <= A and 1 <= q[1] <= B and 1 <= q[2] <= C:
                    m |= 1 << self.idx(q)
            self.nbr[self.idx((x, y, z))] = m
        self.floor = sum(1 << self.idx((x, y, 1)) for x in range(1, A + 1) for y in range(1, B + 1))
        # shapes (rotations) and placements per piece
        self.shapes: List[set] = []
        self.places: List[List[int]] = []          # placement masks
        self.ring: List[Dict[int, int]] = []       # mask -> cells 6-adjacent to it
        self.cover: List[List[List[int]]] = []     # piece -> cell -> masks covering it
        for p in self.pieces:
            shapes = {normalise([tuple(int(v) for v in r @ np.array(c)) for c in p]) for r in ROTS}
            self.shapes.append(shapes)
            masks, ring, cover = [], {}, [[] for _ in range(self.ncell)]
            for s in shapes:
                sx = max(c[0] for c in s); sy = max(c[1] for c in s); sz = max(c[2] for c in s)
                for ox, oy, oz in product(range(1, A - sx + 1), range(1, B - sy + 1), range(1, C - sz + 1)):
                    m = 0
                    for x, y, z in s:
                        m |= 1 << self.idx((x + ox, y + oy, z + oz))
                    if m in ring:
                        continue
                    r = 0
                    for i in bits(m):
                        r |= self.nbr[i]
                    ring[m] = r & ~m
                    masks.append(m)
                    for i in bits(m):
                        cover[i].append(m)
            self.places.append(masks)
            self.ring.append(ring)
            self.cover.append(cover)
        self.count = lru_cache(maxsize=None)(self._count)
        self.exposed = lru_cache(maxsize=None)(self._exposed)
        self.solvable = lru_cache(maxsize=None)(self._solvable)

    # -- cells -----------------------------------------------------------------
    def idx(self, c: Cell) -> int:
        x, y, z = c
        return (z - 1) * self.A * self.B + (y - 1) * self.A + (x - 1)

    def cell(self, i: int) -> Cell:
        z, r = divmod(i, self.A * self.B)
        y, x = divmod(r, self.A)
        return (x + 1, y + 1, z + 1)

    def mask_of(self, cells: Sequence[Cell]) -> Optional[int]:
        m = 0
        for c in cells:
            if not (1 <= c[0] <= self.A and 1 <= c[1] <= self.B and 1 <= c[2] <= self.C):
                return None
            m |= 1 << self.idx(c)
        return m

    # -- rules -----------------------------------------------------------------
    def check_place(self, occ: int, unplaced: int, p: int, cells: Sequence[Cell]) -> Tuple[Optional[int], str]:
        """(mask, "OK") or (None, error) — the game's own checks, in its order."""
        if not (unplaced >> p) & 1:
            return None, "PieceAlreadyPlaced"
        if len(cells) != self.sizes[p]:
            return None, "WrongCount"
        if normalise(cells) not in self.shapes[p]:
            return None, "ShapeMismatch"
        m = self.mask_of(cells)
        if m is None:
            return None, "OutOfBounds"
        if m & occ:
            return None, "Collision"
        if not self.supported(occ, p, m):
            return None, "Floating"
        return m, "OK"

    def supported(self, occ: int, p: int, m: int) -> bool:
        return bool(m & self.floor) or bool(self.ring[p][m] & occ)

    def legal_moves(self, occ: int, unplaced: int) -> List[Tuple[int, int]]:
        out = []
        for p in bits(unplaced):
            for m in self.places[p]:
                if not m & occ and self.supported(occ, p, m):
                    out.append((p, m))
        return out

    # -- oracle ----------------------------------------------------------------
    CAP = 10 ** 4          # counts at or above CAP are reported as CAP

    def _count(self, occ: int, unplaced: int) -> int:
        if occ == self.full:
            return 1 if unplaced == 0 else 0
        free = ~occ & self.full
        c = (free & -free).bit_length() - 1
        total = 0
        if not self.solvable(occ, unplaced):
            return 0
        for p in bits(unplaced):
            for m in self.cover[p][c]:
                if not m & occ:
                    total += self.count(occ | m, unplaced & ~(1 << p))
                    if total >= self.CAP:
                        return self.CAP
        return total

    def _solvable(self, occ: int, unplaced: int) -> bool:
        if occ == self.full:
            return unplaced == 0
        if self.exposed(occ, unplaced):
            return False
        free = ~occ & self.full
        c = (free & -free).bit_length() - 1
        for p in bits(unplaced):
            for m in self.cover[p][c]:
                if not m & occ and self.solvable(occ | m, unplaced & ~(1 << p)):
                    return True
        return False

    def _exposed(self, occ: int, unplaced: int) -> bool:
        if occ == self.full:
            return False
        coverable = 0
        for p in bits(unplaced):
            fits = [m for m in self.places[p] if not m & occ]
            if not fits:
                return True
            for m in fits:
                coverable |= m
        return (~occ & self.full) & ~coverable != 0

    def solution_order(self) -> List[Tuple[int, int]]:
        """The stored solution as (piece, mask) in a placeable order (support first)."""
        sol = self.data["solution"]
        todo = {int(k): self.mask_of([(c[0] + 1, c[1] + 1, c[2] + 1) for c in v]) for k, v in sol.items()}
        occ, out = 0, []
        while todo:
            p = min((p for p, m in todo.items() if self.supported(occ, p, m)),
                    key=lambda p: (todo[p] & -todo[p]).bit_length())
            out.append((p, todo.pop(p)))
            occ |= out[-1][1]
        return out


def bits(m: int):
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


_LEVELS: Dict[str, Level] = {}


def level(name: str) -> Level:
    if name not in _LEVELS:
        _LEVELS[name] = Level(name)
    return _LEVELS[name]
