"""
PipeFit 3D — exhaustive solution search + difficulty metrics (rules v2).

A solution is a water-tight pipe network joining every portal, plus the
support columns it needs, that fits the inventory.  With two portals the
network is a single route; with three it is a tree with one T junction
(every leaf-to-leaf path of such a tree passes through the junction, so
"route between the first two portals + branch from one of its cells to the
third" enumerates every tree exactly once).

Supports: pipes are processed bottom-up; a pipe with nothing solid below
gets a support column down to the floor or to the next solid cell.  A
column of height h is packed into S2 (two cells) and S1 pieces.

Metrics:
  n_solutions   distinct pipe networks that fit (capped at CAP)
  min_pipes     fewest pipe pieces any solution needs
  staircase_ok  the obvious "step toward the exit, then up, repeat" route
                (with supports under every pipe) solves the level
  n_networks    pipe networks of valid piece shapes within the pipe budget,
                ignoring piece types and supports (n_networks())
  score         log2(n_networks / n_solutions) + plan_length / 4; tiers
                (TIERS) are one ladder for the whole game: pieces <= 8 / 9-12 /
                >= 10 and score <= 4.5 / 5.5-8.5 / >= 9.5
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Set, Tuple

from .game_core import (
    DIRS, DIR_OPPOSITE, LevelSpec, ORIENTATIONS, PieceType, PlacedPiece,
    Vec3, portal_inward,
)

CAP = 500


def _nb(c: Vec3, d: int) -> Vec3:
    v = DIRS[d][1]
    return (c[0] + v[0], c[1] + v[1], c[2] + v[2])


def _dir(a: Vec3, b: Vec3) -> int:
    d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    return next(i for i, v in DIRS if v == d)


def _man(a: Vec3, b: Vec3) -> int:
    return sum(abs(x - y) for x, y in zip(a, b))


class Board:
    def __init__(self, spec: LevelSpec):
        self.spec = spec
        self.grid = spec.grid_size
        self.portals = list(spec.entrances) + list(spec.exits)
        self.blocked = set(spec.obstacles) | set(self.portals)

    def free(self, c: Vec3) -> bool:
        A, B, C = self.grid
        return (1 <= c[0] <= A and 1 <= c[1] <= B and 1 <= c[2] <= C
                and c not in self.blocked)

    def inner(self, portal: Vec3) -> Vec3:
        return _nb(portal, portal_inward(portal, self.grid))


def piece_of(dirs: Set[int]) -> Optional[Tuple[PieceType, int]]:
    mask = sum(1 << d for d in dirs)
    for pt in (PieceType.PIPE_I, PieceType.PIPE_L, PieceType.PIPE_T):
        if mask in ORIENTATIONS[pt]:
            return pt, ORIENTATIONS[pt].index(mask)
    return None


def network_dirs(bd: Board, paths: List[Tuple[Vec3, List[Vec3], Optional[Vec3]]]
                 ) -> Dict[Vec3, Set[int]]:
    """Connector directions of every pipe cell.  Each path is
    (start_portal_or_None, cells, end_portal_or_None)."""
    dirs: Dict[Vec3, Set[int]] = {}
    for p_from, cells, p_to in paths:
        for i, c in enumerate(cells):
            ds = dirs.setdefault(c, set())
            if i > 0:
                ds.add(_dir(c, cells[i - 1]))
            elif p_from is not None:
                ds.add(DIR_OPPOSITE[portal_inward(p_from, bd.grid)])
            if i < len(cells) - 1:
                ds.add(_dir(c, cells[i + 1]))
            elif p_to is not None:
                ds.add(DIR_OPPOSITE[portal_inward(p_to, bd.grid)])
    return dirs


def support_columns(bd: Board, pipes: Set[Vec3]) -> Optional[List[List[Vec3]]]:
    """Support columns (bottom-up cell lists) the pipes need, or None."""
    ok: Set[Vec3] = set()
    cols: List[List[Vec3]] = []
    for c in sorted(pipes, key=lambda c: c[2]):
        if c[2] == 1 or (c[0], c[1], c[2] - 1) in ok:
            ok.add(c)
            continue
        col = []
        z = c[2] - 1
        while z >= 1 and (c[0], c[1], z) not in ok:
            cell = (c[0], c[1], z)
            if cell in pipes or not bd.free(cell):
                return None
            col.append(cell)
            z -= 1
        col.reverse()
        ok.update(col)
        ok.add(c)
        cols.append(col)
    return cols


def pack(cols: List[List[Vec3]], s1: int, s2: int) -> Optional[Tuple[int, int]]:
    """(S1 used, S2 used) with as many S2 as possible, or None."""
    halves = sum(len(c) // 2 for c in cols)
    total = sum(len(c) for c in cols)
    a = min(s2, halves)
    b = total - 2 * a
    return (b, a) if b <= s1 else None


def fits(bd: Board, dirs: Dict[Vec3, Set[int]]) -> Optional[Dict]:
    """Pieces + supports for this network if the inventory allows it."""
    inv = bd.spec.inventory
    count = {PieceType.PIPE_I: 0, PieceType.PIPE_L: 0, PieceType.PIPE_T: 0}
    pieces = {}
    for c, ds in dirs.items():
        po = piece_of(ds)
        if po is None:
            return None
        count[po[0]] += 1
        pieces[c] = po
    if any(count[pt] > inv.get(pt, 0) for pt in count):
        return None
    cols = support_columns(bd, set(dirs))
    if cols is None:
        return None
    used = pack(cols, inv.get(PieceType.SUPPORT_S1, 0), inv.get(PieceType.SUPPORT_S2, 0))
    if used is None:
        return None
    return {"pieces": pieces, "cols": cols}


def plan_of(bd: Board, sol: Dict) -> List[PlacedPiece]:
    """Placements for a solution: supports bottom-up first, then pipes."""
    s2 = bd.spec.inventory.get(PieceType.SUPPORT_S2, 0)
    out: List[PlacedPiece] = []
    for col in sol["cols"]:
        i = 0
        while i < len(col):
            if s2 > 0 and i + 1 < len(col):
                out.append(PlacedPiece(PieceType.SUPPORT_S2, col[i], 0))
                s2 -= 1
                i += 2
            else:
                out.append(PlacedPiece(PieceType.SUPPORT_S1, col[i], 0))
                i += 1
    for c in sorted(sol["pieces"], key=lambda c: (c[2], c)):
        pt, o = sol["pieces"][c]
        out.append(PlacedPiece(pt, c, o))
    return out


def routes(bd: Board, s: Vec3, t: Vec3, max_len: int, avoid: Set[Vec3] = frozenset(),
           cap: int = 100000) -> Iterator[List[Vec3]]:
    """Simple cell paths s..t (inclusive) of at most max_len cells."""
    if not bd.free(s) or not bd.free(t) or s in avoid or t in avoid:
        return
    path = [s]
    seen = {s}
    found = [0]

    def dfs(cur):
        if found[0] >= cap:
            return
        if cur == t:
            found[0] += 1
            yield list(path)
            return
        if len(path) + _man(cur, t) > max_len:
            return
        for d in range(6):
            n = _nb(cur, d)
            if n in seen or n in avoid or not bd.free(n):
                continue
            path.append(n)
            seen.add(n)
            yield from dfs(n)
            path.pop()
            seen.discard(n)
    yield from dfs(s)


def solutions(spec: LevelSpec, cap: int = CAP) -> List[Dict]:
    bd = Board(spec)
    inv = spec.inventory
    budget = sum(inv.get(pt, 0) for pt in (PieceType.PIPE_I, PieceType.PIPE_L, PieceType.PIPE_T))
    portals = bd.portals
    out: List[Dict] = []
    if len(portals) == 2:
        a, b = portals
        for r in routes(bd, bd.inner(a), bd.inner(b), budget):
            f = fits(bd, network_dirs(bd, [(a, r, b)]))
            if f:
                out.append(f)
                if len(out) >= cap:
                    break
        return out
    if len(portals) != 3:
        return out
    a, b, c = portals
    tc = bd.inner(c)
    for r in routes(bd, bd.inner(a), bd.inner(b), budget - 1):
        on = set(r)
        # The route's own I / L pieces (the junction cell becomes the T):
        # a branch only adds pieces, so a route over the inventory is dead.
        kinds = [piece_of(ds)[0] for ds in
                 (lambda nd: [nd[x] for x in r])(network_dirs(bd, [(a, r, b)]))]
        n_i = kinds.count(PieceType.PIPE_I)
        n_l = kinds.count(PieceType.PIPE_L)
        rest = budget - len(r)
        for i, j in enumerate(r):
            if (n_i - (kinds[i] == PieceType.PIPE_I) > inv.get(PieceType.PIPE_I, 0)
                    or n_l - (kinds[i] == PieceType.PIPE_L) > inv.get(PieceType.PIPE_L, 0)):
                continue
            for d in range(6):
                start = _nb(j, d)
                if start in on or not bd.free(start) or 1 + _man(start, tc) > rest:
                    continue
                for br in routes(bd, start, tc, rest, avoid=on):
                    dirs = network_dirs(bd, [(a, r, b), (None, [j] + br, c)])
                    f = fits(bd, dirs)
                    if f:
                        out.append(f)
                        if len(out) >= cap:
                            return out
    return out


def n_networks(spec: LevelSpec, cap: int = 20000) -> int:
    """Pipe networks with a valid piece in every cell that use at most the
    inventory's number of pipes, ignoring piece types and supports: the
    space a player picks from.  solutions() is the subset that also fits
    the exact inventory and its supports."""
    bd = Board(spec)
    inv = spec.inventory
    budget = sum(inv.get(pt, 0) for pt in (PieceType.PIPE_I, PieceType.PIPE_L, PieceType.PIPE_T))
    portals = bd.portals
    n = 0
    shaped = lambda dirs: all(piece_of(ds) for ds in dirs.values())
    if len(portals) == 2:
        a, b = portals
        for r in routes(bd, bd.inner(a), bd.inner(b), budget, cap=cap):
            n += shaped(network_dirs(bd, [(a, r, b)]))
        return n
    a, b, c = portals
    tc = bd.inner(c)
    for r in routes(bd, bd.inner(a), bd.inner(b), budget - 1, cap=cap):
        on = set(r)
        rest = budget - len(r)
        for j in r:
            for d in range(6):
                start = _nb(j, d)
                if start in on or not bd.free(start) or 1 + _man(start, tc) > rest:
                    continue
                for br in routes(bd, start, tc, rest, avoid=on):
                    n += shaped(network_dirs(bd, [(a, r, b), (None, [j] + br, c)]))
                    if n >= cap:
                        return n
    return n


def staircase(bd: Board, s: Vec3, t: Vec3) -> Optional[List[Vec3]]:
    """Step horizontally toward t along the larger remaining axis, then one
    step vertically, and repeat."""
    path = [s]
    cur = s
    vertical = False
    for _ in range(60):
        if cur == t:
            return path
        dx, dy, dz = (t[0] - cur[0], t[1] - cur[1], t[2] - cur[2])
        step = None
        if vertical and dz:
            step = (0, 0, 1 if dz > 0 else -1)
        elif abs(dx) >= abs(dy) and dx:
            step = (1 if dx > 0 else -1, 0, 0)
        elif dy:
            step = (0, 1 if dy > 0 else -1, 0)
        elif dz:
            step = (0, 0, 1 if dz > 0 else -1)
        vertical = not vertical
        if step is None:
            continue
        nxt = (cur[0] + step[0], cur[1] + step[1], cur[2] + step[2])
        if not bd.free(nxt) or nxt in path:
            return None
        path.append(nxt)
        cur = nxt
    return None


def analyze(spec: LevelSpec) -> Dict:
    bd = Board(spec)
    sols = solutions(spec)
    stair_ok = False
    if len(bd.portals) == 2:
        a, b = bd.portals
        st = staircase(bd, bd.inner(a), bd.inner(b))
        stair_ok = st is not None and fits(bd, network_dirs(bd, [(a, st, b)])) is not None
    return {
        "n_solutions": len(sols),
        "capped": len(sols) >= CAP,
        "min_pipes": min((len(s["pieces"]) for s in sols), default=None),
        "staircase_ok": stair_ok,
        "n_portals": len(bd.portals),
    }


def idle_obstacles(spec: LevelSpec, m: Dict) -> int:
    """Obstacles whose removal changes nothing (same solution count, same
    staircase verdict): pure decoration, which the generator rejects."""
    import dataclasses
    idle = 0
    for o in spec.obstacles:
        rest = [c for c in spec.obstacles if c != o]
        s2 = dataclasses.replace(spec, obstacles=rest,
                                 obstacle_shapes=[[c] for c in rest])
        m2 = analyze(s2)
        if (m2["n_solutions"] == m["n_solutions"]
                and m2["staircase_ok"] == m["staircase_ok"]):
            idle += 1
    return idle


# One ladder for the whole game, not per category: every easy level places
# fewer pieces and scores lower than every medium / hard level, and score
# bands leave a gap so every medium scores below every hard.
TIERS: Dict[str, Dict] = {
    "easy":   dict(max_solutions=CAP, staircase=True, min_pipes=3, plan=(3, 8),
                   score=(0, 4.5)),
    "medium": dict(max_solutions=20, staircase=False, min_pipes=5, plan=(9, 12),
                   score=(5.5, 8.5)),
    "hard":   dict(max_solutions=12, staircase=False, min_pipes=7, plan=(10, 20),
                   score=(9.5, 99)),
}


def score(m: Dict) -> float:
    """How hard picking the network is: log2(networks / solutions), the bits
    the inventory and supports take away from "any route that looks right",
    plus a quarter point per piece to place."""
    return round(math.log2(max(m["n_networks"], 1) / max(m["n_solutions"], 1))
                 + 0.25 * m.get("plan_length", 0), 2)


def fits_tier(m: Dict, tier: str) -> bool:
    t = TIERS[tier]
    return (m["n_solutions"] >= 1
            and m["n_solutions"] <= t["max_solutions"]
            and (m["min_pipes"] or 0) >= t["min_pipes"]
            and (t["staircase"] or not m["staircase_ok"])
            and t["plan"][0] <= m.get("plan_length", 0) <= t["plan"][1])
