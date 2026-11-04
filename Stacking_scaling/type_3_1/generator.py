"""
Domino Chain Reaction 3D — level generator (rules v2).

Construction:
  1. START (green) on the floor, facing a random direction.
  2. Main chain: dominoes one cell apart, optionally stepping up or down one
     level onto grey STEP blocks, ending at a gold GOAL -- or at a SPLITTER
     whose two side branches end at goals (or one goal and a trap).  With
     split_depth > 1 a branch may end at a further splitter, building a tree
     of forks (the hard tier uses three levels).
  3. Decoy lines: an empty, supported gap followed by dominoes and a red
     PROTECTED piece.  They are placed beside START (wrong push direction),
     after the goal (the wave keeps travelling past a falling goal) and on
     a splitter's second side.  Filling a decoy gap wastes a patch or loses.
  4. K dominoes are carved out of the chain; the player gets exactly K.
     With `min_raised`, that many must sit on a step (z >= 2), so the
     height of a patch is part of the answer.
  5. The exact search (difficulty.solutions) must find exactly one winning
     (placement, direction) and it must use all K patches; the level is
     then assigned to a tier by measured difficulty (difficulty.TIERS).

Every column holds at most one standing piece (plus STEP blocks under it),
and a new column may only touch the column it continues from, so nothing
hits anything by accident and the picture shows exactly what is connected.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from .difficulty import analyze, fits_tier, solutions
from .game_core import DIR_NAME_TO_VEC, LevelSpec, PieceType, Vec3, perp_dirs

DIRS = ["+x", "-x", "+y", "-y"]
Col = Tuple[int, int]


def _step(c: Col, d: str) -> Col:
    v = DIR_NAME_TO_VEC[d]
    return (c[0] + v[0], c[1] + v[1])


class _Board:
    def __init__(self, grid: Vec3):
        self.grid = grid
        self.pieces: Dict[Vec3, PieceType] = {}     # anchor -> type
        self.used: Set[Col] = set()                 # columns with anything
        self.gaps: Set[Col] = set()                 # columns kept empty

    def inside(self, c: Col, b: int) -> bool:
        A, B, C = self.grid
        return 1 <= c[0] <= A and 1 <= c[1] <= B and 1 <= b and b + 1 <= C

    def free(self, c: Col) -> bool:
        return c not in self.used and c not in self.gaps

    def clear(self, c: Col, allowed: Set[Col]) -> bool:
        """No other occupied/gap column touches c (besides `allowed`)."""
        for d in DIRS:
            n = _step(c, d)
            if n not in allowed and (n in self.used or n in self.gaps):
                return False
        return True

    def blocks(self, c: Col, b: int) -> None:
        for z in range(1, b):
            self.pieces[(c[0], c[1], z)] = PieceType.BLOCK

    def put(self, pt: PieceType, c: Col, b: int) -> None:
        self.blocks(c, b)
        self.pieces[(c[0], c[1], b)] = pt
        self.used.add(c)


def _line(bd: _Board, c: Col, b: int, d: str, n: int, end: PieceType,
          rng: random.Random, p_step: float) -> Optional[List[Tuple[Col, int]]]:
    """Place n pieces continuing from (c, b) in direction d; the last is
    `end`.  Heights change by at most one level per step."""
    out = []
    for i in range(n):
        nc = _step(c, d)
        nb = b
        if rng.random() < p_step:
            nb = max(1, b + rng.choice((-1, 1)))
        if not bd.inside(nc, nb) or not bd.free(nc) or not bd.clear(nc, {c}):
            return None
        bd.put(end if i == n - 1 else PieceType.DOMINO, nc, nb)
        out.append((nc, nb))
        c, b = nc, nb
    return out


def _forks(bd: _Board, c: Col, b: int, d_in: str, depth: int,
           rng: random.Random, p_step: float, both_goals: bool):
    """Both sides of the splitter at (c, b), hit while travelling d_in:
    the first side always leads on (to a goal, or to a deeper splitter while
    depth > 1), the second to a goal / deeper splitter if `both_goals`, else
    to a goal or a trap at random.  (chain cells, ends, end dirs) or None."""
    chain, ends, end_dirs = [], [], []
    sides = perp_dirs(d_in)
    rng.shuffle(sides)
    for i, sd in enumerate(sides):
        if i == 1 and not both_goals and rng.random() < 0.5:
            if not _trap(bd, c, b, sd, rng):
                return None
            continue
        deeper = depth > 1 and (i == 0 or rng.random() < 0.5)
        line = _line(bd, c, b, sd, rng.randint(2, 4),
                     PieceType.SPLITTER if deeper else PieceType.GOAL, rng, p_step)
        if line is None:
            return None
        chain += line
        if deeper:
            sub = _forks(bd, line[-1][0], line[-1][1], sd, depth - 1, rng, p_step,
                         rng.random() < 0.5)
            if sub is None:
                return None
            chain += sub[0]
            ends += sub[1]
            end_dirs += sub[2]
        else:
            ends.append(line[-1])
            end_dirs.append(sd)
    return chain, ends, end_dirs


def _trap(bd: _Board, c: Col, b: int, d: str, rng: random.Random) -> bool:
    """Gap (kept empty, on a step if needed) then 1-2 dominoes and a P."""
    g = _step(c, d)
    if not bd.inside(g, b) or not bd.free(g) or not bd.clear(g, {c}):
        return False
    bd.gaps.add(g)
    snapshot = (dict(bd.pieces), set(bd.used))
    # Longest trap that fits, down to "gap then the red piece".
    for n in range(rng.randint(2, 3), 0, -1):
        if _line(bd, g, b, d, n, PieceType.PROTECTED, rng, 0.0) is not None:
            bd.blocks(g, b)              # the gap's step, if raised
            return True
        bd.pieces, bd.used = dict(snapshot[0]), set(snapshot[1])
    bd.gaps.discard(g)
    return False


def generate_level(grid_size: Vec3 = (6, 5, 3),
                   seed: int = 0,
                   n_patches: int = 2,
                   main_len: Tuple[int, int] = (3, 5),
                   use_height: bool = False,
                   use_split: bool = False,
                   two_goals: bool = False,
                   n_decoys: int = 0,
                   n_protected: int = 0,
                   min_raised: int = 0,
                   p_step: Optional[float] = None,
                   split_depth: int = 1,
                   tier: Optional[str] = None,
                   score_band: Optional[Tuple[float, float]] = None,
                   max_attempts: int = 300,
                   **_legacy) -> Optional[Tuple[LevelSpec, dict]]:
    """Generate one level; see the module docstring.  `n_protected` adds
    plain P pieces away from every wave line (pure scenery)."""
    rng = random.Random(seed)
    A, B, C = grid_size
    if p_step is None:
        p_step = 0.4 if use_height else 0.0
    for attempt in range(max_attempts):
        bd = _Board(grid_size)
        L = rng.randint(*main_len)
        # START and a direction with room for the whole chain (a random
        # pick ran into the board edge ~80% of the time).
        room = []
        for x in range(1, A + 1):
            for y in range(1, B + 1):
                for d in DIRS:
                    v = DIR_NAME_TO_VEC[d]
                    run = 0
                    while (1 <= x + v[0] * (run + 1) <= A
                           and 1 <= y + v[1] * (run + 1) <= B):
                        run += 1
                    if run >= L + (0 if use_split else 1):
                        room.append(((x, y), d))
        if not room:
            return None
        start, d0 = rng.choice(room)
        bd.put(PieceType.START, start, 1)
        chain: List[Tuple[Col, int]] = []

        if use_split:
            main = _line(bd, start, 1, d0, L, PieceType.SPLITTER, rng, p_step)
            if main is None:
                continue
            chain += main
            tc, tb = main[-1]
            sides = perp_dirs(d0)
            rng.shuffle(sides)
            if split_depth >= 2:
                # a tree of splitters: every fork sends one side on toward a
                # goal (or a deeper splitter), the other to a goal or a trap
                sub = _forks(bd, tc, tb, d0, split_depth, rng, p_step, two_goals)
                if sub is None:
                    continue
                chain += sub[0]
                ends, end_dirs = sub[1], sub[2]
            else:
                a = _line(bd, tc, tb, sides[0], rng.randint(2, 4), PieceType.GOAL,
                          rng, p_step)
                if a is None:
                    continue
                chain += a
                ends, end_dirs = [a[-1]], [sides[0]]
                if two_goals:
                    b2 = _line(bd, tc, tb, sides[1], rng.randint(2, 4),
                               PieceType.GOAL, rng, p_step)
                    if b2 is None:
                        continue
                    chain += b2
                    ends.append(b2[-1])
                    end_dirs.append(sides[1])
                elif not _trap(bd, tc, tb, sides[1], rng):
                    continue
        else:
            main = _line(bd, start, 1, d0, L, PieceType.GOAL, rng, p_step)
            if main is None:
                continue
            chain += main
            ends, end_dirs = [main[-1]], [d0]

        # Decoys: beside START first (the wrong push directions), then past
        # a goal (the wave keeps going after the goal falls).
        spots = [(start, 1, d) for d in DIRS if d != d0]
        rng.shuffle(spots)
        spots += [(c, b, d) for (c, b), d in zip(ends, end_dirs)]
        placed = 0
        for c, b, d in spots:
            if placed >= n_decoys:
                break
            placed += _trap(bd, c, b, d, rng)
        if placed < n_decoys:
            continue

        # Scenery P pieces away from every line.
        for _ in range(n_protected):
            for _try in range(20):
                c = (rng.randint(1, A), rng.randint(1, B))
                if bd.free(c) and bd.clear(c, set()):
                    bd.put(PieceType.PROTECTED, c, 1)
                    break

        # Carve the patches out of the chain's plain dominoes.
        carvable = [(c, b) for c, b in chain
                    if bd.pieces[(c[0], c[1], b)] == PieceType.DOMINO]
        if len(carvable) < n_patches:
            continue
        carved = rng.sample(carvable, n_patches)
        patches = sorted((c[0], c[1], b) for c, b in carved)
        if sum(1 for p in patches if p[2] >= 2) < min_raised:
            continue                       # steps must matter: a patch on a step
        fixed = [(pt, pos) for pos, pt in sorted(bd.pieces.items())
                 if pos not in patches]

        spec = LevelSpec(grid_size=grid_size, fixed_pieces=fixed,
                         inventory={PieceType.DOMINO: n_patches},
                         push_budget=1)
        sols = solutions(spec, n_patches)
        if sols != [(frozenset(patches), d0)]:
            continue                       # not unique, or a patch is spare
        spec.meta = {"solution_patches": [list(p) for p in patches]}
        m = analyze(spec, n_patches, samples=600, seed=seed)
        if tier is not None and not fits_tier(m, n_patches, tier, score_band):
            continue

        start_pos = (start[0], start[1], 1)
        meta = {
            "rules": "v2",
            "seed": seed,
            "attempt": attempt,
            "tier": tier,
            "grid_size": list(grid_size),
            "patch_count": n_patches,
            "fixed_count": len(fixed),
            "use_height": use_height,
            "use_split": use_split,
            "n_decoys": n_decoys,
            "origin": list(start_pos),
            "origin_dir": d0,
            "solution_patches": [list(p) for p in patches],
            "patch_pieces": [("D", list(p)) for p in patches],
            "pushes": [(list(start_pos), d0)],
            "push_budget": 1,
            "difficulty": m,
        }
        spec.meta = dict(meta)
        return spec, meta
    return None


def level_has_features(spec: LevelSpec, meta: dict, **_kw) -> bool:
    """Kept for old callers: v2 levels are built with their features."""
    return True


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    r = generate_level(seed=seed)
    if r is None:
        raise SystemExit("generation failed")
    spec, meta = r
    for pt, pos in spec.fixed_pieces:
        print(f"  {pt.value}@{pos}")
    print("solution:", meta["patch_pieces"], meta["pushes"])
    print("difficulty:", meta["difficulty"])
