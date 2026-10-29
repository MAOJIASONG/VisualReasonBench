"""
Domino Chain Reaction 3D — exact solution search + difficulty metrics.

A patch only matters if the wave reaches it, and the wave only reaches a
patch through a column where it would otherwise stop.  So all relevant
placements can be enumerated exactly by branching at every point where the
wave stops in front of an empty, supported column ("put a patch here or
not"), for each of the four push directions.  Patches placed anywhere else
never change the outcome.

Metrics:
  n_solutions     winning (relevant placement, direction) pairs -- 1 means
                  the level has a unique solution
  min_patches     fewest patches any winning placement needs (== the
                  inventory means every patch is load-bearing)
  gaps            empty supported columns sitting between two pieces in a
                  straight line ("obvious holes")
  decoy_gaps      gaps that are NOT part of the solution
  p_random        win rate of random legal placement + random direction
  p_gapfill       win rate of "fill the holes nearest START, push START at
                  its neighbour" (the obvious visual shortcut)
  p_wavefill      win rate of following the wave and filling whichever
                  hole it stops at (solves straight chains; fails when a
                  splitter or decoy forces a real choice)
  raised          solution patches standing on a step (z >= 2)
  score           (patches - 1) + 0.5 * (decoy_gaps + raised)
                  + log2(1 / p_wavefill)

Tiers (TIERS) are one ladder for the whole game, not per category:
  easy 1-2 patches, <= 1 decoy hole, score <= 3.1
  medium 3 patches, 2 decoy holes,   score 4-5.6
  hard 4 patches,   >= 3 decoy holes, score >= 6
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .cascade import simulate_push
from .game_core import (
    DIR_NAME_TO_VEC, DIRS4, GameState, LevelSpec, PieceType, PlacedPiece,
    Vec3, in_grid, init_state, piece_is_standing,
)

Patch = Vec3                      # base cell of a patch domino


def _state_with(spec: LevelSpec, patches) -> GameState:
    st = init_state(spec)
    for pos in patches:
        piece = PlacedPiece(PieceType.DOMINO, pos)
        st.pieces[pos] = piece
        st.pieces[(pos[0], pos[1], pos[2] + 1)] = piece
    return st


def start_of(spec: LevelSpec) -> Vec3:
    return next(pos for pt, pos in spec.fixed_pieces if pt == PieceType.START)


def can_stand(st: GameState, base: Vec3) -> bool:
    """A domino could be placed with this base (free, in grid, supported)."""
    x, y, z = base
    top = (x, y, z + 1)
    if not (in_grid(base, st.spec.grid_size) and in_grid(top, st.spec.grid_size)):
        return False
    if st.piece_at_cell(base) or st.piece_at_cell(top):
        return False
    if z == 1:
        return True
    under = st.piece_at_cell((x, y, z - 1))
    return under is not None and under.type == PieceType.BLOCK


def outcome(st: GameState) -> str:
    anchors = st.anchors()
    if any(p.type == PieceType.PROTECTED and p.toppled for p in anchors):
        return "lose"
    if all(p.toppled for p in anchors if p.type == PieceType.GOAL):
        return "win"
    return "open"


def _frontier(st: GameState) -> List[Patch]:
    """Empty supported bases where the last cascade stopped and a patch
    would have been hit."""
    out: List[Patch] = []
    for ev in st.last_cascade:
        if ev["kind"] != "wave_stop":
            continue
        src = st.piece_at_cell(tuple(ev["from"]))
        dx, dy, _ = DIR_NAME_TO_VEC[ev["dir"]]
        x, y, z = src.pos
        for b in (z - 1, z, z + 1):
            base = (x + dx, y + dy, b)
            if b >= 1 and can_stand(st, base) and base not in out:
                out.append(base)
    return out


def solutions(spec: LevelSpec, k: int
              ) -> List[Tuple[FrozenSet[Patch], str]]:
    """Every winning (relevant patch set, push direction) with <= k patches."""
    start = start_of(spec)
    wins: Set[Tuple[FrozenSet[Patch], str]] = set()
    for d, _ in DIRS4:
        seen: Set[FrozenSet[Patch]] = set()
        todo = [frozenset()]
        while todo:
            S = todo.pop()
            if S in seen:
                continue
            seen.add(S)
            st = _state_with(spec, S)
            simulate_push(st, start, d)
            res = outcome(st)
            if res == "win":
                wins.add((S, d))
                continue
            if res == "lose" or len(S) >= k:
                continue
            for b in _frontier(st):
                todo.append(S | {b})
    return sorted(wins, key=lambda w: (len(w[0]), w[1], sorted(w[0])))


def gaps(spec: LevelSpec) -> List[Patch]:
    """Empty supported bases lying between two standing pieces in a line,
    close enough in height that a domino there would link them."""
    st = init_state(spec)
    standing = {p.pos for p in st.anchors() if piece_is_standing(p.type)}
    A, B, C = spec.grid_size
    out = []
    for x in range(1, A + 1):
        for y in range(1, B + 1):
            for b in range(1, C):
                if not can_stand(st, (x, y, b)):
                    continue
                for (dx, dy) in ((1, 0), (0, 1)):
                    lo = [p for p in standing if p[:2] == (x - dx, y - dy)
                          and abs(p[2] - b) <= 1]
                    hi = [p for p in standing if p[:2] == (x + dx, y + dy)
                          and abs(p[2] - b) <= 1]
                    if lo and hi:
                        out.append((x, y, b))
                        break
    return out


def _legal_bases(spec: LevelSpec) -> List[Patch]:
    st = init_state(spec)
    A, B, C = spec.grid_size
    return [(x, y, b) for x in range(1, A + 1) for y in range(1, B + 1)
            for b in range(1, C) if can_stand(st, (x, y, b))]


def _wins(spec: LevelSpec, patches, d: str) -> bool:
    st = _state_with(spec, patches)
    simulate_push(st, start_of(spec), d)
    return outcome(st) == "win"


def analyze(spec: LevelSpec, k: int, samples: int = 1500,
            seed: int = 0) -> Dict:
    sols = solutions(spec, k)
    canon = set(map(tuple, (spec.meta or {}).get("solution_patches", [])))
    g = gaps(spec)
    decoys = [c for c in g if c not in canon]

    rng = random.Random(seed)
    legal = _legal_bases(spec)
    wins = 0
    for _ in range(samples):
        while True:   # k distinct legal bases whose cells do not overlap
            placed = rng.sample(legal, min(k, len(legal)))
            cells = [c for b in placed for c in (b, (b[0], b[1], b[2] + 1))]
            if len(cells) == len(set(cells)):
                break
        d = rng.choice([n for n, _ in DIRS4])
        wins += _wins(spec, placed, d)

    # Gap-fill shortcut: fill the k holes nearest START, then push START at
    # whatever now stands next to it (averaged over ties).
    start = start_of(spec)
    dist = lambda c: abs(c[0] - start[0]) + abs(c[1] - start[1])
    ranked = sorted(g, key=dist)
    fills = [ranked[:k]] if len(ranked) <= k else [
        list(c) for c in itertools.combinations(ranked, k)
        if all(dist(x) <= dist(ranked[k - 1]) for x in c)]
    trials, ok = 0, 0
    for fill in fills:
        st = _state_with(spec, fill)
        dirs = [n for n, _ in DIRS4
                if any(st.piece_at_cell(c) and piece_is_standing(st.piece_at_cell(c).type)
                       for c in ((start[0] + v[0], start[1] + v[1], start[2] + dz)
                                 for v in (DIR_NAME_TO_VEC[n],) for dz in (-1, 0, 1)))]
        for d in dirs or [n for n, _ in DIRS4]:
            trials += 1
            ok += _wins(spec, fill, d)

    p_wave = wave_follow(spec, k)
    m = {
        "n_solutions": len(sols),
        "p_wavefill": round(p_wave, 4),
        "min_patches": min((len(s) for s, _ in sols), default=None),
        "gaps": len(g),
        "decoy_gaps": len(decoys),
        "raised": sum(1 for c in canon if c[2] >= 2),
        "p_random": round(wins / samples, 4),
        "p_gapfill": round(ok / max(1, trials), 4),
    }
    m["score"] = score(m, k)
    return m


def score(m: Dict, k: int) -> float:
    """One number for "how much has to be worked out": every patch past the
    first, half a point per hole that must stay empty and per patch on a
    step, plus the bits of choice a wave-follower faces (log2 1/p_wavefill,
    capped at 6)."""
    bits = math.log2(1 / max(m["p_wavefill"], 1 / 64))
    return round((k - 1) + 0.5 * m["decoy_gaps"] + 0.5 * m.get("raised", 0) + bits, 2)


def wave_follow(spec: LevelSpec, k: int) -> float:
    """Win probability of: push START toward an adjacent piece or hole; each
    time the wave stops in front of holes, fill one of them (uniformly at
    random) and re-run, until patches run out.  This solves any straight
    chain, so it isolates whether a level needs reasoning about splitters
    and decoys."""
    start = start_of(spec)
    st0 = init_state(spec)
    dirs = []
    for n, _ in DIRS4:
        v = DIR_NAME_TO_VEC[n]
        col = [(start[0] + v[0], start[1] + v[1], start[2] + dz) for dz in (-1, 0, 1)]
        if any(st0.piece_at_cell(c) and piece_is_standing(st0.piece_at_cell(c).type)
               for c in col if in_grid(c, spec.grid_size)) or any(
               can_stand(st0, c) for c in col if c[2] >= 1):
            dirs.append(n)

    def p(S, d) -> float:
        st = _state_with(spec, S)
        simulate_push(st, start, d)
        res = outcome(st)
        if res != "open" or len(S) >= k:
            return 1.0 if res == "win" else 0.0
        front = [b for b in _frontier(st) if b not in S]
        if not front:
            return 0.0
        return sum(p(S | {b}, d) for b in front) / len(front)

    return sum(p(frozenset(), d) for d in dirs) / max(1, len(dirs))


# One ladder for the whole game: every easy level is below every medium level
# on pieces to place, holes that must stay empty and score, and every medium
# below every hard (score bands leave a gap).
TIERS: Dict[str, Dict] = {
    "easy":   dict(patches=(1, 2), decoy_gaps=(0, 1), score=(0, 3.1), max_gapfill=1.0),
    "medium": dict(patches=(3, 3), decoy_gaps=(2, 2), score=(4, 5.6), max_gapfill=0.5,
                   max_wavefill=0.5),
    "hard":   dict(patches=(4, 4), decoy_gaps=(3, 9), score=(6, 99), max_gapfill=0.0,
                   max_wavefill=0.34),
}


def fits_tier(m: Dict, k: int, tier: str,
              score_band: Optional[Tuple[float, float]] = None) -> bool:
    """`score_band` overrides the tier's own score band."""
    t = TIERS[tier]
    lo, hi = t["patches"]
    band = score_band or t["score"]
    if not band[0] <= m["score"] <= band[1]:
        return False
    return (lo <= k <= hi
            and m["n_solutions"] == 1
            and m["min_patches"] == k
            and t["decoy_gaps"][0] <= m["decoy_gaps"] <= t["decoy_gaps"][1]
            and m["p_gapfill"] <= t["max_gapfill"]
            and m["p_wavefill"] <= t.get("max_wavefill", 1.0))
