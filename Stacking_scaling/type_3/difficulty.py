"""
FluidNet 3D - exact difficulty measurement.

Every subset of valves is enumerated (levels have <= ~10 valves), so all
numbers below are exact, not sampled.  A game ends only on WIN (every
target full) or LOSE (fluid touches a protected cell); a player keeps
opening valves until one of those happens.

Reported metrics:
  min_plan          fewest valves in a winning set
  n_min_sets        how many different minimal winning sets exist
  prefilled         targets already full before any move (should be 0)
  traps             valves in no winning set (opening them eventually loses)
  nonlocal_traps    traps that are not next to a protected cell, i.e. the
                    danger is only visible by tracing pipes further on
  p_random          win probability of a player opening random valves
  p_frontier        ... of "open a valve that touches fluid and is not next
                    to a red cell" (the obvious local rule)
  p_greedy          ... of "open the valve nearest to an empty target"
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Tuple

from .game_core import CellType, GameState, LevelSpec, face_neighbors
from .propagation import _compute_full_fluid

Pos = Tuple[int, int, int]


def analyze(spec: LevelSpec) -> Dict:
    V: List[Pos] = list(spec.valves)
    n = len(V)
    full = (1 << n) - 1
    fluid: Dict[int, Dict[Pos, str]] = {}
    lose: Dict[int, bool] = {}
    win: Dict[int, bool] = {}
    for m in range(1 << n):
        opened = {V[i] for i in range(n) if m >> i & 1}
        f = _compute_full_fluid(GameState(spec=spec, opened_valves=opened))
        fluid[m] = f
        lose[m] = any(p in f for p in spec.protected)
        win[m] = not lose[m] and all(f.get(t) == spec.cells[t].color
                                     for t in spec.targets)

    win_sets = [m for m in range(1 << n) if win[m]]
    used = 0
    for m in win_sets:
        used |= m
    min_plan = min((bin(m).count("1") for m in win_sets), default=None)
    n_min_sets = sum(1 for m in win_sets if bin(m).count("1") == min_plan)

    red_adjacent = {i for i, v in enumerate(V)
                    if any(nb in spec.protected
                           for nb in face_neighbors(v, spec.grid_size))}
    traps = [i for i in range(n) if not used >> i & 1]
    nonlocal_traps = [i for i in traps if i not in red_adjacent]

    def play(choose: Callable[[int], List[int]]) -> float:
        @lru_cache(None)
        def p(m: int) -> float:
            if m == full:
                return 0.0
            cands = choose(m)
            total = 0.0
            for i in cands:
                nm = m | 1 << i
                total += 1.0 if win[nm] else (0.0 if lose[nm] else p(nm))
            return total / len(cands)
        return 1.0 if win[0] else p(0)

    def closed(m: int) -> List[int]:
        return [i for i in range(n) if not m >> i & 1]

    def frontier(m: int) -> List[int]:
        f = fluid[m]
        c = closed(m)
        wet_safe = [i for i in c if i not in red_adjacent and any(
            nb in f for nb in face_neighbors(V[i], spec.grid_size))]
        safe = [i for i in c if i not in red_adjacent]
        return wet_safe or safe or c

    def greedy(m: int) -> List[int]:
        f = fluid[m]
        empty = [t for t in spec.targets if f.get(t) != spec.cells[t].color]
        c = closed(m)
        if not empty:
            return c
        d = {i: min(sum(abs(a - b) for a, b in zip(V[i], t)) for t in empty)
             for i in c}
        best = min(d.values())
        return [i for i in c if d[i] == best]

    prefilled = sum(1 for t in spec.targets
                    if fluid[0].get(t) == spec.cells[t].color)
    return {
        "n_valves": n,
        "min_plan": min_plan,
        "n_min_sets": n_min_sets,
        "n_win_sets": len(win_sets),
        "prefilled": prefilled,
        "traps": len(traps),
        "nonlocal_traps": len(nonlocal_traps),
        "p_random": round(play(closed), 4),
        "p_frontier": round(play(frontier), 4),
        "p_greedy": round(play(greedy), 4),
    }


# Tier = measured difficulty band (bands are disjoint).
TIERS: Dict[str, Dict] = {
    "easy":   dict(plan=(1, 2), nonlocal_traps=0, max_frontier=1.0, max_greedy=1.0),
    "medium": dict(plan=(2, 3), nonlocal_traps=1, max_frontier=0.6, max_greedy=1.0),
    "hard":   dict(plan=(3, 5), nonlocal_traps=2, max_frontier=0.35, max_greedy=0.35),
}


def fits_tier(metrics: Dict, tier: str) -> bool:
    t = TIERS[tier]
    lo, hi = t["plan"]
    return (metrics["min_plan"] is not None
            and lo <= metrics["min_plan"] <= hi
            and metrics["n_min_sets"] == 1
            and metrics["prefilled"] == 0
            and metrics["traps"] >= 1
            and metrics["nonlocal_traps"] >= t["nonlocal_traps"]
            and metrics["p_frontier"] <= t["max_frontier"]
            and metrics["p_greedy"] <= t["max_greedy"])
