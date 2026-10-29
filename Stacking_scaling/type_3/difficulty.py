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
  p_lookahead       ... of a player who simulates ONE valve ahead: opens a
                    valve that does not lose right now and adds fluid.  It
                    only fails on combination traps (two valves that are
                    each safe but lose together, e.g. two pumps meeting)
  shortest_ok       does "for each tank open the valves on its route with
                    the fewest valves" win?  (False = the obvious route is
                    a trap, e.g. a back door into a red cell)
  filler            valves that are neither in the winning set nor fatal
                    when added to it (dead weight; the generator wants 0)
"""

from __future__ import annotations

import math

from functools import lru_cache
from typing import Callable, Dict, List, Tuple

from collections import deque

from .game_core import ActionCode, CellType, GameState, LevelSpec, face_neighbors
from .propagation import _compute_full_fluid, propagate_fluid

Pos = Tuple[int, int, int]


def outcome(spec: LevelSpec, opened) -> Tuple[bool, bool, Dict[Pos, str]]:
    """(lose, win, fluid) once exactly `opened` valves are open.  Losing =
    a red cell touched, a tank reached by the wrong colour, or two colours
    meeting."""
    st = GameState(spec=spec, opened_valves=set(opened))
    lose = propagate_fluid(st).code != ActionCode.OK
    fluid = _compute_full_fluid(st)
    win = not lose and all(fluid.get(t) == spec.cells[t].color
                           for t in spec.targets)
    return lose, win, fluid


def shortest_plan(spec: LevelSpec) -> List[Pos]:
    """Valves on each tank's fewest-valve route from a same-colour pump
    (0-1 BFS; ties broken by route length, then coordinates)."""
    g = spec.grid_size
    plan: List[Pos] = []
    for t in spec.targets:
        col = spec.cells[t].color
        best: Dict[Pos, Tuple[int, int]] = {}
        prev: Dict[Pos, Pos] = {}
        dq = deque()
        for s in spec.sources:
            if spec.cells[s].color == col:
                best[s] = (0, 0)
                dq.append(s)
        while dq:
            c = dq.popleft()
            cv_, cl = best[c]
            for nb in sorted(face_neighbors(c, g)):
                cell = spec.cells.get(nb)
                if cell is None or cell.type == CellType.PROTECTED:
                    continue
                if cell.type in (CellType.SOURCE, CellType.TARGET) and cell.color != col:
                    continue
                w = 1 if cell.type == CellType.VALVE and nb not in plan else 0
                cand = (cv_ + w, cl + 1)
                if nb not in best or cand < best[nb]:
                    best[nb] = cand
                    prev[nb] = c
                    (dq.appendleft if w == 0 else dq.append)(nb)
        if t not in best:
            continue
        c = t
        while c in prev:
            if spec.cells[c].type == CellType.VALVE and c not in plan:
                plan.append(c)
            c = prev[c]
    return plan


def analyze(spec: LevelSpec) -> Dict:
    V: List[Pos] = list(spec.valves)
    n = len(V)
    full = (1 << n) - 1
    fluid: Dict[int, Dict[Pos, str]] = {}
    lose: Dict[int, bool] = {}
    win: Dict[int, bool] = {}
    for m in range(1 << n):
        lose[m], win[m], fluid[m] = outcome(
            spec, {V[i] for i in range(n) if m >> i & 1})

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

    def lookahead(m: int) -> List[int]:
        c = closed(m)
        safe = [i for i in c if not lose[m | 1 << i]]
        grow = [i for i in safe if len(fluid[m | 1 << i]) > len(fluid[m])]
        return grow or safe or c

    best = min((m for m in win_sets if bin(m).count("1") == min_plan),
               default=None)
    filler = 0
    if best is not None:
        filler = sum(1 for i in range(n) if not best >> i & 1
                     and not lose[best | 1 << i])
    sp = shortest_plan(spec)
    sp_mask = sum(1 << V.index(v) for v in sp)

    prefilled = sum(1 for t in spec.targets
                    if fluid[0].get(t) == spec.cells[t].color)
    m = {
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
        "p_lookahead": round(play(lookahead), 4),
        "shortest_ok": bool(win[sp_mask]),
        "filler": filler,
        "n_colors": len(spec.colors),
    }
    m["score"] = score(m)
    return m


def score(m: Dict) -> float:
    """Bits of insight a level asks for: log2 1/P(frontier) (how much blind
    opening is punished) + 2 * log2 1/P(lookahead) (combination traps a
    one-step simulator cannot see) + 1 if the fewest-valve route loses."""
    bits = lambda p: math.log2(1 / max(p, 1 / 256))
    return round(bits(m["p_frontier"]) + 2 * bits(m["p_lookahead"])
                 + (0 if m["shortest_ok"] else 1), 2)


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
