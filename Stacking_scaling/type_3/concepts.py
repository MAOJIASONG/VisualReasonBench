"""
FluidNet 3D — concept-driven level design.

Every level is built around ONE idea the player has to spot (its "concept"),
and every valve is a real decision: it is either in the unique winning set,
or adding it to that set loses.  No filler valves.

Concepts
  trace      Tree network.  Trap valves sit far from the red cell they feed,
             so the danger only shows when you trace the pipe.
  series     One tank behind several valves in a row; side spurs between
             them lead to red cells.  Open the whole row, none of the spurs.
  backdoor   The network has a loop.  The route to a tank with the fewest
             valves runs through pipes that also reach a red cell somewhere
             else -- take the longer, clean route.
  two_pumps  Two pumps, two colours.  Each tank wants its own colour and the
             two colours must never meet.  A valve that is safe on its own
             can lose once the other colour arrives (combination trap).
  mixed      Two pumps plus red cells: both kinds of trap in one level.

Tiers are measured (difficulty.analyze) and form one ladder for the whole
game (TIER_SCORE): every easy level has fewer valves and a lower
difficulty.score than every medium level, and every medium than every hard.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from .difficulty import analyze
from .game_core import Cell, CellType, LevelSpec, Vec3, face_neighbors

Pos = Tuple[int, int, int]
COLORS = ("blue", "orange")


# ---------------------------------------------------------------------------
# Network growth: every new pipe cell touches only the cell it grew from, so
# the picture shows exactly which pipes are connected.
# ---------------------------------------------------------------------------

class Net:
    def __init__(self, grid: Pos):
        self.grid = grid
        self.adj: Dict[Pos, Set[Pos]] = {}
        self.tree: Dict[Pos, int] = {}           # cell -> pump index
        self.parent: Dict[Pos, Pos] = {}
        self.bridges: List[List[Pos]] = []

    def add(self, c: Pos, frm: Optional[Pos], tree: int) -> None:
        self.adj.setdefault(c, set())
        self.tree[c] = tree
        if frm is not None:
            self.adj[c].add(frm)
            self.adj[frm].add(c)
            self.parent[c] = frm

    def ok(self, c: Pos, allowed: Set[Pos]) -> bool:
        A, B, C = self.grid
        if not (1 <= c[0] <= A and 1 <= c[1] <= B and 1 <= c[2] <= C):
            return False
        if c in self.adj:
            return False
        return all(nb in allowed or nb not in self.adj
                   for nb in face_neighbors(c, self.grid))

    def grow(self, start: Pos, length: int, rng: random.Random,
             straight: float = 0.65) -> List[Pos]:
        out: List[Pos] = []
        cur, prev = start, self.parent.get(start)
        tree = self.tree[start]
        for _ in range(length):
            opts = [c for c in face_neighbors(cur, self.grid) if self.ok(c, {cur})]
            if not opts:
                break
            nxt = None
            if prev is not None:
                d = tuple(a - b for a, b in zip(cur, prev))
                s = tuple(a + b for a, b in zip(cur, d))
                if s in opts and rng.random() < straight:
                    nxt = s
            nxt = nxt or rng.choice(opts)
            self.add(nxt, cur, tree)
            out.append(nxt)
            prev, cur = cur, nxt
        return out

    def bridge(self, u: Pos, v: Pos, rng: random.Random,
               max_len: int = 7) -> Optional[List[Pos]]:
        """A new pipe from u to v (a loop), touching nothing else."""
        path: List[Pos] = []
        budget = [400]

        def dist(c: Pos) -> int:
            return sum(abs(a - b) for a, b in zip(c, v))

        def dfs(cur: Pos) -> bool:
            budget[0] -= 1
            if budget[0] < 0 or len(path) >= max_len:
                return False
            opts = [c for c in face_neighbors(cur, self.grid)
                    if c not in path and self.ok(c, {cur, v})
                    and all(nb == cur or nb not in path
                            for nb in face_neighbors(c, self.grid))]
            rng.shuffle(opts)
            opts.sort(key=dist)
            for c in opts:
                path.append(c)
                if v in face_neighbors(c, self.grid) and len(path) >= 2:
                    return True
                if dfs(c):
                    return True
                path.pop()
            return False

        if not dfs(u):
            return None
        prev = u
        for c in path:
            self.add(c, prev, self.tree[u])
            prev = c
        self.adj[prev].add(v)
        self.adj[v].add(prev)
        self.bridges.append(list(path))
        return path

    def tips(self, roots: Set[Pos]) -> List[Pos]:
        return [c for c, n in self.adj.items() if len(n) == 1 and c not in roots]

    def inline(self, roots: Set[Pos]) -> List[Pos]:
        """Cells on a straight-through run (two neighbours) -- valve spots."""
        return [c for c, n in self.adj.items() if len(n) == 2 and c not in roots]

    def path_to_root(self, c: Pos) -> List[Pos]:
        out = [c]
        while out[-1] in self.parent:
            out.append(self.parent[out[-1]])
        return out


# ---------------------------------------------------------------------------
# Concept parameters per tier and acceptance rules
# ---------------------------------------------------------------------------

# One ladder for the whole game.  Valves: easy <= 4, medium 5-6, hard 7-8
# (the `valves` ranges below); score bands leave a gap between tiers.
TIER_SCORE: Dict[str, Tuple[float, float]] = {
    "easy": (0, 3), "medium": (4, 6), "hard": (7, 99),
}

CONCEPTS: Dict[str, Dict[str, Dict]] = {
    "trace": {
        "easy":   dict(grid=(4, 4, 4), pumps=1, targets=1, reds=1, loops=0, valves=(2, 3), plan=(1, 2)),
        "medium": dict(grid=(5, 5, 4), pumps=1, targets=2, reds=2, loops=0, valves=(5, 6), plan=(2, 3),
                       nonlocal_traps=1, max_frontier=0.6),
        "hard":   dict(grid=(5, 5, 5), pumps=1, targets=2, reds=3, loops=0, valves=(7, 8), plan=(3, 5),
                       nonlocal_traps=2, max_frontier=0.35, max_greedy=0.35),
    },
    "series": {
        "easy":   dict(grid=(4, 4, 4), pumps=1, targets=1, reds=1, loops=0, valves=(3, 3), plan=(2, 2)),
        "medium": dict(grid=(5, 5, 4), pumps=1, targets=1, reds=2, loops=0, valves=(5, 6), plan=(3, 3),
                       nonlocal_traps=1),
        "hard":   dict(grid=(5, 5, 5), pumps=1, targets=1, reds=3, loops=0, valves=(7, 8), plan=(4, 4),
                       nonlocal_traps=2, max_frontier=0.35),
    },
    "backdoor": {
        "easy":   dict(grid=(4, 4, 4), pumps=1, targets=1, reds=1, loops=1, valves=(3, 4), plan=(1, 3),
                       shortest_fails=True),
        "medium": dict(grid=(5, 5, 4), pumps=1, targets=2, reds=1, loops=1, valves=(5, 6), plan=(2, 4),
                       shortest_fails=True, max_frontier=0.6),
        "hard":   dict(grid=(5, 5, 5), pumps=1, targets=2, reds=2, loops=2, valves=(7, 8), plan=(3, 5),
                       shortest_fails=True, max_frontier=0.35, max_greedy=0.5),
    },
    "two_pumps": {
        "easy":   dict(grid=(5, 5, 4), pumps=2, targets=2, reds=0, loops=1, valves=(3, 4), plan=(2, 3)),
        "medium": dict(grid=(5, 5, 5), pumps=2, targets=2, reds=1, loops=1, valves=(5, 6), plan=(2, 4),
                       max_lookahead=0.75),
        "hard":   dict(grid=(6, 5, 5), pumps=2, targets=3, reds=1, loops=2, valves=(7, 8), plan=(3, 5),
                       max_lookahead=0.5, max_frontier=0.35),
    },
    "mixed": {
        "easy":   dict(grid=(5, 5, 5), pumps=2, targets=2, reds=1, loops=1, valves=(4, 4), plan=(2, 4),
                       nonlocal_traps=1),
        "medium": dict(grid=(6, 5, 5), pumps=2, targets=2, reds=2, loops=1, valves=(5, 6), plan=(3, 5),
                       nonlocal_traps=2, max_lookahead=0.75),
        "hard":   dict(grid=(6, 6, 5), pumps=2, targets=3, reds=2, loops=2, valves=(7, 8), plan=(3, 6),
                       nonlocal_traps=2, max_lookahead=0.75),
    },
}

INSIGHT = {
    "trace": "Trace each pipe to its end: a valve far from any red cell can still feed one.",
    "series": "The tank sits behind a row of valves: open the whole row, but none of the side spurs.",
    "backdoor": "The route with the fewest valves also reaches a red cell through a loop; take the clean detour.",
    "two_pumps": "Each tank needs its own colour and the colours must never meet: a valve that is safe alone can lose once the other colour arrives.",
    "mixed": "Two pumps and red cells: every route must avoid the red cells and the other colour.",
}


def accepts(m: Dict, p: Dict, tier: str) -> bool:
    """`score_band` in a concept's params overrides TIER_SCORE[tier]."""
    lo, hi = p["plan"]
    band = p.get("score_band", TIER_SCORE[tier])
    return (m["min_plan"] is not None and lo <= m["min_plan"] <= hi
            and m["n_min_sets"] == 1
            and m["prefilled"] == 0
            and m["traps"] >= 1
            and m["filler"] == 0
            and m["nonlocal_traps"] >= p.get("nonlocal_traps", 0)
            and m["p_frontier"] <= p.get("max_frontier", 1.0)
            and m["p_greedy"] <= p.get("max_greedy", 1.0)
            and m["p_lookahead"] <= p.get("max_lookahead", 1.0)
            and (not p.get("shortest_fails") or not m["shortest_ok"])
            and band[0] <= m["score"] <= band[1])


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _build(p: Dict, rng: random.Random) -> Optional[LevelSpec]:
    A, B, C = p["grid"]
    net = Net(p["grid"])
    pumps: List[Pos] = []
    for k in range(p["pumps"]):
        for _ in range(30):
            s = (rng.randint(1, A), rng.randint(1, B), 1)
            if net.ok(s, set()) and all(sum(abs(a - b) for a, b in zip(s, q)) >= 3
                                         for q in pumps):
                break
        else:
            return None
        net.add(s, None, k)
        pumps.append(s)
        if len(net.grow(s, rng.randint(2, 3), rng)) < 2:
            return None
    roots = set(pumps)

    # Branches until there are enough tips.  With two pumps the tips are
    # shared out so each pump feeds at least one tank.
    need = p["targets"] + p["reds"]
    for _ in range(150):
        if len(net.tips(roots)) >= need + p["loops"] + 1:
            break
        cand = [c for c in net.adj if c not in roots and len(net.adj[c]) <= 2]
        if not cand:
            return None
        j = rng.choice(cand)
        br = net.grow(j, rng.randint(2, 5), rng)
        if len(br) < 2:
            for c in br:                       # too short to read: undo
                for nb in net.adj.pop(c):
                    net.adj.get(nb, set()).discard(c)
                net.tree.pop(c, None); net.parent.pop(c, None)
    # Loops: join a tip to a far part of the network.  For two pumps the
    # first loop joins the two pumps' networks.
    for li in range(p["loops"]):
        tips = net.tips(roots)
        rng.shuffle(tips)
        done = False
        for u in tips:
            far = [v for v in net.adj if v != u and v not in roots
                   and len(net.adj[v]) <= 2
                   and (p["pumps"] == 1 or li > 0 or net.tree[v] != net.tree[u])
                   and v not in net.path_to_root(u)[:5]
                   and sum(abs(a - b) for a, b in zip(u, v)) <= 4]
            rng.shuffle(far)
            for v in far[:6]:
                if net.bridge(u, v, rng) is not None:
                    done = True
                    break
            if done:
                break
        if not done:
            return None

    tips = net.tips(roots)
    if len(tips) < need:
        return None
    rng.shuffle(tips)

    def far(a: Pos, others) -> bool:
        return all(max(abs(x - y) for x, y in zip(a, b)) >= 2 for b in others)

    # Tanks first (with two pumps: one per pump first), then red cells; all
    # spaced from the pumps and from each other.
    picked: List[Pos] = []
    for k in range(p["pumps"]):
        for t in tips:
            if net.tree[t] == k and far(t, picked + pumps):
                picked.append(t)
                break
    for t in tips:
        if len(picked) < need and t not in picked and far(t, picked + pumps):
            picked.append(t)
    if len(picked) < need:
        return None
    targets = picked[:p["targets"]]
    reds = picked[p["targets"]:need]
    tips = picked + [t for t in tips if t not in picked]
    # leftover tips become plain pipe ends -- they read as dead ends; drop them
    for t in tips[need:]:
        while t in net.adj and len(net.adj[t]) == 1 and t not in roots:
            (nb,) = net.adj.pop(t)
            net.adj[nb].discard(t)
            net.tree.pop(t, None)
            t = nb
    if any(len(n) == 0 for n in net.adj.values()):
        return None

    # Valves: one on each tank's / red cell's own stretch, one on each loop,
    # then random extra spots; valves never touch each other.
    inline = [c for c in net.inline(roots) if c in net.adj]
    lo, hi = p["valves"]
    want = rng.randint(lo, hi)
    valves: List[Pos] = []

    def free(c: Pos) -> bool:
        return c not in valves and far(c, valves + pumps + targets + reds)

    # Must-have valves: every red cell behind one on its private stretch
    # (no tank uses it), every loop and every tank gated.  No room = rebuild.
    tank_paths = {c for t in targets for c in net.path_to_root(t)}
    seeds: List[List[Pos]] = []
    for r in reds:
        seeds.append([c for c in net.path_to_root(r)[1:]
                      if c in inline and c not in tank_paths])
    for br in net.bridges:
        seeds.append([c for c in br if c in inline])
    for t in targets:
        seeds.append([c for c in net.path_to_root(t)[1:] if c in inline])
    for opts in seeds:
        if any(c in valves for c in opts):
            continue
        opts = [c for c in opts if free(c)]
        if not opts:
            return None
        valves.append(rng.choice(opts))
    pool = [c for c in inline if free(c)]
    rng.shuffle(pool)
    for c in pool:
        if len(valves) >= want:
            break
        if free(c):
            valves.append(c)
    if not (lo <= len(valves) <= hi):
        return None

    cells: Dict[Pos, Cell] = {}
    for c in net.adj:
        if c in roots:
            cells[c] = Cell(Vec3(*c), CellType.SOURCE, COLORS[net.tree[c]])
        elif c in targets:
            cells[c] = Cell(Vec3(*c), CellType.TARGET, COLORS[net.tree[c]])
        elif c in reds:
            cells[c] = Cell(Vec3(*c), CellType.PROTECTED)
        elif c in valves:
            cells[c] = Cell(Vec3(*c), CellType.VALVE)
        else:
            cells[c] = Cell(Vec3(*c), CellType.PIPE)
    return LevelSpec(grid_size=p["grid"], cells=cells)


def spaced(spec: LevelSpec) -> bool:
    """Pumps, tanks, valves and red cells are drawn much fatter than a pipe;
    keep at least one empty cell between any two of them (also diagonally)
    so none hides or seems to touch another."""
    big = list(spec.sources) + list(spec.targets) + list(spec.valves) + list(spec.protected)
    for i, a in enumerate(big):
        for b in big[i + 1:]:
            if max(abs(x - y) for x, y in zip(a, b)) < 2:
                return False
    return True


def generate(concept: str, tier: str, seed: int,
             max_attempts: int = 400) -> Optional[Tuple[LevelSpec, Dict]]:
    p = CONCEPTS[concept][tier]
    rng = random.Random(seed)
    for attempt in range(max_attempts):
        spec = _build(p, rng)
        if spec is None or not spaced(spec):
            continue
        m = analyze(spec)
        if not accepts(m, p, tier):
            continue
        meta = {
            "generator": "concepts-v1",
            "concept": concept,
            "tier": tier,
            "insight": INSIGHT[concept],
            "seed": seed,
            "attempt": attempt,
            "difficulty": m,
        }
        return spec, meta
    return None


if __name__ == "__main__":
    import sys
    import time
    concept = sys.argv[1] if len(sys.argv) > 1 else "backdoor"
    for tier in ("easy", "medium", "hard"):
        t0 = time.time()
        r = None
        for s in range(20):
            r = generate(concept, tier, s, max_attempts=200)
            if r:
                break
        print(concept, tier, "seed", s if r else None, round(time.time() - t0, 1), "s",
              r[1]["difficulty"] if r else "FAILED")
