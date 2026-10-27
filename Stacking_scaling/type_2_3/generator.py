"""
PipeFit 3D — level generator (rules v2).

  1. Portals on the side walls (never on the floor / ceiling, never in a
     wall's corner column): an entrance low, the exit(s) higher up on other
     walls.
  2. Pipe route: a biased random walk from the entrance's inner cell to the
     exit's (not always the "step, climb, step, climb" staircase).  With a
     third portal, a branch leaves the route at a T junction.
  3. Supports: every pipe needs something solid below (floor, support or a
     supported pipe); missing columns are filled bottom-up and packed into
     S2 / S1 pieces.
  4. Obstacles go on the staircase shortcut (so it fails) or, failing that,
     next to the route.
  5. Inventory = exactly the pieces of that network, plus `spare` extras.
  6. The exhaustive search (difficulty.solutions) must find the network,
     and the measured difficulty must match the requested tier.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from .difficulty import (
    Board, analyze, fits, fits_tier, network_dirs, plan_of, staircase,
)
from .game_core import DIRS, LevelSpec, PieceType, Vec3, portal_inward


def _nb(c: Vec3, d: int) -> Vec3:
    v = DIRS[d][1]
    return (c[0] + v[0], c[1] + v[1], c[2] + v[2])


def _wall_cells(grid: Vec3, wall: Tuple[str, int], zs: List[int]) -> List[Vec3]:
    A, B, C = grid
    axis, val = wall
    if axis == "x":
        return [(val, y, z) for y in range(2, B) for z in zs]
    return [(x, val, z) for x in range(2, A) for z in zs]


def _walk(bd: Board, s: Vec3, t: Vec3, max_len: int, rng: random.Random,
          avoid: Set[Vec3]) -> Optional[List[Vec3]]:
    """Biased random walk s -> t; no cell revisited, none in `avoid`."""
    for _ in range(40):
        path, seen = [s], {s}
        cur = s
        while cur != t and len(path) < max_len:
            opts = [_nb(cur, d) for d in range(6)]
            opts = [n for n in opts if bd.free(n) and n not in seen and n not in avoid
                    and sum(abs(a - b) for a, b in zip(n, t)) <= max_len - len(path) - 1]
            if not opts:
                break
            closer = [n for n in opts if sum(abs(a - b) for a, b in zip(n, t))
                      < sum(abs(a - b) for a, b in zip(cur, t))]
            cur = rng.choice(closer) if closer and rng.random() < 0.65 else rng.choice(opts)
            path.append(cur)
            seen.add(cur)
        if cur == t:
            return path
    return None


def generate_level(grid_size: Vec3 = (5, 4, 4),
                   seed: int = 0,
                   n_exits: int = 1,
                   n_obstacles: int = 1,
                   extra_len: Tuple[int, int] = (1, 4),
                   spare: int = 1,
                   min_supports: int = 0,
                   tier: Optional[str] = None,
                   max_attempts: int = 300,
                   **_legacy) -> Optional[Tuple[LevelSpec, dict]]:
    rng = random.Random(seed)
    A, B, C = grid_size
    walls = [("x", 1), ("x", A), ("y", 1), ("y", B)]
    for attempt in range(max_attempts):
        ws = rng.sample(walls, 1 + n_exits)
        ent = rng.choice(_wall_cells(grid_size, ws[0], [1]))
        exits = [rng.choice(_wall_cells(grid_size, w, list(range(2, C + 1))))
                 for w in ws[1:]]
        spec0 = LevelSpec(grid_size=grid_size, entrances=[ent], exits=exits)
        bd = Board(spec0)
        s, t = bd.inner(ent), bd.inner(exits[0])
        if len({s, t} | {bd.inner(x) for x in exits}) != 1 + n_exits:
            continue
        man = sum(abs(a - b) for a, b in zip(s, t))
        route = _walk(bd, s, t, man + 1 + rng.randint(*extra_len), rng, set())
        if route is None:
            continue
        paths = [(ent, route, exits[0])]
        if n_exits == 2:
            t2 = bd.inner(exits[1])
            ok = False
            for i in rng.sample(range(1, len(route) - 1), len(route) - 2):
                j = route[i]
                for d in rng.sample(range(6), 6):
                    st = _nb(j, d)
                    if not bd.free(st) or st in route:
                        continue
                    man2 = sum(abs(a - b) for a, b in zip(st, t2))
                    br = _walk(bd, st, t2, man2 + 1 + rng.randint(0, 2), rng, set(route))
                    if br is not None:
                        paths.append((None, [j] + br, exits[1]))
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue

        # Obstacles: block the staircase shortcut first.
        obstacles: List[Vec3] = []
        net_cells = {c for _, cells, _ in paths for c in cells}
        for _ in range(n_obstacles):
            bd_o = Board(LevelSpec(grid_size, [ent], exits, obstacles=list(obstacles)))
            stair = staircase(bd_o, s, t)
            cand = [c for c in (stair or []) if c not in net_cells and c not in (s, t)]
            if not cand:
                cand = [_nb(c, d) for c in route for d in range(4)]
                cand = [c for c in cand if bd_o.free(c) and c not in net_cells
                        and c not in obstacles]
            if not cand:
                break
            obstacles.append(rng.choice(cand))

        bd = Board(LevelSpec(grid_size, [ent], exits, obstacles=obstacles))
        dirs = network_dirs(bd, paths)
        bd.spec.inventory = {pt: 99 for pt in PieceType}
        sol = fits(bd, dirs)
        if sol is None:
            continue                       # T junction / supports impossible
        plan = plan_of(bd, sol)
        if sum(len(c) for c in sol["cols"]) < min_supports:
            continue                       # category wants real support work
        inv: Dict[PieceType, int] = {}
        for p in plan:
            inv[p.type] = inv.get(p.type, 0) + 1
        for _ in range(spare):
            pt = rng.choice([PieceType.PIPE_I, PieceType.PIPE_L, PieceType.SUPPORT_S1])
            inv[pt] = inv.get(pt, 0) + 1
        spec = LevelSpec(grid_size=grid_size, entrances=[ent], exits=exits,
                         obstacles=obstacles,
                         obstacle_shapes=[[o] for o in obstacles],
                         inventory=inv)
        m = analyze(spec)
        m["plan_length"] = len(plan)       # pieces to place (tier size cap)
        if m["n_solutions"] == 0:
            continue
        if tier is not None and not fits_tier(m, tier):
            continue
        meta = {
            "rules": "v2",
            "seed": seed,
            "attempt": attempt,
            "tier": tier,
            "plan": [[p.type.value, list(p.pos), p.orientation] for p in plan],
            "plan_length": len(plan),
            "n_pipes": sum(1 for p in plan if p.type.value in ("I", "L", "T")),
            "n_supports": sum(1 for p in plan if p.type.value in ("S1", "S2")),
            "difficulty": m,
        }
        spec.meta = dict(meta)
        return spec, meta
    return None


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    r = generate_level(seed=seed)
    if r is None:
        raise SystemExit("generation failed")
    spec, meta = r
    print(spec.entrances, spec.exits, spec.obstacles)
    print({pt.value: n for pt, n in spec.inventory.items()})
    print(meta["difficulty"])
