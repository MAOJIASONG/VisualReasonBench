"""
FluidNet 3D - Procedural Level Generator

V3 (current): valve tree -- see generate_level() below for the design and
the measurements that motivated it.  The notes that follow describe the V2
snake generator, whose helpers (_backbone_path, _add_side_branches) are kept
for reference.

V2 (backbone-snake)

Design goals (driven by user feedback):
 1. The pipe network should look like long pipes traversing the grid,
    not a compact blob.
 2. Cells should distribute uniformly across X/Y/Z; no axis underused.
 3. Every PROTECTED cell must be face-adjacent to a VALVE (otherwise the
    red zone is irrelevant — fluid could never reach it).
 4. The generator must produce solvable, non-trivial puzzles where:
       * the empty-set is not a solution    (>=1 valve must be opened)
       * opening every valve is unsafe     (forces selective reasoning)
       * at least one valve is dangerous   (a real trap exists)

Algorithm
---------
A. Build a long random DFS-backbone path that snakes through the grid.
   We bias toward cells far from the path centroid so coverage is uniform.
B. Optionally add 1-2 short side branches off the backbone to give
   junction structure (so flow has choices).
C. Place SOURCE at one path endpoint, TARGETs near the other end /
   on side branches.  Place VALVEs at "choke" cells along the backbone.
D. Sprout PROTECTED cells as single-cell side stubs adjacent to valves.
E. Brute-force verify safety/solvability/non-triviality.
"""

import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Set

from .game_core import CellType, Vec3, Cell, LevelSpec, face_neighbors
from .loader import create_game_state
from .propagation import _compute_full_fluid


COLOR_PALETTE = ["blue", "red", "green", "orange", "purple", "cyan"]


# ---------------------------------------------------------------------------
# Backbone path builder
# ---------------------------------------------------------------------------

def _backbone_path(grid_size: Tuple[int, int, int],
                   start: Tuple[int, int, int],
                   max_length: int,
                   rng: random.Random,
                   spread_bias: float = 0.85) -> List[Tuple[int, int, int]]:
    """
    Build a long simple path via DFS-with-backtracking.  At every step we
    prefer extending toward cells FAR from the current path centroid so
    the path snakes outward and covers the grid uniformly.

    Returns a list of cells (the path), or fewer if we cannot reach
    `max_length` from this start.
    """
    visited: Set[Tuple[int, int, int]] = {start}
    path: List[Tuple[int, int, int]] = [start]
    while len(path) < max_length:
        current = path[-1]
        candidates = [n for n in face_neighbors(current, grid_size)
                      if n not in visited]
        if not candidates:
            # Dead end — backtrack.
            if len(path) <= 1:
                break
            path.pop()
            continue
        # Spread mode: sort candidates by distance-from-path-centroid (descending).
        if rng.random() < spread_bias:
            cx = sum(p[0] for p in path) / len(path)
            cy = sum(p[1] for p in path) / len(path)
            cz = sum(p[2] for p in path) / len(path)
            candidates.sort(
                key=lambda p: -(((p[0] - cx) ** 2 +
                                  (p[1] - cy) ** 2 +
                                  (p[2] - cz) ** 2)
                                 + rng.random() * 0.3)
            )
            nxt = candidates[0]
        else:
            nxt = rng.choice(candidates)
        path.append(nxt)
        visited.add(nxt)
    return path


def _add_side_branches(path: List[Tuple[int, int, int]],
                       grid_size: Tuple[int, int, int],
                       n_branches: int,
                       rng: random.Random,
                       max_branch_len: int = 3) -> Set[Tuple[int, int, int]]:
    """Sprout `n_branches` short side branches off the backbone."""
    on_path = set(path)
    extra: Set[Tuple[int, int, int]] = set()
    attempts = 0
    while len(extra) < n_branches and attempts < n_branches * 10:
        attempts += 1
        anchor = rng.choice(path[2:-2]) if len(path) > 4 else rng.choice(path)
        # Walk a short branch.
        cur = anchor
        branch_len = rng.randint(1, max_branch_len)
        for _ in range(branch_len):
            free = [n for n in face_neighbors(cur, grid_size)
                    if n not in on_path and n not in extra]
            if not free:
                break
            cur = rng.choice(free)
            extra.add(cur)
    return extra


# ---------------------------------------------------------------------------
# Spread / sanity helpers
# ---------------------------------------------------------------------------

def _spread_ok(cells_iter, grid_size: Tuple[int, int, int],
               min_ratio: float = 0.75) -> bool:
    pts = list(cells_iter)
    if not pts:
        return False
    A, B, C = grid_size
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    span_x = max(xs) - min(xs) + 1
    span_y = max(ys) - min(ys) + 1
    span_z = max(zs) - min(zs) + 1
    return (span_x >= int(A * min_ratio) and
            span_y >= int(B * min_ratio) and
            span_z >= int(C * min_ratio))


def _evenly_pick_indices(n_total: int, n_picks: int,
                         skip_endpoints: int = 1,
                         rng: Optional[random.Random] = None) -> List[int]:
    """Return n_picks evenly-spaced indices in [skip_endpoints, n_total - 1 - skip_endpoints]."""
    lo, hi = skip_endpoints, n_total - 1 - skip_endpoints
    if lo > hi or n_picks <= 0:
        return []
    span = hi - lo
    if n_picks == 1:
        return [lo + span // 2]
    step = span / (n_picks - 1)
    picks = [int(round(lo + i * step)) for i in range(n_picks)]
    # Tiny perturbation for variety.
    if rng is not None:
        picks = [max(lo, min(hi, p + rng.randint(-1, 1))) for p in picks]
    seen, out = set(), []
    for p in picks:
        while p in seen and p < hi:
            p += 1
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Solver helpers (used for difficulty verification)
# ---------------------------------------------------------------------------

def _violates(spec: LevelSpec, opened: Set[Tuple[int, int, int]]) -> bool:
    from .game_core import GameState
    state = GameState(spec=spec, opened_valves=set(opened))
    fluid = _compute_full_fluid(state)
    for p in spec.protected:
        if p in fluid:
            return True
    for t in spec.targets:
        col = spec.cells[t].color
        got = fluid.get(t)
        if got is not None and got != col:
            return True
    return False


def _targets_filled(spec: LevelSpec, opened: Set[Tuple[int, int, int]]) -> bool:
    from .game_core import GameState
    state = GameState(spec=spec, opened_valves=set(opened))
    fluid = _compute_full_fluid(state)
    for t in spec.targets:
        if fluid.get(t) != spec.cells[t].color:
            return False
    return True


def _find_safe_subsets(spec: LevelSpec) -> List[Set[Tuple[int, int, int]]]:
    valves = spec.valves
    safe: List[Set[Tuple[int, int, int]]] = []
    for k in range(len(valves) + 1):
        for combo in combinations(valves, k):
            s = set(combo)
            if _violates(spec, s):
                continue
            if _targets_filled(spec, s):
                safe.append(s)
    return safe


def _dangerous_valves(spec: LevelSpec,
                      safe: List[Set[Tuple[int, int, int]]]) -> List[Tuple[int, int, int]]:
    appear = set().union(*safe) if safe else set()
    return [v for v in spec.valves if v not in appear]


# ---------------------------------------------------------------------------
# Main generator (V3: valve tree)
# ---------------------------------------------------------------------------
#
# Lessons from measuring the V2 snake generator over 880 levels:
#   * face-adjacency connects ANY two touching cells, so a folded snake had
#     ~5 hidden shortcut edges: targets were often full before the first move
#     and most of the drawn network was irrelevant;
#   * the obvious local rule "open a wet valve that is not next to a red
#     cell" won 86-98% of levels, so tiers did not measure reasoning.
#
# V3 grows the network as a TREE under a "no touching" rule: a new cell may
# touch only its parent.  The drawn pipes are then exactly the graph.
# Targets and protected cells are branch tips; every target path is gated by
# valves (so nothing is full at the start) and every protected tip hangs
# behind its own "trap" valve, placed far from the red cell so that the
# danger is only visible by tracing the branch.  Tiers are assigned from
# exact measurements (difficulty.py), not from the input parameters.

def _no_touch(c: Tuple[int, int, int], parent: Tuple[int, int, int],
              occupied: Set[Tuple[int, int, int]],
              grid_size: Tuple[int, int, int]) -> bool:
    if c in occupied:
        return False
    return all(nb == parent or nb not in occupied
               for nb in face_neighbors(c, grid_size))


def _grow_branch(start: Tuple[int, int, int], length: int,
                 occupied: Set[Tuple[int, int, int]],
                 parent: Dict[Tuple[int, int, int], Tuple[int, int, int]],
                 grid_size: Tuple[int, int, int], rng: random.Random,
                 straight_bias: float = 0.6) -> List[Tuple[int, int, int]]:
    """Grow up to `length` cells from `start`; straight runs read as pipes."""
    cells: List[Tuple[int, int, int]] = []
    cur, prev = start, parent.get(start)
    for _ in range(length):
        opts = [c for c in face_neighbors(cur, grid_size)
                if _no_touch(c, cur, occupied, grid_size)]
        if not opts:
            break
        straight = None
        if prev is not None:
            d = tuple(a - b for a, b in zip(cur, prev))
            cand = tuple(a + b for a, b in zip(cur, d))
            straight = cand if cand in opts else None
        if straight is not None and rng.random() < straight_bias:
            nxt = straight
        else:
            nxt = rng.choice(opts)
        occupied.add(nxt)
        parent[nxt] = cur
        cells.append(nxt)
        prev, cur = cur, nxt
    return cells


def _path_to(cell, parent):
    out = [cell]
    while out[-1] in parent:
        out.append(parent[out[-1]])
    return out[::-1]                      # source ... cell


def generate_level(grid_size: Tuple[int, int, int] = (5, 5, 5),
                   n_targets: int = 2,
                   n_valves: int = 5,
                   n_protected: int = 2,
                   seed: int = 0,
                   max_attempts: int = 200,
                   tier: Optional[str] = None) -> Optional[Tuple[LevelSpec, dict]]:
    """Generate one level.  With `tier` set, only levels whose measured
    difficulty falls in that band (difficulty.TIERS) are accepted."""
    from .difficulty import analyze, fits_tier

    rng = random.Random(seed)
    A, B, C = grid_size
    n_leaves = n_targets + n_protected

    for attempt in range(max_attempts):
        # 1) Source on the floor, away from the edges when possible.
        source = (rng.randint(1, A), rng.randint(1, B), 1)
        occupied: Set[Tuple[int, int, int]] = {source}
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        children: Dict[Tuple[int, int, int], int] = {}

        # 2) Trunk, then side branches from interior cells until we have
        #    enough tips.  Branch length 3-6 keeps each run readable.
        trunk = _grow_branch(source, rng.randint(2, 4), occupied, parent,
                             grid_size, rng)
        if len(trunk) < 2:
            continue
        net = [source] + trunk
        for c in trunk:
            children[parent[c]] = children.get(parent[c], 0) + 1
        guard = 0
        while guard < 60:
            guard += 1
            tips = [c for c in net if c != source and children.get(c, 0) == 0]
            if len(tips) >= n_leaves:
                break
            # Junction: a non-source cell with at most one child so far.
            cands = [c for c in net[1:] if children.get(c, 0) <= 1]
            if not cands:
                break
            j = rng.choice(cands)
            br = _grow_branch(j, rng.randint(3, 6), occupied, parent,
                              grid_size, rng)
            if len(br) < 2:
                for c in br:              # too short to read: undo
                    occupied.discard(c); parent.pop(c, None)
                continue
            for c in br:
                children[parent[c]] = children.get(parent[c], 0) + 1
            net += br
        tips = [c for c in net if c != source and children.get(c, 0) == 0]
        if len(tips) != n_leaves:
            continue
        if not _spread_ok(net, grid_size, min_ratio=0.6):
            continue

        # 3) Roles: deepest tips become targets, the rest protected.
        depth = {c: len(_path_to(c, parent)) - 1 for c in net}
        tips.sort(key=lambda c: -depth[c] + rng.random() * 2)
        targets = tips[:n_targets]
        protected = tips[n_targets:]
        target_path = set()
        for t in targets:
            target_path |= set(_path_to(t, parent)[1:-1])
        inline = {c for c in net[1:] if children.get(c, 0) == 1}

        # 4) Trap valves: one per protected tip, on the part of its path that
        #    no target uses, as far from the red cell as possible.
        traps: List[Tuple[int, int, int]] = []
        ok = True
        for p in protected:
            private = [c for c in _path_to(p, parent)[1:-1]
                       if c not in target_path and c in inline]
            if not private:
                ok = False
                break
            # far end of the private stretch first (non-local danger)
            pick = private[0] if rng.random() < 0.8 else private[-1]
            if pick not in traps:
                traps.append(pick)
        if not ok:
            continue

        # 5) Required valves on the target paths: at least one per target,
        #    the rest spread along shared stretches.
        req_pool = sorted(target_path & inline, key=lambda c: depth[c])
        required: List[Tuple[int, int, int]] = []
        for t in targets:
            own = [c for c in _path_to(t, parent)[1:-1]
                   if c in inline and c not in traps]
            if not own:
                ok = False
                break
            if not any(c in required for c in own):
                required.append(rng.choice(own[len(own) // 2:]))
        if not ok:
            continue
        n_extra = max(0, n_valves - len(traps) - len(required))
        spare = [c for c in req_pool if c not in required and c not in traps
                 and parent.get(c) not in required]
        rng.shuffle(spare)
        required += spare[:n_extra]
        valves = sorted(set(required) | set(traps))

        # 5b) Valves must not touch each other (two touching valves read as
        #     one block).
        if any(nb in valves for v in valves
               for nb in face_neighbors(v, grid_size)):
            continue

        # 6) Materialise.
        color = COLOR_PALETTE[0]
        cells: Dict[Tuple[int, int, int], Cell] = {}
        for c in net:
            if c == source:
                cells[c] = Cell(pos=Vec3(*c), type=CellType.SOURCE, color=color)
            elif c in targets:
                cells[c] = Cell(pos=Vec3(*c), type=CellType.TARGET, color=color)
            elif c in protected:
                cells[c] = Cell(pos=Vec3(*c), type=CellType.PROTECTED)
            elif c in valves:
                cells[c] = Cell(pos=Vec3(*c), type=CellType.VALVE)
            else:
                cells[c] = Cell(pos=Vec3(*c), type=CellType.PIPE)
        spec = LevelSpec(grid_size=grid_size, cells=cells)

        # 7) Measure; accept by tier.
        m = analyze(spec)
        if m["min_plan"] is None or m["prefilled"] or m["traps"] == 0:
            continue
        if tier is not None and not fits_tier(m, tier):
            continue

        safe = _find_safe_subsets(spec)
        min_safe = min(safe, key=len)
        meta = {
            "seed": seed,
            "attempt": attempt,
            "generator": "valve-tree-v3",
            "tier": tier,
            "network_cells": len(net),
            "min_valves_needed": len(min_safe),
            "n_safe_subsets": len(safe),
            "dangerous_valves": [list(v) for v in _dangerous_valves(spec, safe)],
            "min_safe_example": [list(v) for v in sorted(min_safe)],
            "difficulty": m,
        }
        return spec, meta

    return None


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_and_save(out_path: str, **kwargs) -> Optional[Tuple[LevelSpec, dict]]:
    from .loader import save_level_to_json
    result = generate_level(**kwargs)
    if result is None:
        return None
    spec, meta = result
    save_level_to_json(spec, out_path, meta=meta)
    return spec, meta


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_level.json"
    r = generate_and_save(
        out, grid_size=(5, 5, 5), n_targets=2, n_valves=5,
        n_protected=2, seed=42,
    )
    if r is None:
        print("generation failed")
    else:
        spec, meta = r
        print(f"saved to {out}")
        print(f"meta: {meta}")
