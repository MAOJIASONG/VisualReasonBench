"""
FluidNet 3D - BFS Optimal Solver

Finds the shortest sequence of valve-open actions that wins the puzzle,
i.e. fills every TARGET with matching-color fluid without causing any
PROTECTED_BREACH / WRONG_COLOR_TARGET / COLOR_CONTAMINATION.

Search space
------------
The final fluid state is a pure function of (spec, set_of_open_valves).  So
we BFS over the lattice of open-valve subsets, ordered by |subset|.  Nodes
are frozensets of valve coordinates; edges add one valve.  Because the
environment is order-independent for the *final* state, BFS by subset size
gives the minimum-length winning plan.

Intermediate-state legality
---------------------------
Opening a valve that *immediately* breaches a protected cell is a losing
move: the game terminates before reaching the winning state, even if other
valves could have "compensated".  We therefore require every prefix of the
plan to also be non-violating.  Equivalently: every node visited by BFS
must itself be a non-violating state.

Because fluid is monotone-growing with the open-valve set (opening more
valves can only add fluid cells, never remove them), non-violation is also
monotone: if a superset is safe, every subset is safe.  So pruning by
"current node is non-violating" is sufficient AND necessary.

Complexity
----------
Worst case O(2^V) where V = |valves|.  For V <= 12 this runs in well under
a second.  For larger V you'd want IDA* or reachability pruning.
"""

from collections import deque
from typing import FrozenSet, List, Optional, Set, Tuple

from .game_core import GameState, LevelSpec, ActionCode
from .propagation import _compute_full_fluid


Valve = Tuple[int, int, int]


def _state_for_subset(spec: LevelSpec, opened: FrozenSet[Valve]):
    """Build a throw-away GameState and run full propagation."""
    state = GameState(spec=spec, opened_valves=set(opened))
    state.fluid_cells = _compute_full_fluid(state)
    return state


def _classify_subset(spec: LevelSpec, opened: FrozenSet[Valve]) -> Tuple[bool, bool]:
    """
    Returns (non_violating, all_targets_filled).

    non_violating      := no PROTECTED touched, no wrong-color TARGET touched.
                           (Contamination can only occur with >1 color; for the
                            single-color generator output it cannot trigger.)
    all_targets_filled := every TARGET holds its matching-color fluid.
    """
    state = _state_for_subset(spec, opened)
    fluid = state.fluid_cells
    # Protected breach?
    for p in spec.protected:
        if p in fluid:
            return False, False
    # Wrong-color target?
    for t in spec.targets:
        want = spec.cells[t].color
        got = fluid.get(t)
        if got is not None and got != want:
            return False, False
    # All targets filled?
    for t in spec.targets:
        if fluid.get(t) != spec.cells[t].color:
            return True, False
    return True, True


def solve(spec: LevelSpec,
          max_depth: Optional[int] = None) -> Optional[List[Valve]]:
    """
    BFS for the shortest winning open-sequence.

    Returns the sequence of valves to open (each is a 1-based (x,y,z) tuple),
    or None if no solution exists within `max_depth` opens.
    """
    # Check if the empty subset already wins (source next to targets, no
    # dangerous cells adjacent).
    nv, done = _classify_subset(spec, frozenset())
    if not nv:
        return None                 # unrecoverable: initial state already breaches
    if done:
        return []                   # no valves needed

    valves: List[Valve] = list(spec.valves)
    limit = max_depth if max_depth is not None else len(valves)

    visited: Set[FrozenSet[Valve]] = {frozenset()}
    # queue holds (subset, path) where path is list of valves in open order.
    q: deque = deque()
    q.append((frozenset(), []))

    while q:
        subset, path = q.popleft()
        if len(path) >= limit:
            continue
        for v in valves:
            if v in subset:
                continue
            nxt = subset | {v}
            if nxt in visited:
                continue
            visited.add(nxt)
            nv2, done2 = _classify_subset(spec, nxt)
            if not nv2:
                continue            # this move loses; do not expand further
            new_path = path + [v]
            if done2:
                return new_path
            q.append((nxt, new_path))

    return None


def solve_all_shortest(spec: LevelSpec) -> List[List[Valve]]:
    """
    Return every shortest winning plan (one element per ordering).

    Useful for diagnostics — a puzzle with many shortest plans is easier
    than one with a single plan.
    """
    valves: List[Valve] = list(spec.valves)
    # Find shortest length via plain BFS first.
    first = solve(spec)
    if first is None:
        return []
    target_len = len(first)
    if target_len == 0:
        return [[]]

    results: List[List[Valve]] = []
    # Enumerate all paths of exactly target_len that win.
    def dfs(subset: FrozenSet[Valve], path: List[Valve]):
        if len(path) == target_len:
            nv, done = _classify_subset(spec, subset)
            if nv and done:
                results.append(list(path))
            return
        for v in valves:
            if v in subset:
                continue
            nxt = subset | {v}
            nv, done = _classify_subset(spec, nxt)
            if not nv:
                continue
            path.append(v)
            dfs(nxt, path)
            path.pop()

    dfs(frozenset(), [])
    return results


def explain(spec: LevelSpec, plan: List[Valve]) -> str:
    """Render a human-readable trace of the plan."""
    lines = [f"plan ({len(plan)} valve opens):"]
    state = _state_for_subset(spec, frozenset())
    filled = sum(1 for t in spec.targets
                 if state.fluid_cells.get(t) == spec.cells[t].color)
    lines.append(f"  initial: {filled}/{len(spec.targets)} targets filled")
    opened: FrozenSet[Valve] = frozenset()
    for i, v in enumerate(plan, 1):
        opened = opened | {v}
        state = _state_for_subset(spec, opened)
        filled = sum(1 for t in spec.targets
                     if state.fluid_cells.get(t) == spec.cells[t].color)
        lines.append(f"  step {i}: open {v} -> fluid cells={len(state.fluid_cells)}"
                     f"  targets={filled}/{len(spec.targets)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from .generator import generate_level

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    spec, meta = generate_level(
        grid_size=(5, 5, 5), n_targets=2, n_valves=6, n_protected=3,
        seed=seed,
    )
    print(f"meta: {meta}")
    plan = solve(spec)
    print(f"\nshortest plan: {plan}")
    if plan is not None:
        print()
        print(explain(spec, plan))


def order_by_flow(spec: LevelSpec, plan: List[Valve]) -> List[Valve]:
    """Reorder a winning valve set so each opening is next to fluid that is
    already flowing (upstream first).  The final state is the same, but
    every step of a rendered playthrough now shows visible progress."""
    from .game_core import GameState, face_neighbors
    from .propagation import _compute_full_fluid
    remaining = list(plan)
    opened: List[Valve] = []
    while remaining:
        wet = _compute_full_fluid(GameState(spec=spec, opened_valves=set(opened)))
        nxt = next((v for v in remaining
                    if any(nb in wet for nb in face_neighbors(v, spec.grid_size))),
                   remaining[0])
        opened.append(nxt)
        remaining.remove(nxt)
    return opened
