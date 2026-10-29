"""Shared level configurations and de-duplication for the batch scripts."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

from .generator import generate_level

# Building blocks per category.  Tiers are enforced by measurement against
# one ladder for the whole game (difficulty.TIERS): pieces to place and
# difficulty.score, with a gap between the score bands, so every easy level
# is below every medium one and every medium below every hard one.
CATEGORY: Dict[str, Dict[str, Dict]] = {
    "A_route": {        # one entrance, one exit, open box
        "easy":   dict(grid_size=(5, 4, 4), n_exits=1, n_obstacles=0, extra_len=(0, 2)),
        "medium": dict(grid_size=(5, 5, 4), n_exits=1, n_obstacles=1, extra_len=(3, 5)),
        "hard":   dict(grid_size=(6, 6, 4), n_exits=1, n_obstacles=2, extra_len=(6, 9), min_supports=1),
    },
    "B_obstacles": {    # obstacles block the obvious staircase
        "easy":   dict(grid_size=(5, 4, 4), n_exits=1, n_obstacles=1, extra_len=(1, 3)),
        "medium": dict(grid_size=(5, 5, 4), n_exits=1, n_obstacles=3, extra_len=(3, 5)),
        "hard":   dict(grid_size=(6, 6, 4), n_exits=1, n_obstacles=3, extra_len=(4, 7)),
    },
    "C_branch": {       # two exits: the network needs a T junction
        "easy":   dict(grid_size=(5, 5, 4), n_exits=2, n_obstacles=0, extra_len=(0, 1)),
        "medium": dict(grid_size=(5, 5, 4), n_exits=2, n_obstacles=1, extra_len=(1, 2)),
        "hard":   dict(grid_size=(6, 5, 4), n_exits=2, n_obstacles=2, extra_len=(1, 3)),
    },
    "D_tall": {         # high exits: support columns dominate
        "easy":   dict(grid_size=(4, 4, 5), n_exits=1, n_obstacles=0, extra_len=(0, 2), min_supports=1),
        "medium": dict(grid_size=(5, 4, 5), n_exits=1, n_obstacles=2, extra_len=(2, 4), min_supports=2),
        "hard":   dict(grid_size=(5, 5, 5), n_exits=1, n_obstacles=3, extra_len=(3, 6), min_supports=3),
    },
    "E_mixed": {        # two exits, tall box, obstacles
        "easy":   dict(grid_size=(5, 5, 4), n_exits=2, n_obstacles=1, extra_len=(0, 1)),
        "medium": dict(grid_size=(5, 5, 5), n_exits=2, n_obstacles=2, extra_len=(1, 2), min_supports=1),
        "hard":   dict(grid_size=(6, 5, 5), n_exits=2, n_obstacles=3, extra_len=(1, 3), min_supports=2),
    },
}

# What each category asks the player to see (written into every level).
INSIGHT: Dict[str, str] = {
    "A_route": "Count the parts: the bin holds exactly one route's worth of corners and straights (plus one spare).",
    "B_obstacles": "The obvious step-and-climb route is blocked; find the detour your parts allow.",
    "C_branch": "Two exits: one network with a T junction feeding both.",
    "D_tall": "A high exit: every raised pipe needs a support column under it.",
    "E_mixed": "Two exits, obstacles and supports together.",
}


def layout_key(spec) -> Tuple:
    """Portals + obstacles, normalised over the 8 board symmetries."""
    A, B, _ = spec.grid_size
    pts = ([("E", p) for p in spec.entrances] + [("X", p) for p in spec.exits]
           + [("O", p) for p in spec.obstacles])
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for swap in (False, True):
                t = []
                for name, (x, y, z) in pts:
                    a = x if sx == 1 else A + 1 - x
                    b = y if sy == 1 else B + 1 - y
                    t.append((name, b, a, z) if swap else (name, a, b, z))
                key = tuple(sorted(t))
                best = key if best is None or key < best else best
    return best


def find_level(kwargs: Dict, tier: str, seeds: Iterable[int],
               seen: Optional[Set[Tuple]] = None):
    """First (seed, spec, meta) in `tier` whose layout is not in `seen`."""
    for s in seeds:
        r = generate_level(seed=s, tier=tier, max_attempts=150, **kwargs)
        if r is None:
            continue
        key = layout_key(r[0])
        if seen is not None:
            if key in seen:
                continue
            seen.add(key)
        return (s,) + r
    return None


def make_level(slot: Tuple[str, str, str], seed: int):
    """chain_search entry point: slot = (label, category, tier)."""
    _, cat, tier = slot
    return generate_level(seed=seed, tier=tier, max_attempts=150, **CATEGORY[cat][tier])
