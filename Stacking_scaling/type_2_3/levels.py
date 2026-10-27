"""Shared level configurations and de-duplication for the batch scripts."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

from .generator import generate_level

# Building blocks per category; tiers are enforced by measurement
# (difficulty.TIERS).
CATEGORY: Dict[str, Dict[str, Dict]] = {
    "A_route": {        # one entrance, one exit, open box
        "easy":   dict(grid_size=(5, 4, 4), n_exits=1, n_obstacles=0, extra_len=(0, 2)),
        "medium": dict(grid_size=(5, 5, 4), n_exits=1, n_obstacles=1, extra_len=(2, 4)),
        "hard":   dict(grid_size=(6, 5, 4), n_exits=1, n_obstacles=2, extra_len=(4, 7), min_supports=1),
    },
    "B_obstacles": {    # obstacles block the obvious staircase
        "easy":   dict(grid_size=(5, 4, 4), n_exits=1, n_obstacles=1, extra_len=(1, 3)),
        "medium": dict(grid_size=(5, 5, 4), n_exits=1, n_obstacles=3, extra_len=(2, 4)),
        "hard":   dict(grid_size=(6, 5, 4), n_exits=1, n_obstacles=4, extra_len=(4, 7)),
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
        "easy":   dict(grid_size=(5, 5, 5), n_exits=2, n_obstacles=1, extra_len=(0, 1)),
        "medium": dict(grid_size=(5, 5, 5), n_exits=2, n_obstacles=2, extra_len=(1, 2), min_supports=1),
        "hard":   dict(grid_size=(6, 5, 5), n_exits=2, n_obstacles=3, extra_len=(1, 3), min_supports=2),
    },
}


def layout_key(spec) -> Tuple:
    return (tuple(spec.grid_size), tuple(sorted(spec.entrances)),
            tuple(sorted(spec.exits)), tuple(sorted(spec.obstacles)))


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
