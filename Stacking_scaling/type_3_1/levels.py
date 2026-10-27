"""Shared level configurations and de-duplication for the batch scripts."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

from .generator import generate_level

# Building blocks per category.  Tiers are enforced by measurement
# (difficulty.TIERS), these only shape what the generator builds.
CATEGORY: Dict[str, Dict[str, Dict]] = {
    # straight chain; easy = fill the hole(s)
    "A_line": {
        "easy":   dict(grid_size=(6, 5, 3), n_patches=1, main_len=(3, 5)),
        "medium": dict(grid_size=(7, 5, 3), n_patches=2, main_len=(4, 5), n_decoys=2),
        "hard":   dict(grid_size=(8, 6, 3), n_patches=3, main_len=(5, 6), n_decoys=3),
    },
    # chain with decoy lines (gap -> dominoes -> red piece) around it
    "B_decoy": {
        "easy":   dict(grid_size=(5, 5, 3), n_patches=2, main_len=(3, 5), n_decoys=1),
        "medium": dict(grid_size=(7, 6, 3), n_patches=2, main_len=(4, 5), n_decoys=2, n_protected=1),
        "hard":   dict(grid_size=(8, 7, 3), n_patches=3, main_len=(4, 6), n_decoys=3, n_protected=1),
    },
    # height changes on STEP blocks
    "C_steps": {
        "easy":   dict(grid_size=(5, 5, 3), n_patches=2, main_len=(3, 5), n_decoys=1, use_height=True),
        "medium": dict(grid_size=(7, 5, 3), n_patches=2, main_len=(4, 5), n_decoys=2, use_height=True),
        "hard":   dict(grid_size=(8, 6, 4), n_patches=3, main_len=(5, 6), n_decoys=2, use_height=True),
    },
    # a splitter sends the wave both ways
    "D_split": {
        "easy":   dict(grid_size=(6, 6, 3), n_patches=2, main_len=(2, 3), use_split=True, two_goals=True),
        "medium": dict(grid_size=(7, 6, 3), n_patches=3, main_len=(3, 4), n_decoys=1, use_split=True),
        "hard":   dict(grid_size=(8, 7, 3), n_patches=3, main_len=(3, 5), n_decoys=2, use_split=True),
    },
    # everything together
    "E_full": {
        "easy":   dict(grid_size=(6, 6, 3), n_patches=2, main_len=(2, 4), n_decoys=1, use_height=True, use_split=True),
        "medium": dict(grid_size=(7, 7, 4), n_patches=3, main_len=(3, 5), n_decoys=2, use_height=True, use_split=True),
        "hard":   dict(grid_size=(8, 8, 4), n_patches=4, main_len=(4, 5), n_decoys=2, use_height=True, use_split=True, two_goals=True),
    },
}


def layout_key(spec) -> Tuple:
    """Fixed pieces + solution, normalised over the 8 board symmetries and
    translation, so mirrored / rotated / shifted copies compare equal."""
    pts = [(pt.value, pos) for pt, pos in spec.fixed_pieces]
    pts += [("_", tuple(p)) for p in (spec.meta or {}).get("solution_patches", [])]
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for swap in (False, True):
                t = []
                for name, (x, y, z) in pts:
                    a, b = (y, x) if swap else (x, y)
                    t.append((name, sx * a, sy * b, z))
                mx = min(p[1] for p in t)
                my = min(p[2] for p in t)
                key = tuple(sorted((n, a - mx, b - my, z) for n, a, b, z in t))
                best = key if best is None or key < best else best
    return best


def find_level(kwargs: Dict, tier: str, seeds: Iterable[int],
               seen: Optional[Set[Tuple]] = None):
    """First (seed, spec, meta) in `tier` whose layout is not in `seen`."""
    for s in seeds:
        r = generate_level(seed=s, tier=tier, max_attempts=200, **kwargs)
        if r is None:
            continue
        key = layout_key(r[0])
        if seen is not None:
            if key in seen:
                continue
            seen.add(key)
        return (s,) + r
    return None
