"""Shared level configurations and de-duplication for the batch scripts."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

from .generator import generate_level

# Building blocks per category.  Tiers are enforced by measurement against
# one ladder for the whole game (difficulty.TIERS): every easy level has fewer
# pieces, fewer decoy holes and a lower score than every medium level, and
# every medium than every hard.  Every hard level is a three-level splitter
# tree on steps (9-10 patches); the category sets the emphasis (more decoys,
# more raised patches, a second goal per fork).
CATEGORY: Dict[str, Dict[str, Dict]] = {
    # straight chain; easy = fill the hole(s)
    "A_line": {
        "easy":   dict(grid_size=(6, 5, 3), n_patches=1, main_len=(3, 5)),
        "medium": dict(grid_size=(8, 5, 3), n_patches=3, main_len=(5, 6), n_decoys=2),
        "hard":   dict(grid_size=(12, 12, 5), n_patches=9, main_len=(3, 5), n_decoys=3, use_split=True, split_depth=3, use_height=True, min_raised=4, p_step=0.5),
    },
    # chain with decoy lines (gap -> dominoes -> red piece) around it
    "B_decoy": {
        "easy":   dict(grid_size=(5, 5, 3), n_patches=2, main_len=(3, 5), n_decoys=1),
        "medium": dict(grid_size=(7, 6, 3), n_patches=3, main_len=(4, 6), n_decoys=2),
        "hard":   dict(grid_size=(12, 12, 5), n_patches=9, main_len=(3, 5), n_decoys=4, use_split=True, split_depth=3, use_height=True, min_raised=4, p_step=0.5),
    },
    # height changes on STEP blocks
    "C_steps": {
        "easy":   dict(grid_size=(5, 5, 3), n_patches=1, main_len=(3, 5), use_height=True, min_raised=1),
        "medium": dict(grid_size=(7, 5, 3), n_patches=3, main_len=(4, 6), n_decoys=2, use_height=True, min_raised=1),
        "hard":   dict(grid_size=(12, 12, 5), n_patches=9, main_len=(3, 5), n_decoys=3, use_split=True, split_depth=3, use_height=True, min_raised=5, p_step=0.7),
    },
    # a splitter sends the wave both ways
    "D_split": {
        "easy":   dict(grid_size=(6, 6, 3), n_patches=2, main_len=(2, 3), use_split=True, two_goals=True),
        "medium": dict(grid_size=(7, 6, 3), n_patches=3, main_len=(3, 4), n_decoys=1, use_split=True),
        "hard":   dict(grid_size=(12, 12, 5), n_patches=10, main_len=(3, 5), n_decoys=3, use_split=True, split_depth=3, two_goals=True, use_height=True, min_raised=4, p_step=0.5),
    },
    # everything together
    "E_full": {
        "easy":   dict(grid_size=(6, 6, 3), n_patches=1, main_len=(2, 4), use_height=True, min_raised=1, use_split=True),
        "medium": dict(grid_size=(7, 6, 3), n_patches=3, main_len=(3, 4), n_decoys=1, use_height=True, min_raised=1, use_split=True),
        "hard":   dict(grid_size=(12, 12, 5), n_patches=10, main_len=(3, 5), n_decoys=3, use_split=True, split_depth=3, two_goals=True, use_height=True, min_raised=5, p_step=0.6),
    },
}

# What each category asks the player to see (written into every level).
INSIGHT: Dict[str, str] = {
    "A_line": "Find the holes in the line and fill each one; push START toward the goal.",
    "B_decoy": "Not every hole is yours: some lead to a red piece and must stay empty.",
    "C_steps": "A piece on a step needs the step under it, and a falling piece only reaches one level up or down.",
    "D_split": "The splitter sends the wave out to both sides: plan both branches.",
    "E_full": "Holes, decoys, steps and a splitter together: follow the wave all the way.",
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


def make_level(slot: Tuple[str, str, str], seed: int):
    """chain_search entry point: slot = (label, category, tier)."""
    _, cat, tier = slot
    return generate_level(seed=seed, tier=tier, max_attempts=200, **CATEGORY[cat][tier])
