"""
Domino Chain Reaction 3D — Level generator.

Algorithm (forward-build):
  1. Pick a source cell (ground boundary, random).
  2. Pick an initial direction (±x / ±y).
  3. Walk the cascade by "steps":
       - STRAIGHT step: place a DOMINO at the next cell, continue same dir.
       - RAMP step: place a RAMP; next cell is at z+1 same (x,y).
       - SPLIT step: place a SPLITTER at this cell; branch into 2 orthogonal
                     sub-walks.  Each sub-walk must end at a GOAL.
  4. End every branch at a GOAL (z constant, random length).
  5. Add PROTECTED dominos on cells adjacent to the skeleton (making
     wrong-direction pushes catastrophic).
  6. Carve K patch cells out of the DOMINO / RAMP pieces; those become
     inventory the player must re-place.  Patches are NOT in fixed_pieces.
  7. Validate:
       * With the canonical patches + canonical push sequence → WIN.
       * Without the patches (leave holes) → push fails to topple all goals.
  8. Return (spec, meta).

All steps must respect physical-support rules: every base cell must have
ground or another piece's TOP directly below it.  We pre-check before
committing each step.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from .cascade import simulate_push
from .game_core import (
    DIR_NAME_TO_VEC, GameState, LevelSpec, PieceType, Phase, PlacedPiece, Vec3,
    in_grid, perp_dirs, piece_is_standing, piece_occupied_cells,
)
from .mechanics import (
    step_place, submit_placement, step_push, submit,
)


_DIRS = ["+x", "-x", "+y", "-y"]


def _top_cell_of_anchor(pt: PieceType, pos: Vec3) -> Vec3:
    if piece_is_standing(pt):
        return (pos[0], pos[1], pos[2] + 1)
    return pos


def _occupied_after(cells_used: Set[Vec3], pt: PieceType, pos: Vec3) -> bool:
    return any(c in cells_used for c in piece_occupied_cells(pt, pos))


class _BuildCtx:
    """Mutable scratch state while building the skeleton."""
    def __init__(self, grid_size: Vec3, rng: random.Random):
        self.grid_size = grid_size
        self.rng = rng
        self.cells_used: Set[Vec3] = set()
        self.tops: Set[Vec3] = set()              # top cells (can carry base)
        self.pieces: List[Tuple[PieceType, Vec3]] = []   # in build order
        self.anchors: Set[Vec3] = set()
        # 1-cell blocks added only to hold up the piece after a ramp; they
        # are never carved into patches (that would leave a fixed piece
        # floating in the initial state).
        self.pedestals: Set[Vec3] = set()

    def snapshot(self):
        return (set(self.cells_used), set(self.tops), list(self.pieces),
                set(self.anchors), set(self.pedestals))

    def restore(self, snap) -> None:
        (self.cells_used, self.tops, self.pieces, self.anchors,
         self.pedestals) = (set(snap[0]), set(snap[1]), list(snap[2]),
                            set(snap[3]), set(snap[4]))

    def can_place(self, pt: PieceType, pos: Vec3) -> bool:
        """Check footprint is in-grid + free + has support."""
        cells = piece_occupied_cells(pt, pos)
        for c in cells:
            if not in_grid(c, self.grid_size):
                return False
            if c in self.cells_used:
                return False
        # support
        x, y, z = pos
        if z == 1:
            return True
        below = (x, y, z - 1)
        return below in self.tops

    def place(self, pt: PieceType, pos: Vec3) -> None:
        assert self.can_place(pt, pos), f"{pt}@{pos} can't be placed"
        for c in piece_occupied_cells(pt, pos):
            self.cells_used.add(c)
        self.tops.add(_top_cell_of_anchor(pt, pos))
        self.pieces.append((pt, pos))
        self.anchors.add(pos)


def _next_base(pos: Vec3, dir_name: str, z_after: int) -> Vec3:
    """The next base cell when a pulse leaves `pos` in `dir_name` at height z_after."""
    dx, dy, _ = DIR_NAME_TO_VEC[dir_name]
    return (pos[0] + dx, pos[1] + dy, z_after)


def _walk_branch(ctx: _BuildCtx, start_pos: Vec3, dir_name: str,
                 remaining_len: int, allow_ramp: bool, allow_split: bool,
                 max_depth: int = 0, min_len: int = 1
                 ) -> Optional[List[Tuple[PieceType, Vec3]]]:
    """
    Recursively build a branch that ends at a GOAL.  remaining_len is the
    MAX number of further pieces we can add; min_len = remaining pieces
    still required (so we don't stop too early).

    Each call extends the branch by one piece at the NEXT base cell.
    Returns the sequence of pieces added, or None if dead-ended.
    """
    if remaining_len <= 0:
        return None

    z = start_pos[2]
    next_base = _next_base(start_pos, dir_name, z)

    # Option to end branch here by placing a GOAL, if min_len satisfied
    can_end = min_len <= 1 and ctx.can_place(PieceType.GOAL, next_base)

    # Decide what to place at next_base
    choices: List[str] = []
    if remaining_len >= 2 and (min_len >= 2 or ctx.rng.random() < 0.70):
        choices.append("D")
    if allow_ramp and remaining_len >= 2 and ctx.rng.random() < 0.35:
        choices.append("R")
    if (allow_split and remaining_len >= 3 and max_depth == 0
            and ctx.rng.random() < 0.40):
        choices.append("T")
    if can_end:
        choices.append("G")
    if not choices:
        return None

    ctx.rng.shuffle(choices)
    for choice in choices:
        snapshot = ctx.snapshot()

        def _rollback():
            ctx.restore(snapshot)

        if choice == "G":
            if ctx.can_place(PieceType.GOAL, next_base):
                ctx.place(PieceType.GOAL, next_base)
                return [(PieceType.GOAL, next_base)]

        elif choice == "D":
            if not ctx.can_place(PieceType.DOMINO, next_base):
                continue
            ctx.place(PieceType.DOMINO, next_base)
            rest = _walk_branch(ctx, next_base, dir_name,
                                 remaining_len - 1, allow_ramp, allow_split,
                                 max_depth, max(0, min_len - 1))
            if rest is not None:
                return [(PieceType.DOMINO, next_base)] + rest
            _rollback()

        elif choice == "R":
            if not ctx.can_place(PieceType.RAMP, next_base):
                continue
            ctx.place(PieceType.RAMP, next_base)
            ramp_pos = next_base
            z_out = ramp_pos[2] + 1
            virt = (ramp_pos[0], ramp_pos[1], z_out)
            # The cascade re-emerges at z+1, so the next piece stands one
            # level up and needs something under it.  Without this step
            # block every ramp branch failed the support check and was
            # rolled back, i.e. ramps could never be generated.
            landing = _next_base(virt, dir_name, z_out)
            under = (landing[0], landing[1], ramp_pos[2])
            if (in_grid(landing, ctx.grid_size)
                    and under not in ctx.tops
                    and ctx.can_place(PieceType.RAMP, under)):
                ctx.place(PieceType.RAMP, under)
                ctx.pedestals.add(under)
            rest = _walk_branch(ctx, virt, dir_name,
                                 remaining_len - 1,
                                 allow_ramp=False,
                                 allow_split=allow_split,
                                 max_depth=max_depth,
                                 min_len=max(0, min_len - 1))
            if rest is not None:
                return [(PieceType.RAMP, next_base)] + rest
            _rollback()

        elif choice == "T":
            if not ctx.can_place(PieceType.SPLITTER, next_base):
                continue
            ctx.place(PieceType.SPLITTER, next_base)
            perps = perp_dirs(dir_name)
            sub_budget = max(2, (remaining_len - 1) // 2)
            sub1 = _walk_branch(ctx, next_base, perps[0],
                                 sub_budget, allow_ramp=False,
                                 allow_split=False, max_depth=max_depth + 1,
                                 min_len=1)
            if sub1 is None:
                _rollback()
                continue
            sub2 = _walk_branch(ctx, next_base, perps[1],
                                 sub_budget, allow_ramp=False,
                                 allow_split=False, max_depth=max_depth + 1,
                                 min_len=1)
            if sub2 is None:
                _rollback()
                continue
            return [(PieceType.SPLITTER, next_base)] + sub1 + sub2

    return None


def _build_skeleton(grid_size: Vec3, rng: random.Random,
                    target_len: int, allow_ramp: bool, allow_split: bool
                    ) -> Optional[Tuple[List[Tuple[PieceType, Vec3]],
                                         Vec3, str, Set[Vec3]]]:
    """Build a cascade skeleton.

    Returns (pieces_in_order, origin_anchor, origin_dir, pedestal_anchors)."""
    A, B, C = grid_size
    for _ in range(60):
        ctx = _BuildCtx(grid_size, rng)
        # Source = a z=1 cell with random starting dir that points inward
        x = rng.randint(1, A)
        y = rng.randint(1, B)
        # Pick a direction that has room to walk.  We need enough cells
        # PAST the origin to fit (target_len - 1) more pieces at minimum.
        # Accept direction if there's at least floor(target_len/2) cells
        # of clearance (min_len) — walker can splice in ramps to climb z.
        min_needed = max(2, target_len // 2)
        dir_candidates = list(_DIRS)
        rng.shuffle(dir_candidates)
        chosen: Optional[str] = None
        for d in dir_candidates:
            if d == "+x" and (A - x) >= min_needed: chosen = d; break
            if d == "-x" and (x - 1) >= min_needed: chosen = d; break
            if d == "+y" and (B - y) >= min_needed: chosen = d; break
            if d == "-y" and (y - 1) >= min_needed: chosen = d; break
        if chosen is None:
            continue
        origin_pos = (x, y, 1)
        if not ctx.can_place(PieceType.DOMINO, origin_pos):
            continue
        ctx.place(PieceType.DOMINO, origin_pos)
        branch = _walk_branch(ctx, origin_pos, chosen,
                               remaining_len=target_len - 1,
                               allow_ramp=allow_ramp,
                               allow_split=allow_split,
                               min_len=max(1, target_len // 2))
        if branch is None:
            continue
        # Must contain at least one GOAL
        if not any(p[0] == PieceType.GOAL for p in ctx.pieces):
            continue
        return (list(ctx.pieces), origin_pos, chosen, set(ctx.pedestals))
    return None


# ---------------------------------------------------------------------------
# Protected placement + patch extraction
# ---------------------------------------------------------------------------

def _add_protected(pieces: List[Tuple[PieceType, Vec3]],
                    grid_size: Vec3, rng: random.Random,
                    n_protected: int) -> List[Tuple[PieceType, Vec3]]:
    """
    Add `n_protected` PROTECTED dominos at cells face-adjacent (in xy) to the
    skeleton but NOT on the skeleton.  Need physical support & grid bounds.
    Returns the updated pieces list.
    """
    used: Set[Vec3] = set()
    for pt, pos in pieces:
        for c in piece_occupied_cells(pt, pos):
            used.add(c)
    tops: Set[Vec3] = set()
    for pt, pos in pieces:
        tops.add(_top_cell_of_anchor(pt, pos))
    tops.add((-1, -1, 0))  # sentinel (never matches)

    # Candidates: cells adjacent to any skeleton base-cell (same z ±x/±y)
    # Prefer cells near the END of the skeleton (harder to avoid).
    candidates: List[Vec3] = []
    for pt, pos in pieces:
        if pt == PieceType.SPLITTER:
            continue  # don't surround splitters (too easy to hit)
        for d in _DIRS:
            dx, dy, _ = DIR_NAME_TO_VEC[d]
            nb = (pos[0] + dx, pos[1] + dy, pos[2])
            if not in_grid(nb, grid_size):
                continue
            if nb in used:
                continue
            # Needs physical support
            z = nb[2]
            if z == 1:
                ok = True
            else:
                ok = (nb[0], nb[1], nb[2] - 1) in tops
            if not ok:
                continue
            # Check upper cell also free
            upper = (nb[0], nb[1], nb[2] + 1)
            if not in_grid(upper, grid_size) or upper in used:
                continue
            candidates.append(nb)

    rng.shuffle(candidates)
    placed: List[Vec3] = []
    new_used = set(used)
    new_tops = set(tops)
    for c in candidates:
        if len(placed) >= n_protected:
            break
        # Also keep protected dominos from clashing with each other
        if c in new_used or (c[0], c[1], c[2] + 1) in new_used:
            continue
        placed.append(c)
        pieces = pieces + [(PieceType.PROTECTED, c)]
        new_used.add(c)
        new_used.add((c[0], c[1], c[2] + 1))
        new_tops.add((c[0], c[1], c[2] + 1))
    return pieces


def _extract_patches(pieces: List[Tuple[PieceType, Vec3]],
                     rng: random.Random, n_patches: int,
                     keep_fixed: Set[Vec3] = frozenset(),
                     ) -> Tuple[List[Tuple[PieceType, Vec3]],
                                List[Tuple[PieceType, Vec3]]]:
    """
    Split `pieces` into (fixed_pieces, patch_pieces).  Only DOMINO and RAMP
    are patchable.  GOAL, SPLITTER, PROTECTED are always fixed.

    To keep the patch meaningful, we avoid carving the ORIGIN piece
    (first in the list) so the player can always find something to push.
    Anchors in `keep_fixed` (support pedestals) are never carved.
    """
    # Candidate patch indices: D or R, not the first piece
    candidates = [i for i, (pt, pos) in enumerate(pieces)
                  if pt in (PieceType.DOMINO, PieceType.RAMP) and i > 0
                  and pos not in keep_fixed]
    rng.shuffle(candidates)
    patch_indices = set(candidates[:n_patches])
    fixed = [p for i, p in enumerate(pieces) if i not in patch_indices]
    patches = [p for i, p in enumerate(pieces) if i in patch_indices]
    return fixed, patches


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _replay(spec: LevelSpec, patches: List[Tuple[PieceType, Vec3]],
            pushes: List[Tuple[Vec3, str]]) -> str:
    """Return the terminal ActionCode.value from replaying patches+pushes."""
    from .game_core import init_state
    st = init_state(spec)
    for pt, pos in patches:
        r = step_place(st, pt, pos)
        if r.code.value != "OK":
            return f"PLACE_FAIL:{r.code.value}:{pt.value}@{pos}"
    r = submit_placement(st)
    if r.code.value != "OK":
        return f"SUBMIT_PLACE:{r.code.value}"
    for pos, d in pushes:
        r = step_push(st, pos, d)
        if r.code.value != "OK":
            return f"PUSH_FAIL:{r.code.value}"
    r = submit(st)
    return r.code.value


def level_has_features(spec: LevelSpec, meta: dict,
                       allow_ramp: bool = False, allow_split: bool = False,
                       strict: bool = True, **_ignored) -> bool:
    """True iff a generated level actually shows the mechanics it was
    generated for: ramp-only configs need a ramp, split-only configs a
    splitter.  When both are allowed, `strict` requires a splitter (the
    generator never puts a ramp and a splitter in the same level: splitter
    branches are built without ramps, and a splitter after a ramp has no
    support for its side branches); `strict=False` accepts either.

    `allow_*` only *permit* a mechanic, so batch scripts use this to skip
    seeds whose level never ended up using it.
    """
    types = {pt for pt, _ in spec.fixed_pieces}
    types |= {PieceType(t) for t, _ in meta.get("patch_pieces", [])}
    has_r = PieceType.RAMP in types
    has_t = PieceType.SPLITTER in types
    if allow_ramp and allow_split:
        return has_t if strict else (has_r or has_t)
    if allow_ramp:
        return has_r
    if allow_split:
        return has_t
    return True


def generate_level(grid_size: Vec3 = (5, 4, 4),
                   seed: int = 0,
                   target_skeleton_len: int = 6,
                   n_patches: int = 2,
                   n_protected: int = 1,
                   allow_ramp: bool = True,
                   allow_split: bool = False,
                   max_attempts: int = 60
                   ) -> Optional[Tuple[LevelSpec, dict]]:
    """Generate one level.  Returns (spec, meta) or None."""
    rng = random.Random(seed)

    for attempt in range(max_attempts):
        built = _build_skeleton(grid_size, rng, target_skeleton_len,
                                 allow_ramp, allow_split)
        if built is None:
            continue
        skeleton, origin_pos, origin_dir, pedestals = built

        # Add protected dominos
        pieces = _add_protected(skeleton, grid_size, rng, n_protected)

        # Extract patches (only from D/R; skip origin)
        fixed, patches = _extract_patches(pieces, rng, n_patches,
                                          keep_fixed=pedestals)
        if len(patches) < n_patches:
            continue

        # Ensure the origin is FIXED (we never carve origin; belt-and-braces)
        if origin_pos not in [p[1] for p in fixed]:
            continue

        # Push budget: at least 1 (single origin).  For splitters we may
        # want more, but single push suffices because the splitter itself
        # fires orthogonal branches.  We keep budget = max(1, #goals / 2).
        push_budget = 1

        # Inventory from patches
        inventory: Dict[PieceType, int] = {}
        for pt, _ in patches:
            inventory[pt] = inventory.get(pt, 0) + 1

        spec = LevelSpec(
            grid_size=grid_size,
            fixed_pieces=fixed,
            inventory=inventory,
            push_budget=push_budget,
            meta={},
        )

        canonical_pushes = [(origin_pos, origin_dir)]
        # Validate: replay patches+pushes → WIN
        verdict = _replay(spec, patches, canonical_pushes)
        if verdict != "WIN":
            continue

        # Validate: WITHOUT patches (just submit_placement w/ missing pieces
        # fails the inventory-not-empty check → can't even get to PUSH).
        # Equivalent check: replay with [] patches but inventory={} forced
        # AND the patches' cells left empty — cascade can't traverse gap.
        # We simulate by creating a "no-patch" spec and checking cascade.
        nopatch_spec = LevelSpec(
            grid_size=grid_size,
            fixed_pieces=fixed,
            inventory={},
            push_budget=push_budget,
        )
        verdict_nopatch = _replay(nopatch_spec, [], canonical_pushes)
        if verdict_nopatch == "WIN":
            # Patches aren't load-bearing — try again
            continue

        # Near-miss check: at least one alternative push direction either
        # fails to topple all goals OR topples a P.  (We skip this check
        # for level with 0 protected and target_skeleton_len small, since
        # it's naturally trivially "correct-direction-only".)
        if n_protected >= 1:
            near_miss_found = False
            for d in _DIRS:
                if d == origin_dir:
                    continue
                alt = _replay(spec, patches, [(origin_pos, d)])
                if alt != "WIN":
                    near_miss_found = True
                    break
            if not near_miss_found:
                continue

        meta = {
            "seed": seed,
            "attempt": attempt,
            "skeleton_len": len(skeleton),
            "pieces_total": len(pieces),
            "fixed_count": len(fixed),
            "patch_count": len(patches),
            "n_protected": n_protected,
            "allow_ramp": allow_ramp,
            "allow_split": allow_split,
            "origin": list(origin_pos),
            "origin_dir": origin_dir,
            "patch_pieces": [(pt.value, list(pos)) for pt, pos in patches],
            "pushes": [(list(pos), d) for pos, d in canonical_pushes],
            "push_budget": push_budget,
        }
        spec.meta = dict(meta)
        return spec, meta

    return None


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    r = generate_level(seed=seed)
    if r is None:
        print("generation failed")
        raise SystemExit(1)
    spec, meta = r
    print("generated level:")
    print(f"  grid:     {spec.grid_size}")
    print(f"  fixed:    {len(spec.fixed_pieces)} pieces")
    for pt, pos in spec.fixed_pieces:
        print(f"    {pt.value}@{pos}")
    print(f"  patches:  {meta['patch_pieces']}")
    print(f"  origin:   {meta['origin']} dir={meta['origin_dir']}")
