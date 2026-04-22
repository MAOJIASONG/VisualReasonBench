# Domino Chain Reaction 3D — Design Spec

**Task name:** `type_3_domino` (CHAIN benchmark, T3 × T2 hybrid emphasizing T3)
**Date:** 2026-04-22
**Status:** Design — awaiting implementation plan

---

## 1. Positioning within the CHAIN benchmark

| Task | Type | Core reasoning |
|---|---|---|
| Burr Puzzle | T1 | Hidden interlock — disassembly order |
| Stacking | T2 | Polycube exact cover — delayed spatial dead-end |
| PipeFit 3D | T2×T3 | Static graph connectivity + anchor chain (structural, submit-time) |
| **Domino Chain Reaction** | **T3×T2** (T3-heavy) | **Dynamic causal cascade** — push triggers non-local toppling through a chain |

Key differentiator vs PipeFit: PipeFit's T3 element is a *static* anchor graph evaluated once at submit. Domino's T3 element is a *dynamic* cascade that temporally unfolds from a push action — local push A at one cell knocks down pieces across the grid, possibly toppling a protected piece far from A.

The task has a T2 component (player must place `K` "patch" pieces into a partial arrangement), but the dominant reasoning demand is T3 (predicting the shape of the cascade before committing to a push).

---

## 2. Game mechanics

### 2.1 Grid, pieces, states

- Grid `A × B × C` (integer, z=1 is the ground).
- Pieces:

| Symbol | Shape (cells) | Role |
|---|---|---|
| `D` | 1×1×2 standing | Basic domino |
| `T` | 1×1×2 standing, cross-capped | Splitter — when toppled, emits two orthogonal pulses |
| `R` | 1×1×1 ramp block | Level-transition — pulse arrives at z, continues at z+1 |
| `G` | 1×1×2 standing, gold | Goal domino (must be toppled to win) |
| `P` | 1×1×2 standing, red | Protected domino (toppling it = instant loss) |

Physical support rule (reused from PipeFit):
- A standing piece (`D`/`T`/`G`/`P`) occupies `(x,y,z)` and `(x,y,z+1)`. The cell at `(x,y,z-1)` must be ground (`z=1`), another piece's cap, or a ramp top.
- A ramp occupies one cell; same support rule for the cell below.

Each piece has state ∈ `{standing, toppled}`. Ramps use the same flag but "toppled ramp" just means "already activated once" (cascades cannot traverse it twice).

### 2.2 Cascade simulation (`cascade.py`)

A cascade is a set of **pulses**. Each pulse is a tuple `(pos, dir, z)`:
- `pos`: the current `(x,y)` from which the pulse tries to reach the next cell
- `dir`: unit vector in ±x or ±y
- `z`: the z-level of the pulse (1 ≤ z ≤ C)

Pulse step algorithm:
1. `target = pos + dir` at height `z`.
2. Look up what piece occupies that target (using the standing-piece column at `(target.x, target.y)` with base-z = `z`).
3. Dispatch by piece type:
   - **Empty / out-of-bounds** → pulse dies.
   - **Already toppled** → pulse dies.
   - **D / G / P (standing)** → piece topples. If P → set `GAME_OVER=True`. Emit a new pulse `(target, dir, z)` (continues in same direction, same z).
   - **T (standing)** → piece topples. Emit TWO pulses: `(target, perp(dir)[0], z)` and `(target, perp(dir)[1], z)`. **The straight-through direction is NOT continued.**
   - **R (standing)** → ramp activates (state → toppled). Emit one pulse `(target, dir, z+1)`. Ramp can only be activated from a pulse whose `z` equals the ramp's base `z`.

   where `perp(dir)` = the two horizontal unit vectors orthogonal to `dir`:
   - `dir = +x` → `{+y, -y}`
   - `dir = -x` → `{+y, -y}`
   - `dir = +y` → `{+x, -x}`
   - `dir = -y` → `{+x, -x}`

4. Simulation terminates when all pulses have died.
5. If `GAME_OVER` was raised at any point, cascade still runs to completion (so the renderer can show the full damage), but the push is marked as losing.

Determinism: the order of pulse processing doesn't affect the final toppled set, because each piece can be toppled at most once and once toppled its effect is fixed.

### 2.3 Push action

- `push(piece_id, dir)` starts a cascade.
- Valid only if `piece` is a standing `D`, `T`, `G`, or `P` (not `R`). The initial push topples `piece` and emits a pulse `(piece.pos, dir, piece.z)`.
  - **Pushing G** is legal — it counts toward winning.
  - **Pushing P** is legal but instantly loses.
- `dir` ∈ {+x, -x, +y, -y}.
- After simulation: update all piece states; record the set of newly-toppled pieces and whether GAME_OVER fired.

### 2.4 Two-phase game loop

**Phase 1 — Placement (same style as PipeFit)**
- Inventory: `{D: n₁, T: n₂, R: n₃}` (goal and protected are always pre-placed as fixed pieces).
- Actions: `step_place(pt, pos, orientation)`, `step_remove(pos)`, `submit_placement()`.
  - `step_place` enforces the same physical-support rule as PipeFit.
  - Free undo via `step_remove`.
- `submit_placement()` locks: every inventory slot must be consumed (optional: allow leftover, but then generator must also allow it).

**Phase 2 — Pushing**
- Actions: `step_push(piece_id, dir)`, `submit()`.
- Budget `K` pushes total.
- Each `step_push` simulates the cascade synchronously and updates state.
- `submit()` evaluates win condition:
  - WIN ⟺ every `G` is toppled ∧ no `P` is toppled ∧ push count ≤ `K`.
  - Otherwise LOSE.
- Pushes are **irreversible** (unlike placements). This is the T3 commitment — once you push, the cascade is spent.

### 2.5 Action codes (extends PipeFit's `ActionCode`)

```
OK                        # generic success
CELL_OCCUPIED             # placement conflict
NO_INVENTORY              # inventory exhausted
NO_PHYSICAL_SUPPORT       # placement violates support rule (reused)
BAD_PHASE                 # can't place during push phase (or vice versa)
NOT_STANDING              # push target is already toppled
BUDGET_EXCEEDED           # ran out of pushes
```

---

## 3. Generator strategy (backward)

Goal: produce levels where

1. Without the correct patches, no push sequence wins.
2. With the correct patches in the correct positions, at least one push sequence wins.
3. There is always a protected piece P whose distance to a "near miss" cascade is ≤ 2 — so the player cannot just throw pieces in randomly and luck into a win.

### 3.1 Algorithm

1. **Pick a skeleton** — a cascade tree.
   - Root = one push.
   - Each non-leaf branches at splitters (T-pieces) or continues straight.
   - Leaves = goal dominoes `G` (1 to 3 leaves, depending on difficulty).
   - Allow occasional z-transitions via ramps.
   - Skeleton lives on integer cells in the grid.

2. **Lay the skeleton on the grid**:
   - Assign piece types per cell (`D`, `T`, `R`, or `G`).
   - Every piece along a straight segment = `D`. Every branch junction = `T`. Every z-transition = `R`.
   - Endpoint(s) = `G`.

3. **Add protected pieces** `P`:
   - For each skeleton interior cell, consider its neighbors (same z, ±x/±y) that are NOT on the skeleton.
   - Place up to `n_P` protected pieces on those neighbors, prioritizing cells near branch points — so if the player pushes in the wrong direction or places a splitter incorrectly, the cascade veers into `P`.

4. **Carve out patches**:
   - Choose `K_patch` interior skeleton cells (prefer `D` and `R` pieces; avoid `T` and `G`).
   - These become the player's inventory.
   - The rest of the skeleton = fixed pieces.

5. **Optional distractors**:
   - Add a few standing `D` pieces off-skeleton that the cascade won't reach (visual noise; increases ambiguity).

6. **Compute the canonical push**:
   - The push is: `(skeleton_root_piece, skeleton_root_direction)`.
   - `K` push budget ≥ number of independent trees (for easy/medium: K = 1 with branches via splitters; for hard: K = 2-3 with multiple independent trees).

7. **Forward-validate** (solver/replay):
   - Start from the fixed arrangement + patches placed per the plan + canonical push.
   - Simulate. Confirm `WIN`.
   - Also confirm that if you zero out the patches, the cascade fails to reach a goal (otherwise patches don't matter).
   - For `n_P ≥ 1`, confirm that at least one "near-miss" alternative push reaches P (otherwise P is free/not constraining).

8. **Store in `spec.meta`**:
   ```
   meta = {
     "fixed_pieces": [(type, pos), ...],      # piece type + cell for each pre-placed piece (incl. G, P, T)
     "patch_pieces": [(type, pos), ...],      # the canonical patch placement (solver replays this)
     "pushes": [(pos, dir), ...],             # canonical push sequence (len 1..K)
     "K": 1 | 2 | 3,                          # push budget (hard cap, ≥ len(pushes))
     "skeleton": [...],                       # debugging: raw cascade tree before patch extraction
   }
   ```
   Solver uses `meta["patch_pieces"]` + `meta["pushes"]` to replay. Inventory is computed from the type-counts of `patch_pieces`.

**Patch-eligible types:** patches are always drawn from `{D, R}` (commodity pieces). `T`, `G`, and `P` are always pre-placed/fixed. Rationale: (a) splitters and goals shape the puzzle's causal structure — letting the player choose where to put them would change the problem; (b) `P` is a constraint, not an inventory item.

### 3.2 Difficulty tiers

| Axis | Easy | Medium | Hard |
|---|---|---|---|
| Grid | 4×3×3 | 5×4×4 | 6×5×5 |
| Fixed pieces | 5–7 | 8–12 | 13–18 |
| Patch pieces (K_patch) | 1–2 | 3–4 | 4–6 |
| Ramps `R` | 0 | 1–2 | 2–4 |
| Splitters `T` | 0 | 0–1 | 1–3 |
| Protected `P` | 0–1 | 1–2 | 2–3 |
| Goal `G` | 1 | 1–2 | 2–3 |
| Push budget K | 1 | 1–2 | 2–3 |

---

## 4. Solver / validator

Given the `meta`-encoded canonical plan:

1. Start from initial state (fixed pieces only, per `meta["fixed_pieces"]`).
2. For each `(type, pos)` in `meta["patch_pieces"]`, call `step_place(type, pos)`.
3. Apply `submit_placement()`.
4. For each `(pos, dir)` in `meta["pushes"]`, call `step_push(pos, dir)`.
5. Apply `submit()`; expect WIN.

This is a replay-style validator, not a search solver. Exactly like PipeFit, the generator proves feasibility by construction; the solver rebuilds the plan deterministically.

No free-search solver will be shipped in v1 (state space is too large and not needed for generation). A free-search solver could be added later for an oracle/baseline.

---

## 5. Visual encoding

Palette (close to PipeFit for consistency, with domino-specific accents):

| Element | Appearance |
|---|---|
| Standing D | Slim tall cuboid (0.45 × 0.45 × 1.8), matte white-ivory face with dark edge |
| Standing T | Same as D + cross-shaped brass cap on top (2 small brass bars) |
| Standing R (ramp) | Wedge / triangular prism, warm wood-grain tan |
| Standing G (goal) | Gold-painted domino + 4-pointed gold star on top |
| Standing P (protected) | Red-lacquered domino + white warning-X on top |
| Toppled D/T/G/P | Horizontal cuboid lying in the fall direction, darker (shadowed) face |
| Activated R | Wedge rendered darker, with faint impact arrow showing direction |
| Ground | Light sand grid at z=0.5 plane |

Per-render panels (reused from PipeFit):
- STATUS panel (top-right): phase / push count / game state / violations
- INVENTORY panel (mid-right): remaining patch counts per type
- Legend (bottom-left): gold = goal, red = protected, ramp = lifts 1 level, T = splits
- Camera: `front` + `back` pair via `render_state_pair`

Cascade visualization:
- During/after a push, overlay the cascade path as a faint arrow sequence (ghost trails from push source → each toppled piece in order).
- If GAME_OVER: the fatal P piece gets a bright-red halo ring.

---

## 6. File layout

```
type_3_domino/
├── __init__.py
├── game_core.py          # PieceType, State, ActionCode, LevelSpec, GameState
├── cascade.py            # pulse simulation (splitter / ramp semantics)
├── mechanics.py          # step_place / step_remove / submit_placement / step_push / submit
├── generator.py          # backward skeleton → fixed + patches + push
├── solver.py             # replay validator (uses spec.meta)
├── loader.py             # JSON save/load
├── visualizer_plotly.py  # 3D renderer + cascade overlay
├── play.py               # single-level demo
├── play_batch.py         # 10-level batch (easy/medium/hard × a few seeds)
├── generate_50.py        # 50-puzzle diverse batch
├── images/               # per-seed play output
├── images_batch/         # batch + SUMMARY.md
├── puzzles_50/           # 50 puzzles + SUMMARY.md + index.json
└── README.md
```

---

## 7. Test plan

### 7.1 Mechanics unit tests
- Cascade through a straight 3-domino line: push +x at source → all 3 toppled.
- Splitter: push into T from -x → two orthogonal pulses fire; straight-through does NOT continue.
- Ramp: push a domino into a ramp → cascade re-emerges at z+1 heading same dir.
- Protected: cascade hits P → GAME_OVER, but cascade completes (still updates remaining state for rendering).
- Budget: K=1, push once → any second push fails with BUDGET_EXCEEDED.
- Physical support: `step_place` on thin air → NO_PHYSICAL_SUPPORT.

### 7.2 Generator validation
- Every generated level: replay plan → WIN.
- Every generated level with `n_P ≥ 1`: removing all patches → cascade fails OR never reaches G (so patches matter).
- Every generated level: at least one near-miss push (alternative dir) reaches `P` OR fails to cover all `G` (so the correct push is actually informative).

### 7.3 Rendering smoke tests
- Render initial + after-each-push for 1 easy seed → verify no missing pieces, toppled pieces lie flat, cascade overlay visible.
- Render a GAME_OVER frame → verify P halo is visible.

### 7.4 Batch
- 10-level batch (`play_batch.py`): 3 easy + 4 medium + 3 hard.
- 50-level batch (`generate_50.py`): 5 categories × 10 puzzles each (below), mirroring PipeFit's layout.

Category distribution for 50-batch:

| Code | Description | Count |
|---|---|---|
| A_straight | No ramps, no splitters, no protected | 10 |
| B_ramp | Ramps only (z-transitions), no splitters, no protected | 10 |
| C_split | Splitters only (multi-goal via T), no ramps, no protected | 10 |
| D_hazard | Rich arrangement WITH protected pieces | 10 |
| E_full | Ramps + splitters + protected (full difficulty) | 10 |

---

## 8. Non-goals (v1)

- No free-search solver (use generator's canonical plan for validation only).
- No cross-piece support (each piece sits on its own footprint; no "piece A's top carries piece B").
- No animation of the cascade (static frames per push; arrows suffice).
- No support for non-orthogonal cascade directions or diagonal splitters.
- No stochastic cascades — fully deterministic.
- No 1-cell dominos (smallest domino is 1×1×2).

---

## 9. Key invariants

1. **Deterministic cascade**: given a pushed piece + direction + arrangement, the toppled set is unique (pulse order irrelevant).
2. **Replay succeeds**: for every spec generated by `generator.generate_level`, `solver.solve` returns a plan that `replay` executes to WIN.
3. **Patches are load-bearing**: without the patches in their intended cells, the push plan does not reach all goals (verified per spec).
4. **Protected matters**: for every spec with `n_P ≥ 1`, at least one near-miss push alternative actually topples a P (so choosing the right push is not trivial).
5. **Physical plausibility**: every placement (fixed + patch) has ground or another piece below it before it is placed.

---

## 10. Implementation phasing

See `docs/superpowers/plans/2026-04-22-domino-chain-3d-plan.md` (to be written by writing-plans skill).

Rough order:
1. `game_core.py` primitives (PieceType, state, action codes)
2. `cascade.py` pulse simulator + unit tests
3. `mechanics.py` step/submit glue
4. `visualizer_plotly.py` standing + toppled + ramp + splitter + cascade overlay
5. `generator.py` backward skeleton
6. `solver.py` replay validator
7. `loader.py`, `play.py`, `play_batch.py`
8. `generate_50.py`
9. README (iteration-history style, like PipeFit)
