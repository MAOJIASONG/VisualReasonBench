# Domino Chain Reaction 3D — T3 × T2 Hybrid

A 3D causal-cascade puzzle.  You get a partial arrangement of fixed
dominos (including one or more **gold goals** and a few **red protected
pieces**) plus a small **inventory of patch dominos**.  You place the
patches into legal cells, then **choose one piece to push** — the
cascade propagates through the whole arrangement.  You **WIN** iff
every goal is toppled and no protected domino gets touched.

This is the **T3 × T2 Hybrid** task in the CHAIN benchmark — complementary
to T1 Burr Puzzle, T2 Stacking, and T2×T3 PipeFit 3D.  Where PipeFit's
T3 element is a *static* anchor graph evaluated at submit, Domino's T3
element is a *dynamic* cascade that temporally unfolds from a single
push action.

---

## TL;DR

```bash
cd /mnt/hdfs/yuhao/VisualReasonBench/Stacking_scaling

# 1) Auto-play one level (renders initial + each placement + post-push)
python -m type_3_1.play 3

# 2) 10-level mini batch (easy / medium / hard)
python -m type_3_1.play_batch
#    -> output in type_3_1/images_batch/ + SUMMARY.md

# 3) 50-puzzle diverse batch across 5 categories
python -m type_3_1.generate_50
#    -> output in type_3_1/puzzles_50/ + SUMMARY.md + index.json

# 4) Generate a single level as JSON
python -m type_3_1.generator 7
```

---

## Rules in 60 seconds

- Grid is `A × B × C` (z=1 is ground).
- Special pieces (always FIXED, pre-placed):
  - `G` (**gold goal**, 1×1×2) — must be toppled to WIN.
  - `P` (**red protected**, 1×1×2) — toppling any LOSES instantly.
  - `T` (**splitter**, 1×1×2, cross cap) — when toppled, cascade emits
    two orthogonal pulses in the horizontal plane; does NOT continue
    straight.
- Patch pieces (the player places from inventory):
  - `D` (**domino**, 1×1×2) — falls in the direction it was hit, cascade
    continues in same direction.
  - `R` (**ramp**, 1×1×1) — cascade re-emerges at `z + 1` along the same
    horizontal direction.
- **Phase 1 — PLACE**: `step_place(piece_type, pos)` / `step_remove(pos)`
  / `submit_placement()`.  Free undo.  Physical-support rule enforced
  (no building on thin air).  Submit requires the inventory to be empty.
- **Phase 2 — PUSH**: `step_push(piece_anchor, direction)` / `submit()`.
  Up to `push_budget` pushes.  Pushes are **irreversible**.
- **Cascade**:
  - A cascade is driven by pulses `(from_anchor, dir, z)`.
  - Empty / OOB / already-toppled → pulse dies.
  - `D` / `G` / `P`: topple; one forward pulse in the same direction at
    the same z.  (P triggers GAME_OVER but we still run to completion
    for rendering.)
  - `T`: topple; **two** orthogonal pulses; straight-through dies.
  - `R`: activate; one pulse at `z + 1` in the same direction.
- **WIN** iff every G is toppled AND no P is toppled AND push count ≤
  `push_budget`.

---

## What it tests

- **T3 — Structural Propagation**: a single local push triggers a wave
  that can touch pieces far from the source.  The player must
  pre-simulate the cascade mentally to decide which direction to push.
- **T2 — Spatial Commitment**: the patch pieces must be placed in the
  correct cells.  A missing patch → cascade dies early and a goal stays
  standing.  An extra patch in the wrong place → cascade may wander
  into a protected domino.
- **Splitter routing**: with a `T` on the skeleton, one push sends the
  wave down two orthogonal branches simultaneously — the player must
  verify both branches land on goals, not protected.
- **z-transitions**: ramps lift the cascade one cell; a goal on a
  different z is only reachable via a ramp.
- **Near-miss hazards**: generator places protected dominos face-adjacent
  to the skeleton so that a wrong-direction push catches one.

---

## File layout

```
type_3_1/
├── __init__.py
├── game_core.py           # PieceType, LevelSpec, GameState, Phase, ActionCode
├── cascade.py             # pulse simulator (splitter / ramp semantics)
├── mechanics.py           # step_place / step_remove / step_push / submit
├── generator.py           # forward skeleton builder + patch carving + validation
├── solver.py              # replay validator (uses spec.meta)
├── loader.py              # JSON save / load
├── visualizer_plotly.py   # 3D renderer: ivory dominos, brass splitter, gold goal,
│                          # red protected, tan ramp, cascade arrows, panels
├── play.py                # single-level demo (renders every step)
├── play_batch.py          # 10-level easy / medium / hard batch
├── generate_50.py         # 50-puzzle 5-category batch
├── images/                # per-seed play.py output
├── images_batch/          # 10-level batch + SUMMARY.md
├── puzzles_50/            # 50 puzzles + SUMMARY.md + index.json
└── README.md              # you are here
```

---

## Visual encoding

| Element | Appearance |
|---|---|
| Domino `D` | slim ivory standing slab (0.46 × 0.46 × 2.0) with pips on all four faces |
| Splitter `T` | ivory body + **brass cross cap** on top |
| Goal `G` | **gold** slab + dark **5-point star** on top (distinct from the splitter's cross) |
| Protected `P` | red-lacquered slab + white warning-X on top; **red halo ring** on the floor once toppled (the LOSE indicator) |
| Ramp `R` | direction-free **three-step block** until the cascade uses it (game logic gives a ramp no direction); afterwards a wedge rising in the direction it was crossed |
| Toppled D/T/G/P | lies flat **inside its own cell**, darker shade of its own colour, white arrow = fall direction (goal keeps its star) |
| Cascade trail | steel-blue arrows along the true pulse edges (`from` → hit piece), so splitter branches show as branches |
| Coordinates | `x=…` / `y=…` labels on the floor edges facing the camera, `z=…` on a vertical edge (shared `chain_viz`) |
| Raised pieces | dashed drop line + floor footprint under every piece standing above z=1 |
| Panels | STATUS (phase, pushes, goals / protected toppled, result) + INVENTORY + legend |
| Canvas | 1400 × 1100 ×1.4, always frames the full grid |
| Cameras | `front` + `back` via `render_state_pair` |

## Programmatic API

```python
from type_3_1.generator import generate_level
from type_3_1.solver import solve
from type_3_1.game_core import init_state
from type_3_1.mechanics import step_place, step_push, submit, submit_placement

spec, meta = generate_level(
    grid_size=(5, 4, 4),
    target_skeleton_len=6,
    n_patches=2, n_protected=1,
    allow_ramp=True, allow_split=False,
    seed=3,
)
state = init_state(spec)
plan = solve(spec)                              # uses spec.meta to replay
for act in plan:
    kind = act[0]
    if kind == "place":
        step_place(state, act[1], act[2])
    elif kind == "submit_placement":
        submit_placement(state)
    elif kind == "push":
        step_push(state, act[1], act[2])
    elif kind == "submit":
        res = submit(state)
        print(res.code.value)                   # "WIN"
```

---

## Generator knobs

| Argument | Effect | Typical |
|---|---|---|
| `grid_size` | (A, B, C) board | `(5, 4, 4)` |
| `target_skeleton_len` | how long the cascade chain should be (max) | 4–10 |
| `n_patches` | how many inventory pieces the player must place | 1–4 |
| `n_protected` | red-domino hazards near the skeleton | 0–3 |
| `allow_ramp` | whether to insert z-transitions (R) | `True` / `False` |
| `allow_split` | whether to insert splitters (T) | `True` / `False` |
| `seed` | RNG seed | any int |
| `max_attempts` | rejection-sampling budget | 60 default |

Per-level meta fields stored in `spec.meta`:

| Field | Meaning |
|---|---|
| `skeleton_len` | number of pieces on the cascade chain |
| `pieces_total` | total pieces (skeleton + protected) |
| `fixed_count` | pieces pre-placed at level init |
| `patch_count` | pieces the player must add |
| `n_protected` | number of P hazards |
| `origin` | push-origin anchor (as list) |
| `origin_dir` | push direction |
| `patch_pieces` | `[(piece_type_str, [x,y,z]), …]` — canonical patch layout |
| `pushes` | `[([x,y,z], dir_str), …]` — canonical push sequence |
| `push_budget` | cap on pushes (≥ len(pushes)) |

Hard guarantees:

- **Replay always wins**: generator only returns levels where placing the
  canonical patches and applying the canonical pushes evaluates to WIN.
- **Ramps get a step to land on**: the cascade leaves a ramp one level up,
  so the generator adds a fixed 1-cell step block under the landing cell
  when nothing supports it (and never carves those blocks into patches).
  Before this, every ramp branch failed the support check and was rolled
  back, so no generated level contained a ramp.
- **Batch scripts enforce category mechanics**: `generate_50` /
  `play_batch` only accept seeds whose level actually contains a ramp
  (C_ramp) / splitter (D_split / E_full; E_full falls back to a ramp when
  the grid is too narrow for a splitter) — see
  `generator.level_has_features` — and never reuse a seed for the same
  config (several tier slots share a config, which used to produce
  identical puzzles).
- **Known limitation**: a level never contains both a ramp and a
  splitter.  Splitter branches are built with ramps disabled, and a
  splitter reached through a ramp stands one level up with no support for
  its side branches.
- **Patches are load-bearing**: for every returned level, removing the
  patches and running the same push makes the cascade fail to reach all
  goals (or not reach any) — so patches genuinely matter.
- **Near-miss exists (when `n_protected ≥ 1`)**: at least one of the
  three alternative push directions leads to a non-WIN outcome, so
  picking the correct direction is not trivial.
- **Physically plausible**: every piece (fixed or patch) has ground or
  the top of another piece directly below it at placement time.

---

## The 50-puzzle batch (`puzzles_50/`)

`python -m type_3_1.generate_50` produces 50 puzzles in 5 categories:

| category | pieces | obstacles | count |
|---|---|---|---|
| A_straight | D + G only | 0 P | 10 |
| B_protect  | D + G + P | ≥ 1 P | 10 |
| C_ramp     | D + R + G (+P) | maybe | 10 |
| D_split    | D + T + G (+P) | maybe | 10 |
| E_full     | D + R + T + G + P | 1–3 P | 10 |

Each category is split 3 easy + 4 medium + 3 hard.  Per puzzle we save:

- `level.json` — full `LevelSpec` + `meta`
- `solution.txt` — the solver's plan, step-by-step
- `00_initial_front.png` — starting arrangement
- `99_submit_front.png` — final (WIN) arrangement with cascade trail

Top level:

- `puzzles_50/SUMMARY.md` — markdown table of all 50 puzzles
- `puzzles_50/index.json` — programmatic index

---

## Iteration history & lessons learned

The build went through several rounds of adjustment.  Each lesson is
written so the next CHAIN task can skip the mistake.

### Round 1 — define the cascade semantics first

Pulse = `(from_anchor, dir, z)`.  Splitter emits orthogonal ONLY
(straight-through dies).  Ramp lifts the pulse by 1 z unit.  Each piece
topples at most once, so the toppled set is independent of pulse
processing order — BFS suffices.  Getting these semantics pinned down
before writing the generator makes the rest trivially deterministic.

### Round 2 — forward builder, backward validator

The generator walks the cascade forward: pick source, lay a D-line,
occasionally insert R/T, end at G.  The solver is pure **replay**: it
reads `spec.meta["patch_pieces"]` and `spec.meta["pushes"]` and
replays.  No search solver needed at v1 — the generator proves
feasibility by construction.

### Round 3 — direction-selection needs clearance, not exact length

Initial bug: required `target_len` cells of clearance in one cardinal
direction, but `target_len=6` in a 5-wide grid failed every direction.
Relaxed to `min_needed = max(2, target_len // 2)` cells of clearance +
early-termination in `_walk_branch` (emit a goal when `min_len` is
satisfied rather than requiring exact length).  Most generated
skeletons are 4–6 long with some climbing to 8–10 on hard seeds.

### Round 4 — patches carved from D and R only

`T`, `G`, `P` shape the puzzle's causal structure and should be
**fixed**.  Making them patch-able would let the player trivially
redesign the cascade, which is no longer a reasoning puzzle.
Generator enforces `patches ⊆ {D, R}` and keeps the push origin fixed
too (so the player always has something to push).

### Round 5 — placement requires foundation below

Same lesson as PipeFit: no building in thin air.  Standing pieces at
`z > 1` need another piece's TOP cell directly below their base.
Ramps at `z > 1` same rule.  Solver emits patches in skeleton build
order, which is already foundation-first.

### Round 6 — protected adjacent to skeleton, not ON it

A protected domino ON the skeleton would dead-end the cascade before
reaching the goal, so it would be unavoidably toppled and useless as
a hazard.  Instead, `_add_protected` places P face-adjacent to
skeleton cells (but not on the skeleton), so that a wrong-direction
push catches one but the canonical direction misses it.  The generator
validates at least one near-miss direction leads to a non-WIN.

### Round 7 — visual readability requires role-accurate accents

The first cut rendered all pieces as plain cuboids.  Users couldn't
distinguish goal from protected from splitter at a glance.  Added
role-specific accents: gold star on G, white warning-X on P, brass
cross on T, wood treads on R, and a bright red halo on toppled P
to surface the LOSE indicator clearly.

### Round 8 — cascade overlay is the whole point

The cascade itself is invisible at rest; a static "after push" frame
just shows fallen dominos with no sense of HOW the wave propagated.
Added faint steel-blue line segments connecting push origin through
every toppled piece in fall order, giving the reader a visible trail
of the causal chain.

### Round 9 — `read` tool PNG caching gotcha

Same as PipeFit: Claude Code's Read tool caches PNGs by path.  After
any renderer change, write to a new directory or include a timestamp
nonce in the filename before reading back — otherwise the stale
preview suggests your changes didn't take.

---

## Reusable template for the next CHAIN task

What to reuse from `type_3_1/`:

- **Forward-build generator** with early-termination recursion: a
  simple random walk with "end here" choice at every step is more
  robust than requiring exact target lengths.
- **`spec.meta` carries the canonical plan**: patches + pushes, so the
  solver is just a replay engine.
- **Two-phase game loop** (PLACE + PUSH) with clean phase gating and
  inventory-must-be-empty for phase transition.
- **Placement-time physical-support rule** (never building on thin
  air); plan ordering emits supports before things that rest on them.
- **Role-accurate visual encoding**: different accents for each role
  beat different colors alone.
- **Cascade overlay** when the task has a propagation phase — a trail
  line makes the abstract wave visible.
- **Embedded STATUS + INVENTORY + legend panels** for zero-context
  readability.
- **Rectangular bbox + manual aspect ratio** (never force cube).

What you'll re-derive per task:

- Specific piece catalog + their topple / cascade rules.
- How roles connect: in Domino, cascade = causal edges; in PipeFit,
  pipe graph = connectivity edges; in Burr, interlock = motion
  constraints.
- Win / lose conditions.
- Generator's difficulty triple-lock.

---

## License

MIT — same as the rest of the repo.
