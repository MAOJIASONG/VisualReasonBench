# Domino Chain Reaction 3D — T3 × T2 Hybrid (rules v2)

A 3D causal-cascade puzzle.  A partial domino chain stands on a board with
a green **START** domino, gold **goals** and red **protected** pieces.  You
get a few **patch dominoes**: put them into the right gaps, then push the
START domino in one direction.  You **WIN** iff every goal falls and no red
piece does.

This is the **T3 × T2 Hybrid** task in the CHAIN benchmark — complementary
to T1 Burr Puzzle, T2 Stacking, and T2×T3 PipeFit 3D.  Where PipeFit's
T3 element is a *static* network checked at submit, Domino's T3 element is
a *dynamic* cascade that unfolds from a single push.

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

- Grid `A × B × C` (z=1 is the floor).  Pieces (all fixed except patches):

  | piece | look | behaviour |
  |---|---|---|
  | `S` START | green domino, white disc on top | the **only** piece you may push, once, in a direction you choose |
  | `D` domino | ivory, pips | falls forward and knocks the next piece |
  | `T` splitter | ivory + brass cross | when hit, sends the wave out to **both sides** (left and right of the incoming direction), not straight on |
  | `G` goal | gold + star | topple every goal to win |
  | `P` protected | red + white X | toppling one loses |
  | step | grey block | holds a piece one level up; never falls; a wave that meets only a step stops |

- **Hit rule (physical):** a falling domino tips one cell forward and
  knocks over every standing piece whose body is in that column at its own
  height or one level higher (dominoes are two cells tall).  So a wave can
  climb or descend **one** step; two levels apart do not touch.
- **Phase 1 — PLACE:** `step_place(state, D, pos)` / `step_remove`;
  a patch stands on the floor or directly on a step.  Free undo.
  `submit_placement` needs every patch placed.
- **Phase 2 — PUSH:** `step_push(state, START_pos, dir)` runs the whole
  cascade; `submit` scores it.

Rules v1 (any standing piece pushable, pulses only hit pieces whose base
was at exactly their height, ramps lifted the pulse) let 35/50 of the old
puzzles be won by pushing the goal itself with the patches anywhere; see
"Rules v2" in the iteration history below.

---

## What it tests

- **T3 — Structural propagation:** predict where a single push travels,
  through splitters (two branches at once) and up / down steps.
- **T2 — Spatial commitment:** the patches are exactly as many as the real
  gaps, but the board also shows **decoy gaps** (an empty, supported cell
  between pieces that leads to a red piece).  Filling a decoy wastes a patch
  or loses, so "fill every hole" fails from the medium tier on.
- **Direction:** decoy lines beside START punish pushing the wrong way.

---

## File layout

```
type_3_1/
├── __init__.py
├── game_core.py           # PieceType (S/D/T/G/P/step), LevelSpec, GameState
├── cascade.py             # v2 physical hit rule (±1 level), splitter
├── mechanics.py           # step_place / step_remove / step_push / submit
├── difficulty.py          # exhaustive solution search + baseline players
├── generator.py           # chain + decoys + carved patches, tier by measurement
├── levels.py              # category / tier configs, layout de-duplication
├── solver.py              # replay of the stored solution
├── loader.py              # JSON save / load
├── visualizer_plotly.py   # 3D scene (START, steps, toppled pieces, cascade)
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
| START `S` | green slab, white disc on top |
| Domino `D` | ivory slab (0.46 × 0.46 × 2.0) with pips on all four faces |
| Splitter `T` | ivory body + brass cross cap |
| Goal `G` | gold slab + dark 5-point star |
| Protected `P` | red slab + white X; red halo ring on the floor once toppled (LOSE) |
| Step | grey block filling its cell; pieces stand on top |
| Toppled piece | lies flat inside its own cell, darker shade, white arrow = fall direction |
| Cascade trail | steel-blue arrows along the true wave edges, so branches show as branches |
| Board / frame | green-felt board, `x=… / y=… / z=…` labels (`chain_viz`), game frame with STATUS / PATCH DOMINOES / LEGEND cards (`chain_hud`) |

---

## Programmatic API

```python
from type_3_1.generator import generate_level
from type_3_1.solver import solve
from type_3_1.game_core import init_state
from type_3_1.mechanics import step_place, step_push, submit, submit_placement

from type_3_1.levels import CATEGORY

spec, meta = generate_level(seed=3, tier="medium",
                            **CATEGORY["C_steps"]["medium"])
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

`generator.generate_level(grid_size, seed, n_patches, main_len, use_height,
use_split, two_goals, n_decoys, n_protected, min_raised, tier, score_band)`:

| knob | effect |
|---|---|
| `main_len` | length range of the chain from START |
| `use_height` | chain may step up / down one level on grey steps |
| `use_split` | chain ends in a splitter; one side to a goal, the other to a second goal (`two_goals`) or a decoy |
| `n_decoys` | decoy lines (gap → dominoes → red) beside START / past a goal |
| `n_patches` | dominoes carved out of the chain (= the inventory) |
| `min_raised` | at least this many patches stand on a step (z ≥ 2), so a patch's height is part of the answer (C_steps, E_full use 1) |
| `n_protected` | loose red pieces off every line — pure scenery, no category uses it |
| `tier` | accept only levels whose **measured** difficulty is in that band |
| `score_band` | override the tier's score band (default: `difficulty.TIERS`) |

Hard guarantees (checked by exhaustive search, `difficulty.solutions`):

- exactly **one** winning (patch placement, push direction);
- it uses **all** patches (every patch is load-bearing);
- every column holds at most one standing piece, and a chain cell touches
  only its neighbours in the chain, so nothing interacts by accident.

Measured per level (`meta["difficulty"]`, `difficulty.py`): decoy gaps,
win rate of random play, of "fill the holes nearest START" (gap-fill) and
of "follow the wave and fill whichever hole it stops at" (wave-follow).

Every level also gets one number, `difficulty.score`:

    score = (patches - 1) + 0.5 * (decoy gaps + patches on a step) + log2 1/P(wave-follow)

The tiers (`difficulty.TIERS`) are **one ladder for the whole game**, not per
category: every easy level is below every medium level on patches, decoy
holes and score, and every medium below every hard.  The score bands leave a
gap, so no two tiers touch.

| tier | patches | decoy gaps | score | gap-fill wins | wave-follow wins |
|---|---|---|---|---|---|
| easy | 1–2 | 0–1 | ≤ 3.1 | allowed | — |
| medium | 3 | 2 | 4 – 5.6 | ≤ 50 % | ≤ 50 % |
| hard | 9–10 | ≥ 3 | ≥ 14 | 0 % | ≤ 33 % |

The configs in `levels.CATEGORY` were tuned on a 1500-level sample so every
(category, tier) pair can reach its rung: e.g. E_full easy is a single patch
on a step behind a splitter.  `score_band=` on `generate_level` overrides the
tier's band for experiments.

**Hard tier (2026-11).**  Every hard level is a three-level splitter tree
(`split_depth=3`: a fork's side leads on to a deeper splitter, a goal or a
trap) on a 12 × 12 × 5 board with grey steps (`p_step` 0.5–0.7, 4–5 raised
patches), 9–10 patches and 1–5 goals; the category sets the emphasis (more
decoys, more raised patches, two goals per fork).  The tier was calibrated
against gpt-6.1-sol (Codex, default reasoning effort, one-shot, front + back
views): the earlier hard tier (4 patches, no splitter tree) was solved 14/15,
this one about a quarter of the time; its losses are patches put at the wrong
height (`CellOccupied`, `NoPhysicalSupport`).  See
`../chain_analysis/model_runs/gpt61sol_hard_calibration/`.  `images_batch/`
still holds the earlier demo levels (`play_batch` would now build the new
hard ones).

`generate_50` / `play_batch` search seeds in parallel (`../chain_search.py`).

---

## The 50-puzzle batch (`puzzles_50/`)

`python -m type_3_1.generate_50` (configs in `levels.py`):

| category | what it adds |
|---|---|
| A_line | straight chain |
| B_decoy | decoy lines around the chain |
| C_steps | height changes on grey steps; at least one patch stands on a step |
| D_split | a splitter sends the wave both ways |
| E_full | steps + splitter + decoys; at least one patch on a step |

Every level stores the category's insight (`levels.INSIGHT`) in its meta
and `solution.txt` — the one thing the player has to notice.

Each category is 3 easy + 4 medium + 3 hard; layouts are de-duplicated up
to rotation / reflection / translation and every slot scans its own seeds.
Per puzzle: `level.json`, `solution.txt`, `00_initial_front.png`,
`00_initial_back.png`, `99_submit_front.png`; plus `SUMMARY.md` and `index.json` with the measured
difficulty.  `level.json`'s meta and `solution.txt` contain the answer —
keep them out of model inputs.

---

## Iteration history & lessons learned

### Rules v2 — measured, then redesigned (2026-10)

An exhaustive audit of the v1 puzzles (every placement × every push)
found: 35/50 could be won with the patches *anywhere* by pushing the goal
(or a splitter along its stem); in 40/50 the patch cells were exactly the
straight gaps between pieces; protected pieces floated after a ramp under
them became a patch; the "step" under a raised goal was a working ramp;
and tiers only scaled the grid.  v2 fixes the rules rather than the
numbers: one pushable START, a physical hit rule (±1 level), inert steps
instead of ramps, decoy gaps, and acceptance by exhaustive search (unique
solution, all patches load-bearing, measured tier).  The v1 notes below are
kept for history.

A second audit (rules v2) found that the `C_steps` category often did not
need its steps: one hard level had no step at all, and many put every patch
on the floor; `B_decoy` medium / hard also scattered red pieces that no
wave could reach.  Fix: `min_raised` (a patch must stand on a step) and no
scenery pieces — every piece on the board now matters.

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
