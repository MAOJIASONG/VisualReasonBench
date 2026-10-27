# PipeFit 3D — T2 & T3 Hybrid (Placement × Propagation), rules v2

A 3D assembly puzzle.  Entrances and exits sit on the side walls of a box.
With a bounded inventory of pipe pieces (I / L / T) and support blocks
(S1 / S2) you build **one leak-free pipe network that joins every
entrance and every exit**, and every pipe must rest on something solid.

This is the **T2 & T3 Hybrid** task in the CHAIN benchmark —
complementary to T1 Burr Puzzle, T2 Stacking, and T3 FluidNet.

---

## TL;DR

```bash
cd /mnt/hdfs/yuhao/VisualReasonBench/Stacking_scaling

# 1) Auto-play one level (render every step + counterfactual)
python -m type_2_3.play 1

# 2) Generate 10-level mini batch across easy/medium/hard
python -m type_2_3.play_batch
#    -> output in type_2_3/images_batch/  (with SUMMARY.md)

# 3) Generate 50 diverse puzzles — 30+ multi-portal, 10 with obstacles
python -m type_2_3.generate_50
#    -> output in type_2_3/puzzles_50/  (with SUMMARY.md + index.json)

# 4) Generate a single level as JSON
python -m type_2_3.generator 7
```

---

## Rules in 60 seconds

- Grid `A × B × C`.  Portals (green entrance, orange exits) sit on the
  `x=1 / x=A / y=1 / y=B` walls; each is joined by exactly **one** pipe,
  from the inside of the box.  Obstacles (dark red, white X) block cells.
- Pieces:
  - `I` straight, `L` corner, `T` fork (three ends) — name a pipe by its
    **open ends**, e.g. `place L at (3,2,2) ends=+x,+z`
    (`mechanics.step_place(..., ends=("+x", "+z"))`); orientation indices
    still work.
  - `S1` (1 cell) and `S2` (2 cells tall) support blocks.
- Actions: `place` (any free cell), `remove` (free undo, costs an action),
  `submit` (**one** final check; the game ends WIN or LOSE).
- WIN iff, at submit:
  1. **No leaks:** every pipe end meets another pipe end or a portal.
  2. **Connected:** all entrances and exits are in one network (two exits ⇒
     you need a `T`).
  3. **Supported:** every pipe and support rests on something solid
     directly below — the floor, a support, or another supported pipe.
- `max_actions` = 3 × inventory.

Rules v1 let you win by reaching *one* of two exits, by routing through
the other entrance, with open pipe ends, and with a "one unanchored pipe
may dangle" exception the image never explained; see "Rules v2" below.

---

## What it tests

- **T2 — Spatial commitment:** a tight inventory (exact + at most one
  spare) and obstacles that block the obvious "step toward the exit, climb,
  repeat" staircase; on medium / hard that staircase never works.
- **T3 — Structural propagation:** support is checked bottom-up — removing
  one support can leave a pipe far above hanging.
- **Branching:** with two exits the network is a tree with a `T` junction.

---

## File layout

```
type_2_3/
├── __init__.py
├── game_core.py            # PieceType, orientations, LevelSpec, GameState
├── mechanics.py            # place (orientation or open ends) / remove / submit; v2 rules
├── difficulty.py           # exhaustive network search, staircase baseline, tiers
├── generator.py            # random-walk route (+ T branch), supports, obstacles
├── levels.py               # category / tier configs, layout de-duplication
├── solver.py               # stored plan, else first solution of the search
├── loader.py               # JSON load/save
├── visualizer_plotly.py    # plotly + kaleido renderer (gems, brass flanges)
├── play.py                 # single-level (python -m type_2_3.play N)
├── play_batch.py           # 10-level easy/medium/hard batch
├── generate_50.py          # 50-puzzle diverse batch
├── images/                 # per-seed play.py output
├── images_batch/           # 10-level batch + SUMMARY.md
├── puzzles_50/             # 50 puzzles + SUMMARY.md + index.json
└── README.md               # you are here
```

---

## Visual encoding (v5)

| Element | Appearance |
|---|---|
| Pipe (I/L/T) | metallic dark-blue cylindrical body + **brass flange ring** at every connector tip + dark sphere joint + thin white glint |
| Entrance | **green gem core** inside a decorative **brass flange** (concentric ring + 4 accent dots + 4-point star emblem); direction-agnostic |
| Exit | same as entrance but **orange gem**; same flange + emblem |
| Obstacle | red block — **multi-cell shapes (1×1 or 1×2)** merge into one solid rectangle with scaled white X + warning stripes |
| Support S1 | slim warm-tan pillar (1 cell tall) with darker cap |
| Support S2 | slim warm-tan pillar (2 cells tall) with cap + mid-equator line (so you can see it's a single piece, not two S1 stacked) |
| Problem cell | bright red halo ring at leaking / unsupported cells (only after a FAILED submit) |
| Coordinates | `x=…` / `y=…` / `z=…` labels on the grid edges (shared `chain_viz`) |
| Raised cells | dashed drop line + floor footprint under portals, obstacles and pipes above z=1 |

Per render we also draw:

- **Game frame** (`chain_hud`, theme `pipefit`: teal accent, steel-plate board):
  title bar with PLAYING / WIN / LOSE badge, STATUS card (pieces placed and
  actions with progress bars, problems after submit), PARTS BIN card (left /
  total per piece), LEGEND card with a drawn icon per element, rules strip
- **Two camera angles** (`front` + `back`) via `render_state_pair`

Canvas is 1960 × 1540 and always frames the **full grid**.  (The old
content-fitted bbox was computed from `spec.meta["paths"]`, i.e. the
framing hinted at where the solution runs.)

---

## Programmatic API

```python
from type_2_3.generator import generate_level
from type_2_3.solver import solve
from type_2_3.game_core import PieceType, init_state
from type_2_3.mechanics import step_place, step_remove, submit

from type_2_3.levels import CATEGORY

spec, meta = generate_level(seed=3, tier="hard", **CATEGORY["C_branch"]["hard"])
state = init_state(spec)
plan = solve(spec)                  # supports bottom-up, then pipes
for p in plan:
    r = step_place(state, p.type, p.pos, p.orientation)
    assert r.code.value == "OK"
print(submit(state).code.value)     # WIN
step_place(init_state(spec), PieceType.PIPE_L, (3, 2, 2), ends=("+x", "+z"))
```

---

## Generator knobs

`generator.generate_level(grid_size, seed, n_exits, n_obstacles, extra_len,
spare, tier)` builds a biased random-walk route (plus a branch to a second
exit), fills the support columns it needs, puts obstacles on the staircase
shortcut, and sets the inventory to exactly those pieces plus `spare`.

`difficulty.solutions` enumerates **every** network that fits the
inventory (routes for two portals; route + T-junction branch for three).
Tiers (`difficulty.TIERS`, configs in `levels.py`):

| tier | solutions | min pipes | staircase shortcut | pieces to place |
|---|---|---|---|---|
| easy | ≥ 1 | ≥ 3 | may work | ≤ 10 |
| medium | ≤ 20 | ≥ 5 | fails | ≤ 14 |
| hard | ≤ 6 | ≥ 7 | fails | ≤ 18 |

---

## The 50-puzzle batch (`puzzles_50/`)

`python -m type_2_3.generate_50`: categories A_route, B_obstacles,
C_branch (two exits), D_tall (high exits), E_mixed — 3 easy + 4 medium +
3 hard each, distinct seed ranges, de-duplicated layouts, output folder
wiped first.  `SUMMARY.md` / `index.json` list the measured difficulty.
`level.json`'s meta (`plan`) and `solution.txt` contain the answer — keep
them out of model inputs.

---

## Iteration history & lessons learned

### Rules v2 — measured, then redesigned (2026-10)

An exhaustive audit found the v1 game mostly solved by one recipe ("step
toward the exit, climb, repeat; support every pipe"), two-exit levels won
without a `T` (reach one exit, or route through the other entrance), the
inventory computed from a plan that put supports inside pipe cells,
obstacles that never blocked the intended route, a solver that could not
solve the shipped levels, duplicate / stale puzzles, and on-screen rules
that did not match the code.  v2: one leak-free network joining every
portal, a plain "rest on something solid" support rule, pipes named by
their open ends, a single final submit, varied routes with obstacles on
the staircase, exact inventories, and tiers from an exhaustive search.
The v1 rounds below are kept for history.

The current build is the result of five rounds of user feedback.  Each
lesson is written so the next CHAIN task can skip the mistake.

### Round 1 — define the primitives first

`PieceType`, orientation enumerations, `ConnectorMask`, `LevelSpec`,
`GameState`, `ActionCode` all land in `game_core.py` before anything
else.  Every downstream module reads from here.  Getting this wrong
means rewriting solver + generator + renderer.

### Round 2 — solver before generator

A solver that works on hand-built levels tests mechanics fast.  The
generator uses the solver internally to verify feasibility, so solver
correctness must be established first.

### Round 3 — solver must replay the generator's chosen skeleton

Symptom: generator said 4 supports, solver placed 0.  Cause: solver ran
shortest-path BFS itself and picked a different same-length path with
no room for supports.  Fix: generator stores the chosen path in
`spec.meta["paths"]` and solver replays it.  **The solver should follow
the generator's skeleton, not re-derive it.**

### Round 4 — staircase paths are the sweet spot for support mechanics

Flat paths: 0 supports used (everything at z=1).
Vertical columns: 0 supports possible (pipe occupies its own column).
Staircase (alternating horizontal + vertical): every other pipe lands
on a free column, so supports become necessary.  Transferable rule: if
your task has a stacked axis, design the generator to alternate on it.

### Round 5 — obstacles adjacent to the path, not ON it

Obstacles on the path: any 2 adjacent cells have a same-length parallel
path, so the "detour" is identical length and the obstacle is visual
noise.  Better: obstacles face-adjacent to the chosen path but NOT on
it.  The path stays valid and the obstacle looks like a real constraint
nearby.

### Round 6 — relax "anchored" to "no two adjacent unanchored"

Original rule: every pipe needs a support column from z=1.  Problem:
vertical pipe segments become impossible (pipe column blocks support
column).  Fix: allow a single unanchored pipe dangling between anchored
ones.  This makes the puzzle about **choosing** which cells to skip
support on, not uniformly stacking.

### Round 7 — visual quality is task quality

Users surfaced concrete problems round-by-round:

| Concern | Fix |
|---|---|
| "pipes don't look like pipes" | thicker bodies + brass flange ring at every connector tip + dark joint sphere + white glint |
| "too many red/amber small circles" | removed amber unanchored halos; kept only bright-red halo for the *violation* case |
| "entrance/exit imply a top-down flow" | redesigned both as direction-agnostic gem + flange + star emblem (same shape, different colour) |
| "obstacles are tiny 1×1 cubes" | added multi-cell shapes (1×1 / 1×2) rendered as one merged block |
| "only 1 entrance and 1 exit" | `n_entrances ∈ {1,2}`, `n_exits ∈ {1,2}` — generator sprouts branches and solver emits T-junction pieces automatically |

Each of these was one edit in the visualizer or generator, not an
architectural change.  **Iterate against rendered output, not code.**

### Round 8 — placement must be physically plausible

Originally `step_place` only checked cell-free + inventory.  Users
noticed the solver could place a pipe in mid-air and add supports
later.  Fixed:

- Supports need (x, y, z-1) = ground / support / pipe / portal.
- Pipes need at least one face-neighbour already placed (or be at z=1).
- Added `ActionCode.NO_PHYSICAL_SUPPORT`.

Solver also reorders its plan **foundation-first**: walk the path
entrance → exit, emit support columns before the pipe that needs them.
So `replay(plan)` always succeeds step-by-step via `step_place`, which
matches a human builder's experience.

### Round 9 — Read-tool image caching gotcha

Claude Code's `Read` tool caches PNGs by path.  After any renderer
change, write to a new directory (or include a nonce in the filename)
before reading back — otherwise you'll keep seeing the stale preview
and think your changes didn't take.

### Round 10 — counterfactual frame per level

Every batch level renders a counterfactual: "what happens if we remove
one support after winning?"  This surfaces the violation rendering
(bright red halo) and gives the model concrete examples of a failure
mode — diagnostically invaluable.

---

## Reusable template for the next CHAIN task

What to reuse from `type_2_3/`:

- **Path-first generator**: BFS to compute a skeleton, then assign
  piece types + supports.  Store the skeleton in `spec.meta` so the
  solver replays it.
- **Staircase generator** for any task with a stacked axis.
- **Off-skeleton obstacle placement** (adjacent to path, not on it).
- **Multi-portal via branch-then-merge** (one extra portal = one branch
  off the skeleton; junction cell becomes a 3-connector fork).
- **Free-undo action API**: `place / remove / submit`.
- **Placement-time physical check** so the plan is foundation-first.
- **Embedded panels** (STATUS / INVENTORY / legend) on every render.
- **Rectangular bbox + manual aspect ratio** (never force cube).
- **Counterfactual frame** in batch scripts.
- **Violation-only halo rings** (no "warning" halos for normal state).

What you'll re-derive per task:

- Specific piece catalog + orientation enumeration.
- Which cell roles exist (entrance/exit vs source/target).
- The invariant to check at submit time.
- The difficulty triple-lock (adapt from T3 / T2_3).

---

## License

MIT — same as the rest of the repo.
