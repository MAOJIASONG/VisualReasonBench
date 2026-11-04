# FluidNet 3D — Type 3 Structural Propagation Game

A 3D fluid-pipe-network puzzle.  The agent opens valves one at a time; fluid
propagates by BFS through the connected pipe graph; the agent must fill every
target container without sending fluid into any protected cell.

This is the Type 3 (Structural Propagation) task of the CHAIN benchmark —
complementary to the T1 Burr Puzzle and T2 Stacking tasks already in the
repo.

---

## TL;DR — how to play

```bash
cd /mnt/hdfs/yuhao/VisualReasonBench/Stacking_scaling

# 1) Interactive REPL on a freshly generated level
python -m type_3.game_cli --seed 5 --size 5 5 5 --targets 2 --valves 6 --protected 3

# 2) Auto-play one level with the BFS solver, render every step
python -m type_3.play --seed 70

# 3) Generate 10 concept puzzles spanning easy → hard, render solver steps for each
python -m type_3.play_batch
#    -> output in type_3/images_batch/  (with SUMMARY.md)

# 3b) The 50-puzzle set: 5 concepts x (3 easy + 4 medium + 3 hard)
python -m type_3.generate_50
#    -> output in type_3/puzzles_50/  (SUMMARY.md, index.json)

# 4) Visual style bakeoff (5 rendering styles for the same state)
python -m type_3.render_variants
#    -> output in /tmp/fluidnet_variants/<style>/
```

Inside the interactive REPL:

```
fluidnet> state                 # show current fluid / target / valve status
fluidnet> valves                # list every valve and whether it's open
fluidnet> layer 1               # 2D top-down slice at height z=1
fluidnet> render                # save a 3D render to /tmp/fluidnet_...png
fluidnet> open 3 4 1            # open the valve at cell (3, 4, 1)
fluidnet> quit
```

---

## The rules in 60 seconds

- The board is an `A × B × C` grid of cells: **empty / pipe / source /
  target / valve / protected**.  The pipe network is drawn as tubes: two
  cells are connected exactly when a tube joins them (it may contain loops).
- The **source** (big sphere on a pump) floods everything connected to it.
  Some levels have **two pumps** (blue and orange).
- **Target** tanks (glass boxes) must all fill, each with the colour of its
  frame.  Two colours meeting anywhere, or a tank reached by the wrong
  colour, loses.
- **Protected** cells (red, white X) are fed by a short stub; if fluid
  reaches the cell next to one, you lose.
- **Valves** (amber, hand-wheel) start closed.  Each turn you open one;
  opening is permanent and the fluid immediately floods everything newly
  connected.
- **Win** when every tank is full.  **Lose** only when fluid visibly
  reaches a protected cell.  (There is no hidden "deadlock" loss any more:
  opening a still-dry trap changes nothing until fluid arrives behind it.)

---

## What the game tests

**Structural-propagation reasoning** — the T3 axis of the CHAIN benchmark.
The action (open one valve) is local; its effect is everything newly
connected behind that valve.  Each level is built around one concept
(`concepts.py`):

| concept | what the player has to see |
|---|---|
| `trace` | a valve far from any red cell can still feed one: trace each pipe to its end |
| `series` | the tank sits behind a row of valves: open the whole row, none of the side spurs |
| `backdoor` | the route with the fewest valves also reaches a red cell through a loop: take the clean detour |
| `two_pumps` | each tank wants its own colour and the colours must never meet: a valve that is safe alone can lose once the other colour arrives |
| `mixed` | two pumps plus red cells: both kinds of trap at once |

Design rule for every level: **each valve is a real decision** — it is in
the unique minimal winning set, or adding it to that set loses.  There are
no filler valves.  Pumps, tanks, valves and red cells keep at least one
empty cell between each other (also diagonally) so none hides another.

---

## File layout

```
type_3/
├── __init__.py
├── game_core.py            # CellType, Vec3, Cell, LevelSpec, GameState, ActionResult
├── propagation.py          # multi-source BFS flood-fill + step_open_valve()
├── generator.py            # snake-path backbone generator + 2^|V| solvability check
├── solver.py               # BFS over valve-subset lattice — shortest winning plan
├── loader.py               # JSON load/save of levels
├── visualizer_plotly.py    # main renderer (plotly + kaleido, headless, 1400×1100;
│                           #   grid coordinates + floor guides via ../chain_viz.py)
├── visualizer_3d.py        # legacy matplotlib renderer (kept for reference)
├── render_variants.py      # 5-style bakeoff (voxel/labeled/schematic/triview/plotly)
├── game_cli.py             # interactive REPL  (python -m type_3.game_cli)
├── play.py                 # single-level auto-play with solver  (python -m type_3.play)
├── play_batch.py           # 10-level batch with solver steps    (python -m type_3.play_batch)
├── example.py              # minimal scripted demo
├── images/                 # canonical play.py output (one level, two angles per step)
├── images_batch/           # 10-level batch output  +  SUMMARY.md
├── task.md                 # design document
└── README.md               # you are here
```

---

## Programmatic API

```python
from type_3.generator   import generate_level
from type_3.loader      import create_game_state, save_level_to_json
from type_3.propagation import step_open_valve
from type_3.solver      import order_by_flow, solve

# 1. Generate a playable level in a measured tier.
spec, meta = generate_level(
    grid_size=(5, 5, 5),
    n_targets=2,
    n_valves=6,
    n_protected=3,
    seed=5,
    tier="medium",
)
print(meta["difficulty"])    # min_plan, traps, P(random/frontier/greedy)

# 2. Inspect what the level looks like.
print(spec.sources, spec.targets, spec.valves, spec.protected)

# 3. Shortest winning valve sequence, in flow order.
plan = order_by_flow(spec, solve(spec))

# 4. Play by calling step_open_valve().
state = create_game_state(spec)
for v in plan:
    result = step_open_valve(state, v)
    print(result.code)       # ActionCode.OK / WIN / PROTECTED_BREACH / ...
    print(result.newly_filled)
print(state.terminal_code)   # ActionCode.WIN

# 5. Persist the level.
save_level_to_json(spec, "my_level.json", meta=meta)
```

---

## Generator (V3 valve tree) and difficulty

`generate_level(grid_size, n_targets, n_valves, n_protected, seed, tier)`
grows the network as a **tree under a no-touch rule** (a new cell may touch
only its parent), so the drawn tubes are exactly the connections.  Targets
and protected cells are branch tips; every target path is gated by valves
(nothing is full at the start); every protected tip hangs behind its own
trap valve, placed far from the red cell.

`difficulty.analyze` enumerates every valve subset (exact numbers):

| metric | meaning |
|---|---|
| `min_plan` | fewest valves in a winning set (the minimal set is unique) |
| `traps` / `nonlocal_traps` | valves in no winning set / those not next to a red cell |
| `p_random` | win rate opening random valves |
| `p_frontier` | win rate of "open a wet valve not next to red" |
| `p_greedy` | win rate of "open the valve nearest an empty tank" |

Tiers (`difficulty.TIERS`, bands are disjoint):

| tier | min plan | non-local traps | frontier rule | greedy |
|---|---|---|---|---|
| easy | 1–2 | ≥ 0 | any | any |
| medium | 2–3 | ≥ 1 | ≤ 0.60 | any |
| hard | 3–5 | ≥ 2 | ≤ 0.35 | ≤ 0.35 |

The V2 snake generator measured 98 / 91 / 86 % frontier-rule wins on
easy / medium / hard, and up to 40 % of its levels started with a tank
already full (the folded snake touched itself); V3 measures about
64 / 15 / 3 % with no pre-filled tanks.

V3 levels are trees with one pump, and two stronger baselines win all of
them: `p_lookahead` (simulate one valve ahead, open one that adds fluid and
does not lose) and `shortest_ok` (open each tank's fewest-valve route).
The concept generator (`concepts.py`, used by `play_batch` and
`generate_50`) adds loops and a second pump so that, per concept:

| metric | meaning | concept that defeats it |
|---|---|---|
| `p_lookahead` | win rate of a one-valve-ahead simulator | `two_pumps`, `mixed` (combination traps) |
| `shortest_ok` | does the fewest-valve route per tank win? | `backdoor` |
| `filler` | valves neither needed nor fatal | all concepts require 0 |

Per-concept tier rules are in `concepts.CONCEPTS` (plan length, non-local
traps, frontier / greedy / lookahead ceilings, shortest route must fail).

On top of those, every level gets one number, `difficulty.score`:

    score = log2 1/p_frontier + 2 * log2 1/p_lookahead + (1 if the shortest route loses)

The tiers are **one ladder for the whole game** (`concepts.TIER_SCORE` plus
the `valves` ranges), not per concept: every easy level has fewer valves and
a lower score than every medium level, and every medium than every hard.
The score bands leave a gap, so no two tiers touch:

| tier | valves | score |
|---|---|---|
| easy | 2–4 | ≤ 3 |
| medium | 5–6 | 4 – 6 |
| hard | 9–10 | ≥ 7 |

The concept configs were tuned on a 2100-level sample so every (concept,
tier) pair reaches its rung.  backdoor hard is the rarest (~1 % of seeds),
so `generate_50` searches up to 1200 seeds per slot.

**Hard tier (2026-11).**  Hard levels are built with 10–12 valves and a
6-valve plan; `concepts.prune_filler` then turns every valve that is
neither needed nor fatal back into plain pipe (opening it never changed the
outcome), leaving 9–10 real decisions (trace / series get a fourth red
cell, two_pumps a 6 × 6 × 5 box).  The tier was calibrated against
gpt-6.1-sol (Codex, default reasoning effort, one-shot): the earlier 7–8
valve hard tier was solved about two thirds of the time, and over ~160
calibration games the solve rate fell with valves (7: 0.93, 8: 0.32,
9: 0.24) and with plan length (4: 0.50, 5: 0.32, 6: 0.04).  Most losses are
a valve coordinate misread from the 3D view or one valve of a route left
shut.  See `../chain_analysis/model_runs/gpt61sol_hard_calibration/`.
`images_batch/` still holds the earlier demo levels (`play_batch` would now
build the new hard ones).  A `score_band` entry in a
concept's params overrides the tier band for experiments.

`generate_50` and `play_batch` search seeds in parallel
(`../chain_search.py`); the picks equal a sequential search over the same
seeds.

---

## Visual encoding (V2 plotly renderer)

The default renderer is `visualizer_plotly.render_state` — pure Python,
headless via `kaleido==0.2.1`, output PNG at 1960 × 1540.

| Element | Appearance |
|---|---|
| Source | large glossy sphere on a dark pump pedestal with a light-blue halo ring |
| Target (empty) | translucent colored glass frame |
| Target (filled) | glass frame + colored fluid volume + white water-line |
| Pipe (empty) | dark-steel cylinder (`#2F3640`) — **one** unified color |
| Pipe (with fluid) | colored cylinder matching the fluid color |
| Valve (closed) | amber cube + dark hand-wheel + spokes |
| Valve (open) | colored collar fused to the pipe + brass ring on top |
| Protected | red cube with white **X** mark + warning stripes; a short tube stub shows which pipe feeds it |
| Protected (breached) | dark-red cube with white star-burst cracks |

Every render also includes:

- **Game frame** (`chain_hud`, theme `fluidnet`: ocean-blue background,
  tiled board): title bar with a PLAYING / WIN / LOSE badge, STATUS card
  (targets filled and valves opened with progress bars, protected
  breached), LEGEND card with a drawn icon per element, rules strip, so the
  image is self-contained.
- Grid coordinates (`x=…`, `y=…`, `z=…`) and floor drop lines under the
  key cells (shared `chain_viz`).
- **Two camera angles** per state (`front` and `back`) via `render_state_pair`,
  so anything occluded in one view is visible in the other.
- The **full grid** is always framed (a content-fitted crop would hint at
  where the action is).

---

## Example playthrough (`python -m type_3.play --seed 70 --tier hard --valves 7`)

```
source:    (2, 5, 1)
targets:   (1, 1, 5), (4, 4, 4)
valves:    (1,2,5) (1,3,2) (1,5,3) (1,5,5) (2,3,4) (2,4,5) (3,5,2)
protected: (1,4,1) (3,2,4) (3,4,2)
difficulty: min plan 4 (unique), 3 traps (2 non-local),
            P(random)=0.03  P(frontier)=0.02  P(greedy)=0.00

> open 1 5 3   -> OK
> open 1 5 5   -> OK
> open 1 2 5   -> OK, tank (1,1,5) full
> open 2 4 5   -> Win, tank (4,4,4) full

# Mistake shown in the counterfactual frame:
> open 1 3 2   -> ProtectedBreach at (1, 4, 1)
```

Plans are replayed in flow order (each valve opened next to fluid that is
already flowing), so every frame shows progress.

---

## Why this belongs in CHAIN

The CHAIN benchmark tests causally-structured long-horizon physical planning.
The three task families isolate different coupling mechanisms:

| Family | Existing task | What the agent has to predict |
|---|---|---|
| T1 Constraint Gating | Burr Puzzle | Which pieces unblock which |
| T2 Spatial Commitment | Stacking | Where early placements close off later ones |
| **T3 Structural Propagation** | **FluidNet 3D** | **Where fluid flows through the full graph** |

FluidNet gives T3 its own minimal, purely-visual test: the graph is rendered,
the action is a single cell, and the outcome requires running a BFS mentally.

See `task.md` for the full design rationale.

---

# Iteration history & design lessons

The current state of FluidNet 3D is the result of ~10 rounds of feedback,
mostly about **visual quality** and **task interpretability**.  Recording the
iteration arc here so other CHAIN tasks (T1 Rush Hour 3D, T2 Snake Cube, …)
can skip the same mistakes.

## Round-by-round arc

| Round | What broke | Fix |
|---|---|---|
| 1 | Game logic only — no images | Add matplotlib 3D renderer + BFS solver + auto-play |
| 2 | "Looks too AI / not aesthetic" | Cylinders for pipes, spheres for source, glass boxes for targets |
| 3 | matplotlib hits a quality ceiling, no GPU/X11 | Switch to **plotly + kaleido 0.2.1** (headless WebGL-quality) |
| 4 | One angle hides occluded cells | Render every state from **two opposing cameras** (`front` / `back`) |
| 5 | "Worse than before" | Run a **5-style bakeoff** (`render_variants.py`); pick `B_labeled` |
| 6 | Labels overlap, lots of whitespace | Outward label offsets + tight margins + bigger canvas |
| 7 | Entities visually merge | Shrink every primitive (valve 0.6→0.45, pipe 0.18→0.13), bolder edges |
| 8 | "Too much wasted space, can't tell how to play, orphan red zones" | (a) `_spread_ok` filter on tree, (b) embedded **legend + STATUS panel**, (c) **protected must be valve-adjacent**, (d) fix `_is_deadlocked` to consider any continuation |
| 9 | "Pipes too short, two pipe colors confuse, planes empty" | Replace tree-growth with **DFS snake-path backbone**, **unify pipe color**, **rectangular bbox** with normalised aspect ratio |
| 10 | Density too high after the snake-path change | Cap backbone length at ~27 % of grid volume |
| 11 | Measured audit: up to 40 % of levels started with a tank full (the folded snake touched itself), the "open a wet valve not next to red" rule won 86–98 %, half of all losses were an invisible "deadlock" | **V3 valve tree** (no-touch growth, tips = tanks / red cells, far-away traps), tiers from exact measurement, no hidden deadlock loss, prominent source, stub into red cells, breaching counterfactual |
| 12 | Measured audit: every V3 level is won by "simulate one valve ahead" and by "open the fewest-valve route" (trees only test path tracing) | **Concept generator**: loops (`backdoor`), two pumps (`two_pumps`, `mixed`), no filler valves, spacing between big pieces, 50-puzzle set `puzzles_50/` |

## Recurring user concerns (the "what they actually care about" list)

1. **Visual quality is task quality.**  T3 tests "can you read this graph?",
   so a bad render = a degenerate task.  Treat the renderer as part of the
   benchmark, not as polish.
2. **The image must be self-explanatory.**  Embedded legend + STATUS panel
   so a single PNG plus one paragraph of text is enough to play.
3. **Spatial coverage matters.**  Cells must spread across the grid, not blob
   in one corner — both for visual clarity and for puzzle interest.
4. **Visual noise kills clarity.**  Two shades for the same conceptual thing
   (e.g. dry-pipe joint vs dry-pipe tube) is read as two different things.
   One color per role.
5. **Iterate against rendered output, not code.**  After every change, look
   at the actual PNG.  Don't trust intuition about what a code change will
   look like.

## Hard-won lessons (transferable to T1 / T2 expansions)

### Headless 3D rendering picks

| Library | Headless OK? | Quality | Notes |
|---|---|---|---|
| matplotlib | ✅ | low | OK as last resort; cubes look amateurish |
| plotly + **kaleido 0.2.1** | ✅ | high | **Default choice.** Pin the old kaleido. |
| plotly + kaleido ≥ 1.0 | ❌ | — | Needs Chrome — incompatible with sandboxed envs |
| PyVista / VTK | ❌ | excellent | Needs X11 / Xvfb; usually not available |

### Procedural-generation principles

- **Tree-growth → blobs.**  Use a **DFS snake-path backbone** that prefers the
  candidate farthest from the path centroid.  Forces uniform coverage.
- **Volume target ≈ 25–30 %.**  Less than 20 % feels empty; more than 40 %
  reads as a wall of pipes.
- **Spread filter (≥ 75 % per axis).**  Reject any level whose cells don't
  span most of each axis.
- **Role-adjacency constraints are non-negotiable.**  Protected cells that
  don't touch the network are nonsense — fluid can never reach them.  Reject.
- **Difficulty triple lock.**  `min_plan ≥ 1` AND `open-all → unsafe` AND
  `at least one dangerous action` — without all three, the puzzle is trivial.

### Visual encoding principles (any 3D reasoning task)

- One concept → **one color**.  Joint vs tube of the same pipe must share a
  shade.
- **Smaller entities + bolder edges** beat large overlapping ones.
- **Rectangular bbox**, never forced cube.  Use plotly `aspectmode='manual'`
  with the aspect ratio normalised so its max component = 1.  This crops
  empty planes and keeps the camera distance predictable.
- **Two opposing camera angles** > one perspective + one top-down.
- **Embedded legend + STATUS panel** beat external markdown explanations.

### Game-mechanics gotchas

- The `_is_deadlocked` check must look at **any continuation** of the action
  set, not just one step.  A solver-found plan can have setup moves that
  individually don't fill any target.
- Don't render a "WIN" status from inside the propagation function — let the
  outer loop classify the terminal code.
- Solver and player share the same propagation, but solver enumerates
  subsets while player commits one move at a time.  Validate that every
  prefix of a solver plan is non-violating.

### Workflow gotchas

- Read-tool image previews **cache**.  Use unique filenames or write to a
  fresh directory after every rebuild.
- For comparison sweeps, **stitch into a single grid image** so the user (or
  you) can see all variants in one view.
- After any visualizer change: render `seed=70` once and look at it.  Don't
  ship without inspecting.

## Reusable template for the next CHAIN task

```
type_X/
├── game_core.py            # data structures + ActionCode enum
├── propagation.py          # OR mechanics.py — single step(state, action) function
├── solver.py               # BFS over the action lattice + difficulty meta
├── generator.py            # backbone-style construction + 3-stage filter
│                           #   (spread → role-adjacency → safety)
├── loader.py               # JSON load/save
├── visualizer_plotly.py    # plotly + kaleido, unified palette, legend, status
├── play.py                 # one level + solver + dual cameras
├── play_batch.py           # 10 levels × 3 difficulty tiers
├── render_variants.py      # 5 rendering styles for the bakeoff
├── images/                 # canonical play.py output
├── images_batch/           # play_batch.py output + SUMMARY.md
├── task.md
└── README.md
```

### What to reuse verbatim
- The plotly + kaleido 0.2.1 setup (and the pinned version)
- 1400 × 1100 canvas, rectangular bbox, normalised manual aspect ratio
- Legend strip + STATUS panel layout
- Dual `front`/`back` camera config
- One-color-per-role discipline
- BFS subset-lattice solver pattern (when the action space is enumerable)
- Three-stage generator filter: spread → role-adjacency → safety
- Difficulty triple lock: `min ≥ 1` / `open-all unsafe` / `≥ 1 dangerous`

### What you'll re-derive per task
- The specific snake-path / chain / piece-placement generator
- The action-space shape (subsets here; permutations for stacking; sliding
  states for Rush Hour)
- The role-adjacency rule (here: protected ⊥ valve)
- The terminal-code enum (per task)

## Recommendations for CHAIN as a whole

1. **Unified rendering stack.**  All CHAIN tasks should share a single
   `chain_viz` package with the same palette, legend, and STATUS panel,
   so cross-task comparisons are visually fair.
2. **Unified action API.**  `step(state, action) -> ActionResult` for every
   task; `solve(spec) -> List[Action]` for every task.  Lets the same
   evaluation harness drive all of them.
3. **Unified difficulty metadata.**  `min_actions_needed` /
   `n_safe_subsets` / `dangerous_actions` / `min_safe_example` apply to
   T1, T2, T3 alike — use them for cross-task difficulty calibration.
4. **Always include a counterfactual frame.**  "Open the dangerous valve"
   here, "remove the wrong piece" for stacking, "slide the wrong block"
   for Rush Hour.  Diagnostically invaluable.

---

## License

MIT — same as the rest of the repo.
