# FluidNet 3D — Design Document
**CHAIN Benchmark · Type 3 Structural Propagation**

---

## 1. Why this game — the T3 hypothesis

CHAIN argues that today's VLMs can see physical structure, but break down when
a local action has *non-local* consequences that must be reasoned about
**before** committing.  Type 3 tasks test exactly that:

> A local action (opening one valve) triggers a cascade of state change
> (fluid floods a connected sub-network) that reaches cells many hops away.

Compared to the two already-implemented families:

| Type | Already have | What it tests |
|---|---|---|
| T1 Constraint Gating | Burr Puzzle | Order-of-unblocking ("who traps whom") |
| T2 Spatial Commitment | Stacking | Irreversible space consumption (delayed dead-ends) |
| **T3 Structural Propagation** | **FluidNet 3D** | **Local → non-local cascade through a physical graph** |

FluidNet exposes T3 in its purest form: a pipe network + a valve action.  Each
valve is local (a single cell), each propagation is global (BFS through a
graph), and the agent must **predict the entire flood-fill** before
committing.

---

## 2. Rules in one page

The grid is an `A × B × C` box (default `5 × 5 × 5`).  Each cell has one of
six types:

| Type | Meaning | Visual |
|---|---|---|
| EMPTY | no pipe | absent |
| PIPE | conducts fluid | small gray cube |
| SOURCE | infinite fluid origin, colored | large colored cube |
| TARGET | must be filled with matching color | hollow colored cube |
| VALVE | blocks fluid until opened (irreversible) | yellow cube |
| PROTECTED | fluid must **not** enter | red cube |

Each turn the agent picks a closed VALVE and opens it.  Then fluid propagates:

1. Multi-source BFS from every SOURCE through face-adjacent
   PIPE / TARGET / open-VALVE cells (each color independently).
2. If the BFS would enter a PROTECTED cell → **PROTECTED_BREACH** (loss).
3. If color A's BFS reaches a TARGET of color B ≠ A → **WRONG_COLOR_TARGET** (loss).
4. If two colors meet on the same PIPE / VALVE cell → **COLOR_CONTAMINATION** (loss).
5. Otherwise the game continues.  If every TARGET holds its matching color → **WIN**.

Extra terminal: **NO_PROGRESS_DEADLOCK** — if no remaining closed valve, when
opened, would fill any missing target without violating a rule, the game is
unwinnable and ends.

**Irreversibility.**  Opening a valve is one-way.  Fluid, once it enters a
cell, stays (monotone with respect to the valve set).  The agent cannot
recover from a bad commitment.

---

## 3. Why this is a good T3 benchmark

- **Propagation locus is the whole network.**  A valve on the bottom face can
  send fluid to the top corner of the grid.  The agent must mentally BFS
  through a 3D graph rendered as a picture.
- **Coupling is non-trivial.**  Two valves on the same branch are redundant;
  two valves on different branches decide which distant leaf gets filled.
  Some valves are *dangerous* — opening them alone guarantees a PROTECTED
  breach regardless of what else is open.
- **Planning depth is tunable.**  `min_valves_needed` in the generator meta
  controls the shortest valid opening sequence; `n_safe_subsets` controls how
  many distinct solutions exist (lower = harder).
- **Pure visual observability.**  3D structure is only useful from a rendered
  view; a flat text representation collapses the spatial relations that drive
  propagation.

---

## 4. Data model

See `game_core.py`.  Summary:

```
LevelSpec {
    grid_size: (A, B, C)
    cells: { (x, y, z) -> Cell(pos, type, color?) }   # sparse, no EMPTY
    sources / targets / valves / protected: cached lookups
    colors: sorted list of distinct fluid colors
}

GameState {
    spec: LevelSpec
    opened_valves: Set[(x, y, z)]
    fluid_cells:   Dict[(x, y, z) -> color]
    move_history:  List[(x, y, z)]
    terminal, terminal_code, violation_cell
}
```

One public action: `step_open_valve(state, (x, y, z)) -> ActionResult`.

---

## 5. Level generation (generator.py)

Each level is produced by rejection sampling.  The generator:

1. **Grows a random tree** of pipe cells from a random anchor.
2. **Labels** the tree: 1 SOURCE, `n_targets` TARGETs, `n_valves` VALVEs,
   remaining nodes become plain PIPEs.
3. **Sprouts** `n_protected` PROTECTED cells as single-cell branches off the
   main tree.
4. **Verifies** by brute-force enumeration over all `2^|valves|` subsets:
   - At least one safe subset exists (solvable).
   - The smallest safe subset has size ≥ 1 (forces actual reasoning).
   - Opening **all** valves is unsafe (prevents brute-force bypass).
   - At least one valve is dangerous (appears in no safe subset — a trap).
5. **Records meta**: `min_valves_needed`, `n_safe_subsets`,
   `dangerous_valves`, and a `min_safe_example` sequence for verification and
   debugging.

Brute force is fine while `|valves| ≤ 10`.  For larger levels, replace the
enumeration with a BFS over the valve-subset lattice and prune aggressively.

---

## 6. Propagation (propagation.py)

The whole engine is a single multi-source BFS, rerun from scratch after each
valve open.  The implementation (`_compute_full_fluid`) keeps a single
`cell_color` dict; conflicts on TARGET / PROTECTED / PIPE cells record a
violation flag that the wrapping `step_open_valve` converts into the correct
`ActionCode`.

Because fluid state is pure function of the open-valve set + level spec,
propagation is deterministic and order-independent for the final state.
Order *does* matter for which violation gets reported first (the BFS front
advances outward from sources in layers, so the nearest conflict surfaces).

---

## 7. Visual encoding

Rendered via matplotlib 3D (`visualizer_3d.py`).

- SOURCE / filled TARGET: solid colored cube with dark outline.
- Empty TARGET: low-alpha colored shell with dashed outline.
- PIPE: small gray cube (shrinks when empty, grows & recolors when fluid flows through).
- VALVE closed: full-size yellow cube.  VALVE open: yellow outline + fluid color.
- PROTECTED: red cube; on breach the cube becomes darker red with thicker outline.

The view is chosen so that `+X` points right, `+Y` points into the page, and
`+Z` points up.  The `layer <z>` CLI command prints horizontal 2D slices for
sanity-checking.

---

## 8. Difficulty knobs

| Knob | Effect | Typical range |
|---|---|---|
| `grid_size` | Overall board size | 4×4×4 … 6×6×6 |
| `n_targets` | Number of targets to fill | 1 … 4 |
| `n_valves` | Branching complexity | 3 … 8 |
| `n_protected` | Red-zone density | 1 … 4 |
| `min_valves_needed` (derived) | Shortest solution length | 1 … |valves| |
| `n_safe_subsets` (derived) | How many solutions exist | 1 (hard) … many (easy) |
| `# dangerous_valves` (derived) | How many traps | 0 (trivial) … |valves|/2 |

---

## 9. Interaction protocol (agent-facing)

The agent's observation at turn `t` is a rendered 3D image plus a textual
state digest listing:
- grid size, colors
- each TARGET position with fill state (`filled` / `empty`)
- each VALVE position with open/closed state
- PROTECTED positions
- move history

The agent emits a single action:
```
open(x, y, z)
```
The environment returns:
```
{ code, new_fluid_cells, newly_filled_targets, violation_cell?, terminal? }
```

A canonical episode is a sequence of such actions until any terminal code
(WIN, PROTECTED_BREACH, WRONG_COLOR_TARGET, COLOR_CONTAMINATION,
NO_PROGRESS_DEADLOCK).

---

## 10. Evaluation metrics

Per episode:
- **Success** — WIN reached.
- **Dangerous-valve touches** — number of opens that would immediately lose
  when counted individually; counts even if overridden by other opens.
- **Valve efficiency** — `valves_opened / min_valves_needed`; 1.0 is optimal.
- **Breach distance** — when a PROTECTED_BREACH occurs, graph distance from
  the opened valve to the violating cell (tests how far-field the reasoning
  lapse was).
- **First-move quality** — was the first opened valve in every safe subset?
  (Proxy for "did the agent identify structural bottlenecks before exploring".)

Aggregate across a test set of seeds and difficulty tiers.

---

## 11. Roadmap / extensions

- **Multi-color variant.**  The core engine already supports N colors; the
  generator currently emits only one.  Enabling multiple sources/targets with
  distinct colors adds COLOR_CONTAMINATION as a meaningful failure mode.
- **Orientable valves.**  Each VALVE has a connector type (I / L / T / +)
  chosen at opening time.  Agent picks both cell and orientation.
- **Pressure / flow direction.**  Fluid flows only along gradient pipes; some
  cells are one-way.  Requires a directed-graph BFS.
- **Puzzle packs.**  Serialize 100 levels per difficulty tier via
  `generator.generate_and_save` and load them like `puzzles_full_v9`.
