# Calibrating the hard tier against gpt-6.1-sol

Goal (2026-11-03/04): make the hard tier of `<game>/puzzles_50/` solvable by
OpenAI's gpt-6.1-sol about 20–30 % of the time, while every level stays a
proper puzzle (one winning answer, every element meaningful, exact solver
check).  The FluidNet and Domino hard levels were regenerated with harder
settings; PipeFit's hard tier already sat in the band and is unchanged.
Easy and medium levels are byte-identical to before.

## Protocol

* Model: `gpt-6.1-sol` through the Codex CLI (codex-cli 0.166.0), reasoning
  effort **low** (Codex's default for this model; chosen as the reference
  setting).  Runner: `../gpt6luna_oneshot50/run_codex_oneshot.py`
  (`EFFORT=low`), prompts from `../gpt6luna_oneshot50/make_codex_prompts.py`.
* One-shot: the model sees only the start images (front + back) and answers
  with its whole plan; `play_cli plan` plays it through the game rules.
  One run per level.  Codex runs read-only in an empty folder; the event
  logs of the final and confirmation runs hold one answer per level and no
  command or tool call.
* Two protocol fixes made during calibration, applied to every game and tier:
  Domino now shows a back view as well (on large stepped boards heights were
  ambiguous from the front alone; with it, a depth-3 pilot went from 7/10 to
  10/10 at high effort), and the Domino rules say explicitly that each patch
  is one `place` action at its base cell (one model had placed every patch
  twice, at z and z + 1).

## Result

| game | previous hard tier, effort high | previous hard tier, effort low | new hard tier, effort low | confirmation run |
|---|---|---|---|---|
| FluidNet | 15 / 15 | 9 / 15, 11 / 15 | **4 / 15** | 4 / 15 |
| Domino | 15 / 15 | 14 / 15 | **3 / 15** | 3 / 15 |
| PipeFit | 14 / 15 | 4 / 15, 5 / 15 | (unchanged) **5 / 15** | 3 / 15 |
| all | 44 / 45 = 0.98 | 27 / 45 = 0.60 | **12 / 45 = 0.27** | 10 / 45 = 0.22 |

Provenance of the numbers on the new tier (all at effort low; each run's
sessions name exactly the current hard level folders, and every level file
is older than the session that played it):

| run | levels | wins |
|---|---|---|
| FluidNet final (exported here) | the 15 FluidNet hard levels now in `type_3/puzzles_50/` | 4 / 15 |
| Domino final (exported here) | the 15 Domino hard levels now in `type_3_1/puzzles_50/` | 3 / 15 |
| PipeFit (exported here) | the 15 PipeFit hard levels (unchanged) | 5 / 15 |
| confirmation | all 45 of the above | 10 / 45 |
| third run, after the audit and review | all 45 of the above | 12 / 45 (FluidNet 6, Domino 3, PipeFit 3) |
| superseded FluidNet candidate | an earlier regeneration with 5–6 valve plans, since replaced (CSV set `fluid_final_plan56`) | 6 / 15 |

Over the three runs on the new tier: FluidNet 14/45 (0.31), Domino 9/45
(0.20), PipeFit 11/45 (0.24), all 34/135 = 0.25.  The bold column is the run exported here (`sessions/`,
`traj_<game>.jsonl`, `results_<game>.jsonl`); the confirmation run is in
`calibration_runs.csv` (sets `new_hard_confirm`, `new_hard_run3`).  Every game's level set still passes
`python -m chain_analysis.verify_levels`.

### Checks on the new levels

* `python -m chain_analysis.verify_levels type_3/puzzles_50 type_3_1/puzzles_50 type_2_3/puzzles_50`:
  150 levels, 0 problems (each reference plan replays to a win).
* `python -m chain_analysis.audit_hard`: all 45 hard levels (the 30 new
  FluidNet / Domino ones and the 15 PipeFit ones) pass every logic check.  FluidNet: one minimal winning valve set,
  every valve needed or fatal, no tank full at the start, parts never
  touching, every pipe joined to a pump, every tank and red cell reachable.
  Domino: one winning (patches, push) using all patches and none with
  fewer, every patch load-bearing and standing on the floor or a step, the
  solution topples every goal and no red piece, every red piece ends a decoy
  line (one wrong extra patch topples it), every raised piece stands on a
  step.  PipeFit: the stored plan wins, 1–12 networks fit, the staircase
  shortcut fails, every obstacle matters, the stock is the plan plus at most
  one spare, portals sit on the walls.
* `python -m chain_analysis.audit_hard --selftest` plants 28 defects into
  copies of 9 real hard levels (stray pipe, decorative red cell or piece, a
  trap valve made open pipe, a spare patch, a fixed domino removed, a step
  removed from under a piece, an obstacle that changes nothing, surplus
  stock, an obstacle on the answer route) and the audit flags all 28.
* Contact sheets of all 45 hard boards (front view):
  `hard_boards_fluidnet.jpg`, `hard_boards_domino.jpg`,
  `hard_boards_pipefit.jpg`; every board renders all its parts (pumps,
  tanks, valves, red cells / START, goals, red pieces, steps / portals,
  obstacles).
* No level changed after the scored runs: each run's sessions name exactly
  the current hard folders and every level file predates its session.
* All 210 easy / medium level files (`level.json`, `solution.txt`) are
  identical to git HEAD.
* `REVIEW.md` (`python -m chain_analysis.review_hard`): one entry per hard
  level with its parts, the answer and what every wrong move does, all
  computed through the game rules, for a level-by-level sign-off.  Read in
  full: all 45 answers win; every FluidNet trap valve loses with the answer
  open (breach, or colour mixing on two-pump levels); each Domino stop point
  of the answer's wave is either a decoy gap that topples exactly one red
  piece (32) or open floor past a goal (16, harmless and out of reach: the
  stock has no spare patch), and every wrong push direction stops dead;
  every PipeFit obstacle cuts the number of fitting networks.
* Domino P29_C_steps, checked against the game engine: the solution's wave
  runs START → (5,7) splitter → (5,4) splitter → (2,4,3) splitter → goal
  (2,8,2), every height change ±1, all 9 patches used; each of its 4 decoy
  gaps filled alone topples exactly its own red piece, and the 3 wrong push
  directions stop at a gap beside START.  FluidNet P18_series and
  P38_two_pumps were inspected in both views: every pipe visibly joined,
  valves, tanks and red cells unambiguous.

### What the new hard levels are

* **FluidNet** (`type_3/concepts.py`): 9–10 valves and a unique 6-valve
  plan.  Levels are built with 10–12 valves; `prune_filler` turns every valve
  that is neither needed nor fatal back into plain pipe, so each remaining
  valve is a decision.  trace / series get a fourth red cell, two_pumps a
  6 × 6 × 5 box.
* **Domino** (`type_3_1/levels.py`): a three-level splitter tree
  (`split_depth=3`; each fork leads on to a deeper splitter, a goal or a
  trap) on a 12 × 12 × 5 board with grey steps, 9–10 patches, 1–5 goals.
* **PipeFit**: unchanged (10–20 pieces, 3–12 solutions, the staircase
  shortcut blocked).

### Where gpt-6.1-sol fails

* FluidNet: a valve coordinate misread from the 3D view (`ValveNotFound`,
  the plan then misses a needed valve), or one valve of a route left shut;
  a few breaches / colour mixes.
* Domino: one patch at the wrong height on a stepped column
  (`CellOccupied` on a step, `NoPhysicalSupport` in the air); every other
  patch of those plans is right.
* PipeFit: a leaking joint or a pipe in a portal's own wall cell
  (`EntranceExitOccupied`), sometimes an unsupported pipe.

## How the settings were found

All calibration games (pilots and baselines, both efforts) are in
`calibration_runs.csv` (one row per game: set, effort, level, outcome,
first rejection, level features).  Pilot level files lived in a scratch
folder and are not kept; the CSV keeps their measured features.

* At effort **high** gpt-6.1-sol saturates Domino and PipeFit: Domino pilots
  with splitter trees, steps or extra interacting line segments scored
  7/8 to 14/14, PipeFit pilots up to 30 support-heavy pieces 3/3, 10/10 and
  9/9.  Only much larger combined Domino boards (13 × 13, depth-3 tree +
  extra segments + steps) fell to 3/10.  FluidNet with loops and two
  colours fell to 0.4–0.6.
* At effort **low** the response is steep.  FluidNet, pooled over ~160
  games: 7 valves 0.93, 8 valves 0.32, 9 valves 0.24 solved; shortest plan
  of 4 / 5 / 6 valves: 0.50 / 0.32 / 0.04.  A 9–10 valve tier with 5–6 valve
  plans scored 6/15 twice, so the plan was fixed at 6.  Domino with depth-3
  trees alone scored 6/10; with steps on every category 4/15 and then 3/15.

## Caveats

* One sample per level and model setting; 15 levels per game give a
  standard error of about ±0.11 per game, ±0.07 for the whole tier.
* Calibrated to one model at one effort: at effort high the same tier is
  much easier for gpt-6.1-sol (PipeFit 14/15 on the unchanged levels).
* The Claude Sonnet and gpt-6-luna runs in `../sonnet_hard50/`,
  `../sonnet_oneshot50/` and `../gpt6luna_oneshot50/` were measured on the
  previous hard tier (git `de130a7`).
