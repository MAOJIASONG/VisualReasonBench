# chain_analysis — step-value and search-budget analysis for the CHAIN games

Analysis code for FluidNet (`type_3`), Domino (`type_3_1`) and PipeFit
(`type_2_3`).  Everything runs on the generated level folders
(`puzzles_50/`, `images_batch/`).  Run from `Stacking_scaling/`.

| file | what it does |
|---|---|
| `envs.py` | one pure, hashable-state interface over the three games + an exact oracle: N(s) (winning end states still reachable) and d(s) (fewest actions to win, counting the removals a wrong piece needs) |
| `verify_levels.py` | replays every level's reference plan through the game rules and checks the oracle agrees |
| `review_hard.py` | per-level review sheet of the hard levels: parts, answer and the effect of every wrong move |
| `audit_hard.py` | logic audit of every hard level (unique answer, every valve / patch / obstacle meaningful, no decorative piece); `--selftest` plants defects and checks they are caught |
| `exp1_step_value.py` | Experiment 1: dead ends, consequence delay, solution retention V, distance change Δd |
| `exp2_budget_search.py` | Experiment 2: random / greedy / greedy+undo / DFS / BFS under the same verify and action budget |
| `play_cli.py` | play of one level by an outside agent (a model or a person), closed loop (`act`, an image after every action) or one-shot (`plan`, the whole action list in one call); records the trajectory and exports it for both experiments |
| `RESULTS.md` | baseline results and every model run so far |
| `model_runs/` | model runs: sessions, trajectories, Exp 1 / 2 reports (index in `model_runs/README.md`) |

```bash
python -m chain_analysis.verify_levels type_3/puzzles_50 type_3_1/puzzles_50 type_2_3/puzzles_50
python -m chain_analysis.audit_hard [--selftest]
python -m chain_analysis.exp1_step_value --root type_3/puzzles_50 [--traj runs.jsonl]
python -m chain_analysis.exp2_budget_search --root type_3/puzzles_50 [--models models.jsonl]
```

Reports go to `<game>/analysis_exp1/REPORT.md` and `<game>/analysis_exp2/REPORT.md`
(plus `steps.jsonl` / `runs.jsonl` with every record).

## Actions and verifies

| game | actions | verify (can end the game) | undo |
|---|---|---|---|
| FluidNet | `["open", [x,y,z]]` | every valve that touches a red cell or fills the last tank | none — a dead end needs a reset |
| PipeFit | `["place", "I"/"L"/"T"/"S1"/"S2", [x,y,z], o]`, `["remove", [x,y,z]]`, `["submit"]` | submit | remove |
| Domino | `["place", [x,y,z]]`, `["remove", [x,y,z]]`, `["submit_placement"]`, `["push", "+x"]` | push (push budget is 1, so push = submit) | remove, before submit_placement |

`["reset"]` starts the level over (used inside a trajectory after a failed
verify).

## Experiment 1 — dead ends and step value

* **N(s)** counts distinct winning end states: FluidNet winning valve sets
  (the game stops at the first win), PipeFit pipe networks + support columns
  that extend the placed pieces, Domino (patch set, push direction) pairs.
  Capped at `--cap` (default 500, PipeFit's solution enumeration cap).
* **V(a) = N(s')/N(s)**; V = 0 means the move walked into a dead end.
  Each step also gets **V_rand** (mean V over the legal open / place / push
  moves in the same state) and **V_best** (max V).
* **Δd = d(s) − d(s')**; for an irreversible dead end d = d(s0) + 1 (start
  over); for PipeFit / Domino a wrong piece costs its removal.
* **fatal step** = first step with N(s') = 0; **consequence delay** = steps
  until the game exposes it (failed verify); **repaired** = a later removal
  brings N back above 0.
* Per level: dead-end moves offered along the reference plan (`traps at
  start`, `earliest trap step`, `plan steps with a trap`).

Trajectory input (`--traj`), one episode per line:

```json
{"model": "gpt-5.2", "level": "P07_backdoor_medium_seed13500", "actions": [["open", [2,3,1]], ["open", [4,3,1]]]}
```

`level` is the folder name inside `--root`.  Without `--traj` the script
analyses demo trajectories: the reference plan and one random rollout of
the candidate-move policy per level.

## Experiment 2 — same-budget search

Every baseline plays through the same actions as a model and is charged per
executed action (undo removals and post-reset replays included) and per
verify.  All baselines share one local move generator (`env.candidates`):
FluidNet = frontier rule (valves touching fluid first, valves next to red
last); PipeFit = extend the open pipe end nearest an unlinked portal, then
fill support columns bottom-up, then submit; Domino = fill holes between
standing pieces nearest START, then push toward a neighbour.  None of them
uses the oracle.

Each run gets a large budget (default 64 verifies / 2000 actions); the
verify and action counts at the first win give the success rate under every
smaller budget.  `--models` takes model results (`{"model", "level", "win",
"verifies", "actions"}` per line) and adds the verify budget each baseline
needs to match each model.

## Running a model

`play_cli.py` keeps the game state in a session folder; the player only
sees the rendered images (`obs_<n>_front.png` and `obs_<n>_back.png`; Domino
had the front view only before 2026-11-04, when stacked steps on the larger
boards made heights ambiguous from one side) and a one-line result after
each action.  The rules are the real mechanics of each game package.

```bash
# 1. open a session (the player never sees the level path)
python -m chain_analysis.play_cli start type_3/images_batch/level_09_hard_backdoor_seed9225 runs/s01
# 2. the player looks at runs/s01/obs_000_*.png and acts, one JSON action per call
python -m chain_analysis.play_cli act runs/s01 '["open", [2, 3, 1]]'
python -m chain_analysis.play_cli act runs/s01 '["reset"]'          # after a loss
#    or one-shot: the whole plan in one call, no observation in between
python -m chain_analysis.play_cli plan runs/s02 '[["open", [2, 3, 1]], ["open", [4, 3, 1]]]'
# 3. export every session: exp1 trajectories + exp2 result lines
python -m chain_analysis.play_cli export runs/s* --model sonnet \
    --out runs/traj.jsonl --results runs/results.jsonl
# 4. run both experiments on the level set the sessions came from
python -m chain_analysis.exp1_step_value --root type_3/images_batch --traj runs/traj.jsonl
python -m chain_analysis.exp2_budget_search --root type_3/images_batch --models runs/results.jsonl
```

PipeFit pipes take their ends instead of an orientation index
(`["place", "L", [2,3,1], ["+x", "-y"]]`); Domino's push also runs the final
check.  Each session has a 40-action budget (`--max-actions`); a rejected
action (occupied cell, wrong phase, ...) costs one action and changes
nothing; after a loss the player may `["reset"]` (counted as a failed
verify).  `plan` (one-shot) plays a whole action list from the start state,
skips rejected actions, and closes the session; it must be the session's
first call.  Any agent that can read images and run a shell command can play.

## Model runs

All runs are in `model_runs/` (index, protocols, audit and how to
reproduce: `model_runs/README.md`).

**Current hard tier** (FluidNet and Domino regenerated on 2026-11-04,
PipeFit unchanged), calibrated against gpt-6.1-sol (Codex, default effort
low, one-shot) to a 20–30 % solve rate (`model_runs/gpt61sol_hard_calibration/`):

| game | gpt-6.1-sol low, one-shot | best search baseline at 1 / 64 verifies |
|---|---|---|
| FluidNet | 4 / 15 | 7% / 93% (greedy, greedy + undo) |
| Domino | 3 / 15 | 0% / 0% |
| PipeFit | 5 / 15 | 0% / 1% |

**Previous hard tier** (git `de130a7`), wins out of 15:

| game | Sonnet closed loop: solved / 1st verify | Sonnet one-shot | gpt-6-luna one-shot (Codex) | gpt-6.1-sol one-shot, effort high / low |
|---|---|---|---|---|
| FluidNet | 15 / 13 | 15 | 0 | 15 / 9 |
| Domino | 15 / 13 | 12 | 2 | 15 / 14 |
| PipeFit | 15 / 11 | 10 | 0 | 14 / 4 |

| run | model | protocol | levels |
|---|---|---|---|
| `sonnet_pilot/` | Claude Sonnet | closed loop | 15 demo levels (all solved, 14 on the 1st verify) |
| `sonnet_hard50/` | Claude Sonnet | closed loop | previous 45 hard levels of `puzzles_50` |
| `sonnet_oneshot50/` | Claude Sonnet | one-shot | same 45 |
| `gpt6luna_oneshot50/` | Codex gpt-6-luna | one-shot (images attached to `codex exec`) | same 45 |
| `gpt61sol_hard_calibration/` | Codex gpt-6.1-sol | one-shot | current 45 hard levels, plus every calibration game (`calibration_runs.csv`) |

On the previous hard tier Sonnet (and gpt-6.1-sol at effort high) planned
the levels from the start image alone; their failures were misread cells,
not planning.  Full discussion: `RESULTS.md`.

## Caveats

* N(s) counts end states, not move orders; PipeFit counts networks (support
  packings S1 vs S2 are not distinguished) and Domino counts the relevant
  patch set (an extra patch placed where the wave never goes is free).
* d(s) for PipeFit approximates the support pieces still needed from the
  number of empty support cells and S2 pieces left, and d is measured to the
  *shortest* solution, so a longer reference plan shows a few steps with
  Δd ≤ 0 even though it never leaves the solution set.
* PipeFit's `difficulty.solutions()` stops each route search at 100k routes;
  with three portals it can miss networks (on 1 of 60 levels it missed the
  stored plan, P49_E_mixed_hard).  The oracle adds the stored plan's network,
  but N(s) is a lower bound there, and so is that level's recorded
  `n_solutions`.
* The candidate generator is a heuristic: BFS / DFS explore only its moves,
  so "space exhausted" means exhausted under that generator.
