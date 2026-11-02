# chain_analysis — step-value and search-budget analysis for the CHAIN games

Analysis code for FluidNet (`type_3`), Domino (`type_3_1`) and PipeFit
(`type_2_3`).  Everything runs on the generated level folders
(`puzzles_50/`, `images_batch/`).  Run from `Stacking_scaling/`.

| file | what it does |
|---|---|
| `envs.py` | one pure, hashable-state interface over the three games + an exact oracle: N(s) (winning end states still reachable) and d(s) (fewest actions to win, counting the removals a wrong piece needs) |
| `verify_levels.py` | replays every level's reference plan through the game rules and checks the oracle agrees |
| `exp1_step_value.py` | Experiment 1: dead ends, consequence delay, solution retention V, distance change Δd |
| `exp2_budget_search.py` | Experiment 2: random / greedy / greedy+undo / DFS / BFS under the same verify and action budget |

```bash
python -m chain_analysis.verify_levels type_3/puzzles_50 type_3_1/puzzles_50 type_2_3/puzzles_50
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
