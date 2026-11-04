# Baseline results on the current level sets

The baseline numbers come from `puzzles_50/` of each game (15 easy / 20
medium / 15 hard): Experiment 1 uses the reference plan and one random
rollout per level, Experiment 2 the search baselines.  Full tables:
`<game>/analysis_exp1/REPORT.md`, `<game>/analysis_exp2/REPORT.md`.  A first
model run (Claude Sonnet, 15 episodes on the demo levels) is summarised at
the end and detailed in `model_runs/sonnet_pilot/README.md`.

The model scores in the paper's Table 1 were measured on the earlier
30-level sets, so they are not directly comparable with these numbers.

**Hard tier recalibrated (2026-11-04).**  The FluidNet and Domino hard
levels of `puzzles_50/` were regenerated with harder settings so that
gpt-6.1-sol (Codex, default reasoning effort, one-shot) solves about a
quarter of them; PipeFit's hard tier already sat there and is unchanged.
Easy and medium levels are byte-identical.  The baseline tables below are on
the new levels; the Claude Sonnet and gpt-6-luna sections further down were
measured on the previous hard tier (git `de130a7`).  Calibration details:
the last section and `model_runs/gpt61sol_hard_calibration/README.md`.

## Experiment 1 — dead ends (random rollouts)

| game | tier | runs into a dead end | first fatal step | steps until exposed |
|---|---|---|---|---|
| FluidNet | easy / medium / hard | 73% / 80% / 100% | 1.5 / 2.1 / 2.3 | 0.5 / 0.5 / 1.9 |
| Domino | easy / medium / hard | 100% / 100% / 100% | 1.1 / 1.3 / 1.3 | 2.3 / 3.8 / 10.1 |
| PipeFit | easy / medium / hard | 100% / 95% / 100% | 1.5 / 1.8 / 2.0 | 5.1 / 7.2 / 9.4 |

The delay between the fatal move and the moment the game reveals it grows
with tier in every game; PipeFit hides a mistake longest (until submit).
Every level offers a dead-end move at step 1 (`earliest trap step` = 1).

## Experiment 2 — success (%) by verify budget

| game | method | ≤1 | ≤5 | ≤20 | ≤50 |
|---|---|---|---|---|---|
| FluidNet | random | 16 | 36 | 67 | 80 |
| | greedy | 34 | 51 | 70 | 81 |
| | greedy + undo | 34 | 74 | 96 | 98 |
| | DFS | 15 | 57 | 79 | 85 |
| | BFS | 10 | 30 | 70 | 70 |
| Domino | random | 3 | 4 | 9 | 16 |
| | greedy | 24 | 27 | 34 | 37 |
| | greedy + undo | 24 | 24 | 24 | 28 |
| | DFS | 1 | 4 | 13 | 23 |
| | BFS | 0 | 0 | 0 | 0 |
| PipeFit | random | 0 | 1 | 3 | 9 |
| | greedy | 6 | 7 | 15 | 22 |
| | greedy + undo | 6 | 6 | 14 | 22 |
| | DFS | 0 | 0 | 3 | 6 |
| | BFS | 0 | 0 | 0 | 0 |

* FluidNet is the one game where trial and error alone gets far: with 5
  verifies, greedy + undo solves 74% (47% of hard levels; 7% at one
  verify).  A model score in that range on FluidNet does not by itself show
  planning.
* Domino and PipeFit resist search: no baseline solves a hard Domino level
  within 64 verifies, and PipeFit hard stays at 0–1%.
* Greedy + undo is weaker than plain greedy on Domino: depth-first
  backtracking keeps varying the last patch and push, while the mistake is
  usually the first patch.

## Pilot: Claude Sonnet (15 episodes, demo levels)

Five levels per game (2 easy, 1 medium, 2 hard) from `images_batch/`,
played closed loop through `play_cli.py` with a 40-action budget.

| game | solved | first-verify wins | mean actions | dead-end steps (Exp 1) | baselines at Sonnet's level (Exp 2) |
|---|---|---|---|---|---|
| FluidNet | 5 / 5 | 4 / 5 | 5.0 | 1 run, exposed at once, won after a reset | greedy + undo needs 36 verifies; others > 64 |
| Domino | 5 / 5 | 5 / 5 | 5.6 | 1 probe patch, removed next step | all > 64 verifies |
| PipeFit | 5 / 5 | 5 / 5 | 16.2 | 4 runs, all repaired by removals before submit | all > 64 verifies |

* Sonnet's wins are not reproducible by trial and error at its budget: it
  needed 1–2 verifies where the baselines need dozens or never get there.
* Its mistakes are early and local (a misread coordinate, a pipe in a
  portal's wall cell) and it repairs them before verifying, so they cost
  steps, not attempts.
* Every hard demo level was solved, so under step-by-step visual feedback
  the current hard tier does not stop a strong model.  This is a 15-episode
  pilot; the `puzzles_50` levels, several seeds and other models are needed
  before drawing conclusions.

## Claude Sonnet on the previous hard tier of `puzzles_50` (45 episodes)

All 15 hard levels per game, one episode each; details and audit in
`model_runs/sonnet_hard50/README.md`.

| game | solved | first-verify wins | hard-tier success at 1 verify: Sonnet / best baseline | dead-end steps (Exp 1) |
|---|---|---|---|---|
| FluidNet | 15 / 15 | 13 / 15 | 87% / 13% (greedy) | 2 runs; breach shown at once, won after reset |
| Domino | 15 / 15 | 13 / 15 | 87% / 0% | 8 runs; 7 repaired before the push |
| PipeFit | 15 / 15 | 11 / 15 | 73% / 0% | 10 runs; all repaired before submit, 4 later submit failures |

* The hard tier separates the search baselines sharply (Domino and PipeFit
  hard: 0–4% even with 64 verifies) but not Sonnet under step-by-step visual
  feedback.  Sonnet's remaining costs are perception (misread cells: 2.5
  rejected actions per FluidNet episode) and PipeFit gravity mistakes that
  only a submit reveals.
* Two Domino episodes used a Python homography as a coordinate calculator
  (flagged in the audit); excluding them does not change the picture.

## One-shot: Claude Sonnet and Codex gpt-6-luna (previous hard tier, 45 levels each)

Same 45 hard levels, but the player sees only the start images and submits
its whole plan once (`play_cli plan`); no observation in between.  Details:
`model_runs/sonnet_oneshot50/README.md`, `model_runs/gpt6luna_oneshot50/README.md`.

| game | Sonnet closed loop, 1st verify | Sonnet one-shot | gpt-6-luna one-shot | best baseline at 1 verify |
|---|---|---|---|---|
| FluidNet | 13 / 15 | 15 / 15 | 0 / 15 | 13% (greedy) |
| Domino | 13 / 15 | 12 / 15 | 2 / 15 | 0% |
| PipeFit | 11 / 15 | 10 / 15 | 0 / 15 | 0% |

* Removing feedback costs Sonnet little: its one-shot score is within 1–2
  levels of its closed-loop first attempt, and 17 of its 44 plans equal the
  action list it ended with in closed loop.  On these levels Sonnet plans
  from the start image; the feedback mainly catches misread cells.
* gpt-6-luna fails mostly on reading coordinates: in FluidNet 3.7 of its
  ~4 valves per plan do not exist where it names them, and 4 PipeFit answers
  decline to give a plan.  Its accepted moves are rarely traps (no FluidNet
  dead end), so the gap is perception, not planning.
* So the hard tier separates models (Sonnet vs gpt-6-luna) and search
  baselines, but not by planning depth for Sonnet: its failures, closed loop
  or one-shot, are misread step heights, portal wall cells and supports.

## Current hard tier: calibrated against gpt-6.1-sol (2026-11-04)

The previous hard tier was solved 44/45 by gpt-6.1-sol at effort high and
27/45 at its default effort low (Codex, one-shot).  With effort low as the
reference, FluidNet and Domino hard were regenerated harder and PipeFit's
hard tier (4/15 and 5/15 in two runs) was kept.  Details, failure modes and
every calibration game: `model_runs/gpt61sol_hard_calibration/README.md`.

| game | new hard levels | gpt-6.1-sol low, one-shot | best baseline at 1 / 64 verifies |
|---|---|---|---|
| FluidNet | 9–10 valves, unique 6-valve plan (filler valves pruned) | 4 / 15 | 7% / 93% |
| Domino | three-level splitter tree on steps, 9–10 patches, 1–5 goals | 3 / 15 | 0% / 0% |
| PipeFit | unchanged | 5 / 15 | 0% / 1% |
| all | | 12 / 45 = 0.27 | |

Two more runs on the same 45 levels gave 4 / 3 / 3 = 10 / 45 and
6 / 3 / 3 = 12 / 45; over the three runs FluidNet 0.31, Domino 0.20,
PipeFit 0.24, all 34 / 135 = 0.25.

* What separates the tiers for this model is reading the 3D scene exactly
  (valve coordinates, step heights, portal cells) across a longer plan: a
  single misread action breaks a one-shot plan.  FluidNet's solve rate drops
  from 0.93 at 7 valves to 0.24 at 9 and from 0.50 to 0.04 as the plan grows
  from 4 to 6 valves.
* Search baselines are no better on the new tiers: Domino and PipeFit hard
  stay at 0–1% within 64 verifies; on FluidNet hard greedy + undo needs
  5 verifies for 47% and 64 for 93%.
