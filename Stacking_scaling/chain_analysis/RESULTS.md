# Baseline results on the current level sets

The baseline numbers come from `puzzles_50/` of each game (15 easy / 20
medium / 15 hard): Experiment 1 uses the reference plan and one random
rollout per level, Experiment 2 the search baselines.  Full tables:
`<game>/analysis_exp1/REPORT.md`, `<game>/analysis_exp2/REPORT.md`.  A first
model run (Claude Sonnet, 15 episodes on the demo levels) is summarised at
the end and detailed in `model_runs/sonnet_pilot/README.md`.

The model scores in the paper's Table 1 were measured on the earlier
30-level sets, so they are not directly comparable with these numbers.

## Experiment 1 — dead ends (random rollouts)

| game | tier | runs into a dead end | first fatal step | steps until exposed |
|---|---|---|---|---|
| FluidNet | easy / medium / hard | 67% / 90% / 93% | 1.3 / 2.0 / 1.7 | 0.4 / 0.8 / 1.6 |
| Domino | easy / medium / hard | 93% / 100% / 100% | 1.2 / 1.2 / 1.2 | 2.2 / 3.8 / 4.8 |
| PipeFit | easy / medium / hard | 100% / 95% / 100% | 1.5 / 1.8 / 2.0 | 5.1 / 7.2 / 9.4 |

The delay between the fatal move and the moment the game reveals it grows
with tier in every game; PipeFit hides a mistake longest (until submit).
Every level offers a dead-end move at step 1 (`earliest trap step` = 1).

## Experiment 2 — success (%) by verify budget

| game | method | ≤1 | ≤5 | ≤20 | ≤50 |
|---|---|---|---|---|---|
| FluidNet | random | 16 | 40 | 73 | 87 |
| | greedy | 36 | 54 | 77 | 91 |
| | greedy + undo | 36 | 82 | 96 | 98 |
| | DFS | 15 | 62 | 87 | 96 |
| | BFS | 10 | 30 | 72 | 94 |
| Domino | random | 3 | 4 | 9 | 16 |
| | greedy | 24 | 27 | 34 | 38 |
| | greedy + undo | 24 | 24 | 24 | 28 |
| | DFS | 1 | 4 | 13 | 23 |
| | BFS | 0 | 0 | 0 | 0 |
| PipeFit | random | 0 | 1 | 3 | 9 |
| | greedy | 6 | 7 | 15 | 22 |
| | greedy + undo | 6 | 6 | 14 | 22 |
| | DFS | 0 | 0 | 3 | 6 |
| | BFS | 0 | 0 | 0 | 0 |

* FluidNet is the one game where trial and error alone gets far: with 5
  verifies, greedy + undo solves 82% (73% of hard levels).  A model score in
  that range on FluidNet does not by itself show planning.
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

## Claude Sonnet on the hard tier of `puzzles_50` (45 episodes)

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
