# Baseline results on the current level sets

All numbers come from `puzzles_50/` of each game (15 easy / 20 medium / 15
hard).  No model trajectories have been analysed yet: Experiment 1 uses the
reference plan and one random rollout per level, Experiment 2 uses only the
search baselines.  Full tables: `<game>/analysis_exp1/REPORT.md`,
`<game>/analysis_exp2/REPORT.md`.

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
