# Exp 2 — search under a verify budget: `type_2_3/puzzles_50`

50 levels; seeded methods averaged over 5 seeds; each run capped at 64 verifies / 2000 actions.

## Success rate (%) by verify budget (no action cap)

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 | median verifies to win | median actions to win |
|---|---|---|---|---|---|---|---|---|---|
| random | 0 | 1 | 1 | 1 | 2 | 3 | 9 | 28 | 191 |
| greedy | 6 | 6 | 6 | 7 | 10 | 15 | 22 | 13.5 | 96.0 |
| greedy_undo | 6 | 6 | 6 | 6 | 10 | 14 | 22 | 19 | 191 |
| dfs | 0 | 0 | 0 | 0 | 2 | 3 | 6 | 30 | 293 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – | – |

## Success rate (%) by verify budget, at most 60 actions

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 |
|---|---|---|---|---|---|---|---|
| random | 0 | 1 | 1 | 1 | 2 | 2 | 2 |
| greedy | 6 | 6 | 6 | 7 | 10 | 10 | 10 |
| greedy_undo | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| dfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## By tier (≤5 verifies / unlimited within the run cap)

| method | easy ≤5 | easy all | medium ≤5 | medium all | hard ≤5 | hard all |
|---|---|---|---|---|---|---|
| random | 1 | 25 | 2 | 8 | 0 | 0 |
| greedy | 16 | 47 | 6 | 20 | 0 | 1 |
| greedy_undo | 13 | 47 | 5 | 30 | 0 | 0 |
| dfs | 1 | 13 | 0 | 8 | 0 | 1 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 |

Reading: if a search method matches a model at the same verify budget, the model's score is reachable by trial and error.  If cutting the budget hurts backtracking search much more than the model, the model is planning; if both fall together, the model also relies on retries.
