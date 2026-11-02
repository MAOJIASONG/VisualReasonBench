# Exp 2 — search under a verify budget: `type_3_1/puzzles_50`

50 levels; seeded methods averaged over 5 seeds; each run capped at 64 verifies / 2000 actions.

## Success rate (%) by verify budget (no action cap)

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 | median verifies to win | median actions to win |
|---|---|---|---|---|---|---|---|---|---|
| random | 3 | 3 | 3 | 4 | 6 | 9 | 16 | 19 | 57 |
| greedy | 24 | 24 | 25 | 27 | 30 | 34 | 38 | 1.0 | 4.0 |
| greedy_undo | 24 | 24 | 24 | 24 | 24 | 24 | 28 | 1.0 | 3.0 |
| dfs | 1 | 1 | 2 | 4 | 6 | 13 | 23 | 15 | 75 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – | – |

## Success rate (%) by verify budget, at most 60 actions

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 |
|---|---|---|---|---|---|---|---|
| random | 3 | 3 | 3 | 4 | 6 | 9 | 9 |
| greedy | 24 | 24 | 25 | 27 | 30 | 32 | 32 |
| greedy_undo | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| dfs | 1 | 1 | 2 | 4 | 6 | 11 | 11 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## By tier (≤5 verifies / unlimited within the run cap)

| method | easy ≤5 | easy all | medium ≤5 | medium all | hard ≤5 | hard all |
|---|---|---|---|---|---|---|
| random | 13 | 53 | 0 | 3 | 0 | 0 |
| greedy | 87 | 97 | 3 | 20 | 0 | 4 |
| greedy_undo | 80 | 87 | 0 | 5 | 0 | 0 |
| dfs | 12 | 73 | 0 | 4 | 0 | 0 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 |

Reading: if a search method matches a model at the same verify budget, the model's score is reachable by trial and error.  If cutting the budget hurts backtracking search much more than the model, the model is planning; if both fall together, the model also relies on retries.
