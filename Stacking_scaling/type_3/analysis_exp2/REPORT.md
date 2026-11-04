# Exp 2 — search under a verify budget: `type_3/puzzles_50`

50 levels; seeded methods averaged over 5 seeds; each run capped at 64 verifies / 2000 actions.

## Success rate (%) by verify budget (no action cap)

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 | median verifies to win | median actions to win |
|---|---|---|---|---|---|---|---|---|---|
| random | 16 | 20 | 26 | 36 | 52 | 67 | 80 | 7.0 | 15.0 |
| greedy | 34 | 40 | 44 | 51 | 61 | 70 | 81 | 3.0 | 6.0 |
| greedy_undo | 34 | 58 | 64 | 74 | 92 | 96 | 98 | 2 | 6 |
| dfs | 15 | 33 | 42 | 57 | 70 | 79 | 85 | 4.0 | 10.0 |
| bfs | 10 | 24 | 30 | 30 | 62 | 70 | 70 | 7 | 26 |

## Success rate (%) by verify budget, at most 60 actions

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 |
|---|---|---|---|---|---|---|---|
| random | 16 | 20 | 26 | 36 | 52 | 65 | 66 |
| greedy | 34 | 40 | 44 | 51 | 61 | 69 | 71 |
| greedy_undo | 34 | 58 | 64 | 74 | 92 | 94 | 94 |
| dfs | 15 | 33 | 42 | 57 | 69 | 73 | 73 |
| bfs | 10 | 24 | 30 | 30 | 62 | 66 | 66 |

## By tier (≤5 verifies / unlimited within the run cap)

| method | easy ≤5 | easy all | medium ≤5 | medium all | hard ≤5 | hard all |
|---|---|---|---|---|---|---|
| random | 76 | 100 | 31 | 98 | 4 | 44 |
| greedy | 91 | 100 | 48 | 96 | 15 | 47 |
| greedy_undo | 100 | 100 | 75 | 100 | 47 | 93 |
| dfs | 92 | 100 | 65 | 100 | 12 | 55 |
| bfs | 100 | 100 | 0 | 100 | 0 | 0 |

Reading: if a search method matches a model at the same verify budget, the model's score is reachable by trial and error.  If cutting the budget hurts backtracking search much more than the model, the model is planning; if both fall together, the model also relies on retries.
