# Exp 2 — search under a verify budget: `type_3/puzzles_50`

50 levels; seeded methods averaged over 5 seeds; each run capped at 64 verifies / 2000 actions.

## Success rate (%) by verify budget (no action cap)

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 | median verifies to win | median actions to win |
|---|---|---|---|---|---|---|---|---|---|
| random | 16 | 21 | 28 | 40 | 56 | 73 | 87 | 7 | 15 |
| greedy | 36 | 43 | 46 | 54 | 68 | 77 | 91 | 4 | 8 |
| greedy_undo | 36 | 58 | 66 | 82 | 94 | 96 | 98 | 2 | 5 |
| dfs | 15 | 34 | 44 | 62 | 75 | 87 | 96 | 4.0 | 11.0 |
| bfs | 10 | 24 | 30 | 30 | 62 | 72 | 94 | 8.5 | 30.0 |
| **claude-sonnet-oneshot** | 100 | 100 | 100 | 100 | 100 | 100 | 100 | | |

## Success rate (%) by verify budget, at most 60 actions

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 |
|---|---|---|---|---|---|---|---|
| random | 16 | 21 | 28 | 40 | 56 | 73 | 75 |
| greedy | 36 | 43 | 46 | 54 | 68 | 77 | 82 |
| greedy_undo | 36 | 58 | 66 | 82 | 94 | 94 | 94 |
| dfs | 15 | 34 | 44 | 62 | 75 | 81 | 81 |
| bfs | 10 | 24 | 30 | 30 | 62 | 68 | 68 |

## By tier (≤5 verifies / unlimited within the run cap)

| method | easy ≤5 | easy all | medium ≤5 | medium all | hard ≤5 | hard all |
|---|---|---|---|---|---|---|
| random | 76 | 100 | 31 | 98 | 16 | 69 |
| greedy | 91 | 100 | 48 | 96 | 25 | 83 |
| greedy_undo | 100 | 100 | 75 | 100 | 73 | 93 |
| dfs | 92 | 100 | 65 | 100 | 28 | 92 |
| bfs | 100 | 100 | 0 | 100 | 0 | 100 |

## Verifies a method needs to match each model

| model | model success % | random | greedy | greedy_undo | dfs | bfs |
|---|---|---|---|---|---|---|
| claude-sonnet-oneshot | 100 | >64 | >64 | >64 | >64 | 64 |

Reading: if a search method matches a model at the same verify budget, the model's score is reachable by trial and error.  If cutting the budget hurts backtracking search much more than the model, the model is planning; if both fall together, the model also relies on retries.
