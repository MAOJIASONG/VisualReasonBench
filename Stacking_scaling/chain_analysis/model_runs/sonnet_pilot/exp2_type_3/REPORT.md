# Exp 2 — search under a verify budget: `type_3/images_batch`

10 levels; seeded methods averaged over 5 seeds; each run capped at 64 verifies / 2000 actions.

## Success rate (%) by verify budget (no action cap)

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 | median verifies to win | median actions to win |
|---|---|---|---|---|---|---|---|---|---|
| random | 22 | 24 | 34 | 38 | 44 | 64 | 82 | 10 | 24 |
| greedy | 30 | 36 | 38 | 44 | 58 | 70 | 72 | 3 | 5 |
| greedy_undo | 30 | 60 | 60 | 60 | 90 | 90 | 100 | 2.0 | 4.5 |
| dfs | 20 | 38 | 50 | 62 | 86 | 90 | 96 | 3 | 10 |
| bfs | 20 | 20 | 30 | 30 | 60 | 70 | 90 | 7 | 28 |
| **claude-sonnet** | 80 | 100 | 100 | 100 | 100 | 100 | 100 | | |

## Success rate (%) by verify budget, at most 60 actions

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 |
|---|---|---|---|---|---|---|---|
| random | 22 | 24 | 34 | 38 | 44 | 64 | 68 |
| greedy | 30 | 36 | 38 | 44 | 58 | 70 | 70 |
| greedy_undo | 30 | 60 | 60 | 60 | 90 | 90 | 90 |
| dfs | 20 | 38 | 50 | 62 | 86 | 86 | 86 |
| bfs | 20 | 20 | 30 | 30 | 50 | 60 | 60 |

## By tier (≤5 verifies / unlimited within the run cap)

| method | easy ≤5 | easy all | medium ≤5 | medium all | hard ≤5 | hard all |
|---|---|---|---|---|---|---|
| random | 100 | 100 | 15 | 100 | 7 | 53 |
| greedy | 73 | 100 | 55 | 100 | 0 | 13 |
| greedy_undo | 100 | 100 | 75 | 100 | 0 | 100 |
| dfs | 100 | 100 | 55 | 100 | 33 | 93 |
| bfs | 100 | 100 | 0 | 100 | 0 | 67 |

## Verifies a method needs to match each model

| model | model success % | random | greedy | greedy_undo | dfs | bfs |
|---|---|---|---|---|---|---|
| claude-sonnet | 100 | >64 | >64 | 36 | >64 | >64 |

Reading: if a search method matches a model at the same verify budget, the model's score is reachable by trial and error.  If cutting the budget hurts backtracking search much more than the model, the model is planning; if both fall together, the model also relies on retries.
