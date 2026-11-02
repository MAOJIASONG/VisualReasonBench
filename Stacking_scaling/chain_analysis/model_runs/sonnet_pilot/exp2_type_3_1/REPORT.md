# Exp 2 — search under a verify budget: `type_3_1/images_batch`

10 levels; seeded methods averaged over 5 seeds; each run capped at 64 verifies / 2000 actions.

## Success rate (%) by verify budget (no action cap)

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 | median verifies to win | median actions to win |
|---|---|---|---|---|---|---|---|---|---|
| random | 2 | 2 | 2 | 2 | 4 | 6 | 14 | 24 | 96 |
| greedy | 20 | 20 | 20 | 22 | 26 | 36 | 48 | 10.0 | 44.5 |
| greedy_undo | 20 | 20 | 20 | 20 | 20 | 20 | 30 | 1 | 4 |
| dfs | 0 | 2 | 2 | 4 | 4 | 8 | 20 | 30.0 | 116.0 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – | – |
| **claude-sonnet** | 100 | 100 | 100 | 100 | 100 | 100 | 100 | | |

## Success rate (%) by verify budget, at most 60 actions

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 |
|---|---|---|---|---|---|---|---|
| random | 2 | 2 | 2 | 2 | 4 | 6 | 6 |
| greedy | 20 | 20 | 20 | 22 | 26 | 32 | 32 |
| greedy_undo | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| dfs | 0 | 2 | 2 | 4 | 4 | 8 | 8 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## By tier (≤5 verifies / unlimited within the run cap)

| method | easy ≤5 | easy all | medium ≤5 | medium all | hard ≤5 | hard all |
|---|---|---|---|---|---|---|
| random | 7 | 60 | 0 | 0 | 0 | 0 |
| greedy | 73 | 100 | 0 | 50 | 0 | 7 |
| greedy_undo | 67 | 100 | 0 | 0 | 0 | 0 |
| dfs | 13 | 67 | 0 | 0 | 0 | 0 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 |

## Verifies a method needs to match each model

| model | model success % | random | greedy | greedy_undo | dfs | bfs |
|---|---|---|---|---|---|---|
| claude-sonnet | 100 | >64 | >64 | >64 | >64 | >64 |

Reading: if a search method matches a model at the same verify budget, the model's score is reachable by trial and error.  If cutting the budget hurts backtracking search much more than the model, the model is planning; if both fall together, the model also relies on retries.
