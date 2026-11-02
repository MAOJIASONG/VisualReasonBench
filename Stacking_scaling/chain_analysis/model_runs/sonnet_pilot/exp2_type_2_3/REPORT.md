# Exp 2 — search under a verify budget: `type_2_3/images_batch`

10 levels; seeded methods averaged over 5 seeds; each run capped at 64 verifies / 2000 actions.

## Success rate (%) by verify budget (no action cap)

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 | median verifies to win | median actions to win |
|---|---|---|---|---|---|---|---|---|---|
| random | 0 | 0 | 0 | 4 | 4 | 6 | 8 | 12.5 | 72.5 |
| greedy | 10 | 10 | 12 | 12 | 12 | 16 | 16 | 1 | 5 |
| greedy_undo | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 1 | 5 |
| dfs | 0 | 0 | 0 | 0 | 0 | 2 | 6 | 21 | 184 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – | – |
| **claude-sonnet** | 100 | 100 | 100 | 100 | 100 | 100 | 100 | | |

## Success rate (%) by verify budget, at most 60 actions

| method | ≤1 | ≤2 | ≤3 | ≤5 | ≤10 | ≤20 | ≤50 |
|---|---|---|---|---|---|---|---|
| random | 0 | 0 | 0 | 4 | 4 | 4 | 4 |
| greedy | 10 | 10 | 12 | 12 | 12 | 12 | 12 |
| greedy_undo | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| dfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## By tier (≤5 verifies / unlimited within the run cap)

| method | easy ≤5 | easy all | medium ≤5 | medium all | hard ≤5 | hard all |
|---|---|---|---|---|---|---|
| random | 13 | 27 | 0 | 0 | 0 | 0 |
| greedy | 40 | 47 | 0 | 10 | 0 | 0 |
| greedy_undo | 33 | 33 | 0 | 0 | 0 | 0 |
| dfs | 0 | 20 | 0 | 0 | 0 | 0 |
| bfs | 0 | 0 | 0 | 0 | 0 | 0 |

## Verifies a method needs to match each model

| model | model success % | random | greedy | greedy_undo | dfs | bfs |
|---|---|---|---|---|---|---|
| claude-sonnet | 100 | >64 | >64 | >64 | >64 | >64 |

Reading: if a search method matches a model at the same verify budget, the model's score is reachable by trial and error.  If cutting the budget hurts backtracking search much more than the model, the model is planning; if both fall together, the model also relies on retries.
