# Exp 1 — dead ends and step value: `type_3_1/images_batch`

## Traps per level (along the reference plan)

| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |
|---|---|---|---|---|---|---|---|
| easy | 3 | 3.667 | 1 | 24 | 1 | 2.667 | 11.443 |
| medium | 4 | 5 | 1 | 31 | 1 | 4 | 19.2 |
| hard | 3 | 6 | 1 | 42.333 | 1 | 5 | 28.72 |

## Trajectories

| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet | easy | 2 | 100 | 0 | – | – | 0 | – | 1 | 0.862 | 0 |
| claude-sonnet | medium | 1 | 100 | 100 | 1 | – | 0 | 100 | 0.833 | 0.701 | 1 |
| claude-sonnet | hard | 2 | 100 | 0 | – | – | 0 | – | 1 | 0.908 | 0 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
