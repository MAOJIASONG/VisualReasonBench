# Exp 1 — dead ends and step value: `type_3/images_batch`

## Traps per level (along the reference plan)

| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |
|---|---|---|---|---|---|---|---|
| easy | 3 | 1.667 | 1 | 1 | 1 | 1.667 | 1 |
| medium | 4 | 3.25 | 1 | 2.25 | 1 | 3.25 | 2.25 |
| hard | 3 | 4.333 | 1 | 3.333 | 1 | 4.333 | 3.333 |

## Trajectories

| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet | easy | 2 | 100 | 0 | – | – | 0 | – | 1 | 0.458 | 0 |
| claude-sonnet | medium | 1 | 100 | 0 | – | – | 0 | – | 1 | 0.522 | 0 |
| claude-sonnet | hard | 2 | 100 | 50 | 4 | 0 | 0 | 0 | 0.944 | 0.515 | 0.5 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
