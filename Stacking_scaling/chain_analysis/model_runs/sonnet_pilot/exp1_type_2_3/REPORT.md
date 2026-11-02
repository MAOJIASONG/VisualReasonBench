# Exp 1 — dead ends and step value: `type_2_3/images_batch`

## Traps per level (along the reference plan)

| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |
|---|---|---|---|---|---|---|---|
| easy | 3 | 7 | 2.333 | 1195 | 1 | 7 | 1008.437 |
| medium | 4 | 12 | 9.75 | 1794 | 1 | 12 | 1393.438 |
| hard | 3 | 15.333 | 9.667 | 2156.667 | 1 | 15.333 | 1655.56 |

## Trajectories

| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet | easy | 2 | 100 | 100 | 1 | – | 0 | 100 | 0.792 | 0.789 | 3.5 |
| claude-sonnet | medium | 1 | 100 | 0 | – | – | 0 | – | 0.78 | 0.724 | 0 |
| claude-sonnet | hard | 2 | 100 | 100 | 1 | – | 0 | 100 | 0.718 | 0.706 | 5 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
