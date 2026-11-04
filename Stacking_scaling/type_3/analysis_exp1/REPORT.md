# Exp 1 — dead ends and step value: `type_3/puzzles_50`

## Traps per level (along the reference plan)

| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |
|---|---|---|---|---|---|---|---|
| easy | 15 | 1.8 | 1 | 1.4 | 1 | 1.8 | 1.4 |
| medium | 20 | 3.1 | 1 | 2.15 | 1 | 3.1 | 2.15 |
| hard | 15 | 6 | 1 | 3.4 | 1 | 6 | 3.4 |

## Trajectories

| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | easy | 15 | 27 | 73 | 1.46 | 0.545 | 0 | 0 | 0.433 | -0.089 | 1.13 |
| random | medium | 20 | 20 | 80 | 2.06 | 0.5 | 0 | 0 | 0.525 | 0.007 | 1.2 |
| random | hard | 15 | 0 | 100 | 2.27 | 1.87 | 0 | 0 | 0.381 | -0.227 | 2.87 |
| reference | easy | 15 | 100 | 0 | – | – | 0 | – | 1 | 0.5 | 0 |
| reference | medium | 20 | 100 | 0 | – | – | 0 | – | 1 | 0.532 | 0 |
| reference | hard | 15 | 100 | 0 | – | – | 0 | – | 1 | 0.524 | 0 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
