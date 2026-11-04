# Exp 1 — dead ends and step value: `type_3_1/puzzles_50`

## Traps per level (along the reference plan)

| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |
|---|---|---|---|---|---|---|---|
| easy | 15 | 3.4 | 1 | 24.4 | 1 | 2.4 | 10.611 |
| medium | 20 | 5 | 1 | 30.95 | 1 | 4 | 19.17 |
| hard | 15 | 11.4 | 1 | 115.4 | 1 | 10.4 | 95.377 |

## Trajectories

| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | easy | 15 | 0 | 100 | 1.13 | 2.27 | 0 | 0 | 0.044 | -0.03 | 2.87 |
| random | medium | 20 | 0 | 100 | 1.25 | 3.75 | 0 | 0 | 0.125 | 0.04 | 3.9 |
| random | hard | 15 | 0 | 100 | 1.33 | 10.1 | 0 | 0 | 0.144 | 0.07 | 8.53 |
| reference | easy | 15 | 100 | 0 | – | – | 0 | – | 1 | 0.864 | 0 |
| reference | medium | 20 | 100 | 0 | – | – | 0 | – | 1 | 0.892 | 0 |
| reference | hard | 15 | 100 | 0 | – | – | 0 | – | 1 | 0.937 | 0 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
