# Exp 1 — dead ends and step value: `type_2_3/puzzles_50`

## Traps per level (along the reference plan)

| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |
|---|---|---|---|---|---|---|---|
| easy | 15 | 7.467 | 3.533 | 1796.667 | 1 | 7.467 | 1370.867 |
| medium | 20 | 11.3 | 11.95 | 2081.25 | 1 | 11.3 | 1570.529 |
| hard | 15 | 14.133 | 8 | 2628.267 | 1 | 14.133 | 1953.573 |

## Trajectories

| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | easy | 15 | 0 | 100 | 1.47 | 5.07 | 0 | 0 | 0.107 | 0.103 | 4.8 |
| random | medium | 20 | 5 | 95 | 1.79 | 7.21 | 0 | 0 | 0.151 | 0.145 | 6.2 |
| random | hard | 15 | 0 | 100 | 2 | 9.4 | 0 | 0 | 0.214 | 0.209 | 8.53 |
| reference | easy | 15 | 100 | 0 | – | – | 0 | – | 0.91 | 0.908 | 0.6 |
| reference | medium | 20 | 100 | 0 | – | – | 0 | – | 0.892 | 0.889 | 1.85 |
| reference | hard | 15 | 100 | 0 | – | – | 0 | – | 0.913 | 0.909 | 1.6 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).
