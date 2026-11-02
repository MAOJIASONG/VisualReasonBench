# Exp 1 — dead ends and step value: `type_2_3/puzzles_50`

## Traps per level (along the reference plan)

| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |
|---|---|---|---|---|---|---|---|
| easy | 15 | 7.467 | 3.533 | 1782.133 | 1 | 7.467 | 1363.414 |
| medium | 20 | 11.3 | 11.95 | 2049.1 | 1 | 11.3 | 1557.881 |
| hard | 15 | 14.133 | 8 | 2594.133 | 1 | 14.133 | 1942.095 |

## Trajectories

| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | easy | 15 | 0 | 100 | 1.47 | 5.07 | 0 | 0 | 0.107 | 0.099 | 4.8 |
| random | medium | 20 | 5 | 95 | 1.79 | 7.21 | 0 | 0 | 0.151 | 0.138 | 6.2 |
| random | hard | 15 | 0 | 100 | 2 | 9.4 | 0 | 0 | 0.214 | 0.202 | 8.53 |
| reference | easy | 15 | 100 | 0 | – | – | 0 | – | 0.91 | 0.893 | 0.6 |
| reference | medium | 20 | 100 | 0 | – | – | 0 | – | 0.872 | 0.857 | 2.1 |
| reference | hard | 15 | 100 | 0 | – | – | 0 | – | 0.908 | 0.898 | 1.93 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
