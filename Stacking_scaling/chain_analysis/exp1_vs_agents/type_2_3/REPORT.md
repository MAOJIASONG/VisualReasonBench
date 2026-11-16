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
| Opus base | hard | 15 | 100 | 27 | 1 | – | 1 | 75 | 0.816 | 0.798 | 0.933 |
| Opus plan | hard | 15 | 100 | 53 | 2.5 | 1 | 2 | 62.5 | 0.816 | 0.799 | 1.8 |
| Sonnet base | hard | 15 | 93 | 87 | 1.46 | 6 | 0 | 92.3 | 0.737 | 0.723 | 6.07 |
| Sonnet plan | hard | 15 | 80 | 93 | 1.14 | 9 | 1 | 85.7 | 0.62 | 0.603 | 6.4 |
| luna base | hard | 15 | 27 | 100 | 1.07 | 4.54 | 0 | 26.7 | 0.324 | 0.312 | 13.9 |
| luna plan | hard | 15 | 33 | 87 | 1.39 | 5.83 | 4 | 23.1 | 0.329 | 0.317 | 10.7 |
| sol base | hard | 15 | 100 | 60 | 2.22 | 13 | 1 | 55.6 | 0.795 | 0.778 | 4.47 |
| sol plan | hard | 15 | 100 | 33 | 1.2 | 5 | 0 | 60 | 0.814 | 0.797 | 1.53 |
| terra base | hard | 15 | 80 | 87 | 1.31 | 15 | 0 | 92.3 | 0.634 | 0.62 | 6.4 |
| terra plan | hard | 15 | 73 | 87 | 2.62 | 16 | 1 | 69.2 | 0.671 | 0.657 | 6.73 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 5 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
