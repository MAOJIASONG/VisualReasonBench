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
| Opus base | hard | 15 | 100 | 20 | 1 | 10 | 0 | 33.3 | 0.979 | 0.914 | 0.867 |
| Opus plan | hard | 15 | 100 | 7 | 1 | 11 | 0 | 0 | 0.995 | 0.932 | 0.267 |
| Sonnet base | hard | 15 | 0 | 93 | 1.71 | 9.12 | 5 | 7.14 | 0.368 | 0.299 | 6.87 |
| Sonnet plan | hard | 15 | 33 | 87 | 1.85 | 9.42 | 1 | 0 | 0.486 | 0.413 | 6.2 |
| luna base | hard | 15 | 0 | 100 | 1.2 | 10.2 | 2 | 0 | 0.17 | 0.097 | 18.9 |
| luna plan | hard | 15 | 0 | 100 | 1.6 | 9.69 | 2 | 0 | 0.171 | 0.098 | 20.3 |
| sol base | hard | 15 | 100 | 7 | 1 | – | 0 | 100 | 0.994 | 0.931 | 0.067 |
| sol plan | hard | 15 | 100 | 7 | 1 | – | 0 | 100 | 0.994 | 0.931 | 0.067 |
| terra base | hard | 15 | 93 | 60 | 2 | 8.8 | 0 | 44.4 | 0.946 | 0.882 | 2.4 |
| terra plan | hard | 15 | 100 | 60 | 3.78 | 6.33 | 0 | 33.3 | 0.958 | 0.893 | 1.47 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
