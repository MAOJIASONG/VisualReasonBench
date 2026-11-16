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
| Opus base | hard | 15 | 100 | 13 | 3.5 | 0 | 0 | 0 | 0.985 | 0.508 | 0.133 |
| Opus plan | hard | 15 | 100 | 13 | 3 | 0.5 | 0 | 0 | 0.984 | 0.505 | 0.267 |
| Sonnet base | hard | 15 | 80 | 67 | 2.6 | 0.6 | 0 | 0 | 0.888 | 0.376 | 1.6 |
| Sonnet plan | hard | 15 | 100 | 33 | 1.8 | 1.2 | 0 | 0 | 0.946 | 0.457 | 1.07 |
| luna base | hard | 15 | 0 | 40 | 1.17 | 1 | 0 | 0 | 0.66 | 0.037 | 1.27 |
| luna plan | hard | 15 | 0 | 60 | 1.11 | 0 | 5 | 11.1 | 0.454 | -0.18 | 0.667 |
| sol base | hard | 15 | 100 | 7 | 6 | 0 | 0 | 0 | 0.994 | 0.519 | 0.067 |
| sol plan | hard | 15 | 100 | 13 | 4 | 0 | 0 | 0 | 0.987 | 0.506 | 0.133 |
| terra base | hard | 15 | 93 | 20 | 4.67 | 0 | 0 | 0 | 0.981 | 0.49 | 0.2 |
| terra plan | hard | 15 | 87 | 27 | 5 | 0 | 0 | 0 | 0.96 | 0.466 | 0.467 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
