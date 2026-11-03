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
| claude-sonnet-oneshot | hard | 15 | 67 | 33 | 2.4 | 9 | 0 | 0 | 0.675 | 0.658 | 3 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
