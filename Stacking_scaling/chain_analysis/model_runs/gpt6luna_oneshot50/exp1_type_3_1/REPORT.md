# Exp 1 — dead ends and step value: `type_3_1/puzzles_50`

## Traps per level (along the reference plan)

| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |
|---|---|---|---|---|---|---|---|
| easy | 15 | 3.4 | 1 | 24.4 | 1 | 2.4 | 10.611 |
| medium | 20 | 5 | 1 | 30.95 | 1 | 4 | 19.17 |
| hard | 15 | 6 | 1 | 47.867 | 1 | 5 | 32.411 |

## Trajectories

| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-luna-oneshot | hard | 15 | 13 | 67 | 1.9 | 4.25 | 6 | 0 | 0.468 | 0.395 | 1.87 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
