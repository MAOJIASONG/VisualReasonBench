# Exp 1 — dead ends and step value: `type_3/puzzles_50`

## Traps per level (along the reference plan)

| tier | levels | plan | N(s0) | traps at start | earliest trap step | plan steps with a trap | mean traps / state |
|---|---|---|---|---|---|---|---|
| easy | 15 | 1.8 | 1 | 1.4 | 1 | 1.8 | 1.4 |
| medium | 20 | 3.1 | 1 | 2.15 | 1 | 3.1 | 2.15 |
| hard | 15 | 4.133 | 1 | 3.4 | 1 | 4.133 | 3.4 |

## Trajectories

| model | tier | runs | win % | fatal % | fatal step | delay | hidden at end | repaired % | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet | hard | 15 | 100 | 13 | 3 | 0.5 | 0 | 0 | 0.971 | 0.558 | 0.333 |

Columns: *fatal %* trajectories that ever walk into a dead end; *fatal step* first such step; *delay* steps until the game exposes it (failed verify); *hidden at end* fatal runs whose dead end was never exposed or repaired; *V* = N(s')/N(s); *V − V_rand* mean gain over a random legal move in the same state; *wasted* steps with d(s') ≥ d(s).

Oracle check: 0 won trajectories still marked as stuck in a dead end (0 = the oracle saw every winning line).
