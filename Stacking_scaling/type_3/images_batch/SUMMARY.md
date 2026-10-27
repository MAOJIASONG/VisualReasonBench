# FluidNet 3D — 10-puzzle batch playthrough

| # | label | grid | T | V | P | seed | plan_len | traps | non-local | P(random) | P(frontier) | P(greedy) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 01_easy | (4, 4, 4) | 1 | 3 | 1 | 100 | 2 | 1 | 1 | 0.33 | 0.25 | 0.50 |
| 2 | 02_easy | (4, 4, 4) | 1 | 3 | 2 | 200 | 1 | 2 | 2 | 0.33 | 0.33 | 1.00 |
| 3 | 03_easy | (5, 5, 4) | 2 | 3 | 1 | 300 | 2 | 1 | 0 | 0.33 | 1.00 | 1.00 |
| 4 | 04_medium | (5, 5, 4) | 2 | 5 | 2 | 400 | 3 | 2 | 1 | 0.10 | 0.19 | 0.50 |
| 5 | 05_medium | (5, 5, 5) | 2 | 5 | 2 | 500 | 3 | 2 | 1 | 0.10 | 0.12 | 0.00 |
| 6 | 06_medium | (5, 5, 5) | 2 | 5 | 2 | 609 | 3 | 2 | 1 | 0.10 | 0.25 | 0.33 |
| 7 | 07_medium | (5, 5, 5) | 2 | 6 | 3 | 700 | 3 | 3 | 3 | 0.05 | 0.03 | 0.50 |
| 8 | 08_hard | (5, 5, 5) | 2 | 7 | 3 | 800 | 4 | 3 | 2 | 0.03 | 0.02 | 0.21 |
| 9 | 09_hard | (6, 5, 5) | 2 | 8 | 3 | 900 | 5 | 3 | 2 | 0.02 | 0.04 | 0.17 |
| 10 | 10_hard | (6, 6, 5) | 3 | 8 | 3 | 1000 | 5 | 3 | 2 | 0.02 | 0.02 | 0.00 |

P(random / frontier / greedy) = exact win probability of a player who opens random valves / opens a valve touching fluid that is not next to a red cell / opens the valve nearest an empty tank (difficulty.py).

elapsed: 335.1s