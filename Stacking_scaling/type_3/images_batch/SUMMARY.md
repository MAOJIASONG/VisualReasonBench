# FluidNet 3D — 10-puzzle batch playthrough (concepts)

| # | label | concept | grid | seed | valves | plan | traps | P(random) | P(frontier) | P(lookahead) | shortest route wins | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 01_easy | trace | (4, 4, 4) | 1000 | 2 | 1 | 1 | 0.50 | 0.50 | 1.00 | yes | 1.0 |
| 2 | 02_easy | series | (4, 4, 4) | 2000 | 3 | 2 | 1 | 0.33 | 1.00 | 1.00 | yes | 0.0 |
| 3 | 03_easy | two_pumps | (5, 5, 4) | 3000 | 3 | 2 | 1 | 0.33 | 0.33 | 1.00 | yes | 1.6 |
| 4 | 04_medium | trace | (5, 5, 4) | 4003 | 5 | 3 | 2 | 0.10 | 0.06 | 1.00 | yes | 4.2 |
| 5 | 05_medium | backdoor | (5, 5, 4) | 5027 | 5 | 3 | 2 | 0.10 | 0.12 | 1.00 | no | 4.0 |
| 6 | 06_medium | two_pumps | (5, 5, 5) | 6000 | 6 | 4 | 2 | 0.07 | 0.09 | 0.50 | yes | 5.5 |
| 7 | 07_medium | mixed | (6, 5, 5) | 7002 | 6 | 3 | 3 | 0.05 | 0.07 | 0.50 | yes | 5.7 |
| 8 | 08_hard | series | (5, 5, 5) | 8037 | 7 | 4 | 3 | 0.03 | 0.00 | 1.00 | yes | 8.0 |
| 9 | 09_hard | backdoor | (5, 5, 5) | 9225 | 8 | 4 | 4 | 0.01 | 0.01 | 1.00 | no | 8.6 |
| 10 | 10_hard | mixed | (6, 6, 5) | 10004 | 8 | 5 | 3 | 0.02 | 0.01 | 0.50 | yes | 9.2 |

P(random / frontier / lookahead) = exact win probability of a player who opens random valves / opens a wet valve not next to a red cell / simulates one valve ahead (difficulty.py).

| tier | levels | valves | score |
|---|---|---|---|
| easy | 3 | 2–3 | 0–1.59 |
| medium | 4 | 5–6 | 4–5.74 |
| hard | 3 | 7–8 | 8–9.2 |

elapsed: 3665.9s