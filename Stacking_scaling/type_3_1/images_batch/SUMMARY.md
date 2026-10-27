# Domino Chain Reaction 3D — 10-level batch (rules v2)

| # | label | category | tier | seed | grid | fixed | patches | decoy gaps | solutions | P(random) | P(gap-fill) | P(wave-follow) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | E1_easy_line | A_line | easy | 1000 | (6, 5, 3) | 4 | 1 | 0 | 1 | 0.007 | 1.00 | 0.33 |
| 2 | E2_easy_decoy | B_decoy | easy | 2003 | (5, 5, 3) | 4 | 2 | 1 | 1 | 0.002 | 0.00 | 0.33 |
| 3 | E3_easy_split | D_split | easy | 3000 | (6, 6, 3) | 6 | 2 | 0 | 1 | 0.000 | 1.00 | 0.25 |
| 4 | M1_med_decoy | B_decoy | medium | 4000 | (7, 6, 3) | 8 | 2 | 2 | 1 | 0.000 | 0.00 | 0.33 |
| 5 | M2_med_steps | C_steps | medium | 5000 | (7, 5, 3) | 9 | 2 | 2 | 1 | 0.000 | 0.00 | 0.33 |
| 6 | M3_med_split | D_split | medium | 6000 | (7, 6, 3) | 7 | 3 | 2 | 1 | 0.000 | 0.00 | 0.17 |
| 7 | M4_med_full | E_full | medium | 7000 | (7, 7, 4) | 8 | 3 | 3 | 1 | 0.000 | 0.00 | 0.17 |
| 8 | H1_hard_steps | C_steps | hard | 8000 | (8, 6, 4) | 6 | 3 | 2 | 1 | 0.000 | 0.00 | 0.33 |
| 9 | H2_hard_split | D_split | hard | 9000 | (8, 7, 3) | 11 | 3 | 3 | 1 | 0.000 | 0.00 | 0.06 |
| 10 | H3_hard_full | E_full | hard | 10000 | (8, 8, 4) | 14 | 4 | 2 | 1 | 0.000 | 0.00 | 0.17 |

P(random) = random legal placement + random push direction; P(gap-fill) = fill the holes nearest START and push at its neighbour; P(wave-follow) = follow the wave, filling whichever hole it stops at (see difficulty.py).

elapsed: 99.1s