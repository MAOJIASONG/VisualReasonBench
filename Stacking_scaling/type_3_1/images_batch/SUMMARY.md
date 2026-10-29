# Domino Chain Reaction 3D — 10-level batch (rules v2)

| # | label | category | tier | seed | grid | fixed | patches | decoy gaps | solutions | P(random) | P(gap-fill) | P(wave-follow) | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | E1_easy_line | A_line | easy | 1000 | (6, 5, 3) | 4 | 1 | 0 | 1 | 0.007 | 1.00 | 0.33 | 1.6 |
| 2 | E2_easy_decoy | B_decoy | easy | 2003 | (5, 5, 3) | 4 | 2 | 1 | 1 | 0.002 | 0.00 | 0.33 | 3.1 |
| 3 | E3_easy_split | D_split | easy | 3000 | (6, 6, 3) | 6 | 2 | 0 | 1 | 0.000 | 1.00 | 0.25 | 3.0 |
| 4 | M1_med_decoy | B_decoy | medium | 4007 | (7, 6, 3) | 5 | 3 | 2 | 1 | 0.000 | 0.00 | 0.33 | 4.6 |
| 5 | M2_med_steps | C_steps | medium | 5006 | (7, 5, 3) | 11 | 3 | 2 | 1 | 0.000 | 0.00 | 0.50 | 5.0 |
| 6 | M3_med_split | D_split | medium | 6000 | (7, 6, 3) | 7 | 3 | 2 | 1 | 0.000 | 0.00 | 0.17 | 5.6 |
| 7 | M4_med_full | E_full | medium | 7001 | (7, 6, 3) | 15 | 3 | 2 | 1 | 0.000 | 0.00 | 0.33 | 5.6 |
| 8 | H1_hard_steps | C_steps | hard | 8010 | (8, 6, 4) | 9 | 4 | 3 | 1 | 0.000 | 0.00 | 0.33 | 7.6 |
| 9 | H2_hard_split | D_split | hard | 9000 | (8, 7, 3) | 11 | 4 | 4 | 1 | 0.000 | 0.00 | 0.12 | 8.0 |
| 10 | H3_hard_full | E_full | hard | 10000 | (8, 8, 4) | 16 | 4 | 3 | 1 | 0.000 | 0.00 | 0.25 | 7.5 |

P(random) = random legal placement + random push direction; P(gap-fill) = fill the holes nearest START and push at its neighbour; P(wave-follow) = follow the wave, filling whichever hole it stops at (see difficulty.py).

| tier | levels | patches | decoy gaps | score |
|---|---|---|---|---|
| easy | 3 | 1–2 | 0–1 | 1.59–3.09 |
| medium | 4 | 3–3 | 2–2 | 4.59–5.59 |
| hard | 3 | 4–4 | 3–4 | 7.5–8 |

elapsed: 105.0s