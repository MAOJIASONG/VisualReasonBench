# Experiment 2 on Stacking: search baselines under the same action budget

Budget = actions (each place / pickup).  Baselines know the rules and the true state and never make an illegal move; models' budgets also count rejected actions and `state` calls.  Baselines: mean over seeds of the share of levels solved.

## all (77 levels)

| method | ≤5 | ≤10 | ≤20 | ≤30 | ≤50 | ≤100 | ≤200 | ≤500 | ≤1000 |
|---|---|---|---|---|---|---|---|---|---|
| random | 10% | 15% | 20% | 23% | 27% | 30% | 36% | 46% | 55% |
| greedy | 11% | 15% | 20% | 24% | 30% | 36% | 43% | 53% | 61% |
| greedy_undo | 11% | 18% | 25% | 29% | 35% | 42% | 52% | 64% | 73% |
| dfs_prune | 12% | 22% | 28% | 33% | 38% | 49% | 60% | 72% | 84% |
| **Opus base** | 21% | 35% | 53% | 64% | 64% | 64% | 64% | 64% | 64% |
| **Opus plan** | 18% | 31% | 52% | 64% | 64% | 64% | 64% | 64% | 64% |
| **Sonnet base** | 16% | 25% | 32% | 39% | 39% | 39% | 39% | 39% | 39% |
| **Sonnet plan** | 14% | 21% | 32% | 43% | 43% | 43% | 43% | 43% | 43% |
| **luna base** | 14% | 21% | 26% | 27% | 27% | 27% | 27% | 27% | 27% |
| **luna plan** | 16% | 19% | 26% | 32% | 32% | 32% | 32% | 32% | 32% |
| **sol base** | 26% | 51% | 74% | 79% | 79% | 79% | 79% | 79% | 79% |
| **sol plan** | 25% | 48% | 73% | 77% | 77% | 77% | 77% | 77% | 77% |
| **terra base** | 25% | 44% | 53% | 58% | 58% | 58% | 58% | 58% | 58% |
| **terra plan** | 23% | 40% | 48% | 52% | 52% | 52% | 52% | 52% | 52% |

Actions a baseline needs to match each model's success rate on these levels (model: share won within its own budget; '>1000' = never within the cap):

| model | won | random | greedy | greedy_undo | dfs_prune |
|---|---|---|---|---|---|
| Opus base | 64% | >1000 | >1000 | 496 | 263 |
| Opus plan | 64% | >1000 | >1000 | 496 | 263 |
| Sonnet base | 39% | 240 | 136 | 82 | 52 |
| Sonnet plan | 43% | 322 | 186 | 102 | 67 |
| luna base | 27% | 56 | 38 | 24 | 17 |
| luna plan | 32% | 126 | 76 | 42 | 30 |
| sol base | 79% | >1000 | >1000 | >1000 | 720 |
| sol plan | 77% | >1000 | >1000 | >1000 | 619 |
| terra base | 58% | >1000 | 876 | 316 | 178 |
| terra plan | 52% | 784 | 460 | 188 | 124 |

## easy (10 levels)

| method | ≤5 | ≤10 | ≤20 | ≤30 | ≤50 | ≤100 | ≤200 | ≤500 | ≤1000 |
|---|---|---|---|---|---|---|---|---|---|
| random | 48% | 60% | 78% | 90% | 98% | 100% | 100% | 100% | 100% |
| greedy | 64% | 78% | 88% | 92% | 100% | 100% | 100% | 100% | 100% |
| greedy_undo | 64% | 76% | 90% | 94% | 96% | 98% | 98% | 100% | 100% |
| dfs_prune | 64% | 88% | 94% | 94% | 96% | 100% | 100% | 100% | 100% |
| **Opus base** | 90% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **Opus plan** | 70% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **Sonnet base** | 70% | 90% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **Sonnet plan** | 60% | 80% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **luna base** | 80% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **luna plan** | 80% | 90% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **sol base** | 90% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **sol plan** | 90% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **terra base** | 90% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **terra plan** | 80% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

Actions a baseline needs to match each model's success rate on these levels (model: share won within its own budget; '>1000' = never within the cap):

| model | won | random | greedy | greedy_undo | dfs_prune |
|---|---|---|---|---|---|
| Opus base | 100% | 61 | 46 | 258 | 87 |
| Opus plan | 100% | 61 | 46 | 258 | 87 |
| Sonnet base | 100% | 61 | 46 | 258 | 87 |
| Sonnet plan | 100% | 61 | 46 | 258 | 87 |
| luna base | 100% | 61 | 46 | 258 | 87 |
| luna plan | 100% | 61 | 46 | 258 | 87 |
| sol base | 100% | 61 | 46 | 258 | 87 |
| sol plan | 100% | 61 | 46 | 258 | 87 |
| terra base | 100% | 61 | 46 | 258 | 87 |
| terra plan | 100% | 61 | 46 | 258 | 87 |

## mid (20 levels)

| method | ≤5 | ≤10 | ≤20 | ≤30 | ≤50 | ≤100 | ≤200 | ≤500 | ≤1000 |
|---|---|---|---|---|---|---|---|---|---|
| random | 14% | 25% | 34% | 40% | 46% | 56% | 66% | 81% | 90% |
| greedy | 10% | 18% | 28% | 35% | 48% | 59% | 70% | 84% | 92% |
| greedy_undo | 11% | 26% | 42% | 46% | 53% | 61% | 75% | 85% | 95% |
| dfs_prune | 13% | 35% | 47% | 52% | 59% | 72% | 82% | 91% | 96% |
| **Opus base** | 35% | 55% | 85% | 95% | 95% | 95% | 95% | 95% | 95% |
| **Opus plan** | 35% | 55% | 70% | 90% | 90% | 90% | 90% | 90% | 90% |
| **Sonnet base** | 25% | 45% | 60% | 70% | 70% | 70% | 70% | 70% | 70% |
| **Sonnet plan** | 25% | 40% | 60% | 75% | 75% | 75% | 75% | 75% | 75% |
| **luna base** | 15% | 30% | 45% | 45% | 45% | 45% | 45% | 45% | 45% |
| **luna plan** | 20% | 25% | 45% | 55% | 55% | 55% | 55% | 55% | 55% |
| **sol base** | 50% | 90% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **sol plan** | 45% | 85% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| **terra base** | 45% | 85% | 95% | 95% | 95% | 95% | 95% | 95% | 95% |
| **terra plan** | 50% | 90% | 90% | 90% | 90% | 90% | 90% | 90% | 90% |

Actions a baseline needs to match each model's success rate on these levels (model: share won within its own budget; '>1000' = never within the cap):

| model | won | random | greedy | greedy_undo | dfs_prune |
|---|---|---|---|---|---|
| Opus base | 95% | >1000 | >1000 | 865 | 720 |
| Opus plan | 90% | 926 | 754 | 704 | 340 |
| Sonnet base | 70% | 238 | 186 | 144 | 84 |
| Sonnet plan | 75% | 284 | 258 | 196 | 124 |
| luna base | 45% | 43 | 42 | 29 | 16 |
| luna plan | 55% | 96 | 84 | 52 | 34 |
| sol base | 100% | >1000 | >1000 | >1000 | >1000 |
| sol plan | 100% | >1000 | >1000 | >1000 | >1000 |
| terra base | 95% | >1000 | >1000 | 865 | 720 |
| terra plan | 90% | 926 | 754 | 704 | 340 |

## hard (47 levels)

| method | ≤5 | ≤10 | ≤20 | ≤30 | ≤50 | ≤100 | ≤200 | ≤500 | ≤1000 |
|---|---|---|---|---|---|---|---|---|---|
| random | 0% | 0% | 1% | 2% | 3% | 4% | 10% | 20% | 31% |
| greedy | 0% | 1% | 3% | 4% | 8% | 12% | 20% | 30% | 39% |
| greedy_undo | 0% | 2% | 5% | 8% | 14% | 23% | 33% | 47% | 58% |
| dfs_prune | 0% | 3% | 6% | 12% | 17% | 29% | 43% | 59% | 76% |
| **Opus base** | 0% | 13% | 30% | 43% | 43% | 43% | 43% | 43% | 43% |
| **Opus plan** | 0% | 6% | 34% | 45% | 45% | 45% | 45% | 45% | 45% |
| **Sonnet base** | 0% | 2% | 6% | 13% | 13% | 13% | 13% | 13% | 13% |
| **Sonnet plan** | 0% | 0% | 6% | 17% | 17% | 17% | 17% | 17% | 17% |
| **luna base** | 0% | 0% | 2% | 4% | 4% | 4% | 4% | 4% | 4% |
| **luna plan** | 0% | 2% | 2% | 9% | 9% | 9% | 9% | 9% | 9% |
| **sol base** | 2% | 23% | 57% | 66% | 66% | 66% | 66% | 66% | 66% |
| **sol plan** | 2% | 21% | 55% | 62% | 62% | 62% | 62% | 62% | 62% |
| **terra base** | 2% | 15% | 26% | 34% | 34% | 34% | 34% | 34% | 34% |
| **terra plan** | 0% | 6% | 19% | 26% | 26% | 26% | 26% | 26% | 26% |

Actions a baseline needs to match each model's success rate on these levels (model: share won within its own budget; '>1000' = never within the cap):

| model | won | random | greedy | greedy_undo | dfs_prune |
|---|---|---|---|---|---|
| Opus base | 43% | >1000 | >1000 | 424 | 198 |
| Opus plan | 45% | >1000 | >1000 | 456 | 232 |
| Sonnet base | 13% | 240 | 104 | 48 | 32 |
| Sonnet plan | 17% | 340 | 158 | 70 | 50 |
| luna base | 4% | 95 | 28 | 18 | 14 |
| luna plan | 9% | 152 | 58 | 31 | 22 |
| sol base | 66% | >1000 | >1000 | >1000 | 680 |
| sol plan | 62% | >1000 | >1000 | >1000 | 550 |
| terra base | 34% | >1000 | 812 | 220 | 141 |
| terra plan | 26% | 760 | 339 | 126 | 78 |

