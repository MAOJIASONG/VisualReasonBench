# Experiment 1 on Stacking: dead states and step value

Stacking is reversible (`pickup`), so a wrong placement leads to a *dead state* (the placed pieces fit no completion) that can be left again with pickups. Oracle: exact cover of the free cells by the unplaced pieces (`stacking_oracle/core.py`); a dead state is *hidden* while every remaining piece still fits somewhere and every free cell can still be covered.

## Trajectories

fatal = first accepted placement into a dead state; delay = actions from it until the dead state is visible; repaired = left the dead state again (pickups); failed episodes split by how they ended.

| model | episodes | won | with a fatal step | fatal step (median) | delay to exposure (median) | never exposed | repaired | actions in dead states | failed: ended dead | failed: ended solvable | rejected actions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Opus base | 77 | 64% | 68% | 6.0 | 3.0 | 10% | 54% | 33% | 89% | 11% | 39% |
| Opus plan | 77 | 64% | 65% | 7.0 | 2.5 | 12% | 52% | 32% | 93% | 7% | 42% |
| Sonnet base | 77 | 39% | 64% | 5.0 | 3.0 | 10% | 37% | 41% | 81% | 19% | 48% |
| Sonnet plan | 77 | 43% | 69% | 5.0 | 3.0 | 15% | 32% | 43% | 82% | 18% | 50% |
| luna base | 77 | 27% | 68% | 5.0 | 0.0 | 21% | 35% | 41% | 73% | 27% | 57% |
| luna plan | 77 | 32% | 65% | 6.0 | 1.0 | 12% | 34% | 42% | 79% | 21% | 56% |
| sol base | 77 | 79% | 40% | 8.0 | 1.5 | 16% | 68% | 15% | 81% | 19% | 29% |
| sol plan | 77 | 77% | 40% | 6.0 | 1.0 | 13% | 65% | 16% | 89% | 11% | 25% |
| terra base | 77 | 58% | 53% | 6.0 | 3.0 | 2% | 54% | 31% | 91% | 9% | 36% |
| terra plan | 77 | 52% | 61% | 5.0 | 1.0 | 2% | 62% | 35% | 92% | 8% | 34% |

### By difficulty (won / with a fatal step / ended dead)

| model | easy | mid | hard |
|---|---|---|---|
| Opus base | 10/10 · 0 · 0 | 19/20 · 10 · 1 | 20/47 · 42 · 24 |
| Opus plan | 10/10 · 0 · 0 | 18/20 · 8 · 2 | 21/47 · 42 · 24 |
| Sonnet base | 10/10 · 0 · 0 | 14/20 · 11 · 6 | 6/47 · 38 · 32 |
| Sonnet plan | 10/10 · 0 · 0 | 15/20 · 12 · 5 | 8/47 · 41 · 31 |
| luna base | 10/10 · 1 · 0 | 9/20 · 16 · 10 | 2/47 · 35 · 31 |
| luna plan | 10/10 · 1 · 0 | 11/20 · 14 · 9 | 4/47 · 35 · 32 |
| sol base | 10/10 · 0 · 0 | 20/20 · 2 · 0 | 31/47 · 29 · 13 |
| sol plan | 10/10 · 0 · 0 | 20/20 · 3 · 0 | 29/47 · 28 · 16 |
| terra base | 10/10 · 0 · 0 | 19/20 · 3 · 1 | 16/47 · 38 · 28 |
| terra plan | 10/10 · 1 · 0 | 18/20 · 4 · 1 | 12/47 · 42 · 33 |

Engine check: 0 logged actions where the oracle's rules disagree with the game's verdict (0 expected).
