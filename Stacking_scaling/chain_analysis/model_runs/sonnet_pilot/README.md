# Pilot: Claude Sonnet playing the CHAIN games

15 closed-loop episodes, 5 per game (2 easy, 1 medium, 2 hard), on the demo
levels in `<game>/images_batch/`.  Played on 2026-11-02 by Claude Sonnet
subagents of a Claude Code session, one fresh agent per episode.

## Protocol

* Each agent got the game rules, the action syntax and its session folder,
  and was played through `python -m chain_analysis.play_cli act` only.
* Observation after every action: the rendered image(s) (`front`, plus
  `back` for FluidNet and PipeFit) and a one-line result.  No level file,
  coordinates list or solution was shown.
* Budget 40 actions; after a loss the agent could `["reset"]` (each loss is a
  failed verify).
* Audit: the agents' tool calls were checked afterwards — every shell call
  was `play_cli act` (some agents sent several actions in one shell call
  without looking at the image in between) and every file read was an
  `obs_*.png` of the agent's own session.

## Files

| file | content |
|---|---|
| `sessions/<game>__<level>.json` | full session log: every action, its result, rejected actions |
| `traj_<game>.jsonl` | accepted actions per episode (Experiment 1 input) |
| `results_<game>.jsonl` | win / verifies / actions per episode (Experiment 2 input) |
| `exp1_<game>/REPORT.md` | step-value report on these trajectories |
| `exp2_<game>/REPORT.md` | search baselines on the same 10 demo levels, with Sonnet's row |

Reproduce the reports:

```bash
python -m chain_analysis.exp1_step_value --root type_3/images_batch \
    --traj chain_analysis/model_runs/sonnet_pilot/traj_type_3.jsonl \
    --out chain_analysis/model_runs/sonnet_pilot/exp1_type_3
python -m chain_analysis.exp2_budget_search --root type_3/images_batch \
    --models chain_analysis/model_runs/sonnet_pilot/results_type_3.jsonl \
    --out chain_analysis/model_runs/sonnet_pilot/exp2_type_3
```

## Results

| game | level | tier | won | failed verifies | actions | rejected |
|---|---|---|---|---|---|---|
| FluidNet | level_01_easy_trace | easy | yes | 0 | 1 | 0 |
| FluidNet | level_02_easy_series | easy | yes | 0 | 3 | 1 |
| FluidNet | level_04_medium_trace | medium | yes | 0 | 5 | 2 |
| FluidNet | level_09_hard_backdoor | hard | yes | 0 | 5 | 1 |
| FluidNet | level_10_hard_mixed | hard | yes | 1 | 11 | 2 |
| Domino | E1_easy_line | easy | yes | 0 | 4 | 1 |
| Domino | E2_easy_decoy | easy | yes | 0 | 4 | 0 |
| Domino | M1_med_decoy | medium | yes | 0 | 7 | 0 |
| Domino | H2_hard_split | hard | yes | 0 | 6 | 0 |
| Domino | H3_hard_full | hard | yes | 0 | 7 | 1 |
| PipeFit | level_01_easy | easy | yes | 0 | 8 | 0 |
| PipeFit | level_02_easy | easy | yes | 0 | 16 | 1 |
| PipeFit | level_04_medium | medium | yes | 0 | 10 | 1 |
| PipeFit | level_09_hard | hard | yes | 0 | 24 | 2 |
| PipeFit | level_10_hard | hard | yes | 0 | 23 | 2 |

**15 / 15 solved, 14 on the first verify.**  Most rejected actions were
misread coordinates (a valve or step one cell off, a pipe placed in a
portal's wall cell).

### Experiment 1 — where Sonnet's steps go

| game | runs with a dead-end step | what happened | mean V | V − V_rand |
|---|---|---|---|---|
| FluidNet | 1 / 5 (hard) | opened a valve that fed the red cells; exposed at once (delay 0), won after a reset | 0.94–1.0 | ≈ +0.5 |
| Domino | 1 / 5 (medium) | placed a probe patch to calibrate coordinates, removed it next step | 0.83–1.0 | +0.70 to +0.91 |
| PipeFit | 4 / 5 | first pipes off the solution set (wrong cell next to a portal, hidden obstacle); repaired by removals before submitting | 0.72–0.79 | +0.71 to +0.79 |

Sonnet repairs before it verifies: in Domino and PipeFit every dead end was
undone before the push / submit, so no failed verify was needed.  The cost
shows up as wasted steps instead (PipeFit hard: 5 per episode).

### Experiment 2 — verifies the baselines need to match Sonnet

On Sonnet's 5 levels per game, the baselines reach Sonnet's 100% only with
these verify budgets (`>64` = not within the run cap):

| game | random | greedy | greedy + undo | DFS | BFS |
|---|---|---|---|---|---|
| FluidNet | >64 | >64 | 36 | >64 | >64 |
| Domino | >64 | >64 | >64 | >64 | >64 |
| PipeFit | >64 | >64 | >64 | >64 | >64 |

Sonnet's success is not reachable by blind trial and error at its own
budget (1–2 verifies).

## Caveats

* Pilot size: 5 episodes per game, one run each, demo levels only.  This
  shows the pipeline works end to end; it is not a benchmark score.
* Sonnet solved every level, including the hard ones.  Under step-by-step
  visual feedback the current hard tier does not separate a strong model;
  the `puzzles_50` hard levels and a lower action budget are the next things
  to try.
* Batched actions: some agents sent several actions per shell call, so not
  every action was preceded by a fresh look at the image.
