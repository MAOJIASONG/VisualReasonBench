# Model runs on the CHAIN games

Models playing FluidNet (`type_3`), Domino (`type_3_1`) and PipeFit
(`type_2_3`) through `chain_analysis/play_cli.py`, then scored with
Experiment 1 (step value, dead ends) and Experiment 2 (search baselines
under the same verify budget).  The player only ever sees rendered images;
the level files and the hidden game state stay out of reach.

| run | model | protocol | levels | details |
|---|---|---|---|---|
| `sonnet_pilot/` | Claude Sonnet | closed loop | 15 demo levels (`images_batch/`) | [README](sonnet_pilot/README.md) |
| `sonnet_hard50/` | Claude Sonnet | closed loop | 45 hard levels of `puzzles_50/` | [README](sonnet_hard50/README.md) |
| `sonnet_oneshot50/` | Claude Sonnet | one-shot | same 45 | [README](sonnet_oneshot50/README.md) |
| `gpt6luna_oneshot50/` | Codex gpt-6-luna | one-shot | same 45 | [README](gpt6luna_oneshot50/README.md) |

## Protocols

* **Closed loop** (`play_cli act`): one action per call; after each action
  the player gets the new image(s) and a one-line result.  40-action budget;
  after a loss the player may `["reset"]` (counted as a failed verify).
* **One-shot** (`play_cli plan`): the player sees only the start images
  (`obs_000_*`) and sends its whole plan as one JSON list.  The plan is
  played through the game rules with no observation in between, rejected
  actions are skipped (they still count toward the 40-action cap), and the
  session is closed.  `reset` / `quit` are not allowed in a plan.

## Results on the 45 hard levels (wins out of 15)

| game | Sonnet closed loop: solved | Sonnet closed loop: 1st verify | Sonnet one-shot | gpt-6-luna one-shot | best baseline at 1 verify | best baseline at 64 verifies |
|---|---|---|---|---|---|---|
| FluidNet | 15 | 13 | 15 | 0 | 13% (greedy) | 100% (BFS) |
| Domino | 15 | 13 | 12 | 2 | 0% | 4% (greedy) |
| PipeFit | 15 | 11 | 10 | 0 | 0% | 1% (greedy, DFS) |

* The hard tier separates the search baselines and separates the two
  models, but it does not test Sonnet's planning depth: without any feedback
  Sonnet scores within 1–2 levels of its closed-loop first attempt, and 17 of
  its 44 one-shot plans equal the action list of its final closed-loop
  episode.  It plans from the start image; the feedback mainly catches
  misread cells.
* Sonnet's failures, closed loop or one-shot, are perception: step heights
  in Domino, portal wall cells and supports in PipeFit.
* gpt-6-luna mostly misreads coordinates (FluidNet: 3.7 of ~4 valves per
  plan do not exist where it names them) and declined to give a plan on 4
  PipeFit levels.  Whether Codex downscales the 1960 × 1540 images was not
  checked.
* To make the tier test planning, levels need longer dependency chains
  (traps exposed several steps later, more required actions), not only
  harder-to-read scenes.

The pilot (15 demo levels, closed loop) solved every level, 14 of 15 on the
first verify.

## Audit

Every run was audited from the agents' own logs before scoring:

* Sonnet (Claude Code subagents): every tool call was checked.  Closed loop:
  only `play_cli act` and reads of the agent's own `obs_*.png` (two Domino
  agents in `sonnet_hard50` used a Python homography as a coordinate
  calculator; flagged there).  One-shot: each agent read only its prompt and
  its own `obs_000_*.png`, and ran exactly one `play_cli plan` call.
* Codex: run read-only in an empty folder with the start images attached;
  the event logs show one answer per level and no command or tool call.

## Files in each run

| file | content |
|---|---|
| `sessions/<game>__<level>.json` | full session log: every action, its result, rejected actions (one-shot: also the submitted `plan`) |
| `traj_<game>.jsonl` | accepted actions per episode (Experiment 1 input) |
| `results_<game>.jsonl` | win / verifies / actions / rejected per episode (Experiment 2 input) |
| `exp1_<game>/REPORT.md` | step-value report on these trajectories |
| `exp2_<game>/REPORT.md` | search baselines on the same levels, with the model's row |
| `prompts/` | one-shot runs: the exact instruction given for each level |
| `answers/` | Codex run: each final answer |

## Reproducing a one-shot run

Run from `Stacking_scaling/`.  `<out>/map.txt` lists one
`<session dir> <level dir>` pair per line, with session folders named
`NN_<game>` (e.g. `<out>/01_type_3 type_3/puzzles_50/P08_trace_hard_seed14004`).

```bash
# 1. open every session
while read sess level; do python -m chain_analysis.play_cli start $level $sess; done < <out>/map.txt
# 2a. Claude: write prompts/NN.md, then give each to a fresh agent
#     ("Read <out>/prompts/NN.md and follow its instructions exactly")
mkdir -p <out>/prompts
python chain_analysis/model_runs/sonnet_oneshot50/make_prompts.py <out>
# 2b. Codex: write the prompts and run codex exec on every session (8 in parallel)
python chain_analysis/model_runs/gpt6luna_oneshot50/make_codex_prompts.py <out>
python chain_analysis/model_runs/gpt6luna_oneshot50/run_codex_oneshot.py <out> gpt-6-luna 8
# 3. export and score, per game
python -m chain_analysis.play_cli export <out>/*_type_3 --model <name> \
    --out traj_type_3.jsonl --results results_type_3.jsonl
python -m chain_analysis.exp1_step_value --root type_3/puzzles_50 --traj traj_type_3.jsonl --out exp1_type_3
python -m chain_analysis.exp2_budget_search --root type_3/puzzles_50 --models results_type_3.jsonl --out exp2_type_3
```

Codex needs network access; set `HTTPS_PROXY` in the environment if the
machine reaches the internet only through a proxy.

## Caveats

* One run per level, one sample per model; no seeds or temperatures were
  varied.
* Experiment 2 baselines on the hard tier average 5 seeds per seeded method.
