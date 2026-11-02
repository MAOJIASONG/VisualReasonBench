# Claude Sonnet on the hard tier of `puzzles_50`

All 45 hard levels (15 per game) of `<game>/puzzles_50/`, one closed-loop
episode each, played on 2026-11-02 by Claude Sonnet subagents of a Claude
Code session (one fresh agent per episode).  Same protocol as the pilot
(`../sonnet_pilot/README.md`): the agent sees only the rendered images and
a one-line result after each action, acts through
`python -m chain_analysis.play_cli act`, 40-action budget, may `["reset"]`
after a loss.

Nineteen episodes were cut off once by an API rate limit and resumed from
the same session folder; the game state is kept by `play_cli`, so a resume
continues the same episode.

## Audit

Every agent transcript was checked after the run:

* all game actions went through `play_cli act` (many agents sometimes sent
  several actions in one shell loop, so not every action followed a fresh
  look at the image);
* every file read was an `obs_*.png` of the agent's own session; no agent
  opened a level file, the hidden `.state.json`, source code or another
  session;
* minor rule breaks: 4 agents listed their own session folder (`ls`, which
  does not show the hidden state file) and one filtered the CLI output with
  `grep`;
* **two Domino agents (P19_B_decoy_hard, P48_E_full_hard) ran a numpy
  homography in Python to turn pixel positions into grid cells** — a
  calculator over the image, no hidden information, but outside the allowed
  command.  Both won on the first verify; without them Domino is 13/13 solved,
  11/13 on the first verify.

## Results

| game | solved | first-verify wins | mean actions | mean rejected | most failed verifies |
|---|---|---|---|---|---|
| FluidNet | 15 / 15 | 13 / 15 | 7.3 | 2.5 | 3 (P28_backdoor) |
| Domino | 15 / 15 | 13 / 15 | 9.5 | 1.1 | 2 (P49_E_full) |
| PipeFit | 15 / 15 | 11 / 15 | 18.9 | 1.1 | 1 (4 levels) |

Rejected actions are mostly misread coordinates (FluidNet: a valve guessed
one cell off; PipeFit: a pipe placed in a portal's own wall cell).

### Hard-tier success (%) at 1 / 2 / 5 / 64 verifies

| game | Claude Sonnet | random | greedy | greedy + undo | DFS | BFS |
|---|---|---|---|---|---|---|
| FluidNet | 87 / 93 / 100 / 100 | 3 / 3 / 16 / 69 | 13 / 16 / 25 / 83 | 13 / 27 / 73 / 93 | 0 / 5 / 28 / 92 | 0 / 0 / 0 / 100 |
| Domino | 87 / 93 / 100 / 100 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 4 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| PipeFit | 73 / 100 / 100 / 100 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 1 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 1 | 0 / 0 / 0 / 0 |

Baselines: the 15 hard levels of each game, seeded methods averaged over 5
seeds (`exp2_<game>/runs.jsonl`).  With 64 verifies, BFS needs all 64 to
match Sonnet on FluidNet; on Domino and PipeFit no baseline gets close.

### Experiment 1 — where Sonnet's steps go (hard tier)

| game | runs with a dead-end step | first dead-end step | steps until exposed | repaired before verifying | mean V | V − V_rand | wasted steps |
|---|---|---|---|---|---|---|---|
| FluidNet | 2 / 15 | 3 | 0.5 | – (irreversible) | 0.97 | +0.56 | 0.3 |
| Domino | 8 / 15 | 1.1 | 4 | 7 / 8 | 0.92 | +0.82 | 1.3 |
| PipeFit | 10 / 15 | 1.1 | – | 10 / 10 | 0.79 | +0.77 | 4.1 |

* FluidNet: Sonnet rarely opens a trap valve (2 of 15 runs); when it does,
  the breach shows at once and it wins after a reset.
* Domino and PipeFit: the first move is often off the solution set (a probe
  patch or pipe used to calibrate coordinates, or a misread cell), and it is
  almost always removed before the push / submit.  Domino's one unrepaired
  dead end was exposed 4 steps later by a failed push.
* The failed PipeFit submits came from later mistakes (a pipe left hanging
  in mid-air, a support floating at z = 2, a misread exit cell); the
  per-trajectory "first fatal step" only tracks the first dead end, which
  had been repaired.

## Files

`sessions/` (full logs), `traj_<game>.jsonl`, `results_<game>.jsonl`,
`exp1_<game>/REPORT.md`, `exp2_<game>/REPORT.md` — same layout as the pilot.

## Caveats

* One run per level, one model.  Sonnet solves every hard level with
  step-by-step visual feedback: on these levels the hard tier separates the
  search baselines (0–4% on Domino / PipeFit) but not a strong model.  What
  still costs Sonnet is perception (reading cells off the 3D view) and, on
  PipeFit, support / gravity mistakes caught only at submit.
* To make the tier bite for strong models: lower the action budget, allow
  one verify only, or remove the step-by-step images (the paper's one-shot
  setting).
