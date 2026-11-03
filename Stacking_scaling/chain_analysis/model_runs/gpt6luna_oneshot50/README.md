# Codex gpt-6-luna, one-shot, on the hard tier of `puzzles_50`

The same 45 hard levels and one-shot protocol as `../sonnet_oneshot50/`,
played on 2026-11-02 with OpenAI's `gpt-6-luna` through the Codex CLI
(`codex exec`, codex-cli 0.166.0, reasoning effort high).

* `run_codex_oneshot.py` opens each session, runs `codex exec` read-only in
  an empty folder with the start images attached (`-i obs_000_*.png`), takes
  the JSON list on the last line of the answer and plays it with
  `play_cli plan`.
* Prompts (`prompts/`, from `make_codex_prompts.py`) carry the same game
  rules as the Sonnet prompts; only the interaction part differs (answer with
  the plan instead of running a command).
* `answers/` holds every final answer; the Codex event logs showed one
  agent message per level and no command or tool call.

## Results

| game | won | failed verify | never verified | no plan | rejected actions / episode | accepted actions / episode |
|---|---|---|---|---|---|---|
| FluidNet | 0 / 15 | 0 | 15 | 0 | 3.7 | 0.5 |
| Domino | 2 / 15 | 4 | 9 | 0 | 2.4 | 3.6 |
| PipeFit | 0 / 15 | 11 | 0 | 4 | 1.6 | 7.5 |

* FluidNet: most valves named in the plans do not exist at those coordinates
  (`ValveNotFound`); the few accepted openings never fed a red cell (Exp 1:
  no dead end, V = 1), but no plan connected all tanks.
* Domino: patches placed on wrong cells or heights were rejected, so most
  plans never reached the push.
* PipeFit: 4 answers declined to give a plan ("the portal cells are not
  clear enough to place a leak-free network"); the rest submitted and failed
  (Exp 1: 73% of runs leave the solution set, at step 1.4 on average).

## Caveats

* The model reads the same 1960 × 1540 renders as Sonnet, but through Codex's
  image input; whether Codex downscales them was not checked, and the
  failure pattern (coordinates off) is what heavy downscaling would cause.
* One run per level.
