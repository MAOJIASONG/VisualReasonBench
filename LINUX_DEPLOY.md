# Running CHAIN on Linux (all 5 task families)

This is the `stacking_end` branch of `MAOJIASONG/VisualReasonBench`. It includes the
**Linux Unity build of the Puzzle (Luban) environment** in `Luban/` (committed; Unity's
debug symbols `LuBanSuoEnv_BurstDebugInformation_DoNotShip/` and the screenshots the player
saves under `LuBanSuoEnv_Data/StreamingAssets/` are not).

| Task family | Code | Renderer | Needs a display? |
|---|---|---|---|
| Puzzle (Luban lock) | `src/phyvpuzzle/environment/luban_env.py` + `Luban/LuBanSuoEnv.x86_64` (Unity 2022.3.62f2) | Unity player, socket `127.0.0.1:9999` | **Yes** — use `xvfb-run` on a headless server |
| Stacking | `src/phyvpuzzle/environment/stacking_game_env.py` | matplotlib (Agg) | No |
| FluidNet / Domino / PipeFit | `Stacking_scaling/type_3`, `type_3_1`, `type_2_3` via `Stacking_scaling/chain_analysis/play_cli.py` | matplotlib (Agg) | No |

## 1. Get the code on the Linux machine

```bash
git clone -b stacking_end https://github.com/MAOJIASONG/VisualReasonBench.git CHAIN-linux
cd CHAIN-linux && chmod +x Luban/LuBanSuoEnv.x86_64
```

The clone includes `Luban/` (~106 MB). The Stacking puzzles (`puzzles_full_v9`, used by the
agent harness and the Stacking oracle) are not in this repository: put them in
`Stacking_scaling/puzzles_full_v9/` or point `STACK_PUZZLE_DIR` at them.

Requirements: x86_64 Linux (not ARM), Python ≥ 3.10.

## 2. One-time setup

```bash
cd ~/programs/2026/CHAIN-linux
conda create -n chain python=3.10 -y && conda activate chain   # or any venv
bash scripts/linux/setup_linux.sh      # apt deps (xvfb, GL libs), pip install -e ., .env template
```

Then fill in `OPENAI_API_KEY` in `.env` (OpenRouter base URL is pre-filled).

## 3. Smoke test the Puzzle environment (no API calls)

```bash
bash scripts/linux/run_luban_linux.sh --smoke
```

Expected: `OK: Unity server is listening on 127.0.0.1:9999`. If it fails, check
`logs/unity_smoke.log` (missing GL/X11 libraries are the usual cause).

## 4. Run experiments

Puzzle (32 levels, indices 0–31; up to 8 parallel env_ids served by one Unity process):

```bash
bash scripts/linux/run_luban_linux.sh --config eval_configs/luban.yaml --levels 0 --runs-per-level 1
bash scripts/linux/run_luban_linux.sh --config eval_configs/luban.yaml --level-range 0-31 --workers 8
```

Stacking:

```bash
python -m phyvpuzzle.stacking_runner --config eval_configs/<stacking config>.yaml --level-range 0-76 --workers 8
```

FluidNet / Domino / PipeFit (step-by-step CLI driven by an agent; see
`Stacking_scaling/chain_analysis/README.md` and `model_runs/README.md`):

```bash
cd Stacking_scaling
python -m chain_analysis.play_cli tools fluidnet
python -m chain_analysis.play_cli start type_3/puzzles_50/<level_dir> /tmp/sess1
python -m chain_analysis.play_cli act /tmp/sess1 '<json action>'
```

Puzzle with the agent harnesses (Claude Code / Codex CLI) on a Slurm cluster: one Unity
player per job serves up to 8 agents in parallel (env_id 1–8), under the `Xvfb` module:

```bash
sbatch --chdir=cc_runs/luban32 Stacking_scaling/chain_analysis/cc_harness/luban/run_luban_lanes.sbatch \
    0-31 luban32 base claude:opus claude:sonnet codex:gpt-6.1-sol:high
```

Shortest solutions per Puzzle level (BFS over `move` commands with the game's own collision
code, replayed in Unity to check): `tools/luban_solver/` (see its README; needs .NET 8).

## Notes

- The Linux Unity build is dated 2026-01-24 (same Unity version as the editor source,
  2022.3.62f2). The newest Windows build is 2026-03-08; if levels changed in between,
  rebuild Linux from the newer source.
- `luban_env.py` and `start_luban.py` pick the binary by OS (`.x86_64` on Linux,
  `.exe` on Windows, `.app` on macOS); override with `environment.exe_path` in the YAML.
