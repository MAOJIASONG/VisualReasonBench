# Luban (Puzzle) level solver — prototype

Searches for the shortest sequence of game `move` commands that solves a Luban level
(the 32 burr-puzzle levels of the Unity game in `Luban/`), and replays it in the game to check.

## How it matches the game

- **Levels**: `levels/Puzzle_<k>.json` are the game's own level files, extracted from
  `Luban/LuBanSuoEnv_Data/resources.assets` by `extract_levels.py` (`level_index` k = `Puzzle_k`).
- **Collisions**: `game_core/` is the game's geometry code (exact fractions, convex polyhedra),
  decompiled from `Luban/LuBanSuoEnv_Data/Managed/Assembly-CSharp.dll` and compiled unchanged;
  `UnityShim.cs` stands in for the few UnityEngine types it touches. `Move1SBeforeCross` in
  `Program.cs` is a copy of the game's function that decides how far a piece may move.
- **Moves**: what the API offers an agent — one piece, along ±x/±y/±z, any distance up to the
  blocking point. Pieces start at `position1` / `switchAxisTypeXYZ1`, as in the game.
- **Solved**: the game marks a piece finished when it overlaps the finish zone (scene object
  `胜利`: six slabs 150–450 units from the mean start position along ±x/±y/±z, 600 units wide);
  the level is solved when N−1 pieces are finished. Finished pieces stay in the scene and still
  block other pieces. (Moving a piece too far, e.g. 500+, jumps past the zone and does not count.)

## Search

Breadth-first over game commands (each `move` = 1 step), so the first solution found has the
fewest commands. To keep the state space small:
- a piece only stops where one of its faces lines up with a face of another piece (between such
  points nothing changes for axis-aligned pieces), at its blocking point, or in the finish zone;
- if some piece can go straight into the finish zone, only that move is taken (every piece needs
  one such move; this keeps the shortest length unless a parked piece blocks another's exit).
  `--no-greedy` turns this off, `--grid` stops at every grid step instead of alignments.

Not searched yet: **rotations** (the game checks them with Unity physics in 1° steps, not with
the exact geometry) and moving several pieces together (the API cannot).

## Usage

```bash
dotnet build -c Release
dotnet bin/Release/net8.0/LubanSolver.dll levels/Puzzle_1.json --out results/Puzzle_1.json
dotnet bin/Release/net8.0/LubanSolver.dll levels/Puzzle_5.json --group-check   # which groups could move
```

Replay in Unity on the cluster (starts the player under Xvfb, sends the moves, prints `is_solved`):

```bash
srun ... bash replay_cluster.sh results/Puzzle_0.json results/Puzzle_1.json
srun ... bash replay_cluster.sh --cmd python probe_jump.py 0 0 z -100 -200 -500   # finish-zone probe
```
