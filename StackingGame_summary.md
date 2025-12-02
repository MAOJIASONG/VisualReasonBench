# Stacking Game Integration Summary

## What was added
- Introduced a discrete, non-PyBullet environment wrapper: `src/phyvpuzzle/environment/stacking_game_env.py`
  - Tools for LLM: `list_puzzles`, `load_puzzle`, `place_piece`, `place_piece_by_cells`, `pickup_piece`, `get_piece_info`, `finish`
  - PIL rendering via stacking_game 3D visualizer (matplotlib); auto fallback to an internal 2x2x2 demo if puzzle assets are missing
- Added task wrapper: `src/phyvpuzzle/tasks/stacking_game_task.py`
  - System/user prompts tailored for stacking_game
  - Rule-based success: complete when the box is fully filled
  - Passes puzzle hints (size/id/seed) into the environment before reset
- Registry wiring: exported new env/task in `environment/__init__.py` and `tasks/__init__.py`
- Runner improvement: uses environment-provided object summaries when available (works for non-PyBullet envs)

## How to use
- In YAML: set `environment.type: stacking_game` and `task.type: stacking_game`
- Configure puzzles via:
  - Environment: `puzzle_dir` (default `Stacking_scaling/puzzles_full_v9`), `default_size`, `default_puzzle_id`
  - Task: `puzzle_size`, `puzzle_id`, optional `init_seed`
- At runtime: call tools
  - `list_puzzles(size?)` to browse available puzzles (2x2x2 to 4x4x4 supported)
  - `load_puzzle(size, puzzle_id, seed?)` to reset to a chosen level
  - `place_piece` / `place_piece_by_cells` / `pickup_piece` / `get_piece_info` to solve; `finish` to mark done

## Notes
- Works without PyBullet; relies on `Stacking_scaling/stacking_game` modules and matplotlib for rendering
- If the dataset path is missing, a built-in 2x2x2 demo level is loaded

## Testing
- Not run (no automated tests executed).
