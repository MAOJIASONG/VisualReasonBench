# Polycube Puzzle Batch Generator

## Overview

This directory contains a sophisticated **3D polycube puzzle generator** (`polypuzzle_batch.py`) that creates physically feasible assembly puzzles with various difficulty levels and constraints. The generator produces puzzles where pieces must be assembled into rectangular boxes following specific rules.

## Key Features

### 1. **Constraint-Based Generation**
- **No 2×3 planes**: Prevents flat 2×3 rectangular surfaces that would make pieces too trivial
- **No isolated pieces**: Ensures every piece touches at least one other piece in the solution
- **Connected pieces**: Each piece is a connected set of voxels (no floating cubes)
- **Linear assembly**: Most puzzles require sequential assembly where each piece can be inserted from a specific direction

### 2. **Difficulty Control**
- Uses **DLX (Dancing Links X)** algorithm to solve exact cover problems
- Tracks **visited nodes** during solving as a difficulty metric
- Configurable minimum difficulty thresholds per puzzle size
- Staged generation: attempts 4-voxel minimum first, falls back to 3-voxel if needed

### 3. **Physical Feasibility**
- Verifies each puzzle has a valid assembly order
- Checks that pieces can be inserted/removed in specific directions (+x, -x, +y, -y, +z, -z)
- Ensures no geometric impossibilities (e.g., pieces blocking each other)

### 4. **Deduplication**
- Uses shape signatures based on **canonical piece representations**
- Considers all 24 rotations of each piece
- Atomic file-based deduplication prevents duplicate puzzles across parallel workers

## Algorithm Components

### Geometry Operations
- **24 rotation matrices**: All rigid rotations preserving handedness
- **Normalization**: Translates shapes to start at (0,0,0)
- **Placement caching**: Pre-computes all valid placements of rotated pieces

### Partition Generation
- **Guided partition**: Uses region-growing from random seeds
- **Min/max constraints**: Controls piece size (typically 3-5 voxels, max 5 voxels per piece)
- **Piece count limits**: Adjusts number of pieces based on box volume

### Exact Cover Solver (DLX)
- **Anchor piece optimization**: Chooses piece with fewest placements, fixes it at origin
- **MRV heuristic**: Selects columns with minimum remaining values first
- **Piece-use priority**: Prioritizes filling pieces before voxel positions

### Assembly Order Finding
- Iteratively finds pieces that can be removed in one of 6 directions
- Reverses removal order to get assembly sequence
- Returns `None` if no linear assembly exists

## Configuration

### Puzzle Size Settings (PER_SIZE_CFG)

| Size   | Linear | Non-linear | Min Difficulty (nodes) | Max Pieces |
|--------|--------|------------|------------------------|------------|
| 2×2×2  | 1      | 1          | 0                      | 2          |
| 2×2×3  | 10     | 0          | 60                     | 3          |
| 2×3×3  | 10     | 0          | 90                     | 5          |
| 3×3×3  | 10     | 0          | 140                    | 6          |
| 2×2×4  | 10     | 0          | 80                     | 4          |
| 2×3×4  | 10     | 0          | 120                    | 6          |
| 3×3×4  | 10     | 0          | 160                    | 8          |
| 2×4×4  | 10     | 0          | 150                    | 7          |
| 3×4×4  | 10     | 0          | 200                    | 10         |
| 4×4×4  | 10     | 0          | 260                    | 13         |

**Note**: Only 2×2×2 generates both linear and non-linear assembly variants.

## Output Structure

```
puzzles_full_v9/
├── SUMMARY.json                    # Global summary of all puzzles
├── 2x2x2/
│   ├── puzzle_001/
│   │   ├── puzzle_001_2x2x2.json  # Puzzle definition
│   │   ├── assembly_2x2x2.png     # 3D visualization of solution
│   │   └── pieces_2x2x2.png       # Individual piece visualization
│   └── _signatures/                # Deduplication markers
├── 3x3x3/
│   └── ...
└── ...
```

### Puzzle JSON Format

```json
{
  "box": [3, 3, 3],                    // Dimensions (a, b, c)
  "min_piece": 4,                       // Minimum voxels per piece used
  "max_piece_cells": 5,                 // Maximum voxels per piece
  "no_2x3_plane": true,                 // Constraint flag
  "no_isolated_piece": true,            // Constraint flag
  "linear_assembly": true,              // Whether linear assembly is possible
  "pieces": [                           // List of pieces (voxel coordinates)
    [[0,1,2], [0,2,2], [1,1,2], [2,1,2]],
    ...
  ],
  "solution": {                         // Piece placements in solution
    "0": [[0,0,0], [0,0,1], ...],
    ...
  },
  "assembly_order": [                   // Sequence of (piece_id, direction)
    [1, "+x"],
    [0, "-x"]
  ],
  "stats": {
    "visited_nodes": 511,               // DLX search complexity
    "solutions": 1
  },
  "signature": "4d92019..."             // SHA1 hash for deduplication
}
```

## Generated Results

Based on the `puzzles_full_v9` output:

- **Total puzzles**: 48
- **Distribution**:
  - 2×2×2: 1 puzzle (easiest, proof-of-concept)
  - 2×3×4: 4 puzzles
  - 2×4×4: 8 puzzles
  - 3×3×3: 5 puzzles
  - 3×3×4: 10 puzzles
  - 3×4×4: 10 puzzles
  - 4×4×4: 10 puzzles (hardest)

**Note**: Some sizes (2×2×3, 2×2×4, 2×3×3) generated 0 puzzles, likely due to:
- Strict difficulty thresholds not being met
- Limited valid configurations satisfying all constraints
- Random seed and attempt limits

## Usage

### Basic Usage
```bash
python polypuzzle_batch.py --out ./puzzles_full_v9 --workers 8 --tries 5000
```

### Command-Line Arguments
- `--out`: Output directory (default: `./puzzles_full_9`)
- `--workers`: Number of parallel processes (default: half of CPU cores)
- `--seed`: Base random seed (default: 8001)
- `--tries`: Maximum attempts per puzzle (default: 5000)
- `--node-scale`: Multiply difficulty thresholds (default: 1.0)
- `--node-offset`: Add to difficulty thresholds (default: 0)
- `--no-min3-fallback`: Disable 3-voxel minimum fallback (forces 4-voxel minimum)

### Example: Easier Puzzles
```bash
python polypuzzle_batch.py --node-scale 0.5 --tries 10000
```

### Example: Harder Puzzles
```bash
python polypuzzle_batch.py --node-offset 100 --no-min3-fallback
```

## Multiprocessing

The generator uses Python's `multiprocessing.Pool` to parallelize puzzle generation across puzzle sizes and instances. Each worker:
1. Generates random partitions with constraints
2. Attempts to solve via DLX
3. Validates physical feasibility
4. Atomically reserves a unique signature
5. Saves puzzle to dedicated subfolder with visualizations

## Visualization

The code generates two types of 3D visualizations using matplotlib:

1. **Assembly view** (`assembly_*.png`): Complete assembled puzzle with colored pieces
2. **Piece sheet** (`pieces_*.png`): Grid layout showing individual pieces

Both use true 3D voxel rendering with:
- Proper aspect ratios (true cube visualization)
- Edge highlighting for depth perception
- Color-coded pieces for identification

## Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import lru_cache
```

## Algorithm Complexity

- **Partition generation**: O(V) where V = volume of box
- **DLX solving**: Exponential worst-case, but heavily optimized with:
  - Anchor piece fixing (reduces search space)
  - MRV heuristic (prunes early)
  - Placement caching (O(1) lookups)
- **Assembly order**: O(P × D) where P = pieces, D = directions

## Plan Versions

- **Plan C** (current): Global linear assembly requirement, except 2×2×2 which also generates non-linear variants for testing L-shape combinations
- Previous versions likely had different constraint combinations or generation strategies

## Future Improvements

Potential enhancements:
1. Adaptive difficulty tuning based on success rates
2. More sophisticated difficulty metrics (e.g., backtracking depth)
3. Symmetry-aware deduplication
4. Interactive web-based puzzle viewer
5. Puzzle verification and rating system
6. Support for non-rectangular target shapes

---

**Created**: 2025-10-21
**Version**: v9 (Plan C)
**Author**: Based on polypuzzle_batch.py analysis


---

## CHAIN 3D tasks: shared rendering (`chain_viz.py`)

`type_2_3` (PipeFit), `type_3` (FluidNet) and `type_3_1` (Domino) render
through the helpers in `chain_viz.py`, so all three share:

- the full-grid frame (no content-dependent cropping),
- `x=… / y=… / z=…` cell labels on the edges facing each camera, matching
  the 1-based cell coordinates the actions use,
- dashed drop lines + floor footprints under raised objects,
- the `front` / `back` cameras and a per-game theme (board colours,
  background, accent),
- the game-style frame from `chain_hud.py`: title bar with a
  PLAYING / WIN / LOSE badge, STATUS card with progress bars, inventory
  card, LEGEND card with a drawn icon for every piece type, and a rules
  strip.  plotly renders only the 3D scene (transparent); PIL composes the
  frame around it (1960 × 1540 px).

Rendering needs `plotly<6` with `kaleido==0.2.1` (plotly 6+ requires
kaleido>=1, which needs a separate Chrome install):

```bash
pip install "plotly<6" "kaleido==0.2.1"
```

## CHAIN 3D tasks: levels, verification and analysis (`chain_analysis/`)

Each game ships 50 levels (`<game>/puzzles_50/`, 15 easy / 20 medium /
15 hard) and a 10-level demo batch (`<game>/images_batch/`).  The level
files (`level.json`, `index.json`) are in git; the images are not —
regenerate them with `python -m <game>.generate_50` and
`python -m <game>.play_batch` (same seeds give the same levels).

```bash
# every reference plan replays to a win through the game rules
python -m chain_analysis.verify_levels type_3/puzzles_50 type_3_1/puzzles_50 type_2_3/puzzles_50
# Experiment 1 (dead ends, step value) and 2 (search under a verify budget)
python -m chain_analysis.exp1_step_value --root type_3/puzzles_50
python -m chain_analysis.exp2_budget_search --root type_3/puzzles_50
# let a model / person play one level closed loop (an image after every action)
python -m chain_analysis.play_cli start type_3/images_batch/level_01_easy_trace_seed1000 runs/s01
python -m chain_analysis.play_cli act runs/s01 '["open", [3, 4, 4]]'
# ... or one-shot: the whole plan in one call, from the start image only
python -m chain_analysis.play_cli start type_3/images_batch/level_01_easy_trace_seed1000 runs/s02
python -m chain_analysis.play_cli plan runs/s02 '[["open", [3, 4, 4]]]'
```

Hard tier.  The hard levels of each game's `puzzles_50/` are calibrated
against gpt-6.1-sol (Codex, default reasoning effort, one-shot from the start
images): it solves about a quarter of them.  Every level keeps one winning
answer, checked by exhaustive search.

| game | hard levels (since 2026-11-04) | gpt-6.1-sol one-shot | best search baseline at 1 / 64 verifies |
|---|---|---|---|
| FluidNet | 9–10 valves, unique 6-valve plan | 4 / 15 | 7% / 93% |
| Domino | three-level splitter tree on steps, 9–10 patches | 3 / 15 | 0% / 0% |
| PipeFit | 10–20 pieces, 3–12 solutions (unchanged) | 5 / 15 | 0% / 1% |

On the previous hard tier (git `de130a7`) gpt-6.1-sol solved 27/45 at the
same effort and 44/45 at effort high; Claude Sonnet solved every level
closed loop and 37/45 one-shot, Codex gpt-6-luna 2/45 one-shot.  gpt-6.1-sol's losses
on the new tier are single misread actions in a long plan: a valve
coordinate, a step height, a portal cell.

Details: `chain_analysis/README.md`; baseline results and discussion:
`chain_analysis/RESULTS.md`; model runs (sessions, audits, Exp 1 / 2 reports,
how to reproduce): `chain_analysis/model_runs/README.md`.
