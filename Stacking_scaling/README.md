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
