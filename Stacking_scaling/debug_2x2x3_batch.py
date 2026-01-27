
import sys
import random
import math
import hashlib
import os
from collections import defaultdict, deque
from typing import List, Set, Tuple

# Import everything from the module
sys.path.append("/home/yuhao/VisualReasonBench/Stacking_scaling")
from polypuzzle_batch import generate_one_constrained_staged, PER_SIZE_CFG, puzzle_signature

def test_2x2x3_batch():
    a,b,c = 2,2,3
    cfg = PER_SIZE_CFG.get((2,2,3), {"difficulty_nodes": 1, "max_pieces": 3})
    print(f"Testing 2x2x3 with cfg: {cfg} (forcing difficulty=1)")
    
    unique_sigs = set()
    seed_base = 1000
    
    for i in range(50): # Try 50 attempts to find 10 unique
        seed = seed_base + i * 37
        result = generate_one_constrained_staged(
            a,b,c,
            max_pieces=cfg["max_pieces"],
            max_piece_cells=5,
            difficulty_nodes=1, # FORCE LOW DIFFICULTY
            seed=seed,
            tries=2000,
            require_linear_assembly=True,
            allow_min3=True
        )
        
        if result:
            sig = puzzle_signature(result["pieces"])
            if sig not in unique_sigs:
                unique_sigs.add(sig)
                print(f"Found unique puzzle #{len(unique_sigs)} (nodes={result['stats']['visited_nodes']})")
            else:
                print(f"Duplicate signature (nodes={result['stats']['visited_nodes']})")
        else:
            print("Failed to generate.")
            
        if len(unique_sigs) >= 10:
            print("Successfully found 10 unique puzzles.")
            break
            
    print(f"Total unique found: {len(unique_sigs)}")

if __name__ == "__main__":
    test_2x2x3_batch()
