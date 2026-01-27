
import sys
import random
import math
from collections import defaultdict, deque
from typing import List, Set, Tuple

# Import everything from the module
sys.path.append("/home/yuhao/VisualReasonBench/Stacking_scaling")
from polypuzzle_batch import generate_one_constrained_staged, PER_SIZE_CFG

def test_2x2x3():
    a,b,c = 2,2,3
    cfg = PER_SIZE_CFG.get((2,2,3), {"difficulty_nodes": 10, "max_pieces": 4})
    print(f"Testing 2x2x3 with cfg: {cfg}")
    
    seed = 42
    tries = 1000
    
    # Try to generate one
    result = generate_one_constrained_staged(
        a,b,c,
        max_pieces=cfg["max_pieces"],
        max_piece_cells=5, # Standard
        difficulty_nodes=cfg["difficulty_nodes"],
        seed=seed,
        tries=tries,
        require_linear_assembly=True,
        allow_min3=True
    )
    
    if result:
        print("Success!")
        print(f"Stats: {result['stats']}")
        print(f"Min piece used: {result['min_piece_used']}")
    else:
        print("Failed to generate 2x2x3 puzzle.")
        
        # Debugging: Try with Difficulty 0
        print("Retrying with difficulty 0...")
        result = generate_one_constrained_staged(
            a,b,c,
            max_pieces=cfg["max_pieces"],
            max_piece_cells=5,
            difficulty_nodes=0,
            seed=seed,
            tries=tries,
            require_linear_assembly=True,
            allow_min3=True
        )
        if result:
            print("Success with difficulty 0!")
            print(f"Stats: {result['stats']}")
        else:
            print("Still failed with difficulty 0.")

if __name__ == "__main__":
    test_2x2x3()
