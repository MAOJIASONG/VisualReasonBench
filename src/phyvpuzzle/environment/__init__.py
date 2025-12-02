"""
Physics environment implementations for PhyVPuzzle.

This package contains physics simulation environments using PyBullet
for different types of puzzles. All environments inherit from the unified
PhysicsEnvironment base class and add task-specific functionality.
"""

# Normal imports to ensure proper environment registration
from phyvpuzzle.environment.base_env import PhysicsEnvironment
from phyvpuzzle.environment.domino_env import DominoEnvironment, DominoConfig
from phyvpuzzle.environment.puzzle_env import PuzzleEnvironment, PuzzleConfig
from phyvpuzzle.environment.stacking_game_env import StackingGameEnvironment, StackingGameConfig

__all__ = [
    "PhysicsEnvironment",
    "DominoEnvironment", 
    "DominoConfig",
    "LegoEnvironment",
    "LegoConfig", 
    "PuzzleEnvironment",
    "PuzzleConfig",
    "StackingGameEnvironment",
    "StackingGameConfig",
]
