"""
Physics environment implementations for PhyVPuzzle.

This package contains physics simulation environments using PyBullet
for different types of puzzles. All environments inherit from the unified
PhysicsEnvironment base class and add task-specific functionality.
"""

from .base_env import PhysicsEnvironment, CameraConfig, ObjectInfo
from .domino_env import DominoEnvironment, DominoConfig
from .luban_env import LubanEnvironment

__all__ = [
    "PhysicsEnvironment",
    "CameraConfig",
    "ObjectInfo",
    "DominoEnvironment",
    "DominoConfig", 
    "LubanEnvironment"
]
