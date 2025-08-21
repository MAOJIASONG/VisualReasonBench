"""
Task implementations for PhyVPuzzle benchmark.

This package contains specific puzzle task implementations including:
- Domino falling tasks
- Luban lock assembly tasks  
- Pagoda stacking tasks
- Custom puzzle tasks
"""

from .base_task import PuzzleTask
from .domino_task import DominoTask
from .luban_task import LubanTask

__all__ = [
    "PuzzleTask",
    "DominoTask",
    "LubanTask"
]
