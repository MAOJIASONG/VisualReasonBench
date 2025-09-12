"""
Task implementations for PhyVPuzzle benchmark.

This package contains specific puzzle task implementations including:
- Domino falling tasks
- Luban lock assembly tasks  
- Pagoda stacking tasks
- Custom puzzle tasks
"""

# Normal imports to ensure proper task registration
from phyvpuzzle.tasks.base_task import PhysicsTask
from phyvpuzzle.tasks.domino_dont_fall_task import DominoDontFallTask
from phyvpuzzle.tasks.luban_task import LubanTask
from phyvpuzzle.tasks.pagoda_task import PagodaTask

__all__ = [
    "PhysicsTask",
    "DominoDontFallTask",
    "LubanTask",
    "PagodaTask"
]
