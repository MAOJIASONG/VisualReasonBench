"""
Task implementations for PhyVPuzzle benchmark.

This package contains specific puzzle task implementations including:
- Domino falling tasks
- Luban lock assembly tasks  
- Pagoda stacking tasks
- Jigsaw puzzle assembly tasks
- Custom puzzle tasks
"""

# Normal imports to ensure proper task registration
from phyvpuzzle.tasks.base_task import PhysicsTask
from phyvpuzzle.tasks.domino_dont_fall import DominoDontFallTask
from phyvpuzzle.tasks.three_by_three_stacking import ThreeByThreeStackingTask

__all__ = [
    "PhysicsTask",
    "DominoDontFallTask",
    "ThreeByThreeStackingTask"
]
