"""
Domino Task Implementation

Provides a simple domino toppling task for demonstration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_task import BaseTask, TaskConfiguration, TaskType, TaskDifficulty


class DominoTask(BaseTask):
    """Simple domino toppling task.

    Goal: topple all dominoes on the table by applying pushes.
    """

    def __init__(self, config: Optional[TaskConfiguration] = None):
        cfg = config or TaskConfiguration(
            task_type=TaskType.DOMINOES,
            difficulty=TaskDifficulty.EASY,
            max_steps=20,
            time_limit=120.0,
        )
        super().__init__(cfg)
        self.domino_names: List[str] = []
        self.initial_upright_threshold = 0.95

    def setup_task(self, environment) -> bool:
        self.environment = environment
        # Create a simple row of box primitives to simulate dominoes
        self.domino_names = []
        x0, y0, z0 = 0.0, 0.0, 0.5
        spacing = 0.12
        num = 5
        for i in range(num):
            name = f"domino_{i+1}"
            self.environment.create_primitive_object(
                object_name=name,
                shape_type="box",
                size=(0.02, 0.05, 0.12),
                position=(x0 + i * spacing, y0, z0),
                color=(0.9, 0.9, 0.1, 1.0),
                mass=0.2,
            )
            self.current_objects[name] = {
                "type": "domino",
            }
            self.domino_names.append(name)
        # A small cube to push
        self.environment.create_primitive_object(
            object_name="pusher",
            shape_type="box",
            size=(0.03, 0.03, 0.03),
            position=(x0 - 0.2, y0, z0),
            color=(0.2, 0.6, 0.9, 1.0),
            mass=0.3,
        )
        return True

    def get_task_description(self) -> str:
        return "Topple all dominoes by interacting with the scene."

    def check_completion(self) -> bool:
        # Consider domino toppled if its up-vector z is small; approximate via base orientation
        import pybullet as p
        toppled = 0
        for name in self.domino_names:
            obj = self.environment.objects.get(name)
            if not obj:
                continue
            _, orn = p.getBasePositionAndOrientation(obj.object_id)
            # Convert quaternion to up-vector z magnitude heuristic: squared z-axis dot world z
            # For simplicity, check if height dropped below initial height
            pos, _ = p.getBasePositionAndOrientation(obj.object_id)
            if pos[2] < 0.2:
                toppled += 1
        return toppled == len(self.domino_names)

    def evaluate_state(self) -> float:
        import pybullet as p
        toppled = 0
        for name in self.domino_names:
            obj = self.environment.objects.get(name)
            if not obj:
                continue
            pos, _ = p.getBasePositionAndOrientation(obj.object_id)
            if pos[2] < 0.2:
                toppled += 1
        return toppled / max(1, len(self.domino_names))

    def get_optimal_solution(self) -> List[str]:
        return ["push pusher towards domino_1"]

    def reset_task(self) -> None:
        if self.environment:
            self.environment.reset()
            self.setup_task(self.environment)

    def get_task_specific_context(self) -> Dict[str, Any]:
        return {
            "num_dominoes": len(self.domino_names),
            "objects": list(self.domino_names),
            "hint": "You can call tools: observe, move, push, pick, place.",
        }
