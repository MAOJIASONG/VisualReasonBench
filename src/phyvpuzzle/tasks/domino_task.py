"""
Domino Task Implementation

Provides a simple domino toppling task for demonstration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base_task import BaseTask, TaskConfiguration, TaskType, TaskDifficulty
from .domino_tools import DominoTools


class DominoTask(BaseTask):
    """Domino toppling task using URDF domino models.

    Goal: topple all dominoes on the table by applying pushes.
    Supports procedural layouts via configuration parameters:
    - parameters.num_dominoes: int, number of dominoes (default: 5)
    - parameters.layout: str, layout name among {"line", "L", "S"} (default: "line")
    - parameters.spacing: float, spacing between dominoes (default: 0.12)
    """

    def __init__(self, config: Optional[TaskConfiguration] = None):
        # Adjust parameters based on difficulty
        if config:
            if config.difficulty == TaskDifficulty.VERY_EASY:
                config.parameters = {"num_dominoes": 3, "layout": "line", "spacing": 0.12}
                config.max_steps = 5  # Default to 5 rounds for very-easy
            elif config.difficulty == TaskDifficulty.EASY:
                config.parameters = config.parameters or {"num_dominoes": 5, "layout": "line", "spacing": 0.12}
                config.max_steps = config.max_steps or 10
        
        cfg = config or TaskConfiguration(
            task_type=TaskType.DOMINOES,
            difficulty=TaskDifficulty.EASY,
            max_steps=20,
            time_limit=120.0,
        )
        super().__init__(cfg)
        self.domino_names: List[str] = []
        self.initial_upright_threshold = 0.95
        self.tools = None  # Will be initialized in setup_task
        self.use_vlm_completion_check = True  # Use VLM to check completion
        self.target_image = None  # Will store the target/requirement image

    def setup_task(self, environment) -> bool:
        from math import pi
        import pybullet as p
        import os

        self.environment = environment
        self.domino_names = []

        # Parameters
        params = self.config.parameters or {}
        num = int(params.get("num_dominoes", 5))
        layout = str(params.get("layout", "line")).lower()
        spacing = float(params.get("spacing", 0.12))

        # Domino URDF path (reuse same URDF for all instances)
        base_dir = os.path.dirname(os.path.dirname(__file__))  # .../phyvpuzzle
        urdf_path = os.path.join(
            base_dir,
            "environment",
            "phobos_models",
            "domino",
            "Domino_1",
            "urdf",
            "Domino_1.urdf",
        )

        # Layout generation
        def positions_for_layout(n: int) -> List[Tuple[float, float, float]]:
            x0, y0, z0 = 0.0, 0.0, 0.5
            pts: List[Tuple[float, float, float]] = []
            if layout == "line":
                for i in range(n):
                    pts.append((x0 + i * spacing, y0, z0))
            elif layout == "l":
                half = max(1, n // 2)
                for i in range(half):
                    pts.append((x0 + i * spacing, y0, z0))
                for j in range(n - half):
                    pts.append((x0 + (half - 1) * spacing, y0 + (j + 1) * spacing, z0))
            elif layout == "s":
                for i in range(n):
                    dy = ((-1) ** i) * (spacing * 0.3)
                    pts.append((x0 + i * spacing, y0 + dy, z0))
            else:
                # Fallback to line
                for i in range(n):
                    pts.append((x0 + i * spacing, y0, z0))
            return pts

        positions = positions_for_layout(num)

        # Create dominoes as URDF instances
        for i, pos in enumerate(positions):
            name = f"domino_{i+1}"
            # Always use primitive objects for now to avoid URDF issues
            self.environment.create_primitive_object(
                    object_name=name,
                    shape_type="box",
                    size=(0.15, 0.015, 0.25),
                    position=pos,
                    color=(0.9, 0.9, 0.1, 1.0),
                    mass=0.2,
                )
            self.current_objects[name] = {"type": "domino"}
            self.domino_names.append(name)

        # Initialize tools after dominoes are created
        self.tools = DominoTools(self.environment)
        self.tools.set_domino_names(self.domino_names)
        
        # Make tools available to the environment
        environment.domino_tools = self.tools

        # A small cube to push
        x0 = positions[0][0]
        y0 = positions[0][1]
        z0 = positions[0][2]
        self.environment.create_primitive_object(
            object_name="pusher",
            shape_type="box",
            size=(0.03, 0.03, 0.03),
            position=(x0 - 0.2, y0, z0),
            color=(0.2, 0.6, 0.9, 1.0),
            mass=0.3,
        )
        
        # Capture initial state image for VLM comparison
        if self.use_vlm_completion_check:
            import time
            import pybullet as p
            # Let physics settle
            for _ in range(10):
                p.stepSimulation()
            time.sleep(0.1)
            # Capture initial image
            self.initial_image = environment.render()
        
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
