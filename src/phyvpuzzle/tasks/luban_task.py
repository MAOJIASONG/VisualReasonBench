"""
Luban Lock disassembly task implementation (Unity-backed).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from phyvpuzzle.core import register_task, register_task_config, TaskConfig
from phyvpuzzle.core.base import Observation, TaskResult
from phyvpuzzle.tasks.base_task import PhysicsTask

from phyvpuzzle.environment.luban_env import LubanEnvironment

@register_task_config("luban_disassembly")
@dataclass
class LubanTaskConfig(TaskConfig):
    urdf_root: str = "src/phyvpuzzle/environment/phobos_models/luban-3-piece"
    level_index: int = 0
    target_displacement_threshold: float = 0.08
    ruled_evaluation: bool = True

@register_task("luban_disassembly")
class LubanDisassemblyTask(PhysicsTask):
    """
    Task: Disassemble a Luban Lock.
    Loads pieces using RuntimeAssembly logic (Fixed Constraints).
    """

    def __init__(self, config: LubanTaskConfig):
        super().__init__(config)
        self.target_piece_id: Optional[int] = None
        self.initial_positions: Dict[int, Tuple[float, float, float]] = {}

    def _calculate_optimal_steps(self) -> int:
        # Feasible disassembly paths do not establish a shortest-path bound.
        # Metrics skip negative values instead of comparing every level to one move.
        return -1

    def configure_environment(self, environment) -> Observation:  # type: ignore[override]
        """Configure Unity-backed environment (no PyBullet setup)."""
        if not isinstance(environment, LubanEnvironment):
            raise ValueError("LubanTask requires LubanEnvironment")
        self.environment = environment
        observation = self.environment.reset()
        self._cache_initial_state(observation)
        return observation

    def _configure_environment(self) -> None:
        """No-op for Unity-backed environment."""
        return

    def _cache_initial_state(self, observation: Observation) -> None:
        self.initial_positions = {}
        for obj in observation.state.objects:
            self.initial_positions[obj.object_id] = obj.position
        if observation.state.objects:
            self.target_piece_id = observation.state.objects[0].object_id

    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        if not task_results: return []
        latest = task_results[-1]
        last_obs = self._extract_last_observation(latest)
        is_solved = None
        if last_obs is not None:
            if hasattr(last_obs, "state"):
                is_solved = last_obs.state.metadata.get("is_solved")
            elif isinstance(last_obs, dict):
                meta = last_obs.get("state", {}).get("metadata", {})
                is_solved = meta.get("is_solved")
        latest.success = bool(is_solved)
        latest.metadata["feedback"] = f"Solved={is_solved}"
        return task_results

    def _extract_last_observation(self, task_result: TaskResult):
        if not task_result.trajectory:
            return None
        last = task_result.trajectory[-1]
        if isinstance(last, dict):
            observations = last.get("observations", [])
            return observations[-1] if observations else None
        try:
            _, obs = last
            return obs
        except Exception:
            return None

    def _get_initial_system_prompt(self) -> str:
        prompt = """You are a spatial reasoning expert solving a "Luban Lock".
OBJECTS:
- Pieces are mechanically interlocked and must be moved via tools.

GOAL:
Disassemble the lock: separate all pieces and move the released pieces out of
the central puzzle area. Continue until the environment reports solved=true.
"""
        environment = getattr(self, "environment", None)
        if getattr(getattr(environment, "config", None), "action_mode", "native") == "restored_original_rules":
            prompt += """
AVAILABLE ACTIONS:
- Move one piece or a selected group along an axis or an integer direction.
- Rotate one piece or a group by 45 or 90 degrees, with an optional pivot.
- The state lists applicable_special_rules. Each listed rule is one original
  combined translation-and-rotation action whose conditions currently match.
- A blocked action leaves the puzzle unchanged. Inspect the result before
  deciding the next move. Group movement may release an assembly that still
  needs further disassembly; moving that assembly away does not finish the task.
"""
        return prompt

    def _get_initial_instruction(self) -> str:
        return "Disassemble the lock using legal moves. Separate the pieces and continue until the environment confirms success."
