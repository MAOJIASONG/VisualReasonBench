"""
Stacking-game task for VisualReasonBench.

This task asks the agent to pack all polycube pieces into the target box using
the stacking_game environment tools.
"""

from dataclasses import dataclass
from typing import List, Optional

from phyvpuzzle.core import TaskConfig, register_task, register_task_config
from phyvpuzzle.core.base import Observation, TaskResult
from phyvpuzzle.tasks.base_task import PhysicsTask


@register_task_config("stacking_game")
@dataclass
class StackingGameTaskConfig(TaskConfig):
    """Configuration for the stacking_game task."""

    puzzle_size: str = "2x2x2"
    puzzle_id: str = "puzzle_001"
    ruled_evaluation: bool = True
    allow_random_puzzle: bool = False
    init_seed: Optional[int] = None


@register_task("stacking_game")
class StackingGameTask(PhysicsTask):
    """Task wrapper for the stacking_game environment."""

    def __init__(self, config: StackingGameTaskConfig):
        super().__init__(config)

    def _calculate_optimal_steps(self) -> int:
        # Practical upper bound for small puzzles; used for metrics only.
        return 24

    def configure_environment(self, environment) -> Observation:  # type: ignore[override]
        """Configure environment with the puzzle specified in the task config."""
        self.environment = environment
        # Pass puzzle selection hints to the environment before reset.
        if hasattr(environment, "current_size"):
            environment.current_size = self.config.puzzle_size
        if hasattr(environment, "current_puzzle_id"):
            environment.current_puzzle_id = self.config.puzzle_id
        if hasattr(environment, "_task_seed"):
            environment._task_seed = self.config.init_seed
        observation = environment.reset()
        return observation

    def _configure_environment(self) -> None:
        """Unused because configure_environment is fully overridden."""
        return None

    def _evaluate_success(self, task_results: List[TaskResult]) -> List[TaskResult]:
        """Rule-based success: puzzle cells must be fully filled."""
        for task_result in task_results:
            is_complete = False
            if task_result.trajectory:
                # Traverse backwards to find the latest observation with metadata.
                for step in reversed(task_result.trajectory):
                    observations = step.get("observations", []) if isinstance(step, dict) else []
                    if observations:
                        last_obs = observations[-1]
                        try:
                            meta = last_obs.state.metadata if hasattr(last_obs, "state") else {}
                            is_complete = bool(meta.get("is_complete"))
                        except Exception:
                            is_complete = False
                        break
            task_result.success = is_complete
            task_result.metadata["is_complete"] = is_complete
        return task_results

    def _get_initial_system_prompt(self) -> str:
        """Provide system instructions tailored for stacking_game."""
        return (
            "You are a precise 3D packing assistant inside a discrete grid box. "
            "All coordinates are 1-based integers (x,y,z). Use the provided tools:\n"
            "- list_puzzles(size?): inspect available puzzles.\n"
            "- load_puzzle(size, puzzle_id, seed?): load/reset a level.\n"
            "- get_piece_info(piece_id): inspect voxel layout and rotations.\n"
            "- place_piece(piece_id, rotation, position): place by rotation index (0-23) and minimum corner.\n"
            "- place_piece_by_cells(piece_id, cells): place by explicit cell list when rotations are unclear.\n"
            "- pickup_piece(piece_id): remove a placed piece.\n"
            "- finish(): call only when you believe the box is fully filled.\n"
            "Avoid floating pieces: every placement must be supported by the floor (z=1) or other pieces."
        )

    def _get_initial_instruction(self) -> str:
        """User-facing task instruction."""
        cfg = self.config
        return (
            f"Pack every piece into the {cfg.puzzle_size} box for puzzle '{cfg.puzzle_id}'. "
            "You must fill all cells without collisions. "
            "Inspect pieces, plan a feasible order, place them, and call finish once solved."
        )
