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
        # return (
        #     "You are a 3D stacking game player. "
        # )
        return ""

    def _get_initial_instruction(self) -> str:
        """User-facing task instruction."""
        cfg = self.config

        instruction = f"""
You are solving a 3D packing puzzle.

**Goal:** Pack every piece into the **{cfg.puzzle_size}** box for puzzle **`{cfg.puzzle_id}`**. You must fill **all {len(self.environment.game_state.spec.box)} cells** with **no collisions** and **no out-of-bounds** placements.

**Critical rules:**

1. **First, carefully read the list of available tools** (their names, required arguments, and what state they return).
2. **Use the current image / board progress** as the ground-truth state of what is already placed and what remains.
3. Work in a safe order: **inspect pieces → plan a feasible sequence → place pieces step-by-step**, verifying after each placement.
4. If a placement fails (collision / invalid), **undo or adjust** and try a different orientation/order.
5. Only call **`finish`** when the box is fully filled and valid.

**Output format constraints:**

* Do **not** output multiple actions.
* Your response must end with **exactly one tool call** wrapped as:

  * `<action> XXX </action>`
* `XXX` must be the **exact tool invocation content** (use the tool’s required schema/arguments).

**Now do the next step:** based on the current image/board, start by inspecting the remaining pieces (or the most ambiguous piece) to plan the packing order.

<action>{{"tool":"inspect_pieces","puzzle_id":"puzzle_001"}}</action>        
        """
        # return (
        #     f"Pack every piece into the {cfg.puzzle_size} box for puzzle '{cfg.puzzle_id}'. "
        #     "You must fill all cells without collisions. "
        #     "Inspect pieces, plan a feasible order, place them, and call finish once solved."
        # )
        return instruction
