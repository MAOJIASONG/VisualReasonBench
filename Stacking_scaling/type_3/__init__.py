"""
FluidNet 3D — Type 3 Structural Propagation Game

Agent opens valves in a 3D pipe network; fluid propagates via BFS through the
connected graph and must fill all target reservoirs without entering protected
cells or mixing colors.

This game belongs to the CHAIN T3 family: a local action (opening one valve)
causes a non-local state change (fluid may reach cells many hops away).
"""

from .game_core import (
    CellType, ActionCode, Vec3, Cell, LevelSpec, GameState, ActionResult,
)
from .propagation import propagate_fluid, step_open_valve, check_terminal
from .loader import load_level_from_json, save_level_to_json, create_game_state

__all__ = [
    "CellType", "ActionCode", "Vec3", "Cell", "LevelSpec", "GameState",
    "ActionResult", "propagate_fluid", "step_open_valve", "check_terminal",
    "load_level_from_json", "save_level_to_json", "create_game_state",
]
