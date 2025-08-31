# Environment Harness

This document explains how the physics environment layer is organized, which tools it exposes, how configuration is loaded and used, and how to extend or debug environments. It reflects the current implementation status in the repository.

## Overview

- Base class: `src/phyvpuzzle/environment/base_env.py` (`PhysicsEnvironment`) centralizes PyBullet setup, common tools, rendering, and state/observation plumbing.
- Inheritance model: Puzzle environments subclass the base (e.g., `DominoEnvironment`, `LubanEnvironment`) and add task‑specific tools or overrides.
- Agent‑friendly semantics: Single‑pick invariant with auto‑swap on `pick`. In constrained Luban mode, `move/rotate` auto‑release after success; in free‑physics modes (e.g., Domino) the agent should call `release` explicitly.

## File Structure (Environment Layer)

- `src/phyvpuzzle/environment/base_env.py`: Unified harness and default tool implementations, including `place`.
- `src/phyvpuzzle/environment/domino_env.py`: Domino setup (arrangements, success detection) + domino‑specific tools.
- `src/phyvpuzzle/environment/luban_env.py`: Luban lock with constrained‑mode overrides (pick/move/rotate/push semantics).
- `src/phyvpuzzle/utils/multi_view_renderer.py`: Named cameras and multi‑view rendering used by environments.
- `docs/luban_constrained_mode_spec.md`: Detailed semantics and configuration for Luban constrained mode.

## Lifecycle & State

- Init: Reads environment keys (GUI, gravity, time step, rendering), builds camera/renderer, then `_initialize_pybullet()`.
- Reset: Clears bodies, resets physics, reloads plane/table, calls `_setup_task_environment()` to add task objects, re‑enables gravity, returns initial `Observation`.
- Step: Executes one action, steps physics, updates `State`, checks done, returns next `Observation` plus feedback.
- Observation: Main image plus named per‑view images (`front`, `side`, `top`) when multi‑view is enabled.
- Customization points: `_setup_task_environment()`, `_get_current_state()`, `_create_observation()`, `_evaluate_success()`, `_is_done()`.

## Base Tools (from `base_env.py`)

- `pick(object_id, reset_orientation=False)`: Enforces single‑pick with auto‑swap; returns `picked_objects` and `constraint_id`.
- `release(object_id)`: Removes the pick constraint on that object.
- `move(object_id, position=[x,y,z])`: Moves a picked object via constraint; free‑physics keeps selection active.
- `rotate(object_id, axis={'x'|'y'|'z'}, angle=radians)`: Rotates a picked object via constraint; free‑physics keeps selection active.
- `place(object_id, target_id, offset_xy=[dx,dy], offset_local=true, align_orientation={'keep'|'align_target'|'snap_90'}, clearance=0.001, release_after=false, stabilize_target=true, hold_max_force=300.0, hold_erp=0.4)`: Places a picked object on top of a target (details below).
- `push(object_id, force, direction=[x,y,z])`: Applies an external force (disabled in Luban constrained mode; see below).
- `observe(angle=radians)`: Repositions the perspective camera around the scene.
- `wait(duration=seconds)`: Steps physics for a duration to let things settle.
- `check_solution()`: Returns current success status from `_evaluate_success()`.

Notes:
- All tool results are dicts with at minimum `status` (`success|error`) and `message`; some include `picked_objects`, `constraint_id`, or final poses.
- `get_tool_schemas()` returns JSON schemas for all tools above; environments may add more via `_get_task_specific_tool_schemas()`.
- `get_available_actions()` lists base actions; it currently includes a placeholder `pull` not implemented in the base. Calling it will return an unknown‑tool error unless a subclass implements it.

## Tools Cheat Sheet

- pick: object_id (string), reset_orientation (bool=false) → status, message, picked_objects [string], constraint_id (int), auto_swapped (bool)
- release: object_id (string) → status, message, picked_objects [string]
- move: object_id (string), position [x,y,z] (meters) → status, message
- rotate: object_id (string), axis {'x'|'y'|'z'}, angle (radians) → status, message
- place: object_id (string), target_id (string), offset_xy [dx,dy]=[0,0], offset_local=true, align_orientation {'keep'|'align_target'|'snap_90'} (default align_target), clearance=0.001, release_after=false, stabilize_target=true, hold_max_force=300.0, hold_erp=0.4 → status, message, final_position [x,y,z], final_orientation [x,y,z,w]
- push: object_id (string), force (N), direction [x,y,z] (unit) → status, message; disabled in Luban constrained mode
- observe: angle (radians) → status, message
- wait: duration (seconds) → status, message
- check_solution: no args → status, solved (bool), message

## Configuration: Keys Used by the Harness

- Environment keys respected by `PhysicsEnvironment.__init__`: `gui`, `gravity`, `time_step`, `urdf_base_path`, `render_width`, `render_height`, `multi_view`, `load_table`.
- Example YAML:

  ```yaml
  environment:
    type: pybullet
    gui: false
    urdf_base_path: src/phyvpuzzle/environment/phobos_models
    gravity: -9.81
    time_step: 0.00416667
    render_width: 512
    render_height: 512
    multi_view: true
    load_table: true
  ```

- Luban‑specific config is read by `LubanEnvironment` under `environment.luban_constrained`:

  ```yaml
  environment:
    gui: true
    luban_constrained:
      enabled: true
      move_step_m: 0.01
      rotate_step_deg: 5.0
      max_force: 300.0
      erp: 0.4
      settle_steps: 8
      guard_contacts: true
  ```

## Place Tool (implemented)

`place` is fully implemented in `base_env.py` and available in all free‑physics environments.

- Arguments: `object_id`, `target_id`, `offset_xy=[dx,dy]`, `offset_local`, `align_orientation={'keep'|'align_target'|'snap_90'}`, `clearance`, `release_after`, `stabilize_target`, `hold_max_force`, `hold_erp`.
- Semantics:
  - Requires `object_id` to be picked and `target_id` to exist.
  - Computes target top from `getAABB(target_id)` and object half‑height from `getAABB(object_id)`; places at `top_z + half_height + clearance`.
  - `offset_xy` applies in the target’s top plane; if `offset_local=true`, it rotates with the target’s yaw.
  - `align_orientation` options: keep current yaw, copy target yaw, or snap yaw to nearest 90°.
  - Stabilization: if `stabilize_target=true`, a temporary world‑fixed constraint holds the target during placement and is always removed afterwards.
  - Basic penetration guard: raises height slightly if penetration increases during final set‑down.
  - `release_after=true` frees the pick after placement (free‑physics only).

Typical use: `pick A` → `place A on B` (optionally `release_after: true`). In Luban constrained mode, prefer `pick+move/rotate` (place remains free‑physics‑oriented).

Important: PLACE is experimental and not fully tested. Expect to adjust offsets, clearance, and stabilization for your scene. If it misbehaves, treat this implementation as a starting point and adapt it to your task (see HANDOVER notes for open design items).

## Payload Examples

Minimal tool call payloads (arguments passed to `environment.execute_tool_call(name, arguments)`):

```json
{ "name": "pick", "arguments": { "object_id": "block_1" } }
```

```json
{ "name": "move", "arguments": { "object_id": "block_1", "position": [0.10, 0.00, 0.60] } }
```

```json
{ "name": "rotate", "arguments": { "object_id": "block_1", "axis": "z", "angle": 1.5708 } }
```

```json
{ "name": "place", "arguments": {
  "object_id": "block_1",
  "target_id": "platform",
  "offset_xy": [0.0, 0.0],
  "offset_local": true,
  "align_orientation": "align_target",
  "clearance": 0.001,
  "release_after": true,
  "stabilize_target": true
} }
```

```json
{ "name": "push", "arguments": { "object_id": "domino_1", "force": 5.0, "direction": [1, 0, 0] } }
```

```json
{ "name": "observe", "arguments": { "angle": 1.0 } }
```

```json
{ "name": "wait", "arguments": { "duration": 0.5 } }
```

```json
{ "name": "check_solution", "arguments": {} }
```

Example results:

```json
{
  "status": "success",
  "message": "Picked up block_1 using constraint 42",
  "picked_objects": ["block_1"],
  "constraint_id": 42,
  "auto_swapped": false
}
```

```json
{
  "status": "success",
  "message": "Placed block_1 on top of platform and released",
  "final_position": [0.10, 0.00, 0.52],
  "final_orientation": [0, 0, 0, 1]
}
```

## Extending Environments

- Subclass: Create `MyEnv(PhysicsEnvironment)` in `src/phyvpuzzle/environment/`.
- Setup: Implement `_setup_task_environment()` to load pieces, create any constraints, and seed object tracking.
- State: Override `_get_current_state()`, `_evaluate_success()`, and optionally `_get_state_description()`.
- Tools:
  - Add JSON schemas via `_get_task_specific_tool_schemas()`.
  - Handle calls in `_execute_task_specific_tool(tool_name, arguments)`.
  - Extend `get_available_actions()` to expose your new actions.
- Example: See `DominoEnvironment` for task‑specific tools (`push_domino`, `push_specific_domino`, `reset_dominoes`) and success logic.

## Common Errors & Recovery Patterns

- Must pick first: `move/rotate/place` require the object to be picked. Call `pick(object_id)` first.
- Auto‑swap on pick: Picking a new object auto‑releases the previous pick; this is expected.
- Luban push disabled: In constrained mode, `push` returns `status:error`; use `pick + move/rotate` instead.
- Guard aborts (Luban): Interpolated `move/rotate` may return `status:error` if penetration increases. Recover by trying smaller deltas, a different path/orientation, or by repositioning other pieces.
- Unknown tool: Ensure the action appears in `get_available_actions()` and has a schema in `get_tool_schemas()`.
- Unknown object: Verify the `object_id`/`target_id` exists in `environment.objects`.
- Visual debugging: Use `environment.gui: true`; for Luban, set `environment.verbose_logging: true`.

## Quick Pointers

- Base harness: `src/phyvpuzzle/environment/base_env.py`
- Domino env: `src/phyvpuzzle/environment/domino_env.py`
- Luban env: `src/phyvpuzzle/environment/luban_env.py`
- Renderer: `src/phyvpuzzle/utils/multi_view_renderer.py`

Onboarding tip: skim this doc, then open the files above in order. For Luban specifics, read `docs/luban_constrained_mode_spec.md`.
