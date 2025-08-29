# Environment Harness

This guide explains how the physics “environment harness” is structured, the key design decisions behind tool semantics, and exactly where to look when debugging or extending environments.

## What The Harness Does

- **Base class:** `src/phyvpuzzle/environment/base_env.py` (`PhysicsEnvironment`) centralizes PyBullet setup, common tools, rendering, and observation/state plumbing.
- **Unification:** All puzzle environments inherit this base and either extend it with task‑specific tools or override methods when behavior must differ (e.g., Luban constrained mode).
- **Minimal agent burden:** Tools implement single‑pick invariant and auto‑swap on pick. Auto‑release after move/rotate happens only in Luban constrained mode; in free‑physics, the VLM explicitly calls `release`.

## Files & Entry Points

- `src/phyvpuzzle/environment/base_env.py`: Base harness and default tool implementations.
- `src/phyvpuzzle/environment/domino_env.py`: Domino environment (patterns, success detection, tools).
- `src/phyvpuzzle/environment/luban_env.py`: Luban environment with constrained‑mode overrides.
- `src/phyvpuzzle/runner.py`: Wires config → environment → task → agent; drives the loop and logging.
- `src/phyvpuzzle/utils/multi_view_renderer.py`: Camera presets and multi‑view rendering used by environments.
  (Deprecated: no separate second-hand manager module.)
- `docs/luban_constrained_mode_spec.md`: Detailed spec for Luban constrained mode behavior and configuration.

## Lifecycle & State

- `__init__`: Reads config (GUI, gravity, time step, rendering), sets up camera/renderer, and calls `_initialize_pybullet()`.
- `_initialize_pybullet()`: Connects to PyBullet (`GUI` or `DIRECT`), sets gravity/time step, loads a plane and (optionally) a table.
- `reset()`: Clears bodies, resets physics, (re)loads ground/table, calls `_setup_task_environment()` to add task objects and any constraints, then re‑enables gravity and returns an initial `Observation`.
- `step(Action)`: Executes one environment action, steps physics, updates `State`, checks done, and returns the next `Observation` plus feedback.
- `Observation`: Includes a main image and, if enabled, named per‑view images (`front`, `side`, `top`). See renderer in `utils/multi_view_renderer.py`.
- `State/Observation` creation: `_get_current_state()`, `_create_observation()`, `_evaluate_success()`, `_is_done()` are the main customization points.

## Tool System (Design Decisions)

- **JSON schemas:** `get_tool_schemas()` returns standard tools for all envs; subclasses add task‑specific tools with `_get_task_specific_tool_schemas()`.
- **Execution:** `execute_tool_call(tool_name, arguments)` dispatches to `_tool_*` handlers. Subclasses can override `_execute_task_specific_tool()`.
- **Single‑pick invariant:** Exactly one piece may be picked. Calling `pick` auto‑releases an existing pick before creating a new one (auto‑swap).
- **Auto‑release:** Only in Luban constrained mode. In free‑physics environments (e.g., Domino), `move/rotate` keep the pick active and the VLM should call `release`.
- **Placement‑only stabilization:** No automatic holding for `move/rotate/push`. The `place` tool can optionally apply a temporary world‑fixed hold on the target piece during placement for stability; the hold is always removed after the operation.
- **Common tools (base_env):**
  - `pick(object_id, reset_orientation=False)`
  - `release(object_id)`
  - `move(object_id, position=[x,y,z])`
  - `rotate(object_id, axis={'x'|'y'|'z'}, angle=radians)`
  - `push(object_id, force, direction=[x,y,z])`
  - `observe(angle=radians)`
  - `wait(duration=seconds)`
  - `check_solution()`
- **Returns:** All tools return a dict with `status` (`success|error`) and `message`. Some also return `picked_objects` and/or `constraint_id`.

Note on agent wiring: `OpenAIAgent` supports native tools but `_get_available_tools()` currently returns `[]`. The runner can be extended to inject `environment.get_tool_schemas()` so the model uses native function‑calling instead of text parsing.

## Place Tool (Plan)

- **Intent:** Provide a simple way to place a picked object directly on top of a target object under free‑physics.
- **API:** `place(object_id, target_id, offset_xy=[0,0], offset_local=true, align_orientation='align_target'|'keep'|'snap_90', clearance=0.001, release_after=false, stabilize_target=true, hold_max_force=300.0, hold_erp=0.4)`.
- **Semantics:**
  - Requires `object_id` to be currently picked; `target_id` must exist.
  - If `stabilize_target=true`: create a temporary world‑fixed `JOINT_FIXED` constraint on `target_id` at its current pose (`changeConstraint(maxForce=hold_max_force, erp=hold_erp)`), then remove it in a finally block after placement.
  - Computes target top Z from `getAABB(target_id)`, object half‑height from `getAABB(object_id)`; places at `top_z + half_height + clearance`.
  - `offset_xy` applies on target’s top plane; when `offset_local=true`, offset is rotated by target yaw.
  - `align_orientation`: keep yaw, copy target yaw, or snap yaw to nearest 90°.
  - Applies pose via active pick constraint and steps a few frames to settle. No auto‑release; `release_after` triggers an explicit release.
  
- **How placement computes pose (IDs + live state only):**
  - Core idea: rely on Bullet’s collision AABB and the live base pose to infer a correct “on top” placement without mesh anchors.
  - Target top: `t_min, t_max = getAABB(target_id)` → `top_z = t_max[2]`; target XY center `t_xy = ((t_min.x+t_max.x)/2, (t_min.y+t_max.y)/2)`; target yaw from `getBasePositionAndOrientation`.
  - Object extents: `o_min, o_max = getAABB(object_id)` → `half_h = (o_max[2]-o_min[2])/2`; object AABB center `o_center` and live base pose `o_pos, o_orn`.
  - Origin offset correction: `delta_xy = (o_pos.xy - o_center.xy)`, `delta_z = (o_pos.z - o_center.z)`; preserves visible alignment when moving.
  - XY target: start with `t_xy`; if `offset_local=true`, rotate `offset_xy` by target yaw into world XY; else use as world XY. Then `place_xy_world = t_xy + offset_world` and `pivot_xy = place_xy_world + delta_xy`.
  - Z target: `pivot_z = top_z + half_h + clearance + delta_z`.
  - Orientation: `keep` uses `o_orn`; `align_target` copies target yaw; `snap_90` snaps yaw to nearest 90° relative to target yaw.
  - Apply via `changeConstraint(..., jointChildPivot=[pivot_xy, pivot_z], jointChildFrameOrientation=...)`, step a few frames; if penetration grows, lift `pivot_z` slightly and retry a couple times.
  - Why it works: AABB is computed on collision geometry in world space; the origin offset ensures the object’s physical bottom sits on the target’s top despite URDF origin peculiarities; yaw-only changes keep Z extent stable.
  - Optional robustness: vertical ray cast along −Z to find the exact hit point on `target_id` at the intended XY; for tilted targets, derive the plane normal from target orientation and optionally project along that normal instead of world −Z.
- **Error handling:** Invalid IDs or not picked → error; if penetration increases after placement, lift slightly and retry a couple times, else return error with guidance.
- **Scope:** Implemented for free‑physics environments first (e.g., stacking puzzles). Luban constrained mode can return a clear “not supported” for now.

## Domino Environment (Free‑Physics)

- **Where:** `src/phyvpuzzle/environment/domino_env.py`
- **What:** Loads 3D dominoes (URDF or primitives), arranges patterns, and exposes domino‑specific tools.
- **Patterns:** `line`, `curve`, `zigzag`, `circle` via `_calculate_domino_positions()`; initial placement in `_arrange_dominoes()`.
- **Task‑specific tools:**
  - `push_domino(force=..., direction=[...])`: push the first domino.
  - `push_specific_domino(domino_id, force, direction)`: push a named domino.
  - `reset_dominoes()`: restore initial upright pose.
- **Success:** `_count_fallen_dominoes()` classifies fallen via tilt; success when ≥80% fall (`_evaluate_success`).
- **Config surface:** `DominoConfig.from_difficulty(...)` maps `TaskDifficulty` to counts/patterns; environment reads `urdf_base_path`, `gravity`, render settings, and `load_table`.
- **Holding:** Domino runs in free physics; there is no automatic holding.

## Luban Environment (Constrained Mode)

- **Where:** `src/phyvpuzzle/environment/luban_env.py` and `docs/luban_constrained_mode_spec.md`.
- **Why constrained:** Remove gravity/instability from wooden interlocks so VLM reasoning focuses on geometry. All pieces are “world‑fixed” by baseline constraints; manipulation moves/rotates that baseline.
- **Key overrides (when constrained enabled):**
  - `pick`: selects a piece by syncing its baseline constraint to the live pose (keeps single‑pick semantics; auto‑swap supported).
  - `move`: interpolates translation from current pose to target (`move_step_m`); aborts early if `guard_contacts` detects increasing penetration; auto‑deselects.
  - `rotate`: interpolates rotation by `rotate_step_deg`; same guard; auto‑deselects.
  - `push`: disabled in constrained mode (returns `status:error`).
  - `release`: clears selection but never removes the baseline constraint.
- **Guarding contacts:** `_calculate_penetration_metric()` sums negative contact distances; movement/rotation abort when penetration worsens consistently.
- **Config (under `environment.luban_constrained`):**
  - `enabled` (bool, default true)
  - `move_step_m` (meters per interp step)
  - `rotate_step_deg` (degrees per interp step)
  - `max_force` (for `changeConstraint(..., maxForce=...)`)
  - `erp` (for `changeConstraint(..., erp=...)`)
  - `settle_steps` (simulation steps between increments)
  - `guard_contacts` (abort if penetration increases)
- **Free‑physics fallback:** If `enabled: false`, Luban inherits base behavior; pushes work and there is no automatic holding.

Example YAML snippet:

```yaml
environment:
  type: pybullet
  gui: true
  luban_constrained:
    enabled: true
    move_step_m: 0.01
    rotate_step_deg: 5
    max_force: 300
    erp: 0.4
    settle_steps: 8
    guard_contacts: true
```

## Automatic Holding

- There is no global automatic holding. The `place` tool can optionally stabilize the target piece with a temporary world‑fixed hold during placement; this hold is always removed after the operation.

## Runner & Wiring Notes

- **Where:** `src/phyvpuzzle/runner.py`.
- **Current routing:** Domino tasks use `DominoEnvironment`; other types currently fall back to `PhysicsEnvironment`.
- **Extending:** Route `luban_lock` to `LubanEnvironment` and inject tool schemas to the agent for native function calling.
- **Tool schemas to agent (TODO):** Update `OpenAIAgent._get_available_tools()` or pass `tools=environment.get_tool_schemas()` from the runner when constructing the request.

## Extending With A New Environment

- **Subclass:** Create `MyEnv(PhysicsEnvironment)` in `src/phyvpuzzle/environment/`.
- **Setup:** Implement `_setup_task_environment()` to load pieces and any constraints.
- **State:** Override `_get_current_state()`, `_evaluate_success()`, and optionally `_get_state_description()`.
- **Tools:** Add JSON schemas via `_get_task_specific_tool_schemas()` and handle them in `_execute_task_specific_tool()`.
- **Actions list:** Extend `get_available_actions()` if exposing new actions.
- **Config:** Read keys you need from `config` in `__init__`.

## Common Errors & Recovery Patterns

- **“must be picked” errors:** Always call `pick(object_id)` before `move/rotate` in free‑physics and in Luban constrained mode.
- **Luban push disabled:** In constrained mode, `push` returns an error; use `pick + move/rotate` instead.
- **Guard aborts (Luban):** Movement/rotation may return `status:error` when penetration increases. Retry with smaller steps, different path/orientation, or reposition other pieces first.
- **Tool not found:** Ensure your action name is included in `get_available_actions()` and has a schema in `get_tool_schemas()`.
- **Debugging visuals:** Set `environment.gui: true`; increase logging with `environment.verbose_logging: true` (Luban); enable saving images in the runner config.

## Quick Pointers (Code)

- Base harness: `src/phyvpuzzle/environment/base_env.py`
- Domino env: `src/phyvpuzzle/environment/domino_env.py`
- Luban env: `src/phyvpuzzle/environment/luban_env.py`
- Runner: `src/phyvpuzzle/runner.py`
  (No separate second-hand module; placement stabilization is inline.)
- Renderer: `src/phyvpuzzle/utils/multi_view_renderer.py`
- Agent (OpenAI): `src/phyvpuzzle/agents/openai_agent.py` (tools injection TODO)

If you’re onboarding: skim this file, then open the code paths above in that order. For Luban specifics, read `docs/luban_constrained_mode_spec.md` next.
