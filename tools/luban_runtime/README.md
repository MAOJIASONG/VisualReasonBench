# Isolated Luban runtime repairs

`prepare_runtime.py` creates a **new** runtime directory and records input/output hashes.
It does not edit an existing package, launch the game, or claim that a level is solved.
The supported input is the committed Linux player at commit
`65538f43c98b13aeadaae95a57c5a9850c11ef79`; unknown hashes are rejected.

## Prepare and run

```bash
python tools/luban_solver/restore_original_rules.py \
  --out artifacts/luban/original_rules
python tools/luban_runtime/prepare_runtime.py \
  --source Luban --output artifacts/luban/runtime_final \
  --dotnet /path/to/dotnet \
  --original-rules artifacts/luban/original_rules
xvfb-run -a artifacts/luban/runtime_final/LuBanSuoEnv.x86_64 \
  -job-worker-count 2 -logFile /tmp/luban-repaired.log
```

The restored rule files must come from `tools/luban_solver/restore_original_rules.py`
and retain the current 32 levels' geometry and starting arrangements. The bundled
`original_rule_pack.json` contains the source-derived conditional rules, so preparation
does not require the uploaded archive. An optional `--source` on the restoration
command cross-checks an extracted copy. The runtime output directory must not already
exist; use a different output for a later build. The rule-file,
engine, extension, patcher-source, input, and output hashes are in `repair_manifest.json`.
`original_engine_source.sha256` identifies the exact common source used for the Unity
and standalone builds; compare this digest with search receipts before replay. Their
compiled DLL hashes differ because the targets differ. Original-rule packaging always
builds fresh and fails if engine sources change during preparation.
The DLL files are built by .NET 8 (NuGet restores Mono.Cecil if not cached). The game extension targets .NET Standard 2.1 and uses
the game's actual Unity assemblies; the native Unity executable is unchanged.

## Native protocol fixes

* Server response wait: 5 seconds → 300 seconds. Six-view captures on mounted storage
  sometimes exceeded the original wait and returned an empty line. Original geometry
  precomputation also takes longer under shared CPU contention: level 30 took 21.7 seconds
  in the full-level check and about 179 seconds during a heavily loaded cold start.
  The finite budget accommodates that observed computation; the wrapper waits 310 seconds.
  On timeout, inspect state instead of replaying the request. The diagnostic logs are in
  [the cold-start diagnosis](../../artifacts/luban/cold_load_diagnosis/summary.json)
  when the validation artifacts are available. The v6 → v7 game DLL differs in only
  the three bytes needed to replace 120000 with 300000; the engine, extension, and
  all 32 rule files have identical hashes.
* Timeout now returns JSON with `code=response_timeout` and
  `execution_may_have_occurred=true`. **Do not blindly retry state-changing actions.**
  Check state first. This does not implement transaction cancellation.
* Accept-loop polling: 100 ms → 5 ms, with a bounded sleep (not a busy loop). Actual
  throughput must be measured separately; this parameter alone proves no speedup.
* Native `rotate` forwards the requested finite multiple of 90°, rather than clamping
  every nonzero angle to ±90°. It rejects invalid fractions/nonfinite values. Native
  multi-turn operations can still partially progress before a later turn is blocked;
  inspect the returned state after failure. Restored mode uses explicit ±45/±90 steps.

## Restored original-rule mode

This mode restores legal action capabilities found in the uploaded original source,
including groups, 45° rotations and condition-gated simultaneous translation/rotation.
It is explicit and must be used consistently within a session.

```json
{"command":"original_start","env_id":1,"params":{"level_index":4}}
{"command":"original_action","env_id":1,"params":{"kind":"special","rule_index":1}}
{"command":"original_state","env_id":1,"params":{}}
{"command":"capture","env_id":1,"params":{}}
```

Action parameters follow the restored engine:

* `kind=move`, `piece_ids=[...]`, `axis=x|y|z`, `distance=<logical units>`.
* `kind=move_vector`, `piece_ids=[...]`, `vector=[dx,dy,dz]`, and positive `distance`.
  The integer direction must be primitive (nonzero, greatest common divisor 1). The
  displacement is direction multiplied by distance, checked by the original geometry.
* `kind=rotate`, `piece_ids=[...]`, `axis=x|y|z`, `angle=±45|±90`, optional
  `pivot=[x,y,z]` in logical units (default is the first selected piece's origin).
* `kind=special`, `rule_index=<zero-based original moveRotates index>`.

Use `distance_exact` instead of `distance` when replaying double-precision solver
distances; the old `distance` field stores a float. Pivot coordinates use doubles.

Initial coordinates and orientation are checked against the native game before enabling
the mode. Every action first passes the restored original legality engine. Only an
accepted action's affected transforms are applied to the native scene. There is no
external state-restore endpoint, no reference-solution state assignment, and no write to
Unity's finished-piece set or victory flag. The two fixed updates before an action
response let native trigger volumes observe the changed scene.

Responses distinguish `engine_state` / `engine_zone_estimate` from
`native_unity_is_solved` / `native_finished_piece_ids`. The engine's estimate is not an
independent success receipt. A solution certificate must also establish disassembly;
moving an intact group into a finish volume is not sufficient evidence.
`native_positions` reports native logical coordinates, actual rendered quaternions,
`visual_logical_position` read back from the Transform, and `scene_world_bounds`
(union of enabled renderer/collider bounds in world coordinates). Compare every pair
of actual scene bounds independently when certifying disassembly. Scene bounds are
read-only evidence and do not come from the search engine.

The existing native geometry stores only 24 axis-aligned orientations. After a 45°
restored action its collision state cannot represent that orientation, so native `move`
and `rotate` are rejected while restored mode is active. `start_level` or `reset` leaves
restored mode and reinstantiates the native level. Native screenshot capture remains
available in either mode.
Legacy `get_state` stays readable in restored mode, with `orientation_is_base_only=true`;
use `original_state` for the complete orientation including 45-degree rotation.

## Project integration

The normal `LubanEnvironment` supports `action_mode: restored_original_rules`; native
mode remains the default. `eval_configs/luban_restored.yaml` selects a runtime prepared
at `artifacts/luban/runtime_final`. Prepare that directory with the command above, changing
`--output` for another location and setting the configuration's `exe_path` to match.
The wrapper exposes single/group moves, vector moves, ±45/±90 rotations with optional
pivots, and original special rules. It keeps rendered quaternions and half-rotation
state in observations, and requires native victory plus pairwise scene separation for
`is_solved`. Failed reset/state/capture calls are surfaced rather than reusing a stale
observation. Communication failures never automatically replay actions.
The wrapper allows 90 seconds for startup, launches two Unity job workers by default,
and uses `xvfb-run` on Linux when `DISPLAY` is absent. It closes only its own process
group, including its virtual display, and leaves an already-running server alone.

## Validation

```bash
DOTNET=/path/to/dotnet python -m unittest discover -s tools/luban_runtime/tests -v
```

Package tests cover input/output protection, unsupported-input rejection, deterministic
patch bytes and bounded timeout configuration. They are not gameplay tests. Record
separate actual startup, 32-level capture, legal/blocked actions, rotation, special
precondition, and native-completion results before deployment.
