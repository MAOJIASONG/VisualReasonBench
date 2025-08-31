# HANDOVER

## Supplementary Documentation

**Environment Harness**: `docs/ENVIRONMENT_HARNESS.md`
**Blender**: `docs/BLENDER.md`

Notes:
- The environment harness doc is up to date with current code. See it for base tools, config, and extension points.

## TODO

- [ ] Add color (+ number) combination to current environments using Blender (**IMPORTANT**)
- [ ] Finish up `luban_env.py` constrained implementation (roughly finished; needs proper testing)
  - [ ] Create solutions for luban-style puzzles (luban-*-piece, liliput, tangled-nails, three-plate-burr-puzzle, interlocking cube)
  - [ ] Implement the scoring mechanism for `luban_env.py` where the optimal number of steps is the minimum steps taken
  - [ ] Count number of steps needed for each puzzle; use online solutions where appropriate (see the Google Slides)
- [ ] Route `luban_lock` in runner to `LubanEnvironment` and pass `environment.luban_constrained` settings
- [ ] Inject `environment.get_tool_schemas()` into agent requests (OpenAI/VLLM) for native tool-calling
- [ ] Resolve `pull` action in base env (either implement or remove from `get_available_actions()`)
- [ ] PLACE tool: implemented but provisional — requires deeper design review for stacking tasks (LEGO/Pagoda)
  - [ ] Surface detection: replace naive AABB-top with robust top-surface finding (concavity, sloped tops, multi-part)
  - [ ] Frame semantics: clarify `offset_xy` local vs world; ensure yaw alignment options (`keep|align_target|snap_90`) are consistent and documented
  - [ ] Clearance strategy: dynamic approach/settle vs fixed clearance; minimize penetration while avoiding hover
  - [ ] Stabilization: confirm temporary hold placement (pose, frame); tune `hold_max_force`/`hold_erp`; guarantee cleanup on all error paths
  - [ ] Collision guard: formalize thresholds and backoff when penetration increases; potentially multi-try with reduced step size
  - [ ] Constrained mode: decide semantics (disable? passthrough?) to avoid conflicting with baseline constraints
  - [ ] Return contract: include final poses, contact metadata, and explicit `released` flag; ensure idempotency where feasible
  - [ ] Error taxonomy: standardized messages (`must_be_picked`, `unknown_object`, `stabilization_failed`, `penetration_guard_abort`)
  - [ ] Tests/demos: author scenarios for 3x3 stacking puzzle and LEGO pagoda; edge cases (thin targets, tiny clearances, non-flat tops)
  - [ ] Documentation: update tool schema description and examples in `ENVIRONMENT_HARNESS.md` after design decisions
