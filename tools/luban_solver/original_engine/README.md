# Restored original LuBan rules

This library adapts the `BurrPuzzle/DataCore` source from the supplied 2022.3 editor archive. It preserves original puzzle geometry, original orientation tables, exact rational coordinates, group translation, original sampled rotation checks, and conditional `moveRotates`. It does not relocate starting pieces, change meshes, or provide a forced-win action.

`source_manifest.txt` records each imported source file's original SHA-256. Adaptations to imported files are limited to removing the asset/UI loader, making the existing orientation-hull selector accessible for state restore, an overload exposing the existing movement implementation for primitive integer directions, and making verbose shape-comparison logs optional. The compatibility layer supplies standalone Unity math types; `UseUnity=true` uses the real Unity types instead.

The façade corrects conversion at its boundary: the archive's `BigFraction.DoubleValue()` performs integer division and mishandles negative coordinates. `AsDouble()` divides numerator by denominator, so a self-pivot remains the exact piece origin. Internal state snapshots retain rational strings, all three orientation axes, the half-rotation axis, and the finish-zone state. Rejected actions preserve state. Actions must pass the original collision or conditional-rule predicate before they change state.

Performance caches use exact relative coordinates, piece identities, both orientations, the half-rotation axes, and movement direction. A conservative exact swept-AABB rejection skips only pairs that cannot collide; the remaining pairs use the original SAT implementation. No approximate voxel or bounding-box-only collision rule is substituted.

## Build and checks

```bash
dotnet build tools/luban_solver/original_engine/OriginalEngine.csproj -c Release
dotnet build tools/luban_solver/original_engine/OriginalEngine.csproj -c Release -p:UseUnity=true
dotnet run --project tools/luban_solver/original_engine/cli/CoreCheck.csproj -- artifacts/luban/original_rules/Puzzle_0.json --self-test
dotnet run --project tools/luban_solver/original_engine/cli/CoreCheck.csproj -- artifacts/luban/original_rules/Puzzle_4.json --special4
dotnet run --project tools/luban_solver/original_engine/cli/CoreCheck.csproj -- --compare-native artifacts/luban/original_rules native-probes.json parity-output.json
```

The two targets isolate their intermediate files so runtime and search builds do not overwrite each other's restore results.

## Actions and state

- `move`: piece IDs, axis `0/1/2`, signed logical distance.
- `move_vector`: piece IDs, `Vector` with three integer components whose GCD is one, and positive scalar `Distance`. Displacement is `Vector * Distance`; diagonal distance is not Euclidean length.
- `rotate`: piece IDs, world axis, explicit angle `±45` or `±90`, optional logical pivot. Omitted pivot means the first selected piece's current origin. The original engine's restrictions on half rotations about external pivots remain in force.
- `special`: an original `RuleIndex`; its original complete precondition is checked again immediately before execution.

`Capture` and `Restore` exist for in-process graph search and are not exposed as gameplay commands. `MaxTranslationVector` returns a scalar distance along the supplied primitive vector. `BlockingClosureVector` returns the minimal group closed under current zero-distance blockers in that direction.

`FinishedCount` and `Solved` are search estimates from six original finish slabs. Zone membership is recomputed until N−1 pieces finish; victory then latches, matching the native game's behavior. These values are **not a native game success receipt**. Candidate solutions additionally need independent piece separation and replay through the actual Unity runtime's unmodified completion trigger. Original rotations check sampled orientations; conditional moves execute the original authored rules. Neither is a new continuous-physics proof.

## Recorded validation

`validation/core_checks.json` records group/vector/rotation/restore regressions, all six rotation axis/sign cases, and finish-zone entry/exit/latch checks. `validation/native_parity.json` compares 3,427 previously captured native translation responses (complete levels 0–26 and seven moves from level 27): zero differences in acceptance. `validation/current_core_parity.json` compares all 32 levels against the current decompiled exact geometry implementation: 2,232 initial single-piece ±XYZ limits, zero differences, with a 600-unit query cap. These checks establish initial geometry/movement compatibility; actual solution replay is a separate requirement.

Additional harness source text is saved beside the reports without changing the frozen engine fingerprint. The recorded source identity is `955375b81731551b04ab1d83c8f96c0e67405d7659f418cbd214df6a645eda0b` under the runtime/search shared fingerprint function.
