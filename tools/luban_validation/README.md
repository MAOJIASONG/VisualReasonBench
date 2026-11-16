# LuBan verification

These tools test the real Linux Unity player. They do not call a language model,
restore solution snapshots, or write the game's victory flags.

## Prepare

From the repository root, with Python, Pillow, .NET 8 and `xvfb-run` installed:

```bash
python tools/luban_solver/restore_original_rules.py --out artifacts/luban/original_rules
python tools/luban_runtime/prepare_runtime.py --source Luban \
  --output artifacts/luban/runtime_final --original-rules artifacts/luban/original_rules
dotnet build tools/luban_solver/search/LubanSearch.csproj -c Release
```

The bundled rule pack restores 20 original conditional actions in eight levels.
Restoration checks every original level's content before writing a separate
dataset. Geometry and initial poses are unchanged. Runtime preparation requires
a new output directory, protecting the shipped build.

## Check all 32 levels

```bash
python tools/luban_validation/original_regressions.py \
  --package artifacts/luban/runtime_final --out artifacts/luban/original_checks
```

This tests blocked-action immutability, real 45° visual rotation, rotation/vector
roundtrips, native/rendered coordinate agreement, and six screenshots for each
level. Loading successfully is distinct from solving the puzzle.

## Search and replay

```bash
python tools/luban_solver/search/run_suite.py \
  --input-dir artifacts/luban/original_rules --output-dir artifacts/luban/search \
  --directions grid26 --distances grid --expansion-batch 32 \
  --seconds 600 --max-states 500000 --workers 2
python tools/luban_validation/replay_original.py \
  --package artifacts/luban/runtime_final --candidates artifacts/luban/search/Puzzle_*.json \
  --out artifacts/luban/replayed
```

Search results are candidates in a declared finite action domain. Time/state
limits are inconclusive. See the search README for bounds, resumption and
heuristic options. A real-game receipt requires all of:

- Each action is accepted, starting from the ordinary level initial state.
- Input hashes and the baked engine source fingerprint match the runtime.
- Every final rational coordinate, orientation and half-rotation state matches.
- Actual scene bounding boxes have a positive gap for **every** pair of pieces.
- The actual Unity completion trigger reports victory.

Every request and response is retained. A timed-out action is never retried
automatically, since it might already have executed. The validators refuse to
reuse an existing server, and stop only the process group they launched.

## Test the normal project interface

```bash
python tools/luban_validation/harness_smoke.py --package artifacts/luban/runtime_final \
  --candidate artifacts/luban/solutions/Puzzle_3.json --out artifacts/luban/harness_check
```

This uses `LubanEnvironment.reset/step`, automatic headless startup, ordinary
tool dispatch and composed observations; it also verifies graceful shutdown.
`eval_configs/luban_restored.yaml` enables the same restored-rule mode for the
benchmark runner. The native action mode remains available.

## Summarize matching receipts

```bash
python tools/luban_validation/summarize.py --solutions artifacts/luban/solutions \
  --replays artifacts/luban/live_v6_solutions_batch*/report.json \
  --loads artifacts/luban/original_v5_all32/report.json --out artifacts/luban/summary.json
```

A candidate is counted as game-verified only when its exact SHA-256 matches a
successful replay receipt. The summary is complete only after all 32 loads and
all 32 distinct candidates have passed. `test_checks.py` covers rejection of
intact-group wins, invalid scene bounds and inexact terminal states.
