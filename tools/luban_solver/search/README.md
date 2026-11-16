# Original-rule search

This search uses `../original_engine` to restore the game's original group translations, group/pivot rotations, 45° states and conditional linked translation/rotation rules. It does not change puzzle geometry or start positions. Results are **candidates**, never live-game success receipts.

Build with .NET 8:

```sh
dotnet build tools/luban_solver/search/LubanSearch.csproj -c Release
```

Input files must contain the original rules restored by `../restore_original_rules.py`. For example, from the repository root:

```sh
dotnet tools/luban_solver/search/bin/Release/net8.0/LubanSearch.dll \
  artifacts/luban/original_rules/Puzzle_4.json \
  --out artifacts/luban/search/Puzzle_4.json --seconds 120 --max-states 100000
```

Default best-first search prioritizes finish-zone progress and separated bounding boxes. This is a practical search heuristic, not an admissible shortest-path algorithm. `--strategy bfs` selects breadth-first traversal. `--expansion-batch 64` permits best-first traversal to consider promising child states before enumerating all rotations of their parent; the remaining parent actions stay on the resumable frontier. States reached by a shorter path are reopened so depth bounds do not discard shallower continuations. `--heuristic overlap` instead prioritizes reducing bounding-box penetration after finish count, with pair separation as a smaller tie-breaker; it may help when useful intermediate contacts look worse under the default pair-count score. Whole-assembly moves are excluded; finished pieces remain physical obstacles and may still move.

A candidate must satisfy **both** the original N−1 finish-zone condition and a conservative disassembly certificate: every pair of world-space piece AABBs has positive separation on at least one axis. Simply moving a still-interlocked group into the finish zone is insufficient.

## Explicit search scope

- `--directions axes` (default) uses six axis directions. `--directions grid26` adds the twenty diagonals with components −1, 0, or 1. A vector action moves by `Distance * Vector`; it remains collision checked.
- `--groups closure` (default): single pieces and directional blocking closures. `--groups all` enumerates all nonempty proper subsets (exponential; 62 pieces maximum). `--groups singles` omits groups.
- `--distances events` (default): vertex alignments, positive separation offsets, contact limits, small grid motions and goal-slab landings. `--distances grid --grid 10` enumerates configured positive grid distances up to `--max-distance 600`, plus the contact limit, in both directions.
- `--pivots origins` (default): member origins and their mean. `--pivots vertices` also considers current geometry vertices.
- `--park-finished` omits ordinary moves/rotations containing a finished piece whose AABB is already positively separated from every other piece. After native victory is latched, the last unflagged piece is also eligible if it is independently separated. Such pieces still participate in collision checks; conditional original special actions remain available. This is optional heuristic pruning and may miss paths requiring a separated piece to move again.
- `--park-components` omits rigid translations of a finished group (or any group after native victory) already AABB-separated from every outside piece. Internal group splits and rotations remain available. This optional heuristic prevents wandering of isolated interlocked chunks, but may miss paths needing those chunks repositioned.
- Rotations include ±45° and ±90°, subject to original engine rules. `--no-rotations` is an explicit diagnostic restriction.
- `--radius 1200` bounds positions relative to the initial center; `--max-depth 80` bounds path length. These bounds are part of the reported domain.

Even all-subsets/grid/vertex mode is a **declared finite action domain**, not an exhaustive enumeration of all continuous physical motions. No search result here proves that a physical puzzle is impossible. Even grid26 does not enumerate arbitrary continuous translation vectors, and arbitrary pivot positions are not enumerated.

## Receipts and resume

A result and full checkpoint are written separately. The checkpoint stores exact fractional piece positions, orientations, half-rotation state, finish flags, parents, frontier and the next action of a partially expanded node. It includes SHA-256 hashes of input rules, engine DLL, common engine source and search DLL. The source fingerprint is baked at compile time from sorted `.cs`, `.csproj` and `.props` paths and bytes, excluding `bin` and `obj`, so it is comparable with the separately built Unity runtime. Changed rules or code cannot silently reuse a stale checkpoint.

```sh
dotnet tools/luban_solver/search/bin/Release/net8.0/LubanSearch.dll \
  artifacts/luban/original_rules/Puzzle_4.json \
  --out artifacts/luban/search/Puzzle_4.json \
  --resume artifacts/luban/search/Puzzle_4.json.checkpoint.json \
  --seconds 600 --max-states 500000
```

Keep search-scope options identical when resuming; time and state budgets may change. The search timer begins after puzzle initialization; suite `wall_seconds` also includes initialization. A time limit is checked between engine operations, so one slow geometry operation can exceed the requested deadline. The suite runner imposes an additional external deadline and records its process exit status.

Statuses distinguish `candidate_solution`, `time_limit`, `state_limit`, `interrupted`, `engine_error` and `frontier_exhausted_in_declared_domain`. Frontier exhaustion applies only to the explicitly restricted domain. No `unsolvable` or `shortest` label is emitted.

A fresh-instance replay checks every action and exact final state using the same original-rule model:

```sh
dotnet tools/luban_solver/search/bin/Release/net8.0/LubanSearch.dll replay \
  artifacts/luban/original_rules/Puzzle_4.json artifacts/luban/search/Puzzle_4.json
```

This produces `.model-replay.json`, still with `game_verified=false`. Actual Unity replay must independently accept the commands, confirm the game victory signal, and verify separation; snapshot teleports do not count as replay.

Run all 32 levels with bounded resources:

```sh
python tools/luban_solver/search/run_suite.py --input-dir artifacts/luban/original_rules \
  --output-dir artifacts/luban/search_suite --workers 2 --seconds 120 \
  --directions grid26 --expansion-batch 32
```

Integration regressions cover state-budget interruption, exact resumed paths, fresh-model replay, stale engine/input rejection, and level 4's original linked action:

```sh
python tools/luban_solver/search/test_search.py --levels artifacts/luban/original_rules
```

A known action sequence can be checked without claiming the search discovered it:

```sh
dotnet tools/luban_solver/search/bin/Release/net8.0/LubanSearch.dll certify \
  artifacts/luban/original_rules/Puzzle_19.json actions.json candidate.json
```

The actions file is a JSON array of `ActionSpec` objects. Such receipts explicitly say `candidate_origin=provided_path_revalidated`; they still require live-game replay.

To extend an existing path, use `--initial-actions actions.json`. Every prefix action is replayed from the original input before searching further, and its file hash is recorded in the scope and receipt. This searches only continuations of that supplied prefix.

The verified search portfolio used a general pass with group/diagonal moves and rotations, then coarse-grid continuations for unresolved levels. Useful continuation options are `--directions axes --distances grid --grid 10 --expansion-batch 32 --park-finished --park-components`. A translation-only diagnostic (`--no-rotations`) can avoid spending most of a budget on blocked rotations; failure in that restricted mode says nothing about paths requiring rotation. Every supplied prefix is re-applied from the original initial state before continuing.
