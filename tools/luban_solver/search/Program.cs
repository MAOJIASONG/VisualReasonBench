using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using Luban.Original;

// Every result remains a candidate until independently replayed through the game.
static class Program
{
    internal static readonly JsonSerializerOptions Json = new() { IncludeFields = true, WriteIndented = true };
    public sealed class Options
    {
        public string Level = "", Output = "", Resume = "", Strategy = "best-first", Groups = "closure", Distances = "events", Pivots = "origins", Directions = "axes", InitialActionsFile = "", InitialActionsHash = "", Heuristic = "counts";
        public double Seconds = 120, Grid = 10, MaxDistance = 600, Radius = 1200, Clearance = .001;
        public int MaxStates = 100000, MaxDepth = 80, ExpansionBatch = 0;
        public bool Rotations = true, ParkFinished = false, ParkComponents = false;
        public string Scope() => JsonSerializer.Serialize(new { Strategy, Groups, Distances, Pivots, Directions, Grid, MaxDistance, Radius, Clearance, MaxDepth, Rotations, ExpansionBatch, InitialActionsHash, Heuristic, ParkFinished, ParkComponents });
    }
    public sealed class Node
    {
        public EngineState State = new();
        public int Parent = -1, Depth, ResumeAction;
        public ActionSpec? Action;
        public double Rank;
    }
    public sealed class Checkpoint
    {
        public string Version = "luban-original-search-v4", InputHash = "", EngineHash = "", EngineSourceHash = "", SearchHash = "", Scope = "";
        public List<Node> Nodes = new();
        public List<int> Frontier = new();
        public int Active = -1, NextAction;
        public long ActionsTried, ActionsAccepted, NodesExpanded;
        public double PreviousSeconds;
        public string? LastError;
        public ActionSpec? FailedAction;
        public double[] Center = Array.Empty<double>();
    }
    public static int Main(string[] args)
    {
        try { if (args.Length > 0 && args[0] == "certify") return Certify.Run(args.Skip(1).ToArray()); if (args.Length > 0 && args[0] == "replay") return Replay.Run(args.Skip(1).ToArray()); return Run(Parse(args)); }
        catch (Exception e) { Console.Error.WriteLine(e); return 2; }
    }
    static Options Parse(string[] a)
    {
        if (a.Length == 0) throw new ArgumentException("Usage: LubanSearch level.json --out result.json [--seconds 120 --max-states 100000 --strategy bfs|best-first --groups singles|closure|all --distances events|grid --pivots origins|vertices --resume checkpoint.json]");
        var o = new Options { Level = Path.GetFullPath(a[0]) };
        for (int i = 1; i < a.Length; i++)
        {
            string Value() => ++i < a.Length ? a[i] : throw new ArgumentException("Missing option value");
            switch (a[i])
            {
                case "--out": o.Output = Path.GetFullPath(Value()); break;
                case "--resume": o.Resume = Path.GetFullPath(Value()); break;
                case "--seconds": o.Seconds = double.Parse(Value()); break;
                case "--max-states": o.MaxStates = int.Parse(Value()); break;
                case "--max-depth": o.MaxDepth = int.Parse(Value()); break;
                case "--expansion-batch": o.ExpansionBatch = int.Parse(Value()); break;
                case "--strategy": o.Strategy = Value(); break;
                case "--groups": o.Groups = Value(); break;
                case "--distances": o.Distances = Value(); break;
                case "--pivots": o.Pivots = Value(); break;
                case "--directions": o.Directions = Value(); break;
                case "--heuristic": o.Heuristic = Value(); break;
                case "--initial-actions": o.InitialActionsFile = Path.GetFullPath(Value()); o.InitialActionsHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(o.InitialActionsFile))); break;
                case "--grid": o.Grid = double.Parse(Value()); break;
                case "--max-distance": o.MaxDistance = double.Parse(Value()); break;
                case "--radius": o.Radius = double.Parse(Value()); break;
                case "--clearance": o.Clearance = double.Parse(Value()); break;
                case "--no-rotations": o.Rotations = false; break;
                case "--park-finished": o.ParkFinished = true; break;
                case "--park-components": o.ParkComponents = true; break;
                default: throw new ArgumentException("Unknown option: " + a[i]);
            }
        }
        if (o.Output == "") o.Output = Path.GetFullPath("search-result.json");
        if (!new[] { "bfs", "best-first" }.Contains(o.Strategy) || !new[] { "singles", "closure", "all" }.Contains(o.Groups) || !new[] { "events", "grid" }.Contains(o.Distances) || !new[] { "origins", "vertices" }.Contains(o.Pivots) || !new[] { "axes", "grid26" }.Contains(o.Directions) || !new[] { "counts", "overlap" }.Contains(o.Heuristic)) throw new ArgumentException("Unsupported search mode");
        if (new[] { o.Grid, o.MaxDistance, o.Radius, o.Seconds, o.Clearance }.Any(v => !double.IsFinite(v)) || o.Grid <= 0 || o.MaxDistance <= 0 || o.Radius <= 0 || o.Seconds <= 0 || o.MaxStates < 1 || o.MaxDepth < 1 || o.Clearance <= 0 || o.ExpansionBatch < 0) throw new ArgumentException("Search bounds must be positive");
        return o;
    }
    static int Run(Options o)
    {
        string input = File.ReadAllText(o.Level), hash = Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(input)));
        string engineHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(OriginalEngine).Assembly.Location)));
        string searchHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(Program).Assembly.Location)));
        var engine = OriginalEngine.LoadFromJson(input);
        var clock = Stopwatch.StartNew();
        var c = o.Resume == "" ? new Checkpoint { InputHash = hash, EngineHash = engineHash, EngineSourceHash = EngineSourceFingerprint.Value, SearchHash = searchHash, Scope = o.Scope() } : JsonSerializer.Deserialize<Checkpoint>(File.ReadAllText(o.Resume), Json)!;
        if (c.Version != "luban-original-search-v4" || c.InputHash != hash || c.EngineHash != engineHash || c.EngineSourceHash != EngineSourceFingerprint.Value || c.SearchHash != searchHash || c.Scope != o.Scope()) throw new InvalidOperationException("Checkpoint input/rules/search scope does not match");
        if (c.Nodes.Count == 0)
        {
            var positions = engine.PositionsLogical();
            c.Center = Enumerable.Range(0, 3).Select(a => positions.Average(p => p[a])).ToArray();
            c.Nodes.Add(new Node { State = engine.Capture() });
            if (o.InitialActionsFile != "")
            {
                var prefix = JsonSerializer.Deserialize<ActionSpec[]>(File.ReadAllText(o.InitialActionsFile), Json)!;
                if (prefix.Length > o.MaxDepth || prefix.Length + 1 > o.MaxStates) throw new ArgumentException("Supplied prefix exceeds depth or state bound");
                foreach (var action in prefix)
                {
                    if (action.Kind != "special" && action.PieceIds.Distinct().Count() >= c.Nodes[0].State.Pieces.Length) throw new ArgumentException("Whole-assembly prefix action excluded");
                    var result = engine.Apply(action);
                    if (!result.Success) throw new ArgumentException("Supplied prefix action rejected: " + result.Message);
                    if (!Inside(engine.PositionsLogical(), c.Center, o.Radius)) throw new ArgumentException("Supplied prefix leaves coordinate bound");
                    var state = engine.Capture(); var progress = Evaluate(engine, state, o.Clearance);
                    c.Nodes.Add(new Node { State = state, Parent = c.Nodes.Count - 1, Depth = c.Nodes.Count, Action = action, Rank = Rank(progress, c.Nodes.Count, o) });
                }
            }
            c.Frontier.Add(c.Nodes.Count - 1);
        }
        var seen = new Dictionary<string, int>();
        foreach (var node in c.Nodes) { engine.Restore(node.State); var k = engine.StateKey(); if (!seen.TryGetValue(k, out int depth) || node.Depth < depth) seen[k] = node.Depth; }
        var frontier = new PriorityQueue<int, (double, int)>();
        foreach (int i in c.Frontier) frontier.Enqueue(i, (o.Strategy == "bfs" ? c.Nodes[i].Depth : c.Nodes[i].Rank, i));
        c.Frontier.Clear();
        int found = -1, best = Enumerable.Range(0, c.Nodes.Count).MinBy(i => c.Nodes[i].Rank); bool interrupted = false; string status = "frontier_exhausted_in_declared_domain";
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; interrupted = true; };
        var savedAt = Stopwatch.StartNew();
        while (c.Active >= 0 || frontier.Count > 0)
        {
            if (interrupted) { status = "interrupted"; break; }
            if (clock.Elapsed.TotalSeconds >= o.Seconds) { status = "time_limit"; break; }
            if (c.Active < 0) { c.Active = frontier.Dequeue(); c.NextAction = c.Nodes[c.Active].ResumeAction; }
            var cur = c.Nodes[c.Active]; engine.Restore(cur.State);
            var score = Evaluate(engine, cur.State, o.Clearance);
            if (score.Solved) { found = c.Active; status = "candidate_solution"; break; }
            if (cur.Depth >= o.MaxDepth) { c.Active = -1; c.NextAction = 0; continue; }
            using var actions = Actions(engine, o, c.Center, () => interrupted || clock.Elapsed.TotalSeconds >= o.Seconds).GetEnumerator();
            bool capped = false, yielded = false; int index = 0, batchActions = 0;
            while (true)
            {
                engine.Restore(cur.State);
                if (o.ExpansionBatch > 0 && batchActions >= o.ExpansionBatch)
                {
                    cur.ResumeAction = c.NextAction;
                    frontier.Enqueue(c.Active, (o.Strategy == "bfs" ? cur.Depth : cur.Rank, c.Active));
                    c.Active = -1; c.NextAction = 0; yielded = true; break;
                }
                if (interrupted || clock.Elapsed.TotalSeconds >= o.Seconds) { status = interrupted ? "interrupted" : "time_limit"; capped = true; break; }
                if (savedAt.Elapsed.TotalSeconds > 30) { Save(o, c, frontier, clock.Elapsed.TotalSeconds, status: "running", found: -1, best); savedAt.Restart(); }
                if (!actions.MoveNext())
                {
                    if (interrupted || clock.Elapsed.TotalSeconds >= o.Seconds) { status = interrupted ? "interrupted" : "time_limit"; capped = true; }
                    break;
                }
                if (index++ < c.NextAction) continue;
                var action = actions.Current;
                var result = engine.Apply(action); c.ActionsTried++; batchActions++;
                if (!result.Success)
                {
                    if (result.Message != null && result.Message.Contains("Exception:")) { c.LastError = result.Message; c.FailedAction = action; status = "engine_error"; capped = true; break; }
                    c.NextAction++; continue;
                }
                c.ActionsAccepted++;
                var child = engine.Capture();
                if (!Inside(engine.PositionsLogical(), c.Center, o.Radius)) { c.NextAction++; continue; }
                string key = engine.StateKey();
                if (seen.TryGetValue(key, out int knownDepth) && knownDepth <= cur.Depth + 1) { c.NextAction++; continue; }
                if (c.Nodes.Count >= o.MaxStates) { status = "state_limit"; capped = true; break; }
                seen[key] = cur.Depth + 1; c.NextAction++;
                var evaluated = Evaluate(engine, child, o.Clearance);
                double rank = Rank(evaluated, cur.Depth + 1, o);
                var next = new Node { State = child, Parent = c.Active, Depth = cur.Depth + 1, Action = action, Rank = rank };
                int id = c.Nodes.Count; c.Nodes.Add(next); frontier.Enqueue(id, (o.Strategy == "bfs" ? next.Depth : next.Rank, id));
                if (rank < c.Nodes[best].Rank) best = id;
                if (evaluated.Solved) { found = id; status = "candidate_solution"; capped = true; break; }
                if (savedAt.Elapsed.TotalSeconds > 30) { Save(o, c, frontier, clock.Elapsed.TotalSeconds, status: "running", found: -1, best); savedAt.Restart(); }
            }
            if (capped) break;
            if (yielded) continue;
            c.NodesExpanded++; c.Active = -1; c.NextAction = 0;
            if (c.NodesExpanded % 10 == 0) Console.Error.WriteLine($"expanded={c.NodesExpanded} seen={c.Nodes.Count} frontier={frontier.Count} actions={c.ActionsTried} seconds={clock.Elapsed.TotalSeconds:F1}");
        }
        Save(o, c, frontier, clock.Elapsed.TotalSeconds, status, found, best);
        Console.WriteLine($"{Path.GetFileName(o.Level)}: {status}; states={c.Nodes.Count}; actions={c.ActionsTried}; seconds={clock.Elapsed.TotalSeconds:F2}");
        return found >= 0 ? 0 : 3;
    }
    static bool Inside(double[][] positions, double[] center, double radius) => positions.All(p => Enumerable.Range(0, 3).All(a => Math.Abs(p[a] - center[a]) <= radius));
    static double Rank((bool Solved, int Finished, int Separated, double Overlap) p, int depth, Options o) => o.Heuristic == "overlap"
        ? -p.Finished * 100000.0 + p.Overlap - p.Separated * .1 + depth * .001
        : -p.Finished * 100000.0 - p.Separated * 100.0 + depth * .01;
    public static (bool Solved, int Finished, int Separated, double Overlap) Evaluate(OriginalEngine engine, EngineState state, double clearance)
    {
        var bounds = engine.BoundsLogical(); int separated = 0, n = state.Pieces.Length; double overlap = 0;
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++)
            if (Enumerable.Range(0, 3).Any(a => bounds[i][a + 3] + clearance <= bounds[j][a] || bounds[j][a + 3] + clearance <= bounds[i][a])) separated++;
            else overlap += Math.Max(0, Enumerable.Range(0, 3).Min(a => Math.Min(bounds[i][a + 3] - bounds[j][a], bounds[j][a + 3] - bounds[i][a])));
        int finished = state.Pieces.Count(p => p.Finished);
        return (finished >= n - 1 && separated == n * (n - 1) / 2, finished, separated, overlap);
    }
    static void Save(Options o, Checkpoint c, PriorityQueue<int, (double, int)> frontier, double elapsed, string status, int found, int best)
    {
        c.Frontier = frontier.UnorderedItems.Select(x => x.Element).Order().ToList();
        double oldSeconds = c.PreviousSeconds; c.PreviousSeconds += elapsed;
        string checkpoint = o.Output + ".checkpoint.json";
        Atomic(checkpoint, JsonSerializer.Serialize(c, Json));
        c.PreviousSeconds = oldSeconds;
        int terminal = found >= 0 ? found : best;
        var path = new List<ActionSpec>();
        for (int id = terminal; c.Nodes[id].Parent >= 0; id = c.Nodes[id].Parent) path.Add(c.Nodes[id].Action!);
        path.Reverse();
        var report = new
        {
            schema_version = c.Version, status, level_file = o.Level, input_sha256 = c.InputHash, engine_sha256 = c.EngineHash, engine_source_sha256 = c.EngineSourceHash, search_sha256 = c.SearchHash,
            independently_replayed = false, game_verified = false,
            initial_actions_file = o.InitialActionsFile == "" ? null : o.InitialActionsFile, initial_actions_sha256 = o.InitialActionsHash,
            meaning = status == "candidate_solution" ? "Candidate satisfies modeled finish zone and conservative pairwise separation; real game replay required." : "No conclusion about physical solvability; bounded or restricted action-domain search only.",
            search_scope = JsonSerializer.Deserialize<JsonElement>(c.Scope),
            goal = "At least N-1 pieces have entered original finish slabs, AND all piece AABBs are pairwise separated by positive clearance. Finished pieces stay in collision model and remain movable.",
            excludes_whole_assembly_move = true,
            completeness = "Only the explicitly configured discrete action domain and coordinate/depth bounds. Closure groups, event distances and optional parking of separated finished pieces/chunks are heuristics; no physical-unsolvability or global-shortest-path claim.",
            states_seen = c.Nodes.Count, nodes_expanded = c.NodesExpanded, actions_tried = c.ActionsTried,
            actions_accepted = c.ActionsAccepted, frontier_count = frontier.Count, active_node = c.Active,
            total_seconds = Math.Round(oldSeconds + elapsed, 3), checkpoint_file = checkpoint,
            engine_error = c.LastError, failed_action = c.FailedAction,
            candidate_actions = found >= 0 ? path : null, best_progress_actions = found < 0 ? path : null,
            terminal_state = c.Nodes[terminal].State,
        };
        Atomic(o.Output, JsonSerializer.Serialize(report, Json));
    }
    static void Atomic(string path, string text)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path + ".tmp", text); File.Move(path + ".tmp", path, true);
    }
    static IEnumerable<ActionSpec> Actions(OriginalEngine e, Options o, double[] center, Func<bool> stop)
    {
        foreach (int rule in e.ApplicableSpecialRules()) { if (stop()) yield break; yield return new ActionSpec { Kind = "special", RuleIndex = rule }; }
        if (stop()) yield break;
        var positions = e.PositionsLogical(); var bounds = e.BoundsLogical(); int n = positions.Length;
        var parked = new HashSet<int>();
        var actionState = e.Capture();
        bool victoryLatched = actionState.Pieces.Count(p => p.Finished) >= n - 1;
        bool Separated(int i, int j) => Enumerable.Range(0, 3).Any(a => bounds[i][a + 3] + o.Clearance <= bounds[j][a] || bounds[j][a + 3] + o.Clearance <= bounds[i][a]);
        if (o.ParkFinished)
        {
            var state = e.Capture();
            for (int i = 0; i < n; i++)
                if ((state.Pieces[i].Finished || victoryLatched) && Enumerable.Range(0, n).Where(j => j != i).All(j => Enumerable.Range(0, 3).Any(a => bounds[i][a + 3] + o.Clearance <= bounds[j][a] || bounds[j][a + 3] + o.Clearance <= bounds[i][a]))) parked.Add(i);
        }
        var coordinates = e.Data.PolyhedronList.Select((p, i) => Enumerable.Range(0, 3).Select(a =>
            p.AnglePolyhedron.Points.Select(v => positions[i][a] + (double)(a == 0 ? v.X : a == 1 ? v.Y : v.Z) / 10000).Distinct().Order().ToArray()).ToArray()).ToArray();
        var rotationGroups = new Dictionary<string, int[]>();
        foreach (int[] vector in Directions(o.Directions))
        {
            int nonzero = vector.Count(x => x != 0), singleAxis = Array.FindIndex(vector, x => x != 0);
            foreach (var ids in Groups(e, o, n, vector))
            {
                if (stop()) yield break;
                if (ids.Any(parked.Contains)) continue;
                rotationGroups.TryAdd(string.Join(',', ids), ids);
                // Avoid rigidly translating an already finished isolated chunk; retain its internal splits and rotations.
                if (o.ParkComponents && (victoryLatched || ids.All(i => actionState.Pieces[i].Finished)) && ids.All(i => Enumerable.Range(0, n).Where(j => Array.IndexOf(ids, j) < 0).All(j => Separated(i, j)))) continue;
                double max = nonzero == 1 ? e.MaxTranslation(ids, singleAxis, vector[singleAxis], o.MaxDistance) : e.MaxTranslationVector(ids, vector, o.MaxDistance);
                if (max <= 1e-8) continue;
                var distances = new SortedSet<double>();
                void Add(double d) { d = Math.Round(d, 8); if (d > 1e-8 && d <= max + 1e-8) distances.Add(Math.Min(d, max)); }
                if (o.Distances == "grid")
                {
                    // Explicit finite scalar distance alphabet along the configured vector.
                    for (int k = 1; k <= Math.Floor(max / o.Grid + 1e-8); k++) Add(k * o.Grid);
                }
                else
                {
                    Add(o.Grid); Add(2 * o.Grid); Add(1);
                    for (int axis = 0; axis < 3; axis++)
                    {
                        int sign = vector[axis]; if (sign == 0) continue;
                        foreach (int i in ids) for (int j = 0; j < n; j++)
                        {
                            if (Array.IndexOf(ids, j) >= 0) continue;
                            foreach (double x in coordinates[i][axis]) foreach (double y in coordinates[j][axis]) Add((y - x) / sign);
                            double clear = sign > 0 ? bounds[j][axis + 3] - bounds[i][axis] : bounds[i][axis + 3] - bounds[j][axis];
                            Add((clear + Math.Max(o.Grid, o.Clearance)) / Math.Abs(sign));
                        }
                        foreach (int i in ids) Add((center[axis] + Math.Sign(sign) * 300 - (bounds[i][axis] + bounds[i][axis + 3]) / 2) / sign);
                    }
                }
                Add(max);
                foreach (double d in o.Strategy == "bfs" ? distances : distances.Reverse())
                    yield return nonzero == 1
                        ? new ActionSpec { Kind = "move", PieceIds = ids, Axis = singleAxis, Distance = vector[singleAxis] * d }
                        : new ActionSpec { Kind = "move_vector", PieceIds = ids, Vector = vector, Distance = d };
            }
        }
        if (!o.Rotations) yield break;
        foreach (var ids in rotationGroups.Values)
        {
            var pivots = new Dictionary<string, double[]>();
            void Pivot(double[] p) { pivots.TryAdd(string.Join(',', p.Select(x => x.ToString("R", System.Globalization.CultureInfo.InvariantCulture))), p); }
            foreach (int i in ids) Pivot(positions[i]);
            Pivot(Enumerable.Range(0, 3).Select(a => ids.Average(i => positions[i][a])).ToArray());
            if (o.Pivots == "vertices")
                foreach (int i in ids)
                    foreach (var p in e.Data.PolyhedronList[i].AnglePolyhedron.Points)
                        Pivot(new[] { positions[i][0] + (double)p.X / 10000, positions[i][1] + (double)p.Y / 10000, positions[i][2] + (double)p.Z / 10000 });
            foreach (double[] pivot in pivots.Values)
                for (int axis = 0; axis < 3; axis++) foreach (int angle in new[] { 45, -45, 90, -90 })
                    { if (stop()) yield break; yield return new ActionSpec { Kind = "rotate", PieceIds = ids, Axis = axis, Angle = angle, Pivot = pivot }; }
        }
    }
    static IEnumerable<int[]> Directions(string mode)
    {
        for (int a = 0; a < 3; a++) foreach (int sign in new[] { 1, -1 }) { var v = new int[3]; v[a] = sign; yield return v; }
        if (mode != "grid26") yield break;
        for (int x = -1; x <= 1; x++) for (int y = -1; y <= 1; y++) for (int z = -1; z <= 1; z++)
            if (Math.Abs(x) + Math.Abs(y) + Math.Abs(z) > 1) yield return new[] { x, y, z };
    }
    static IEnumerable<int[]> Groups(OriginalEngine e, Options o, int n, int[] vector)
    {
        if (o.Groups == "all")
        {
            if (n >= 63) throw new ArgumentException("All-subsets mode supports at most 62 pieces");
            ulong end = (1UL << n) - 1;
            for (ulong mask = 1; mask < end; mask++) yield return Enumerable.Range(0, n).Where(i => (mask & (1UL << i)) != 0).ToArray();
            yield break;
        }
        var seen = new HashSet<string>();
        for (int i = 0; i < n; i++)
        {
            int[] single = { i }; seen.Add(i.ToString()); yield return single;
            if (o.Groups != "closure") continue;
            int axis = Array.FindIndex(vector, x => x != 0);
            int[] group = (vector.Count(x => x != 0) == 1 ? e.BlockingClosure(i, axis, vector[axis]) : e.BlockingClosureVector(i, vector)).Distinct().Order().ToArray();
            if (group.Length > 0 && group.Length < n && seen.Add(string.Join(',', group))) yield return group;
        }
    }
}
