// Exhaustive solver for the Luban (burr puzzle) levels of the CHAIN Unity game.
//
//   dotnet run -c Release -- <level.json> [--max-states N] [--out result.json]
//
// It searches the same moves the game API offers to an agent (one piece at a time, along
// +-x/+-y/+-z), using the game's own exact collision code (game_core/, decompiled from the
// Unity build), so every move it finds is one the game accepts. Breadth-first search over
// "move" commands gives the fewest commands needed to take the puzzle apart.
//
// Model of a move: piece i, direction d, distance t. The game accepts any t up to the
// largest distance before the piece would hit another one. The solver tries every multiple
// of the level's grid step up to that limit, plus the limit itself. If nothing blocks the
// piece at all, one command moves it out of the puzzle ("removed"); the level counts as
// solved when all pieces but one are removed (the game's own rule: N-1 pieces finished).
// Rotations are not searched (prototype).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using LuBanSuo.Core;
using LuBanSuo.Data;
using LuBanSuo.Logic;

static class Program
{
    sealed class Move
    {
        public int Piece, Dir;
        public BigFraction Dist;     // internal units (JSON units * 10000)
        public bool Removal;
    }

    sealed class Node
    {
        public BigFraction[] X, Y, Z;  // positions, internal units
        public bool[] Finished;   // piece has entered the finish zone (it stays in the scene)
        public int Parent;
        public Move Move;
        public int Depth;
    }

    static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("usage: LubanSolver <level.json> [--max-states N] [--out result.json]");
            return 2;
        }
        string path = args[0];
        long maxStates = 2_000_000;
        string outPath = null;
        bool debug = false;
        bool alignOnly = true;   // --grid: try every grid step instead
        bool greedyFinish = true; // --no-greedy: also branch on other moves when a piece can finish
        for (int a = 1; a < args.Length; a++)
        {
            if (args[a] == "--max-states") maxStates = long.Parse(args[++a]);
            else if (args[a] == "--out") outPath = args[++a];
            else if (args[a] == "--debug") debug = true;
            else if (args[a] == "--grid") alignOnly = false;
            else if (args[a] == "--no-greedy") greedyFinish = false;
        }

        var data = JsonSerializer.Deserialize<PuzzleData>(File.ReadAllText(path),
            new JsonSerializerOptions { IncludeFields = true });
        var (pieces, dirs) = Load(data);
        int n = pieces.Count;

        // Grid step: gcd of every vertex coordinate and start position (JSON units).
        BigInteger g = 0;
        foreach (var p in data.polyhedrons)
        {
            foreach (var v in p.vertexes)
                foreach (var c in new[] { v.x, v.y, v.z })
                    g = BigInteger.GreatestCommonDivisor(g, new BigInteger(Math.Round(Math.Abs(c))));
            foreach (var f in new[] { p.position1.x, p.position1.y, p.position1.z })
            {
                if (f.Denominator != 1) throw new Exception("non-integer start position");
                g = BigInteger.GreatestCommonDivisor(g, BigInteger.Abs(f.Numerator));
            }
        }
        if (g == 0) g = 1;
        var step = new BigFraction(g * Position.ScaleFactor);

        // A distance after which a piece that is not blocked is clear of every other piece.
        double lo = double.MaxValue, hi = double.MinValue;
        foreach (var p in data.polyhedrons)
            foreach (var v in p.vertexes)
                foreach (var c in new[] { v.x, v.y, v.z })
                {
                    double off = Math.Max(Math.Abs(p.position1.x.Numerator), Math.Max(Math.Abs(p.position1.y.Numerator), Math.Abs(p.position1.z.Numerator)));
                    lo = Math.Min(lo, c - off); hi = Math.Max(hi, c + off);
                }
        BigInteger escapeUnits = new BigInteger(Math.Ceiling((hi - lo) * 3 / (double)g)) * g;
        if (escapeUnits < 2000) escapeUnits = 2000;   // must reach past the finish zone (450)
        var escape = new BigFraction(escapeUnits * Position.ScaleFactor);

        var sw = Stopwatch.StartNew();
        var start = new Node
        {
            X = pieces.Select(p => p.Position.X).ToArray(),
            Y = pieces.Select(p => p.Position.Y).ToArray(),
            Z = pieces.Select(p => p.Position.Z).ToArray(),
            Finished = new bool[n], Parent = -1, Depth = 0,
        };
        // pieces overlapping at the start would make every result meaningless
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (Polyhedron.TestIntersection(pieces[i], pieces[j]))
                    Console.Error.WriteLine($"warning: pieces {i} and {j} overlap at the start");

        if (args.Contains("--group-check"))
        {
            // Which groups of pieces could move together at the start (the game API cannot do this).
            for (int mask = 1; mask < (1 << n) - 1; mask++)
            {
                int size = BitOperations.PopCount((uint)mask);
                if (size < 2 || size > n / 2) continue;
                var grp = Enumerable.Range(0, n).Where(i => (mask >> i & 1) == 1).Select(i => pieces[i]).ToList();
                var rest = Enumerable.Range(0, n).Where(i => (mask >> i & 1) == 0).Select(i => pieces[i]).ToList();
                for (int d = 0; d < dirs.Count; d++)
                {
                    var max = Move1SBeforeCross(grp, dirs[d], escape, rest);
                    if (max.CompareTo(BigFraction.Zero) > 0)
                        Console.WriteLine($"group [{string.Join(",", grp.Select(p => p.Index))}] dir {d} can move {max.DoubleValue() / Position.ScaleFactor}");
                }
            }
            return 0;
        }

        // Finish zone of the game scene ("胜利" walls, see README): around the mean start position
        // of the pieces, six slabs 150..450 units out along +-x/+-y/+-z, 600 units wide.
        var center = new[] { Mean(start.X), Mean(start.Y), Mean(start.Z) };
        Apply(pieces, start);
        for (int i = 0; i < n; i++) start.Finished[i] = InZone(pieces[i], center);

        var nodes = new List<Node> { start };
        var seen = new HashSet<string> { Key(start) };
        int goal = -1, head = 0;
        bool capped = false;
        while (head < nodes.Count)
        {
            var cur = nodes[head];
            if (cur.Finished.Count(r => r) >= n - 1) { goal = head; break; }
            Apply(pieces, cur);
            // If some piece can go straight into the finish zone, do only that: every piece needs
            // one such move anyway, and taking a piece out never blocks the rest of the assembly
            // (except when its parking spot is in another piece's way out), so the shortest
            // solution length is kept.
            if (greedyFinish)
            {
                Node fin = null;
                for (int i = 0; i < n && fin == null; i++)
                {
                    if (cur.Finished[i]) continue;
                    var others = Enumerable.Range(0, n).Where(j => j != i).Select(j => pieces[j]).ToList();
                    for (int d = 0; d < dirs.Count && fin == null; d++)
                    {
                        var land = Landing(pieces[i], center, dirs[d], step);
                        if (land == null) continue;
                        var max = Move1SBeforeCross(new List<Polyhedron> { pieces[i] }, dirs[d], escape, others);
                        if (land.Value.mid.CompareTo(max) > 0) continue;
                        var t = land.Value.mid;
                        var next = Clone(cur, head, new Move { Piece = i, Dir = d, Dist = t, Removal = true });
                        next.X[i] = next.X[i] + t * dirs[d].X;
                        next.Y[i] = next.Y[i] + t * dirs[d].Y;
                        next.Z[i] = next.Z[i] + t * dirs[d].Z;
                        next.Finished[i] = true;
                        fin = next;
                    }
                }
                if (fin != null)
                {
                    Push(nodes, seen, fin);
                    head++;
                    continue;
                }
            }
            for (int i = 0; i < n; i++)
            {
                if (cur.Finished[i]) continue;      // a finished piece stays where it is
                var others = Enumerable.Range(0, n).Where(j => j != i).Select(j => pieces[j]).ToList();
                var assembly = Enumerable.Range(0, n).Where(j => j != i && !cur.Finished[j]).Select(j => pieces[j]).ToList();
                for (int d = 0; d < dirs.Count; d++)
                {
                    var max = Move1SBeforeCross(new List<Polyhedron> { pieces[i] }, dirs[d], escape, others);
                    if (debug && head == 0) Console.Error.WriteLine($"start: piece {i} dir {d} max {max.DoubleValue() / Position.ScaleFactor}");
                    if (max.CompareTo(BigFraction.Zero) <= 0) continue;
                    var clear = assembly.Count > 0 ? Clearance(pieces[i], assembly, dirs[d]) : BigFraction.Zero;
                    // stops inside the assembly
                    var dists = new SortedSet<BigFraction>(Comparer<BigFraction>.Create((x, y) => x.CompareTo(y)));
                    var lim = max.CompareTo(clear) < 0 ? max : clear;
                    if (alignOnly)
                        foreach (var t in AlignStops(pieces[i], assembly, dirs[d], lim)) dists.Add(t);
                    else
                        for (var t = step; t.CompareTo(lim) <= 0; t = t + step) dists.Add(t);
                    if (max.CompareTo(clear) <= 0) dists.Add(max);
                    // one move into the finish zone along this direction, if the way is free
                    var land = Landing(pieces[i], center, dirs[d], step);
                    if (land != null)
                    {
                        if (land.Value.mid.CompareTo(max) <= 0) dists.Add(land.Value.mid);
                        else if (max.CompareTo(land.Value.lo) > 0) dists.Add(max);
                    }
                    foreach (var t in dists)
                    {
                        if (t.CompareTo(BigFraction.Zero) <= 0) continue;
                        var next = Clone(cur, head, new Move { Piece = i, Dir = d, Dist = t });
                        next.X[i] = next.X[i] + t * dirs[d].X;
                        next.Y[i] = next.Y[i] + t * dirs[d].Y;
                        next.Z[i] = next.Z[i] + t * dirs[d].Z;
                        var save = (pieces[i].Position.X, pieces[i].Position.Y, pieces[i].Position.Z);
                        pieces[i].Position.X = next.X[i]; pieces[i].Position.Y = next.Y[i]; pieces[i].Position.Z = next.Z[i];
                        next.Finished[i] = InZone(pieces[i], center);
                        next.Move.Removal = next.Finished[i];
                        (pieces[i].Position.X, pieces[i].Position.Y, pieces[i].Position.Z) = save;
                        Push(nodes, seen, next);
                    }
                }
            }
            head++;
            if (nodes.Count > maxStates) { capped = true; break; }
        }

        // Furthest progress: the state with the most pieces in the finish zone (BFS order, so the
        // first such state is also the shortest way to reach that many), and the deepest state.
        int startFinished = start.Finished.Count(r => r);
        int best = 0, bestFinished = startFinished, maxDepth = 0;
        for (int k = 0; k < nodes.Count; k++)
        {
            int f = nodes[k].Finished.Count(r => r);
            if (f > bestFinished) { bestFinished = f; best = k; }
            if (nodes[k].Depth > maxDepth) maxDepth = nodes[k].Depth;
        }
        int trace = goal >= 0 ? goal : best;

        var moves = new List<Dictionary<string, object>>();
        for (int k = trace; k > 0; k = nodes[k].Parent)
        {
            var m = nodes[k].Move;
            var dir = dirs[m.Dir];
            string axis = dir.X != 0 ? "x" : dir.Y != 0 ? "y" : "z";
            int sign = (int)(dir.X + dir.Y + dir.Z);
            moves.Add(new Dictionary<string, object>
            {
                ["piece_id"] = m.Piece, ["axis"] = axis,
                ["distance"] = sign * m.Dist.DoubleValue() / Position.ScaleFactor,
                ["finishes_piece"] = m.Removal,
            });
        }
        moves.Reverse();

        var result = new Dictionary<string, object>
        {
            ["level"] = Path.GetFileNameWithoutExtension(path),
            ["n_pieces"] = n,
            ["grid_step"] = (long)g,
            ["escape_distance"] = (long)escapeUnits,
            ["status"] = goal >= 0 ? "solved" : capped ? "state_limit" : "unsolvable_without_rotation",
            ["rule"] = "game API moves (one piece, +-x/y/z, no rotation); solved = N-1 pieces in the finish zone",
            ["min_moves"] = goal >= 0 ? nodes[goal].Depth : (int?)null,
            ["states_expanded"] = head,
            ["states_seen"] = nodes.Count,
            ["max_depth_reached"] = maxDepth,
            ["max_pieces_finished"] = (goal >= 0 ? nodes[goal] : nodes[best]).Finished.Count(r => r) - startFinished,
            ["moves_to_max_finished"] = (goal >= 0 ? nodes[goal] : nodes[best]).Depth,
            ["seconds"] = Math.Round(sw.Elapsed.TotalSeconds, 2),
            ["moves"] = moves,   // solution, or (if unsolved) the path to the furthest progress
        };
        string json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        if (outPath != null) File.WriteAllText(outPath, json);
        Console.WriteLine(json);
        return 0;
    }

    static readonly BigFraction ZoneIn = new BigFraction((BigInteger)150 * Position.ScaleFactor);
    static readonly BigFraction ZoneOut = new BigFraction((BigInteger)450 * Position.ScaleFactor);
    static readonly BigFraction ZoneHalf = new BigFraction((BigInteger)300 * Position.ScaleFactor);

    static BigFraction Mean(BigFraction[] v)
    {
        var sum = BigFraction.Zero;
        foreach (var x in v) sum = sum + x;
        return sum / new BigInteger(v.Length);
    }

    // Does the piece overlap one of the six finish-zone slabs?
    static bool InZone(Polyhedron p, BigFraction[] center)
    {
        var (lo, hi) = Bounds(p);
        for (int a = 0; a < 3; a++)
            for (int s = -1; s <= 1; s += 2)
            {
                bool ok = true;
                for (int b = 0; b < 3 && ok; b++)
                {
                    BigFraction zlo, zhi;
                    if (b == a) { zlo = s > 0 ? ZoneIn : -ZoneOut; zhi = s > 0 ? ZoneOut : -ZoneIn; }
                    else { zlo = -ZoneHalf; zhi = ZoneHalf; }
                    var l = lo[b] - center[b]; var h = hi[b] - center[b];
                    ok = l.CompareTo(zhi) < 0 && h.CompareTo(zlo) > 0;
                }
                if (ok) return true;
            }
        return false;
    }

    // Distances t along `dir` that put the piece into the slab ahead of it: (lo, hi) open,
    // and `mid`, a grid point near the middle of that range.
    static (BigFraction lo, BigFraction hi, BigFraction mid)? Landing(Polyhedron p, BigFraction[] center, Direction dir, BigFraction step)
    {
        int a = dir.X != 0 ? 0 : dir.Y != 0 ? 1 : 2;
        int sign = (int)(dir.X + dir.Y + dir.Z);
        var (blo, bhi) = Bounds(p);
        var l = blo[a] - center[a]; var h = bhi[a] - center[a];
        BigFraction tlo = sign > 0 ? ZoneIn - h : l + ZoneIn;
        BigFraction thi = sign > 0 ? ZoneOut - l : h + ZoneOut;
        if (thi.CompareTo(BigFraction.Zero) <= 0) return null;
        var midRaw = (tlo + thi) / new BigInteger(2);
        var k = (midRaw / step).Numerator / (midRaw / step).Denominator;   // floor to grid
        var mid = step * k;
        if (mid.CompareTo(tlo) <= 0) mid = midRaw;
        if (mid.CompareTo(BigFraction.Zero) <= 0) return null;
        return (tlo, thi, mid);
    }

    // Axis-aligned bounds of a piece at its current position (internal units).
    static (BigFraction[] lo, BigFraction[] hi) Bounds(Polyhedron p)
    {
        var pts = p.AnglePolyhedron.Points;
        var pos = new[] { p.Position.X, p.Position.Y, p.Position.Z };
        var lo = new BigFraction[3]; var hi = new BigFraction[3];
        for (int a = 0; a < 3; a++)
        {
            BigInteger mn = pts.Min(q => a == 0 ? q.X : a == 1 ? q.Y : q.Z);
            BigInteger mx = pts.Max(q => a == 0 ? q.X : a == 1 ? q.Y : q.Z);
            lo[a] = pos[a] + mn; hi[a] = pos[a] + mx;
        }
        return (lo, hi);
    }

    // Stopping points for a move along `dir`: the distances (0, max] at which a vertex
    // coordinate of the piece lines up with one of another piece along that axis. Between two
    // such points nothing about which moves are possible can change (for pieces whose faces
    // are axis-aligned), so the shortest solutions only need these stops.
    static List<BigFraction> AlignStops(Polyhedron piece, List<Polyhedron> others, Direction dir, BigFraction max)
    {
        int a = dir.X != 0 ? 0 : dir.Y != 0 ? 1 : 2;
        int sign = (int)(dir.X + dir.Y + dir.Z);
        Func<Polyhedron, IEnumerable<BigFraction>> coords = p =>
        {
            var off = a == 0 ? p.Position.X : a == 1 ? p.Position.Y : p.Position.Z;
            return p.AnglePolyhedron.Points.Select(q => a == 0 ? q.X : a == 1 ? q.Y : q.Z).Distinct().Select(c => off + c);
        };
        var mine = coords(piece).ToList();
        var stops = new SortedSet<BigFraction>(Comparer<BigFraction>.Create((x, y) => x.CompareTo(y)));
        foreach (var o in others)
            foreach (var co in coords(o))
                foreach (var cp in mine)
                {
                    var t = sign > 0 ? co - cp : cp - co;
                    if (t.CompareTo(BigFraction.Zero) > 0 && t.CompareTo(max) <= 0) stops.Add(t);
                }
        return stops.ToList();
    }

    // Distance the piece must travel along `dir` to be past all other pieces.
    static BigFraction Clearance(Polyhedron piece, List<Polyhedron> others, Direction dir)
    {
        int a = dir.X != 0 ? 0 : dir.Y != 0 ? 1 : 2;
        int sign = (int)(dir.X + dir.Y + dir.Z);
        var (plo, phi) = Bounds(piece);
        BigFraction best = null;
        foreach (var o in others)
        {
            var (olo, ohi) = Bounds(o);
            var c = sign > 0 ? ohi[a] - plo[a] : phi[a] - olo[a];
            if (best == null || c.CompareTo(best) > 0) best = c;
        }
        return best ?? BigFraction.Zero;
    }

    // Copy of the game's DataTools.Move1SBeforeCross: how far the pieces may move along
    // `direction` (at most `defaultDirectionFactor`) before touching another piece.
    static BigFraction Move1SBeforeCross(List<Polyhedron> polyhedron1S, Direction direction, BigFraction defaultDirectionFactor, List<Polyhedron> polyhedronList)
    {
        BigFraction best = Polyhedron.MoveMaxPositive;
        foreach (Polyhedron polyhedron in polyhedron1S)
        {
            List<BoundXYZMove> moves = polyhedron.GetMoveBoundXYZMoves(direction, defaultDirectionFactor);
            foreach (Polyhedron item in polyhedronList.Where(p2 => moves.Any(p2.TestBoundXYZMoveIntersection)).ToList())
            {
                if (item.Index != polyhedron.Index)
                {
                    BigFraction d = Polyhedron.MoveBeforeIntersection(polyhedron, item, direction);
                    if (d != null && best.CompareTo(d) > 0) best = d;
                }
            }
        }
        return best.CompareTo(Polyhedron.MoveMaxPositive) == 0 ? defaultDirectionFactor : best;
    }

    // Same construction as the game's DataTools.LoadPolyhedrons.
    static (List<Polyhedron>, List<Direction>) Load(PuzzleData data)
    {
        var dirs = data.directions.Select((d, i) => new Direction(i, d.x, d.y, d.z)).ToList();
        var pieces = new List<Polyhedron>();
        for (int i = 0; i < data.polyhedrons.Count; i++)
        {
            var pd = data.polyhedrons[i];
            var src = pd.vertexes.Select((v, idx) => new Point(idx, 0f - v.x, v.y, v.z, fromJson: true)).ToList();
            var faces = pd.faces.Select(f => f.vertexIndexes.ToList()).ToList();
            var convex = (pd.convexPolyhedrons ?? new List<ConvexPolyhedronData>())
                .Select(c => c.faces.Select(f => f.vertexIndexes.ToList()).ToList()).ToList();
            var hulls = new AnglePolyhedron[OrientationHelper.RotateAxisTypes.Length];
            for (int k = 0; k < hulls.Length; k++)
            {
                var r = OrientationHelper.RotateAxisTypes[k];
                var rot = new SwitchAxisTypeXYZ(r[0], r[1], r[2]);
                hulls[k] = new AnglePolyhedron(src.Select(p => OrientationHelper.RotatePoint(p, rot)).ToList(), faces, convex);
            }
            pieces.Add(new Polyhedron(i, hulls, Pos(pd.position1), Pos(pd.position2), Rot(pd.switchAxisTypeXYZ1), Rot(pd.switchAxisTypeXYZ2)));
        }
        return (pieces, dirs);
    }

    static Position Pos(PositionData p) => p == null ? new Position(0, 0, 0)
        : new Position(Frac(p.x), Frac(p.y), Frac(p.z), fromJson: true);
    static BigFraction Frac(FractionData f) => f == null ? BigFraction.Zero
        : new BigFraction(f.Numerator, f.Denominator == 0 ? 1 : f.Denominator);
    static SwitchAxisTypeXYZ Rot(SwitchAxisTypeXYZData r) => r == null
        ? new SwitchAxisTypeXYZ(SwitchAxisType.XPositive, SwitchAxisType.YPositive, SwitchAxisType.ZPositive)
        : new SwitchAxisTypeXYZ(r.x, r.y, r.z);

    static void Apply(List<Polyhedron> pieces, Node s)
    {
        for (int i = 0; i < pieces.Count; i++)
        {
            pieces[i].Position.X = s.X[i]; pieces[i].Position.Y = s.Y[i]; pieces[i].Position.Z = s.Z[i];
        }
    }

    static Node Clone(Node s, int parent, Move m) => new Node
    {
        X = (BigFraction[])s.X.Clone(), Y = (BigFraction[])s.Y.Clone(), Z = (BigFraction[])s.Z.Clone(),
        Finished = (bool[])s.Finished.Clone(), Parent = parent, Move = m, Depth = s.Depth + 1,
    };

    static string Key(Node s) => string.Join(";", Enumerable.Range(0, s.X.Length)
        .Select(i => $"{(s.Finished[i] ? "F" : "")}{s.X[i]},{s.Y[i]},{s.Z[i]}"));

    static void Push(List<Node> nodes, HashSet<string> seen, Node s)
    {
        if (seen.Add(Key(s))) nodes.Add(s);
    }
}
