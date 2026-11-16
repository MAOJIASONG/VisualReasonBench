using System.Security.Cryptography;
using System.Text.Json;
using Luban.Original;

// Revalidate a supplied path without presenting it as newly discovered by this search.
static class Certify
{
    public static int Run(string[] args)
    {
        if (args.Length != 3) throw new ArgumentException("Usage: LubanSearch certify level.json actions.json output.json");
        string input = File.ReadAllText(args[0]), pathInput = File.ReadAllText(args[1]);
        var actions = JsonSerializer.Deserialize<ActionSpec[]>(pathInput, Program.Json) ?? throw new ArgumentException("Expected an array of ActionSpec values");
        var engine = OriginalEngine.LoadFromJson(input); var receipts = new List<object>(); bool accepted = true;
        foreach (var action in actions)
        {
            // Supplied paths obey the same no-whole-assembly convention as generated paths.
            if (action.Kind != "special" && action.PieceIds.Distinct().Count() >= engine.Capture().Pieces.Length) throw new ArgumentException("Whole-assembly action excluded");
            var result = engine.Apply(action);
            receipts.Add(new { action, accepted = result.Success, message = result.Message });
            if (!result.Success) { accepted = false; break; }
        }
        var final = engine.Capture(); var evaluated = Program.Evaluate(engine, final, .001);
        bool passed = accepted && evaluated.Solved;
        string Hash(string value) => Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(value)));
        var report = new
        {
            schema_version = "luban-original-search-v3", status = passed ? "candidate_solution" : accepted ? "provided_path_goal_not_satisfied" : "provided_path_rejected",
            candidate_origin = "provided_path_revalidated", supplied_path = Path.GetFullPath(args[1]), supplied_path_sha256 = Hash(pathInput),
            level_file = Path.GetFullPath(args[0]), input_sha256 = Hash(input),
            engine_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(OriginalEngine).Assembly.Location))),
            engine_source_sha256 = EngineSourceFingerprint.Value,
            search_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(Program).Assembly.Location))),
            independently_replayed = false, game_verified = false,
            meaning = "Supplied path replayed in the original-rules model; live-game verification still required.",
            search_scope = new { Clearance = .001, Method = "provided path replay, no search performed" },
            goal = "Original N-1 finish-zone state plus every pair of world AABBs separated by positive clearance.",
            excludes_whole_assembly_move = true, all_actions_accepted = accepted, model_goal_satisfied = evaluated.Solved,
            candidate_actions = passed ? actions : null, supplied_actions = actions, terminal_state = final, receipts,
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[2]))!);
        File.WriteAllText(args[2], JsonSerializer.Serialize(report, Program.Json));
        Console.WriteLine($"provided_path accepted={accepted} model_goal={evaluated.Solved} candidate={passed}");
        return passed ? 0 : 4;
    }
}
