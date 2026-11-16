using System.Security.Cryptography;
using System.Text.Json;
using Luban.Original;

static class Replay
{
    public static int Run(string[] args)
    {
        if (args.Length != 2) throw new ArgumentException("Usage: LubanSearch replay level.json candidate.json");
        var data = JsonDocument.Parse(File.ReadAllText(args[1])).RootElement;
        string input = File.ReadAllText(args[0]);
        string inputHash = Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(input)));
        if (data.GetProperty("input_sha256").GetString() != inputHash) throw new InvalidOperationException("Candidate was generated for different input data");
        string engineHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(OriginalEngine).Assembly.Location)));
        if (data.GetProperty("engine_sha256").GetString() != engineHash) throw new InvalidOperationException("Candidate engine version differs; regenerate before replay");
        if (data.GetProperty("engine_source_sha256").GetString() != EngineSourceFingerprint.Value) throw new InvalidOperationException("Candidate source version differs; regenerate before replay");
        var actions = data.GetProperty("candidate_actions").Deserialize<ActionSpec[]>(Program.Json) ?? throw new InvalidOperationException("No candidate actions");
        var engine = OriginalEngine.LoadFromJson(input); var receipts = new List<object>(); bool accepted = true;
        foreach (var action in actions)
        {
            var result = engine.Apply(action);
            receipts.Add(new { action, accepted = result.Success, message = result.Message, state = engine.Capture() });
            if (!result.Success) { accepted = false; break; }
        }
        var final = engine.Capture(); string actualKey = engine.StateKey();
        var expected = data.GetProperty("terminal_state").Deserialize<EngineState>(Program.Json)!;
        engine.Restore(expected); string expectedKey = engine.StateKey(); engine.Restore(final);
        double clearance = data.GetProperty("search_scope").GetProperty("Clearance").GetDouble();
        var evaluated = Program.Evaluate(engine, final, clearance);
        bool matches = actualKey == expectedKey;
        bool passed = accepted && matches && evaluated.Solved;
        string output = args[1] + ".model-replay.json";
        File.WriteAllText(output, JsonSerializer.Serialize(new { model_replay_passed = passed, game_verified = false, note = "Fresh instance of same original-rules model; this is not independent Unity verification.", engine_sha256 = engineHash, engine_source_sha256 = EngineSourceFingerprint.Value, input_sha256 = inputHash, all_actions_accepted = accepted, exact_terminal_state_matches = matches, model_goal_satisfied = evaluated.Solved, receipts }, Program.Json));
        Console.WriteLine($"model_replay_passed={passed} accepted={accepted} exact_state={matches} output={output}");
        return passed ? 0 : 4;
    }
}
