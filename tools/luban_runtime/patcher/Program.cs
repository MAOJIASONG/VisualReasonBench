using System.Security.Cryptography;
using System.Text.Json;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal static class Program
{
    // This is the committed Linux player at 65538f43. Refuse unknown/repatched inputs.
    const string SupportedSha256 = "350e7df0e6a8e15a52ef0eae2a234b177afbb8196354cf6f40b61fb608a2a33a";
    const string TimeoutResponse = "{\"status\":\"error\",\"code\":\"response_timeout\",\"execution_may_have_occurred\":true,\"result\":\"Unity did not complete this request before the deadline. Inspect state before retrying any action.\"}";

    static string Hash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();

    static int Main(string[] args)
    {
        if (args.Length is < 2 or > 4)
            throw new ArgumentException("Usage: RuntimePatcher input.dll NEW-output.dll [response-timeout-ms] [LubanOriginalRuntime.dll]");
        var input = Path.GetFullPath(args[0]);
        var output = Path.GetFullPath(args[1]);
        int timeout = args.Length >= 3 ? int.Parse(args[2]) : 300000;
        if (timeout < 1 || timeout > 600000) throw new ArgumentOutOfRangeException(nameof(timeout));
        if (input == output || File.Exists(output))
            throw new ArgumentException("Output must be a new file, distinct from input; in-place edits are forbidden.");
        if (Hash(input) != SupportedSha256)
            throw new InvalidOperationException("Unsupported input assembly hash. No output was written.");

        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(input)!);
        using var assembly = AssemblyDefinition.ReadAssembly(input,
            new ReaderParameters { AssemblyResolver = resolver, InMemory = true });
        var module = assembly.MainModule;
        var bridge = module.GetType("LuBanSuo.Logic.SocketBridge");
        var handle = bridge.Methods.Single(m => m.Name == "HandleClient");
        var waits = handle.Body.Instructions.Where(i => i.OpCode == OpCodes.Ldc_I4 && Equals(i.Operand, 5000)).ToList();
        if (waits.Count != 1) throw new InvalidOperationException("Expected exactly one original 5000 ms wait.");
        waits[0].Operand = timeout;

        // Initialize the callback result with a structured uncertain-outcome error. The
        // callback overwrites it on success; a timeout never sends the original blank line.
        var emptyDefaults = handle.Body.Instructions.Where(i => i.OpCode == OpCodes.Ldstr && Equals(i.Operand, "")
            && i.Next?.OpCode == OpCodes.Stfld
            && i.Next.Operand is FieldReference f && f.Name == "responseJson").ToList();
        if (emptyDefaults.Count != 1) throw new InvalidOperationException("Expected original empty callback response.");
        emptyDefaults[0].Operand = TimeoutResponse;

        // Keep a bounded sleep, rather than spinning when no client is pending.
        var listen = bridge.Methods.Single(m => m.Name == "ListenLoop");
        var polls = listen.Body.Instructions.Where(i => i.OpCode == OpCodes.Ldc_I4_S
            && Convert.ToInt32(i.Operand) == 100 && i.Next?.Operand is MethodReference m
            && m.Name == "Sleep").ToList();
        if (polls.Count != 1) throw new InvalidOperationException("Expected original 100 ms polling interval.");
        polls[0].Operand = (sbyte)5;

        ForwardValidatedAngle(module);
        if (args.Length == 4) AddOriginalMode(module, Path.GetFullPath(args[3]), resolver);
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        assembly.Write(output);
        using (var check = AssemblyDefinition.ReadAssembly(output))
        {
            var method = check.MainModule.GetType("LuBanSuo.API.EnvControlAPI").Methods.Single(m => m.Name == "RotatePiece");
            if (!method.Body.Instructions.Any(i => i.OpCode == OpCodes.Callvirt && i.Operand is MethodReference m && m.Name == "AgentRotate"))
                throw new InvalidOperationException("Output validation failed.");
        }
        Console.WriteLine(JsonSerializer.Serialize(new {
            input_sha256 = Hash(input), output_sha256 = Hash(output), response_timeout_ms = timeout,
            accept_poll_ms = 5, original_mode_extension = args.Length == 4,
            legacy_action_rules_changed = false,
            fixes = new[] { "structured_timeout", "long_capture_response", "accept_poll_latency", "full_quarter_turn_angle" }
        }, new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }

    static void AddOriginalMode(ModuleDefinition module, string extensionPath, DefaultAssemblyResolver resolver)
    {
        resolver.AddSearchDirectory(Path.GetDirectoryName(extensionPath)!);
        using var extension = AssemblyDefinition.ReadAssembly(extensionPath,
            new ReaderParameters { AssemblyResolver = resolver });
        var bridge = module.GetType("LuBanSuo.Logic.SocketBridge");
        var parameters = bridge.NestedTypes.Single(t => t.Name == "CommandParams");
        foreach (var field in new[] {
            new FieldDefinition("kind", FieldAttributes.Public, module.TypeSystem.String),
            new FieldDefinition("piece_ids", FieldAttributes.Public, new ArrayType(module.TypeSystem.Int32)),
            new FieldDefinition("pivot", FieldAttributes.Public, new ArrayType(module.TypeSystem.Double)),
            new FieldDefinition("vector", FieldAttributes.Public, new ArrayType(module.TypeSystem.Int32)),
            new FieldDefinition("distance_exact", FieldAttributes.Public, module.TypeSystem.Double),
            new FieldDefinition("rule_index", FieldAttributes.Public, module.TypeSystem.Int32)
        })
        {
            if (parameters.Fields.Any(f => f.Name == field.Name)) throw new InvalidOperationException("Extension field already present");
            parameters.Fields.Add(field);
        }
        var handle = extension.MainModule.GetType("Luban.Runtime.OriginalRuntimeExtension").Methods.Single(m => m.Name == "Handle");
        var method = bridge.Methods.Single(m => m.Name == "ExecuteCommand");
        var il = method.Body.GetILProcessor();
        var first = method.Body.Instructions[0];
        il.InsertBefore(first, il.Create(OpCodes.Ldarg_1));
        il.InsertBefore(first, il.Create(OpCodes.Call, module.ImportReference(handle)));
        il.InsertBefore(first, il.Create(OpCodes.Brfalse, first));
        il.InsertBefore(first, il.Create(OpCodes.Ret));
    }

    static void ForwardValidatedAngle(ModuleDefinition module)
    {
        var api = module.GetType("LuBanSuo.API.EnvControlAPI");
        var rotate = api.Methods.Single(m => m.Name == "RotatePiece");
        if (!rotate.Body.Instructions.Any(i => i.OpCode == OpCodes.Ldc_R4 && Equals(i.Operand, 90f)))
            throw new InvalidOperationException("Expected the original +/-90 angle clamp.");
        var abs = (MethodReference)rotate.Body.Instructions.First(i => i.Operand is MethodReference m && m.Name == "Abs").Operand;
        var getLogic = api.Methods.Single(m => m.Name == "GetMainLogic");
        var logic = module.GetType("LuBanSuo.Logic.MainLogic");
        var agentRotate = logic.Methods.Single(m => m.Name == "AgentRotate");
        // Construct references with the target's mscorlib scope, never the patcher's .NET8 scope.
        var isnan = new MethodReference("IsNaN", module.TypeSystem.Boolean, module.TypeSystem.Single) { HasThis = false };
        isnan.Parameters.Add(new ParameterDefinition(module.TypeSystem.Single));
        var isinf = new MethodReference("IsInfinity", module.TypeSystem.Boolean, module.TypeSystem.Single) { HasThis = false };
        isinf.Parameters.Add(new ParameterDefinition(module.TypeSystem.Single));

        rotate.Body = new MethodBody(rotate) { InitLocals = true };
        rotate.Body.Variables.Add(new VariableDefinition(logic));
        var il = rotate.Body.GetILProcessor();
        var failure = il.Create(OpCodes.Ldc_I4_0);
        var success = il.Create(OpCodes.Ldc_I4_1);
        il.Emit(OpCodes.Ldarg, rotate.Parameters[3]); il.Emit(OpCodes.Call, isnan); il.Emit(OpCodes.Brtrue, failure);
        il.Emit(OpCodes.Ldarg, rotate.Parameters[3]); il.Emit(OpCodes.Call, isinf); il.Emit(OpCodes.Brtrue, failure);
        // Reject invalid angles before MainLogic can record a partial/no-op action.
        il.Emit(OpCodes.Ldarg, rotate.Parameters[3]); il.Emit(OpCodes.Ldc_R4, 90f); il.Emit(OpCodes.Rem);
        il.Emit(OpCodes.Call, abs); il.Emit(OpCodes.Ldc_R4, 0.0001f); il.Emit(OpCodes.Bgt, failure);
        il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Call, getLogic); il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Ldloc_0); il.Emit(OpCodes.Brfalse, failure);
        il.Emit(OpCodes.Ldarg, rotate.Parameters[3]); il.Emit(OpCodes.Call, abs); il.Emit(OpCodes.Ldc_R4, 0.0001f); il.Emit(OpCodes.Blt, success);
        il.Emit(OpCodes.Ldloc_0); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Ldarg, rotate.Parameters[3]);
        il.Emit(OpCodes.Callvirt, agentRotate); il.Emit(OpCodes.Ret);
        il.Append(failure); il.Emit(OpCodes.Ret);
        il.Append(success); il.Emit(OpCodes.Ret);
    }
}
