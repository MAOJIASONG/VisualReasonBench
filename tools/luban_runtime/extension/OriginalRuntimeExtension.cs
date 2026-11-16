using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using Luban.Original;
using LuBanSuo.API;
using LuBanSuo.Logic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using NativeFraction = LuBanSuo.Core.BigFraction;
using NativeOrientation = LuBanSuo.Core.SwitchAxisTypeXYZ;
using NativeAxis = LuBanSuo.Core.SwitchAxisType;
using Vector3 = UnityEngine.Vector3;

namespace Luban.Runtime
{
    /// <summary>
    /// Explicit restored-rule mode. No state-restore endpoint and no writes to Unity's
    /// finished-piece set or victory flag. The engine accepts actions; real scene triggers
    /// independently report completion after actual piece transforms are updated.
    /// </summary>
    public static class OriginalRuntimeExtension
    {
        sealed class Session { public OriginalEngine Engine; public int Level; public MainLogic Logic; public bool Busy; }
        static readonly Dictionary<int, Session> Sessions = new Dictionary<int, Session>();

        public static bool Handle(SocketBridge.CommandRequest request)
        {
            string command = request.command?.ToLowerInvariant() ?? "";
            bool originalCommand = command.StartsWith("original_", StringComparison.Ordinal);
            if (Sessions.TryGetValue(request.env_id, out Session inFlight) && inFlight.Busy)
            {
                Reply(request, "error", new { code = "environment_busy", message = "Wait for the previous response before sending another request to this environment." });
                return true;
            }
            if (!originalCommand)
            {
                if (command == "start_level" || command == "reset") Sessions.Remove(request.env_id);
                else if (Sessions.TryGetValue(request.env_id, out Session reading) && (command == "get_state" || command == "state"))
                {
                    var data = JObject.Parse(EnvControlAPI.Instance.GetState(request.env_id));
                    data["mode"] = "restored_original_rules";
                    data["orientation_is_base_only"] = true;
                    data["original_state_command"] = "original_state";
                    Reply(request, "success", data);
                    return true;
                }
                else if (Sessions.ContainsKey(request.env_id) && (command == "move" || command == "rotate"))
                {
                    Reply(request, "error", new { code = "mixed_action_modes", message = "Use original_action/original_state while original-rule mode is active, or start_level to leave it." });
                    return true;
                }
                return false;
            }
            try
            {
                if (request.env_id < 1 || request.env_id > 8) throw new ArgumentException("Original-rule mode requires env_id 1..8.");
                if (command == "original_start")
                {
                    int level = request.@params?.level_index ?? -1;
                    if (level < 0 || level > 31) throw new ArgumentException("level_index must be 0..31");
                    // Rule files belong to this prepared runtime, never a caller-supplied path.
                    string path = Path.Combine(Application.dataPath, "StreamingAssets", "OriginalRules", "Puzzle_" + level + ".json");
                    var engine = OriginalEngine.LoadFromJson(File.ReadAllText(path));
                    EnvControlAPI.Instance.StartLevel(request.env_id, level);
                    Sessions.Remove(request.env_id);
                    var logic = EnvGroupManager.Instance.GetEnv(request.env_id);
                    if (logic == null) throw new InvalidOperationException("Native environment missing");
                    var session = new Session { Engine = engine, Level = level, Logic = logic };
                    CheckAgreement(session);
                    Sessions[request.env_id] = session;
                    session.Busy = true;
                    logic.StartCoroutine(ReplyAfterPhysics(request, session, null));
                    return true;
                }
                if (!Sessions.TryGetValue(request.env_id, out Session current))
                    throw new InvalidOperationException("Call original_start first");
                if (command == "original_state") { ReplyState(request, current, null); return true; }
                if (command != "original_action") throw new ArgumentException("Unknown original command");
                if (request.@params == null) throw new ArgumentException("Missing action params");
                CheckAgreement(current);
                var action = ParseAction(request.@params);
                var accepted = current.Engine.Apply(action);
                if (!accepted.Success) { ReplyState(request, current, accepted); return true; }
                ApplyAcceptedPose(current, accepted.ChangedPieceIds);
                current.Busy = true;
                current.Logic.StartCoroutine(ReplyAfterPhysics(request, current, accepted));
                return true;
            }
            catch (Exception error)
            {
                Reply(request, "error", new { code = "original_action_error", message = error.Message });
                return true;
            }
        }

        static T Field<T>(object value, string name, T fallback)
        {
            var field = value.GetType().GetField(name, BindingFlags.Public | BindingFlags.Instance);
            return field != null && field.GetValue(value) is T result ? result : fallback;
        }

        static ActionSpec ParseAction(SocketBridge.CommandParams p)
        {
            string kind = Field(p, "kind", "");
            if (kind != "move" && kind != "move_vector" && kind != "rotate" && kind != "special") throw new ArgumentException("kind must be move, move_vector, rotate or special");
            int axis = p.axis == "x" ? 0 : p.axis == "y" ? 1 : p.axis == "z" ? 2 : -1;
            if ((kind == "move" || kind == "rotate") && axis < 0) throw new ArgumentException("axis must be x/y/z");
            if (float.IsNaN(p.angle) || float.IsInfinity(p.angle) || p.angle != (int)p.angle)
                throw new ArgumentException("angle must be an integer");
            var pivot = Field<double[]>(p, "pivot", null);
            double exactDistance = Field(p, "distance_exact", 0.0);
            return new ActionSpec {
                Kind = kind, PieceIds = Field(p, "piece_ids", new int[0]), Axis = Math.Max(axis, 0),
                Distance = exactDistance == 0.0 ? p.distance : exactDistance, Angle = (int)p.angle,
                Vector = Field<int[]>(p, "vector", null), Pivot = pivot, RuleIndex = Field(p, "rule_index", -1)
            };
        }

        static void CheckAgreement(Session session)
        {
            if (session.Level != session.Logic.puzzleIndex) throw new InvalidOperationException("Native level changed; restart original session");
            var original = session.Engine.Data.PolyhedronList;
            if (original.Count != session.Logic.pieceControllers.Count) throw new InvalidOperationException("Original/native piece count mismatch");
            foreach (var piece in session.Logic.pieceControllers)
            {
                var native = piece.GetPolyhedron();
                var restored = original[native.Index];
                // The historical engine's DoubleValue rounds negative integers incorrectly.
                // Compare rational coordinates exactly, never through that conversion helper.
                if (native.Position.X.Numerator * restored.Position.X.Denominator != restored.Position.X.Numerator * native.Position.X.Denominator
                    || native.Position.Y.Numerator * restored.Position.Y.Denominator != restored.Position.Y.Numerator * native.Position.Y.Denominator
                    || native.Position.Z.Numerator * restored.Position.Z.Denominator != restored.Position.Z.Numerator * native.Position.Z.Denominator)
                    throw new InvalidOperationException("Original/native coordinates differ for piece " + native.Index
                        + ": native=(" + native.Position.X + "," + native.Position.Y + "," + native.Position.Z
                        + "), restored=(" + restored.Position.X + "," + restored.Position.Y + "," + restored.Position.Z + "); restart original session");
                if (UnityEngine.Quaternion.Angle(piece.transform.localRotation, session.Engine.VisualQuaternion(native.Index)) > 0.05f)
                    throw new InvalidOperationException("Original/native orientation differs for piece " + native.Index
                        + ": native=" + piece.transform.localRotation + ", restored=" + session.Engine.VisualQuaternion(native.Index) + "; restart original session");
                var expectedPosition = new Vector3(
                    (float)(native.Position.X.DoubleValue() - session.Logic.centerOffset.x),
                    (float)(native.Position.Y.DoubleValue() - session.Logic.centerOffset.y),
                    (float)(native.Position.Z.DoubleValue() - session.Logic.centerOffset.z)) * session.Logic.scale;
                if ((piece.transform.localPosition - expectedPosition).sqrMagnitude > 0.000001f)
                    throw new InvalidOperationException("Native scene position drifted for piece " + native.Index + "; restart original session");
            }
        }

        static NativeFraction Fraction(string value)
        {
            string[] parts = value.Split('/');
            return new NativeFraction(BigInteger.Parse(parts[0]), BigInteger.Parse(parts[1]));
        }

        static void ApplyAcceptedPose(Session session, int[] changed)
        {
            var state = session.Engine.Capture();
            foreach (int id in changed)
            {
                var controller = session.Logic.GetPieceController(id);
                var native = controller.GetPolyhedron();
                var piece = state.Pieces.Single(s => s.Id == id);
                native.LastPosition = new LuBanSuo.Core.Position(native.Position);
                native.Position.X = Fraction(piece.X); native.Position.Y = Fraction(piece.Y); native.Position.Z = Fraction(piece.Z);
                native.SetOrientation(new NativeOrientation((NativeAxis)piece.Orientation[0], (NativeAxis)piece.Orientation[1], (NativeAxis)piece.Orientation[2]));
                var rotation = session.Engine.VisualQuaternion(id);
                native.VisualEuler = rotation.eulerAngles;
                controller.transform.localPosition = new Vector3(
                    (float)(native.Position.X.DoubleValue() - session.Logic.centerOffset.x),
                    (float)(native.Position.Y.DoubleValue() - session.Logic.centerOffset.y),
                    (float)(native.Position.Z.DoubleValue() - session.Logic.centerOffset.z)) * session.Logic.scale;
                controller.transform.localRotation = rotation;
            }
            Physics.SyncTransforms();
        }

        static IEnumerator ReplyAfterPhysics(SocketBridge.CommandRequest request, Session session, ActionResult accepted)
        {
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            session.Busy = false;
            ReplyState(request, session, accepted);
        }

        static void ReplyState(SocketBridge.CommandRequest request, Session session, ActionResult accepted)
        {
            var data = new {
                puzzle_index = session.Level, mode = "restored_original_rules",
                engine_state = session.Engine.Capture(), engine_zone_estimate = session.Engine.Solved,
                native_unity_is_solved = session.Logic.isLevelCleared,
                native_positions = session.Logic.pieceControllers.Select(pc => {
                    var p = pc.GetPolyhedron();
                    var local = pc.transform.localPosition;
                    // Read scene bounds from actual rendered meshes. Child colliders include
                    // the native trigger volumes for these same pieces, never engine estimates.
                    var renderers = pc.GetComponentsInChildren<Renderer>().Where(r => r.enabled).ToArray();
                    if (renderers.Length == 0) throw new InvalidOperationException("Piece has no active renderer: " + p.Index);
                    var bounds = renderers[0].bounds;
                    foreach (var renderer in renderers.Skip(1)) bounds.Encapsulate(renderer.bounds);
                    foreach (var collider in pc.GetComponentsInChildren<Collider>().Where(c => c.enabled)) bounds.Encapsulate(collider.bounds);
                    return new { id = p.Index, x = p.Position.X.DoubleValue() / 10000,
                        y = p.Position.Y.DoubleValue() / 10000, z = p.Position.Z.DoubleValue() / 10000,
                        visual_logical_position = new {
                            x = (local.x / (double)session.Logic.scale + session.Logic.centerOffset.x) / 10000,
                            y = (local.y / (double)session.Logic.scale + session.Logic.centerOffset.y) / 10000,
                            z = (local.z / (double)session.Logic.scale + session.Logic.centerOffset.z) / 10000 },
                        scene_world_bounds = new {
                            min = new { x = bounds.min.x, y = bounds.min.y, z = bounds.min.z },
                            max = new { x = bounds.max.x, y = bounds.max.y, z = bounds.max.z } },
                        visual_rotation = new { x = pc.transform.localRotation.x, y = pc.transform.localRotation.y,
                            z = pc.transform.localRotation.z, w = pc.transform.localRotation.w } };
                }).ToArray(),
                native_finished_piece_ids = session.Logic.pieceControllers.Where(p => session.Logic.IsPieceFinished(p.GetPolyhedron().Index)).Select(p => p.GetPolyhedron().Index).ToArray(),
                applicable_special_rules = session.Engine.ApplicableSpecialRules(), action_result = accepted
            };
            Reply(request, accepted != null && !accepted.Success ? "fail" : "success", data);
        }

        static void Reply(SocketBridge.CommandRequest request, string status, object data)
            => request.callback?.Invoke(JsonConvert.SerializeObject(new { status, env_id = request.env_id, data }));
    }
}
