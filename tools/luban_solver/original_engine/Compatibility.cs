namespace BurrPuzzle.DataCore { internal static class SettingLogic { public static bool QuickMoveNegative = false; } internal static class CollectionExtensions { public static void AddRange<T>(this System.Collections.Generic.ICollection<T> dst,System.Collections.Generic.IEnumerable<T> values) { foreach(var value in values) dst.Add(value); } } }

namespace Luban.Original { public static class EngineDiagnostics { public static bool VerboseGeometry = false; } }
