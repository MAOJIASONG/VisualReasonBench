// Minimal stand-ins for the few UnityEngine types used by the game's geometry code
// (game_core/), so it can be compiled as a plain .NET program. Only the orientation
// table in OrientationHelper and a few unused helpers touch them.
using System;

namespace UnityEngine
{
    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static float Magnitude(Vector3 a) => (float)Math.Sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
        public static Vector3 zero => new Vector3(0, 0, 0);
        public static Vector3 one => new Vector3(1, 1, 1);
        public static Vector3 right => new Vector3(1, 0, 0);
        public static Vector3 left => new Vector3(-1, 0, 0);
        public static Vector3 up => new Vector3(0, 1, 0);
        public static Vector3 down => new Vector3(0, -1, 0);
        public static Vector3 forward => new Vector3(0, 0, 1);
        public static Vector3 back => new Vector3(0, 0, -1);
        public static float Distance(Vector3 a, Vector3 b)
        {
            float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
            return (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }
        public static Vector3 operator *(Vector3 a, float s) => new Vector3(a.x * s, a.y * s, a.z * s);
        public static Vector3 operator /(Vector3 a, float s) => new Vector3(a.x / s, a.y / s, a.z / s);
        public static Vector3 operator +(Vector3 a, Vector3 b) => new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
        public static Vector3 operator -(Vector3 a, Vector3 b) => new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
        public override string ToString() => $"({x}, {y}, {z})";
    }

    public struct Quaternion
    {
        public float x, y, z, w;
        public Quaternion(float x, float y, float z, float w) { this.x = x; this.y = y; this.z = z; this.w = w; }
        public static Quaternion identity => new Quaternion(0, 0, 0, 1);

        public static Quaternion AngleAxis(float deg, Vector3 a)
        {
            double h = deg * Math.PI / 360.0, s = Math.Sin(h);
            return new Quaternion((float)(a.x * s), (float)(a.y * s), (float)(a.z * s), (float)Math.Cos(h));
        }

        // Unity's convention: rotate about z, then x, then y.
        public static Quaternion Euler(Vector3 e) =>
            AngleAxis(e.y, Vector3.up) * AngleAxis(e.x, Vector3.right) * AngleAxis(e.z, Vector3.forward);
        public static Quaternion Euler(float x, float y, float z) => Euler(new Vector3(x, y, z));

        public static float Angle(Quaternion a, Quaternion b) { double d=Math.Abs(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w); return (float)(2*Math.Acos(Math.Min(d,1))*180/Math.PI); }
        public static Quaternion Inverse(Quaternion q) => new Quaternion(-q.x, -q.y, -q.z, q.w);

        public static Quaternion operator *(Quaternion a, Quaternion b) => new Quaternion(
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
            a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);

        public static Vector3 operator *(Quaternion q, Vector3 v)
        {
            // v' = q * v * q^-1
            var p = q * new Quaternion(v.x, v.y, v.z, 0) * Inverse(q);
            return new Vector3(p.x, p.y, p.z);
        }

        public static bool operator ==(Quaternion a, Quaternion b) => a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
        public static bool operator !=(Quaternion a, Quaternion b) => !(a == b);
        public override bool Equals(object o) => o is Quaternion q && q == this;
        public override int GetHashCode() => HashCode.Combine(x, y, z, w);
    }

    public class Transform { public Vector3 localEulerAngles; }

    public static class Mathf
    {
        public static float Abs(float v) => Math.Abs(v);
        public static int RoundToInt(float v) => (int)Math.Round(v);
        public static float Sign(float v) => v >= 0 ? 1f : -1f;
    }

    public static class Debug
    {
        public static void Log(object m) { }
        public static void LogWarning(object m) { }
        public static void LogError(object m) => Console.Error.WriteLine(m);
    }
}
