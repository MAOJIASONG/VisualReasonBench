namespace BurrPuzzle.DataCore
{
	public class PolyhedronInitAxis
	{
		public const int AngleStepCount = 16;

		public const int HalfAngleStepCount = 8;

		public readonly List<AnglePolyhedron> NegativeWise = new List<AnglePolyhedron>();

		public readonly List<AnglePolyhedron> PositiveWise = new List<AnglePolyhedron>();
	}
}
