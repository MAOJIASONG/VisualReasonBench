namespace BurrPuzzle.DataCore
{
	public struct MoveRotate
	{
		public int polyhedronIndex;

		public List<PolyhedronCondition> polyhedronConditions;

		public bool isHalf;

		public RotateAxis rotateAxis;

		public int rotateCounts;

		public Position positionAddScaled;
	}
}
