namespace BurrPuzzle.DataCore
{
	public struct PuzzleData
	{
		public List<PolyhedronData> polyhedrons;

		public List<DirectionData> directions;

		public int polyhedronBaseIndex;

		public List<int> polyhedronTypes;

		public List<MoveRotate> moveRotates;

		public bool skipPlay1;

		public bool disableQuickMove;

		public bool disableHalfRotate;
	}
}
