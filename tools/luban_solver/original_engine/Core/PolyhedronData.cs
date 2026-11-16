namespace BurrPuzzle.DataCore
{
	public struct PolyhedronData
	{
		public List<VertexData> vertexes;

		public List<FaceData> faces;

		public PositionData position1;

		public SwitchAxisTypeXYZ switchAxisTypeXYZ1;

		public PositionData position2;

		public SwitchAxisTypeXYZ switchAxisTypeXYZ2;

		public List<ConvexPolyhedronData> convexPolyhedrons;
	}
}
