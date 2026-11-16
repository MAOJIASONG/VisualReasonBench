namespace BurrPuzzle.DataCore
{
	public class Polygon
	{
		public int ID;

		public readonly List<Point> Points;

		public readonly List<LineSegment> LineSegments;

		public readonly Direction Direction;

		public Polygon(int id, List<Point> points, List<LineSegment> lineSegments, Direction normalDirection)
		{
			ID = id;
			Points = points;
			LineSegments = lineSegments;
			Direction = normalDirection;
		}

		public bool ContainTriangle(int point1Index, int point2Index, int point3Index)
		{
			bool flag = false;
			bool flag2 = false;
			bool flag3 = false;
			foreach (Point point in Points)
			{
				if (point.Index == point1Index)
				{
					flag = true;
				}
				else if (point.Index == point2Index)
				{
					flag2 = true;
				}
				else if (point.Index == point3Index)
				{
					flag3 = true;
				}
			}
			return flag && flag2 && flag3;
		}
	}
}
