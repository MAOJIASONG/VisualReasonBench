using System.Collections.Generic;

namespace LuBanSuo.Core;

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
}
