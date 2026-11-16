using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace LuBanSuo.Core;

public class ConvexPolyhedron
{
	public readonly List<Point> Points = new List<Point>();

	public readonly List<Direction> SgnDirections = new List<Direction>();

	public readonly List<Direction> ParallelDirections = new List<Direction>();

	public readonly List<LineSegment> LineSegments = new List<LineSegment>();

	public readonly List<Polygon> Polygons;

	public readonly BoundXYZ Bound = new BoundXYZ();

	public static string LastIntersectionDetail = "";

	public ConvexPolyhedron(List<Polygon> polygons)
	{
		foreach (Polygon polygon in polygons)
		{
			foreach (Point point in polygon.Points)
			{
				FilterAddPoint(point);
			}
			foreach (LineSegment lineSegment in polygon.LineSegments)
			{
				FilterAddDirection(lineSegment.Direction);
				FilterAddLineSegment(lineSegment);
			}
		}
		Polygons = polygons;
	}

	private void FilterAddPoint(Point point)
	{
		if (!Points.Any((Point p) => p.IsSameQuick(point)))
		{
			Points.Add(point);
			Bound.Update(point);
		}
	}

	private void FilterAddDirection(Direction direction)
	{
		if (!SgnDirections.Any((Direction d) => d.IsSameDirection(direction)))
		{
			SgnDirections.Add(direction);
			if (!ParallelDirections.Any((Direction d) => d.IsSameDirection(direction)))
			{
				ParallelDirections.Add(direction);
			}
		}
	}

	private void FilterAddLineSegment(LineSegment lineSegment)
	{
		if (!LineSegments.Any((LineSegment l) => l.IsSameQuick(lineSegment)))
		{
			LineSegments.Add(lineSegment);
		}
	}

	public static bool TestIntersection(ConvexPolyhedron cp1, Position pos1, ConvexPolyhedron cp2, Position pos2)
	{
		LastIntersectionDetail = "";
		BigFraction bigFraction = pos1.X + cp1.Bound.BoundXMin;
		BigFraction bigFraction2 = pos1.X + cp1.Bound.BoundXMax;
		BigFraction bigFraction3 = pos1.Y + cp1.Bound.BoundYMin;
		BigFraction bigFraction4 = pos1.Y + cp1.Bound.BoundYMax;
		BigFraction bigFraction5 = pos1.Z + cp1.Bound.BoundZMin;
		BigFraction other = pos1.Z + cp1.Bound.BoundZMax;
		BigFraction bigFraction6 = pos2.X + cp2.Bound.BoundXMin;
		BigFraction bigFraction7 = pos2.X + cp2.Bound.BoundXMax;
		BigFraction bigFraction8 = pos2.Y + cp2.Bound.BoundYMin;
		BigFraction bigFraction9 = pos2.Y + cp2.Bound.BoundYMax;
		BigFraction bigFraction10 = pos2.Z + cp2.Bound.BoundZMin;
		BigFraction other2 = pos2.Z + cp2.Bound.BoundZMax;
		if (bigFraction.CompareTo(bigFraction7) >= 0 || bigFraction6.CompareTo(bigFraction2) >= 0 || bigFraction3.CompareTo(bigFraction9) >= 0 || bigFraction8.CompareTo(bigFraction4) >= 0 || bigFraction5.CompareTo(other2) >= 0 || bigFraction10.CompareTo(other) >= 0)
		{
			return false;
		}
		LastIntersectionDetail = $"AABB: [{bigFraction},{bigFraction2}]x[{bigFraction3},{bigFraction4}] overlapping [{bigFraction6},{bigFraction7}]x[{bigFraction8},{bigFraction9}]";
		foreach (Direction item in ComputeDirections(cp1, cp2))
		{
			BigFraction[] i = cp1.ComputeInterval(item, pos1);
			BigFraction[] i2 = cp2.ComputeInterval(item, pos2);
			if (!TestIntervalIntersection(i, i2))
			{
				return false;
			}
		}
		return true;
	}

	private static List<Direction> ComputeDirections(ConvexPolyhedron cp1, ConvexPolyhedron cp2)
	{
		List<Direction> list = new List<Direction>();
		foreach (Polygon polygon in cp1.Polygons)
		{
			if (!ContainParallel(list, polygon.Direction))
			{
				list.Add(polygon.Direction);
			}
		}
		foreach (Polygon polygon2 in cp2.Polygons)
		{
			if (!ContainParallel(list, polygon2.Direction))
			{
				list.Add(polygon2.Direction);
			}
		}
		foreach (Direction parallelDirection in cp1.ParallelDirections)
		{
			foreach (Direction parallelDirection2 in cp2.ParallelDirections)
			{
				Direction direction = parallelDirection.Cross(parallelDirection2);
				if (!direction.IsZero() && !ContainParallel(list, direction))
				{
					list.Add(direction);
				}
			}
		}
		return list;
	}

	private static bool ContainParallel(IEnumerable<Direction> directions, Direction direction)
	{
		return directions.Any(direction.IsParallel);
	}

	private static bool TestIntervalIntersection(BigFraction[] i1, BigFraction[] i2)
	{
		if (i1[0].CompareTo(i1[1]) == 0 || i2[0].CompareTo(i2[1]) == 0)
		{
			return false;
		}
		if (i1[0].CompareTo(i2[1]) >= 0 || i2[0].CompareTo(i1[1]) >= 0)
		{
			return false;
		}
		return true;
	}

	private BigFraction[] ComputeInterval(Direction direction, Position position)
	{
		BigFraction bigFraction = Points[0].Dot(direction, position);
		BigFraction bigFraction2 = bigFraction;
		for (int i = 1; i < Points.Count; i++)
		{
			BigFraction bigFraction3 = Points[i].Dot(direction, position);
			if (bigFraction3.CompareTo(bigFraction) < 0)
			{
				bigFraction = bigFraction3;
			}
			if (bigFraction3.CompareTo(bigFraction2) > 0)
			{
				bigFraction2 = bigFraction3;
			}
		}
		return new BigFraction[2] { bigFraction, bigFraction2 };
	}

	public static BigFraction MoveBeforeIntersection(ConvexPolyhedron convexPolyhedron1, Position position1, ConvexPolyhedron convexPolyhedron2, Position position2, Direction move1To2)
	{
		BigFraction bigFraction = new BigFraction((BigInteger)int.MinValue);
		BigFraction other = new BigFraction((BigInteger)int.MaxValue);
		foreach (Direction item in ComputeDirections(convexPolyhedron1, convexPolyhedron2))
		{
			BigFraction[] interval = convexPolyhedron1.ComputeInterval(item, position1);
			BigFraction[] interval2 = convexPolyhedron2.ComputeInterval(item, position2);
			BigFraction[] array = MoveForIntersection(interval, interval2, item, move1To2);
			if (array == null)
			{
				return null;
			}
			if (array[0].CompareTo(bigFraction) > 0)
			{
				bigFraction = array[0];
			}
			if (array[1].CompareTo(other) < 0)
			{
				other = array[1];
			}
		}
		if (bigFraction.CompareTo(other) >= 0)
		{
			return null;
		}
		if (bigFraction.CompareTo(BigFraction.Zero) < 0)
		{
			return null;
		}
		return bigFraction;
	}

	private static BigFraction[] MoveForIntersection(IReadOnlyList<BigFraction> interval1, IReadOnlyList<BigFraction> interval2, Direction direction, Direction move1To2)
	{
		if (interval1[0].CompareTo(interval1[1]) == 0 || interval2[0].CompareTo(interval2[1]) == 0)
		{
			return null;
		}
		BigInteger bigInteger = move1To2.Dot(direction);
		BigFraction bigFraction = interval2[0] - interval1[1];
		BigFraction bigFraction2 = interval2[1] - interval1[0];
		if (bigInteger == 0L)
		{
			if (bigFraction.CompareTo(BigFraction.Zero) >= 0 || bigFraction2.CompareTo(BigFraction.Zero) <= 0)
			{
				return null;
			}
			return new BigFraction[2]
			{
				new BigFraction((BigInteger)int.MinValue),
				new BigFraction((BigInteger)int.MaxValue)
			};
		}
		BigFraction bigFraction3 = new BigFraction(bigInteger);
		if (bigFraction3.CompareTo(BigFraction.Zero) > 0)
		{
			BigFraction bigFraction4 = bigFraction / bigFraction3;
			BigFraction bigFraction5 = bigFraction2 / bigFraction3;
			return new BigFraction[2] { bigFraction4, bigFraction5 };
		}
		BigFraction bigFraction6 = bigFraction2 / bigFraction3;
		BigFraction bigFraction7 = bigFraction / bigFraction3;
		return new BigFraction[2] { bigFraction6, bigFraction7 };
	}

	public BoundXYZMove GetMoveBounds(Position thisPosition, BigFraction moveX, BigFraction moveY, BigFraction moveZ)
	{
		return new BoundXYZMove
		{
			BoundXMin = ((moveX.CompareTo(BigFraction.Zero) < 0) ? (moveX + Bound.BoundXMin + thisPosition.X) : (new BigFraction(Bound.BoundXMin) + thisPosition.X)),
			BoundXMax = ((moveX.CompareTo(BigFraction.Zero) > 0) ? (moveX + Bound.BoundXMax + thisPosition.X) : (new BigFraction(Bound.BoundXMax) + thisPosition.X)),
			BoundYMin = ((moveY.CompareTo(BigFraction.Zero) < 0) ? (moveY + Bound.BoundYMin + thisPosition.Y) : (new BigFraction(Bound.BoundYMin) + thisPosition.Y)),
			BoundYMax = ((moveY.CompareTo(BigFraction.Zero) > 0) ? (moveY + Bound.BoundYMax + thisPosition.Y) : (new BigFraction(Bound.BoundYMax) + thisPosition.Y)),
			BoundZMin = ((moveZ.CompareTo(BigFraction.Zero) < 0) ? (moveZ + Bound.BoundZMin + thisPosition.Z) : (new BigFraction(Bound.BoundZMin) + thisPosition.Z)),
			BoundZMax = ((moveZ.CompareTo(BigFraction.Zero) > 0) ? (moveZ + Bound.BoundZMax + thisPosition.Z) : (new BigFraction(Bound.BoundZMax) + thisPosition.Z))
		};
	}

	public bool TestBoundXYZMoveIntersection(BoundXYZMove boundXYZMove, Position thisPosition)
	{
		if ((thisPosition.X + Bound.BoundXMin).CompareTo(boundXYZMove.BoundXMax) < 0 && (thisPosition.X + Bound.BoundXMax).CompareTo(boundXYZMove.BoundXMin) > 0 && (thisPosition.Y + Bound.BoundYMin).CompareTo(boundXYZMove.BoundYMax) < 0 && (thisPosition.Y + Bound.BoundYMax).CompareTo(boundXYZMove.BoundYMin) > 0 && (thisPosition.Z + Bound.BoundZMin).CompareTo(boundXYZMove.BoundZMax) < 0)
		{
			return (thisPosition.Z + Bound.BoundZMax).CompareTo(boundXYZMove.BoundZMin) > 0;
		}
		return false;
	}
}
