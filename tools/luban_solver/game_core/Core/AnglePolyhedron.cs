using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using UnityEngine;

namespace LuBanSuo.Core;

public class AnglePolyhedron
{
	public readonly List<Point> Points;

	public readonly List<Polygon> Polygons = new List<Polygon>();

	public readonly List<Direction> SgnDirections = new List<Direction>();

	public readonly List<Direction> ParallelDirections = new List<Direction>();

	public readonly List<LineSegment> LineSegments = new List<LineSegment>();

	public readonly List<ConvexPolyhedron> ConvexPolyhedronList = new List<ConvexPolyhedron>();

	public AnglePolyhedron(List<Point> points, List<List<int>> polygonsPointsIndex, List<List<List<int>>> convexPolyhedraPointsIndex)
	{
		Points = points;
		foreach (List<int> item in polygonsPointsIndex)
		{
			List<Point> list = item.Select((int i) => points[i]).ToList();
			List<LineSegment> list2 = new List<LineSegment>();
			for (int num = 0; num < list.Count; num++)
			{
				Point point = list[num];
				Point point2 = list[(num + 1) % list.Count];
				Direction direction = FilterAddDirection(new Direction(point.X - point2.X, point.Y - point2.Y, point.Z - point2.Z));
				list2.Add(FilterAddLineSegment(new LineSegment(point, point2, direction)));
			}
			Direction direction2 = list2[0].Direction.Cross(list2[1].Direction);
			direction2 = FilterAddDirection(direction2);
			Polygons.Add(new Polygon(-1, list, list2, direction2));
		}
		foreach (List<List<int>> item2 in convexPolyhedraPointsIndex)
		{
			List<Polygon> list3 = new List<Polygon>();
			foreach (List<int> item3 in item2)
			{
				List<Point> list4 = item3.Select((int i) => points[i]).ToList();
				List<LineSegment> list5 = new List<LineSegment>();
				for (int num2 = 0; num2 < list4.Count; num2++)
				{
					Point point3 = list4[num2];
					Point point4 = list4[(num2 + 1) % list4.Count];
					Direction direction3 = FilterAddDirection(new Direction(point3.X - point4.X, point3.Y - point4.Y, point3.Z - point4.Z));
					list5.Add(FilterAddLineSegment(new LineSegment(point3, point4, direction3)));
				}
				Direction direction4 = list5[0].Direction.Cross(list5[1].Direction);
				direction4 = FilterAddDirection(direction4);
				list3.Add(new Polygon(-1, list4, list5, direction4));
			}
			ConvexPolyhedronList.Add(new ConvexPolyhedron(list3));
		}
	}

	public AnglePolyhedron(List<ConvexPolyhedron> convexPolyhedrons)
	{
		ConvexPolyhedronList = convexPolyhedrons;
		Points = new List<Point>();
		foreach (ConvexPolyhedron convexPolyhedron in convexPolyhedrons)
		{
			foreach (Point p in convexPolyhedron.Points)
			{
				if (!Points.Any((Point existing) => existing.IsSameQuick(p)))
				{
					Points.Add(p);
				}
			}
			foreach (Polygon poly in convexPolyhedron.Polygons)
			{
				Polygons.Add(poly);
				foreach (LineSegment ls in poly.LineSegments)
				{
					if (!LineSegments.Any((LineSegment existing) => existing.IsSameQuick(ls)))
					{
						LineSegments.Add(ls);
					}
				}
				if (!SgnDirections.Any((Direction existing) => existing.IsSameDirection(poly.Direction)))
				{
					SgnDirections.Add(poly.Direction);
				}
			}
			foreach (Direction sd in convexPolyhedron.SgnDirections)
			{
				if (!SgnDirections.Any((Direction existing) => existing.IsSameDirection(sd)))
				{
					SgnDirections.Add(sd);
				}
			}
		}
	}

	private Direction FilterAddDirection(Direction direction)
	{
		Direction direction2 = SgnDirections.FirstOrDefault((Direction d) => d.IsSameDirection(direction));
		if (direction2 != null)
		{
			return direction2;
		}
		direction.Index = SgnDirections.Count;
		SgnDirections.Add(direction);
		if (!ParallelDirections.Any((Direction d) => d.IsParallel(direction)))
		{
			ParallelDirections.Add(direction);
		}
		return direction;
	}

	private LineSegment FilterAddLineSegment(LineSegment lineSegment)
	{
		LineSegment lineSegment2 = LineSegments.FirstOrDefault((LineSegment l) => l.IsSameQuick(lineSegment));
		if (lineSegment2 != null)
		{
			return lineSegment2;
		}
		lineSegment.Index = LineSegments.Count;
		LineSegments.Add(lineSegment);
		return lineSegment;
	}

	public static bool TestIntersection(AnglePolyhedron ap1, Position pos1, AnglePolyhedron ap2, Position pos2)
	{
		foreach (ConvexPolyhedron convexPolyhedron in ap1.ConvexPolyhedronList)
		{
			foreach (ConvexPolyhedron convexPolyhedron2 in ap2.ConvexPolyhedronList)
			{
				if (ConvexPolyhedron.TestIntersection(convexPolyhedron, pos1, convexPolyhedron2, pos2))
				{
					return true;
				}
			}
		}
		return false;
	}

	public static BigFraction MoveBeforeIntersection(AnglePolyhedron ap1, Position pos1, AnglePolyhedron ap2, Position pos2, Direction move1To2)
	{
		BigFraction bigFraction = new BigFraction((BigInteger)int.MaxValue);
		bool flag = false;
		foreach (ConvexPolyhedron convexPolyhedron in ap1.ConvexPolyhedronList)
		{
			foreach (ConvexPolyhedron convexPolyhedron2 in ap2.ConvexPolyhedronList)
			{
				BigFraction bigFraction2 = ConvexPolyhedron.MoveBeforeIntersection(convexPolyhedron, pos1, convexPolyhedron2, pos2, move1To2);
				if (bigFraction2 != null)
				{
					if (bigFraction2.CompareTo(bigFraction) < 0)
					{
						bigFraction = bigFraction2;
					}
					flag = true;
				}
			}
		}
		if (!flag)
		{
			return null;
		}
		return bigFraction;
	}

	public void SetAngle(Transform transform, int xyz, bool isAdd)
	{
		if (!(transform == null))
		{
			_ = GetAllTestEulerAngles(transform.localEulerAngles, xyz, isAdd).Count;
			_ = 0;
		}
	}

	public static List<UnityEngine.Vector3> GetAllTestEulerAngles(UnityEngine.Vector3 startEuler, int xyz, bool isAdd)
	{
		List<UnityEngine.Vector3> list = new List<UnityEngine.Vector3>();
		float num = (isAdd ? 1f : (-1f));
		for (int i = 1; i <= 90; i++)
		{
			UnityEngine.Vector3 item = startEuler;
			float num2 = (float)i * num;
			switch (xyz)
			{
			case 0:
				item.x += num2;
				break;
			case 1:
				item.y += num2;
				break;
			case 2:
				item.z += num2;
				break;
			}
			list.Add(item);
		}
		return list;
	}
}
