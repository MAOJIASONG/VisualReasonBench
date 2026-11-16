namespace BurrPuzzle.DataCore
{
	public class AnglePolyhedron
	{
		public readonly List<Point> Points;

		public readonly List<Polygon> Polygons = new List<Polygon>();

		public readonly List<Direction> SgnDirections = new List<Direction>();

		public readonly List<Direction> ParallelDirections = new List<Direction>();

		public readonly List<LineSegment> LineSegments = new List<LineSegment>();

		public readonly List<ConvexPolyhedron> ConvexPolyhedronList = new List<ConvexPolyhedron>();

		private void BuildPointsList(List<Point> pointsAll, List<List<int>> polygonsPointsIndex, List<List<List<int>>> convexPolyhedraPointsIndex, List<List<Point>> polygonsPoints, List<List<List<Point>>> convexPolyhedraPoints)
		{
			foreach (List<int> item in polygonsPointsIndex)
			{
				List<Point> list = new List<Point>(item.Count);
				foreach (int item2 in item)
				{
					list.Add(pointsAll[item2]);
				}
				polygonsPoints.Add(list);
			}
			foreach (List<List<int>> item3 in convexPolyhedraPointsIndex)
			{
				List<List<Point>> list2 = new List<List<Point>>(item3.Count);
				foreach (List<int> item4 in item3)
				{
					List<Point> list3 = new List<Point>(item4.Count);
					foreach (int item5 in item4)
					{
						list3.Add(pointsAll[item5]);
					}
					list2.Add(list3);
				}
				convexPolyhedraPoints.Add(list2);
			}
		}

		public AnglePolyhedron(List<Point> points, List<List<int>> polygonsPointsIndex, List<List<List<int>>> convexPolyhedraPointsIndex)
		{
			List<List<Point>> list = new List<List<Point>>(polygonsPointsIndex.Count);
			List<List<List<Point>>> list2 = new List<List<List<Point>>>(convexPolyhedraPointsIndex.Count);
			BuildPointsList(points, polygonsPointsIndex, convexPolyhedraPointsIndex, list, list2);
			Points = points;
			for (int i = 0; i < list.Count; i++)
			{
				List<Point> list3 = list[i];
				List<LineSegment> list4 = new List<LineSegment>(list3.Count);
				Point point = list3[0];
				for (int j = 1; j <= list3.Count; j++)
				{
					Point point2 = list3[j % list3.Count];
					BigInteger x = point.X - point2.X;
					BigInteger y = point.Y - point2.Y;
					BigInteger z = point.Z - point2.Z;
					Direction direction = new Direction(x, y, z);
					direction = FilterAddDirection(direction);
					LineSegment lineSegment = new LineSegment(point, point2, direction);
					lineSegment = FilterAddLineSegment(lineSegment);
					list4.Add(lineSegment);
					point = point2;
				}
				Direction direction2 = null;
				for (int k = 1; k < list4.Count; k++)
				{
					direction2 = list4[0].Direction.Cross(list4[k].Direction);
					if (!direction2.IsZero())
					{
						break;
					}
				}
				if (direction2 == null || direction2.IsZero())
				{
					UnityEngine.Debug.Log("direction zero !!!");
				}
				direction2 = FilterAddDirection(direction2);
				Polygon item = new Polygon(i, list3, list4, direction2);
				Polygons.Add(item);
			}
			foreach (List<List<Point>> item4 in list2)
			{
				List<Polygon> list5 = new List<Polygon>(item4.Count);
				for (int l = 0; l < item4.Count; l++)
				{
					List<Point> list6 = item4[l];
					for (int m = 0; m < list6.Count; m++)
					{
						list6[m] = FilterAddPoint(list6[m]);
					}
					List<LineSegment> list7 = new List<LineSegment>(list6.Count);
					Point point3 = list6[0];
					for (int n = 1; n <= list6.Count; n++)
					{
						Point point4 = list6[n % list6.Count];
						BigInteger x2 = point3.X - point4.X;
						BigInteger y2 = point3.Y - point4.Y;
						BigInteger z2 = point3.Z - point4.Z;
						Direction direction3 = new Direction(x2, y2, z2);
						direction3 = FilterAddDirection(direction3);
						LineSegment lineSegment2 = new LineSegment(point3, point4, direction3);
						lineSegment2 = FilterAddLineSegment(lineSegment2);
						list7.Add(lineSegment2);
						point3 = point4;
					}
					Direction direction4 = list7[0].Direction.Cross(list7[1].Direction);
					direction4 = FilterAddDirection(direction4);
					Polygon item2 = new Polygon(l, list6, list7, direction4);
					list5.Add(item2);
				}
				ConvexPolyhedron item3 = new ConvexPolyhedron(list5);
				ConvexPolyhedronList.Add(item3);
			}
		}

		private Point FilterAddPoint(Point point)
		{
			foreach (Point point2 in Points)
			{
				if (point2.IsSameQuick(point))
				{
					return point2;
				}
			}
			Points.Add(point);
			return point;
		}

		private Direction FilterAddDirection(Direction direction)
		{
			if (direction == null)
			{
				UnityEngine.Debug.Log("74:");
			}
			Direction direction2 = null;
			foreach (Direction sgnDirection in SgnDirections)
			{
				if (sgnDirection == null)
				{
					UnityEngine.Debug.Log("80:" + direction);
				}
				if (sgnDirection.IsSameDirection(direction))
				{
					direction2 = sgnDirection;
				}
			}
			if (direction2 == null)
			{
				direction2 = direction;
				direction.Index = SgnDirections.Count;
				SgnDirections.Add(direction);
			}
			Direction direction3 = null;
			foreach (Direction parallelDirection in ParallelDirections)
			{
				if (parallelDirection.IsParallel(direction))
				{
					direction3 = parallelDirection;
				}
			}
			if (direction3 == null)
			{
				ParallelDirections.Add(direction2);
			}
			return direction2;
		}

		private LineSegment FilterAddLineSegment(LineSegment lineSegment)
		{
			foreach (LineSegment lineSegment2 in LineSegments)
			{
				if (lineSegment2.IsSameQuick(lineSegment))
				{
					return lineSegment2;
				}
			}
			lineSegment.Index = LineSegments.Count;
			LineSegments.Add(lineSegment);
			return lineSegment;
		}

		public static bool TestIntersection(AnglePolyhedron anglePolyhedron1, AnglePolyhedron anglePolyhedron2)
		{
			Position position = new Position(BigFraction.Zero, BigFraction.Zero, BigFraction.Zero);
			foreach (ConvexPolyhedron convexPolyhedron in anglePolyhedron1.ConvexPolyhedronList)
			{
				foreach (ConvexPolyhedron convexPolyhedron2 in anglePolyhedron2.ConvexPolyhedronList)
				{
					if (ConvexPolyhedron.TestIntersection(convexPolyhedron, position, convexPolyhedron2, position))
					{
						return true;
					}
				}
			}
			return false;
		}

		public static bool TestIntersection(AnglePolyhedron anglePolyhedron1, Position position1, AnglePolyhedron anglePolyhedron2, Position position2)
		{
			foreach (ConvexPolyhedron convexPolyhedron in anglePolyhedron1.ConvexPolyhedronList)
			{
				foreach (ConvexPolyhedron convexPolyhedron2 in anglePolyhedron2.ConvexPolyhedronList)
				{
					if (ConvexPolyhedron.TestIntersection(convexPolyhedron, position1, convexPolyhedron2, position2))
					{
						return true;
					}
				}
			}
			return false;
		}

		public static BigFraction MoveBeforeIntersection(AnglePolyhedron anglePolyhedron1, Position position1, AnglePolyhedron anglePolyhedron2, Position position2, Direction move1To2)
		{
			BigFraction bigFraction = Polyhedron.MoveMaxPositive;
			foreach (ConvexPolyhedron convexPolyhedron in anglePolyhedron1.ConvexPolyhedronList)
			{
				foreach (ConvexPolyhedron convexPolyhedron2 in anglePolyhedron2.ConvexPolyhedronList)
				{
					BigFraction bigFraction2 = ConvexPolyhedron.MoveBeforeIntersection(convexPolyhedron, position1, convexPolyhedron2, position2, move1To2);
					if (bigFraction2 != null && bigFraction.CompareTo(bigFraction2) > 0)
					{
						bigFraction = bigFraction2;
					}
				}
			}
			if (bigFraction.Equals(Polyhedron.MoveMaxPositive))
			{
				return null;
			}
			return bigFraction;
		}
	}
}
