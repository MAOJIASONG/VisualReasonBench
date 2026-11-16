namespace BurrPuzzle.DataCore
{
	public class ConvexPolyhedron
	{
		public readonly List<Point> Points = new List<Point>();

		public readonly List<Direction> SgnDirections = new List<Direction>();

		public readonly List<Direction> ParallelDirections = new List<Direction>();

		public readonly List<LineSegment> LineSegments = new List<LineSegment>();

		public readonly List<Polygon> Polygons;

		public readonly BoundXYZ Bound = new BoundXYZ();

		public ConvexPolyhedron(List<Point> points, List<Direction> sgnDirections, List<Direction> parallelDirections, List<LineSegment> lineSegments, List<Polygon> polygons)
		{
			Points = points;
			SgnDirections = sgnDirections;
			ParallelDirections = parallelDirections;
			LineSegments = lineSegments;
			Polygons = polygons;
			foreach (Point point in points)
			{
				Bound.Update(point);
			}
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
			if (!Points.Any((Point pointOld) => pointOld.IsSameQuick(point)))
			{
				Points.Add(point);
				Bound.Update(point);
			}
		}

		private void FilterAddDirection(Direction direction)
		{
			if (!SgnDirections.Any((Direction sgnDirection) => sgnDirection.IsSameDirection(direction)))
			{
				SgnDirections.Add(direction);
				if (!ParallelDirections.Any((Direction parallelDirection) => parallelDirection.IsSameDirection(direction)))
				{
					ParallelDirections.Add(direction);
				}
			}
		}

		private void FilterAddLineSegment(LineSegment lineSegment)
		{
			if (!LineSegments.Any((LineSegment lineSegmentOld) => lineSegmentOld.IsSameQuick(lineSegment)))
			{
				LineSegments.Add(lineSegment);
			}
		}

		private static bool ContainParallel(IEnumerable<Direction> directions, Direction direction)
		{
			return directions.Any(direction.IsParallel);
		}

		private static List<Direction> ComputeDirections(ConvexPolyhedron convexPolyhedron1, ConvexPolyhedron convexPolyhedron2)
		{
			List<Direction> directions = new List<Direction>();
			foreach (Direction item in from polygon in convexPolyhedron1.Polygons
				select polygon.Direction into direction
				where !ContainParallel(directions, direction)
				select direction)
			{
				directions.Add(item);
			}
			foreach (Direction item2 in from polygon in convexPolyhedron2.Polygons
				select polygon.Direction into direction
				where !ContainParallel(directions, direction)
				select direction)
			{
				directions.Add(item2);
			}
			foreach (Direction item3 in from direction in convexPolyhedron1.ParallelDirections.SelectMany((Direction _) => convexPolyhedron2.ParallelDirections, (Direction direction1, Direction direction2) => direction1.Cross(direction2))
				where !direction.IsZero() && !ContainParallel(directions, direction)
				select direction)
			{
				directions.Add(item3);
			}
			return directions;
		}

		public static bool TestIntersection(ConvexPolyhedron convexPolyhedron1, Position position1, ConvexPolyhedron convexPolyhedron2, Position position2)
		{
			BigFraction bigFraction = position1.X + convexPolyhedron1.Bound.BoundXMin;
			BigFraction bigFraction2 = position1.X + convexPolyhedron1.Bound.BoundXMax;
			BigFraction bigFraction3 = position1.Y + convexPolyhedron1.Bound.BoundYMin;
			BigFraction bigFraction4 = position1.Y + convexPolyhedron1.Bound.BoundYMax;
			BigFraction bigFraction5 = position1.Z + convexPolyhedron1.Bound.BoundZMin;
			BigFraction bigFraction6 = position1.Z + convexPolyhedron1.Bound.BoundZMax;
			BigFraction bigFraction7 = position2.X + convexPolyhedron2.Bound.BoundXMin;
			BigFraction bigFraction8 = position2.X + convexPolyhedron2.Bound.BoundXMax;
			BigFraction bigFraction9 = position2.Y + convexPolyhedron2.Bound.BoundYMin;
			BigFraction bigFraction10 = position2.Y + convexPolyhedron2.Bound.BoundYMax;
			BigFraction bigFraction11 = position2.Z + convexPolyhedron2.Bound.BoundZMin;
			BigFraction bigFraction12 = position2.Z + convexPolyhedron2.Bound.BoundZMax;
			if (bigFraction.CompareTo(bigFraction8) >= 0 || bigFraction7.CompareTo(bigFraction2) >= 0 || bigFraction3.CompareTo(bigFraction10) >= 0 || bigFraction9.CompareTo(bigFraction4) >= 0 || bigFraction5.CompareTo(bigFraction12) >= 0 || bigFraction11.CompareTo(bigFraction6) >= 0)
			{
				return false;
			}
			foreach (Direction item in ComputeDirections(convexPolyhedron1, convexPolyhedron2))
			{
				BigFraction[] interval = convexPolyhedron1.ComputeInterval(item, position1);
				BigFraction[] interval2 = convexPolyhedron2.ComputeInterval(item, position2);
				if (!TestIntersection(interval, interval2))
				{
					return false;
				}
			}
			return true;
		}

		public bool TestBoundXYZMoveIntersection(Position thisPosition, BoundXYZMove boundXYZMove)
		{
			BigFraction bigFraction = thisPosition.X + Bound.BoundXMin;
			BigFraction bigFraction2 = thisPosition.X + Bound.BoundXMax;
			BigFraction bigFraction3 = thisPosition.Y + Bound.BoundYMin;
			BigFraction bigFraction4 = thisPosition.Y + Bound.BoundYMax;
			BigFraction bigFraction5 = thisPosition.Z + Bound.BoundZMin;
			BigFraction bigFraction6 = thisPosition.Z + Bound.BoundZMax;
			if (bigFraction.CompareTo(boundXYZMove.BoundXMax) < 0 && boundXYZMove.BoundXMin.CompareTo(bigFraction2) < 0 && bigFraction3.CompareTo(boundXYZMove.BoundYMax) < 0 && boundXYZMove.BoundYMin.CompareTo(bigFraction4) < 0 && bigFraction5.CompareTo(boundXYZMove.BoundZMax) < 0)
			{
				return boundXYZMove.BoundZMin.CompareTo(bigFraction6) < 0;
			}
			return false;
		}

		private static bool TestIntersection(IReadOnlyList<BigFraction> interval1, IReadOnlyList<BigFraction> interval2)
		{
			if (interval1[0].CompareTo(interval1[1]) == 0)
			{
				return false;
			}
			if (interval2[0].CompareTo(interval2[1]) == 0)
			{
				return false;
			}
			if (interval1[0].CompareTo(interval2[1]) >= 0)
			{
				return false;
			}
			if (interval2[0].CompareTo(interval1[1]) >= 0)
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
			BigFraction bigFraction = Polyhedron.MoveMinNegative;
			BigFraction bigFraction2 = Polyhedron.MoveMaxPositive;
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
				if (array[1].CompareTo(bigFraction2) < 0)
				{
					bigFraction2 = array[1];
				}
			}
			if (bigFraction.CompareTo(bigFraction2) >= 0)
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
			if (interval1[0].CompareTo(interval1[1]) == 0)
			{
				return null;
			}
			if (interval2[0].CompareTo(interval2[1]) == 0)
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
					Polyhedron.MoveMinNegative,
					Polyhedron.MoveMaxPositive
				};
			}
			BigFraction bigFraction3 = new BigFraction(bigInteger);
			bigFraction.CompareTo(BigFraction.Zero);
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
	}
}
