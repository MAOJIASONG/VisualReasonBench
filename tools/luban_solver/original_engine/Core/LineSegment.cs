namespace BurrPuzzle.DataCore
{
	public class LineSegment
	{
		public int Index = -1;

		public readonly Point APoint;

		public readonly Point BPoint;

		public readonly Direction Direction;

		public LineSegment(Point aPoint, Point bPoint, Direction direction)
		{
			APoint = aPoint;
			BPoint = bPoint;
			Direction = direction;
		}

		public bool IsSameQuick(LineSegment otherLine)
		{
			if (APoint.Index != otherLine.APoint.Index || BPoint.Index != otherLine.BPoint.Index)
			{
				if (APoint.Index == otherLine.BPoint.Index)
				{
					return BPoint.Index == otherLine.APoint.Index;
				}
				return false;
			}
			return true;
		}

		public bool isOnLine(Point point)
		{
			BigInteger bigInteger = point.X - APoint.X;
			BigInteger bigInteger2 = point.Y - APoint.Y;
			BigInteger bigInteger3 = point.Z - APoint.Z;
			if (bigInteger == 0L && bigInteger2 == 0L && bigInteger3 == 0L)
			{
				return true;
			}
			Direction direction = new Direction(bigInteger, bigInteger2, bigInteger3);
			return Direction.IsParallel(direction);
		}

		public bool IsOnLineSegment(Point point)
		{
			if (!isOnLine(point))
			{
				return false;
			}
			if (Direction.X != 0L)
			{
				return point.X.CompareTo(APoint.X) * point.X.CompareTo(BPoint.X) < 0;
			}
			if (Direction.Y != 0L)
			{
				return point.Y.CompareTo(APoint.Y) * point.Y.CompareTo(BPoint.Y) < 0;
			}
			if (Direction.Z != 0L)
			{
				return point.Z.CompareTo(APoint.Z) * point.Z.CompareTo(BPoint.Z) < 0;
			}
			return false;
		}
	}
}
