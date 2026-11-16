namespace LuBanSuo.Core;

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

	public bool IsSameQuick(LineSegment other)
	{
		if (APoint.Index == other.APoint.Index && BPoint.Index == other.BPoint.Index)
		{
			return true;
		}
		if (APoint.Index == other.BPoint.Index && BPoint.Index == other.APoint.Index)
		{
			return true;
		}
		return false;
	}
}
