using System.Numerics;

namespace LuBanSuo.Core;

public class BoundXYZ
{
	public BigInteger BoundXMin = int.MaxValue;

	public BigInteger BoundXMax = int.MinValue;

	public BigInteger BoundYMin = int.MaxValue;

	public BigInteger BoundYMax = int.MinValue;

	public BigInteger BoundZMin = int.MaxValue;

	public BigInteger BoundZMax = int.MinValue;

	public void Update(Point point)
	{
		if (point.X < BoundXMin)
		{
			BoundXMin = point.X;
		}
		if (point.X > BoundXMax)
		{
			BoundXMax = point.X;
		}
		if (point.Y < BoundYMin)
		{
			BoundYMin = point.Y;
		}
		if (point.Y > BoundYMax)
		{
			BoundYMax = point.Y;
		}
		if (point.Z < BoundZMin)
		{
			BoundZMin = point.Z;
		}
		if (point.Z > BoundZMax)
		{
			BoundZMax = point.Z;
		}
	}
}
