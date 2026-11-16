using System.Numerics;

namespace LuBanSuo.Core;

public class Direction
{
	public int Index = -1;

	public BigInteger X;

	public BigInteger Y;

	public BigInteger Z;

	public BigInteger Factor;

	public BigInteger Factor2;

	public Direction(int index, int x, int y, int z)
		: this(x, y, z)
	{
		Index = index;
	}

	public Direction(BigFraction x, BigFraction y, BigFraction z)
	{
		Factor2 = BigInteger.GreatestCommonDivisor(BigInteger.GreatestCommonDivisor(x.Numerator, y.Denominator), z.Denominator);
		BigInteger bigInteger = x.Denominator / Factor2;
		BigInteger bigInteger2 = y.Denominator / Factor2;
		BigInteger bigInteger3 = z.Denominator / Factor2;
		X = x.Numerator * bigInteger2 * bigInteger3;
		Y = y.Numerator * bigInteger3 * bigInteger;
		Z = z.Numerator * bigInteger * bigInteger2;
		Factor2 *= bigInteger * bigInteger2 * bigInteger3;
		Reduce();
	}

	public Direction Negate()
	{
		return new Direction(-X, -Y, -Z, reduce: false)
		{
			Factor = BigInteger.One,
			Factor2 = BigInteger.One
		};
	}

	private void Reduce()
	{
		BigInteger x = X;
		BigInteger y = Y;
		BigInteger z = Z;
		Factor = 1;
		if (x == 0L)
		{
			if (y == 0L)
			{
				if (z == 0L)
				{
					X = 0;
					Y = 0;
					Z = 0;
				}
				else
				{
					X = x;
					Y = y;
					Factor = BigInteger.Abs(z);
					Z = z.Sign;
				}
				return;
			}
			if (z == 0L)
			{
				X = x;
				Factor = BigInteger.Abs(y);
				Y = y.Sign;
				Z = z;
				return;
			}
			Factor = BigInteger.GreatestCommonDivisor(y, z);
		}
		else if (y == 0L)
		{
			if (z == 0L)
			{
				Factor = BigInteger.Abs(x);
				X = x.Sign;
				Y = y;
				Z = z;
				return;
			}
			Factor = BigInteger.GreatestCommonDivisor(x, z);
		}
		else if (z == 0L)
		{
			Factor = BigInteger.GreatestCommonDivisor(x, y);
		}
		else
		{
			Factor = BigInteger.GreatestCommonDivisor(x, y);
			Factor = BigInteger.GreatestCommonDivisor(Factor, z);
		}
		X = x / Factor;
		Y = y / Factor;
		Z = z / Factor;
	}

	public Direction(BigInteger x, BigInteger y, BigInteger z, bool reduce = true)
	{
		X = x;
		Y = y;
		Z = z;
		if (reduce)
		{
			Reduce();
		}
	}

	public bool IsZero()
	{
		if (X == 0L && Y == 0L)
		{
			return Z == 0L;
		}
		return false;
	}

	public bool IsParallel(Direction direction)
	{
		if (X == 0L)
		{
			if (direction.X == 0L)
			{
				if (Y == 0L)
				{
					return direction.Y == 0L;
				}
				if (direction.Y == 0L)
				{
					return false;
				}
				return Y * direction.Z - Z * direction.Y == 0L;
			}
			return false;
		}
		if (direction.X == 0L)
		{
			return false;
		}
		if (X * direction.Y - Y * direction.X == 0L)
		{
			return X * direction.Z - Z * direction.X == 0L;
		}
		return false;
	}

	public bool IsSameQuick(Direction direction)
	{
		return Index == direction.Index;
	}

	public bool IsSameDirection(Direction direction)
	{
		if (X.Sign != direction.X.Sign || Y.Sign != direction.Y.Sign || Z.Sign != direction.Z.Sign)
		{
			return false;
		}
		return IsParallel(direction);
	}

	public BigInteger Dot(Direction direction)
	{
		return X * direction.X + Y * direction.Y + Z * direction.Z;
	}

	public Direction Cross(Direction direction)
	{
		return new Direction(Y * direction.Z - Z * direction.Y, Z * direction.X - X * direction.Z, X * direction.Y - Y * direction.X);
	}

	public static Direction operator -(Direction a)
	{
		return a.Negate();
	}
}
