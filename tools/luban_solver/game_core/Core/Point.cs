using System;
using System.Collections.Generic;
using System.Numerics;

namespace LuBanSuo.Core;

public class Point
{
	public class CosSinByDegree
	{
		public readonly BigFraction CosDegree;

		public readonly BigFraction SinDegree;

		public CosSinByDegree(BigFraction cosDegree, BigFraction sinDegree)
		{
			CosDegree = cosDegree;
			SinDegree = sinDegree;
		}
	}

	public int Index;

	public BigInteger X;

	public BigInteger Y;

	public BigInteger Z;

	public const int ScaleFactor = 10000;

	private static readonly List<CosSinByDegree> CosSinByDegreesPositive;

	private static readonly List<CosSinByDegree> CosSinByDegreesNegative;

	static Point()
	{
		CosSinByDegreesPositive = new List<CosSinByDegree>(33);
		CosSinByDegreesPositive.Add(new CosSinByDegree(BigFraction.One, BigFraction.Zero));
		for (int i = 1; i <= 32; i++)
		{
			BigFraction cosDegree = new BigFraction(Math.Cos((double)i / 32.0 * Math.PI));
			BigFraction sinDegree = new BigFraction(Math.Sin((double)i / 32.0 * Math.PI));
			CosSinByDegreesPositive.Add(new CosSinByDegree(cosDegree, sinDegree));
		}
		CosSinByDegreesNegative = new List<CosSinByDegree>(33);
		CosSinByDegreesNegative.Add(new CosSinByDegree(BigFraction.One, BigFraction.Zero));
		for (int j = 1; j <= 32; j++)
		{
			BigFraction cosDegree2 = new BigFraction(Math.Cos((double)(-j) / 32.0 * Math.PI));
			BigFraction sinDegree2 = new BigFraction(Math.Sin((double)(-j) / 32.0 * Math.PI));
			CosSinByDegreesNegative.Add(new CosSinByDegree(cosDegree2, sinDegree2));
		}
	}

	public Point(int index, float x, float y, float z, bool fromJson = false)
	{
		Index = index;
		if (fromJson)
		{
			X = (BigInteger)(x * 10000f);
			Y = (BigInteger)(y * 10000f);
			Z = (BigInteger)(z * 10000f);
		}
		else
		{
			X = (BigInteger)x;
			Y = (BigInteger)y;
			Z = (BigInteger)z;
		}
	}

	public Point(int index, BigInteger x, BigInteger y, BigInteger z)
	{
		Index = index;
		X = x;
		Y = y;
		Z = z;
	}

	public bool IsSameQuick(Point point)
	{
		return point.Index == Index;
	}

	public BigFraction Dot(Direction direction, Position movePosition)
	{
		BigFraction bigFraction = new BigFraction(X) + movePosition.X;
		BigFraction bigFraction2 = new BigFraction(Y) + movePosition.Y;
		BigFraction bigFraction3 = new BigFraction(Z) + movePosition.Z;
		return bigFraction * direction.X + bigFraction2 * direction.Y + bigFraction3 * direction.Z;
	}

	public static CosSinByDegree GetCsDegree(int angleSteps)
	{
		if (angleSteps <= 0)
		{
			return CosSinByDegreesNegative[-angleSteps];
		}
		return CosSinByDegreesPositive[angleSteps];
	}

	private static BigInteger GetBigInt(BigFraction bigFraction)
	{
		return bigFraction.Numerator / bigFraction.Denominator;
	}

	public Point RotateX(int angleSteps)
	{
		CosSinByDegree csDegree = GetCsDegree(angleSteps);
		BigInteger bigInt = GetBigInt(csDegree.CosDegree * Y - csDegree.SinDegree * Z);
		BigInteger bigInt2 = GetBigInt(csDegree.SinDegree * Y + csDegree.CosDegree * Z);
		return new Point(Index, X, bigInt, bigInt2);
	}

	public Point RotateY(int angleSteps)
	{
		CosSinByDegree csDegree = GetCsDegree(angleSteps);
		BigInteger bigInt = GetBigInt(csDegree.CosDegree * Z - csDegree.SinDegree * X);
		BigInteger bigInt2 = GetBigInt(csDegree.SinDegree * Z + csDegree.CosDegree * X);
		return new Point(Index, bigInt2, Y, bigInt);
	}

	public Point RotateZ(int angleSteps)
	{
		CosSinByDegree csDegree = GetCsDegree(angleSteps);
		BigInteger bigInt = GetBigInt(csDegree.CosDegree * X - csDegree.SinDegree * Y);
		BigInteger bigInt2 = GetBigInt(csDegree.SinDegree * X + csDegree.CosDegree * Y);
		return new Point(Index, bigInt, bigInt2, Z);
	}
}
