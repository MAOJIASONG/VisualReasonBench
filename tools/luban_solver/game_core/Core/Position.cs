using System.Numerics;
using UnityEngine;

namespace LuBanSuo.Core;

public class Position
{
	public static readonly int ScaleFactor = 10000;

	public BigFraction X { get; set; }

	public BigFraction Y { get; set; }

	public BigFraction Z { get; set; }

	public Position()
	{
	}

	public Position(int x, int y, int z)
	{
		X = new BigFraction((BigInteger)(x * ScaleFactor));
		Y = new BigFraction((BigInteger)(y * ScaleFactor));
		Z = new BigFraction((BigInteger)(z * ScaleFactor));
	}

	public Position(BigFraction x, BigFraction y, BigFraction z, bool fromJson = false)
	{
		X = x;
		Y = y;
		Z = z;
		if (fromJson)
		{
			X = x * ScaleFactor;
			Y = y * ScaleFactor;
			Z = z * ScaleFactor;
		}
	}

	public Position(Position position)
	{
		X = position.X;
		Y = position.Y;
		Z = position.Z;
	}

	public float GetRealX()
	{
		return (float)(X.DoubleValue() / (double)ScaleFactor);
	}

	public float GetRealY()
	{
		return (float)(Y.DoubleValue() / (double)ScaleFactor);
	}

	public float GetRealZ()
	{
		return (float)(Z.DoubleValue() / (double)ScaleFactor);
	}

	public UnityEngine.Vector3 ToVector3()
	{
		return new UnityEngine.Vector3(GetRealX(), GetRealY(), GetRealZ());
	}
}
