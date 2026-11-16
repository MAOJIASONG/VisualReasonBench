using System;
using System.Numerics;

namespace LuBanSuo.Core;

[Serializable]
public class BigFraction
{
	public static readonly BigFraction Zero = new BigFraction((BigInteger)0);

	public static readonly BigFraction One = new BigFraction((BigInteger)1);

	public static bool reduce = true;

	public BigInteger Numerator { get; set; }

	public BigInteger Denominator { get; set; }

	public bool Equals(BigFraction other)
	{
		if (other == null)
		{
			return false;
		}
		if (Numerator.Equals(other.Numerator))
		{
			return Denominator.Equals(other.Denominator);
		}
		return false;
	}

	public override bool Equals(object obj)
	{
		if (obj == null)
		{
			return false;
		}
		if (this == obj)
		{
			return true;
		}
		if (obj.GetType() == GetType())
		{
			return Equals((BigFraction)obj);
		}
		return false;
	}

	public override int GetHashCode()
	{
		return (17 * 23 + Numerator.GetHashCode()) * 23 + Denominator.GetHashCode();
	}

	public BigInteger GetNumerator()
	{
		return Numerator;
	}

	public BigInteger GetDenominator()
	{
		return Denominator;
	}

	public BigFraction()
	{
		Numerator = BigInteger.Zero;
		Denominator = BigInteger.One;
	}

	public BigFraction(BigInteger value)
	{
		Numerator = value;
		Denominator = BigInteger.One;
	}

	public BigFraction(double value, string description = "")
	{
		BigInteger bigInteger = (BigInteger)(value * 10000.0);
		BigInteger bigInteger2 = 10000;
		BigInteger bigInteger3 = BigInteger.GreatestCommonDivisor(bigInteger, bigInteger2);
		Numerator = bigInteger / bigInteger3;
		Denominator = bigInteger2 / bigInteger3;
	}

	public BigFraction(BigInteger numerator, BigInteger denominator, bool reduce = true)
	{
		if (denominator == 0L)
		{
			throw new ArgumentException("denominator should not be zero");
		}
		if (denominator < 0L)
		{
			numerator = -numerator;
			denominator = -denominator;
		}
		if (reduce)
		{
			BigInteger bigInteger = BigInteger.GreatestCommonDivisor(numerator, denominator);
			Numerator = numerator / bigInteger;
			Denominator = denominator / bigInteger;
		}
		else
		{
			Numerator = numerator;
			Denominator = denominator;
		}
	}

	public BigFraction Negate()
	{
		return new BigFraction(-Numerator, Denominator, reduce: false);
	}

	public static BigFraction Add(BigFraction a, BigFraction b)
	{
		BigInteger numerator = a.Numerator * b.Denominator + b.Numerator * a.Denominator;
		BigInteger denominator = a.Denominator * b.Denominator;
		return new BigFraction(numerator, denominator);
	}

	public static BigFraction Subtract(BigFraction a, BigFraction b)
	{
		BigInteger numerator = a.Numerator * b.Denominator - b.Numerator * a.Denominator;
		BigInteger denominator = a.Denominator * b.Denominator;
		return new BigFraction(numerator, denominator);
	}

	public static BigFraction Multiply(BigFraction a, BigFraction b)
	{
		BigInteger numerator = a.Numerator * b.Numerator;
		BigInteger denominator = a.Denominator * b.Denominator;
		return new BigFraction(numerator, denominator);
	}

	public static BigFraction Divide(BigFraction a, BigFraction b)
	{
		if (b.Numerator == 0L)
		{
			throw new DivideByZeroException();
		}
		return Multiply(a, new BigFraction(b.Denominator, b.Numerator));
	}

	public int CompareTo(BigFraction other)
	{
		if (other == null)
		{
			return -2;
		}
		return (Numerator * other.Denominator - other.Numerator * Denominator).Sign;
	}

	public int IntValue()
	{
		return (int)((Numerator * 2 + Denominator) / (Denominator * 2));
	}

	public double DoubleValue()
	{
		return (double)Numerator / (double)Denominator;
	}

	public override string ToString()
	{
		if (Denominator == 1L)
		{
			return Numerator.ToString();
		}
		return $"{Numerator}/{Denominator}";
	}

	public static BigFraction operator +(BigFraction a, BigFraction b)
	{
		return Add(a, b);
	}

	public static BigFraction operator -(BigFraction a, BigFraction b)
	{
		return Subtract(a, b);
	}

	public static BigFraction operator *(BigFraction a, BigFraction b)
	{
		return Multiply(a, b);
	}

	public static BigFraction operator /(BigFraction a, BigFraction b)
	{
		return Divide(a, b);
	}

	public static BigFraction operator +(BigFraction a, BigInteger b)
	{
		return Add(a, new BigFraction(b));
	}

	public static BigFraction operator -(BigFraction a, BigInteger b)
	{
		return Subtract(a, new BigFraction(b));
	}

	public static BigFraction operator *(BigFraction a, BigInteger b)
	{
		return Multiply(a, new BigFraction(b));
	}

	public static BigFraction operator /(BigFraction a, BigInteger b)
	{
		return Divide(a, new BigFraction(b));
	}

	public static BigFraction operator -(BigFraction a)
	{
		return a.Negate();
	}

	public static bool operator ==(BigFraction a, BigFraction b)
	{
		if ((object)a == b)
		{
			return true;
		}
		if ((object)a == null || (object)b == null)
		{
			return false;
		}
		return a.Equals(b);
	}

	public static bool operator !=(BigFraction a, BigFraction b)
	{
		return !(a == b);
	}
}
