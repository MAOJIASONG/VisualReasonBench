namespace BurrPuzzle.DataCore
{
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
			return (float)(X.DoubleValue() * 1.0 / (double)ScaleFactor);
		}

		public float GetRealY()
		{
			return (float)(Y.DoubleValue() * 1.0 / (double)ScaleFactor);
		}

		public float GetRealZ()
		{
			return (float)(Z.DoubleValue() * 1.0 / (double)ScaleFactor);
		}

		private BigFraction GetSwitchAxisValue(SwitchAxisType rotateAxisType)
		{
			return rotateAxisType switch
			{
				SwitchAxisType.XPositive => X, 
				SwitchAxisType.XNegative => -X, 
				SwitchAxisType.YPositive => Y, 
				SwitchAxisType.YNegative => -Y, 
				SwitchAxisType.ZPositive => Z, 
				_ => -Z, 
			};
		}

		public Position SwitchXYZAxis(SwitchAxisType xAxis, SwitchAxisType yAxis, SwitchAxisType zAxis)
		{
			return new Position(GetSwitchAxisValue(xAxis), GetSwitchAxisValue(yAxis), GetSwitchAxisValue(zAxis));
		}

		public static SwitchAxisType GetSwitchXYZAxisReverse(SwitchAxisType axisTypePositive, SwitchAxisType axisTypeNegative, SwitchAxisType xRotateAxis, SwitchAxisType yRotateAxis, SwitchAxisType zRotateAxis)
		{
			if (xRotateAxis == axisTypePositive)
			{
				return SwitchAxisType.XPositive;
			}
			if (xRotateAxis == axisTypeNegative)
			{
				return SwitchAxisType.XNegative;
			}
			if (yRotateAxis == axisTypePositive)
			{
				return SwitchAxisType.YPositive;
			}
			if (yRotateAxis == axisTypeNegative)
			{
				return SwitchAxisType.YNegative;
			}
			if (zRotateAxis == axisTypePositive)
			{
				return SwitchAxisType.ZPositive;
			}
			return SwitchAxisType.ZNegative;
		}

		public Position SwitchXYZAxisReverse(SwitchAxisType xAxisType, SwitchAxisType yAxisType, SwitchAxisType zAxisType)
		{
			SwitchAxisType switchXYZAxisReverse = GetSwitchXYZAxisReverse(SwitchAxisType.XPositive, SwitchAxisType.XNegative, xAxisType, yAxisType, zAxisType);
			SwitchAxisType switchXYZAxisReverse2 = GetSwitchXYZAxisReverse(SwitchAxisType.YPositive, SwitchAxisType.YNegative, xAxisType, yAxisType, zAxisType);
			SwitchAxisType switchXYZAxisReverse3 = GetSwitchXYZAxisReverse(SwitchAxisType.ZPositive, SwitchAxisType.ZNegative, xAxisType, yAxisType, zAxisType);
			return new Position(GetSwitchAxisValue(switchXYZAxisReverse), GetSwitchAxisValue(switchXYZAxisReverse2), GetSwitchAxisValue(switchXYZAxisReverse3));
		}

		public static SwitchAxisType GetReverseSwitchAxisType(SwitchAxisType axisType)
		{
			return axisType switch
			{
				SwitchAxisType.XPositive => SwitchAxisType.XNegative, 
				SwitchAxisType.XNegative => SwitchAxisType.XPositive, 
				SwitchAxisType.YPositive => SwitchAxisType.YNegative, 
				SwitchAxisType.YNegative => SwitchAxisType.YPositive, 
				SwitchAxisType.ZPositive => SwitchAxisType.ZNegative, 
				SwitchAxisType.ZNegative => SwitchAxisType.ZPositive, 
				_ => SwitchAxisType.None, 
			};
		}

		public static SwitchAxisType GetSwitchXYZAxis(SwitchAxisType xAxis1, SwitchAxisType yAxis1, SwitchAxisType zAxis1, SwitchAxisType axis2)
		{
			return axis2 switch
			{
				SwitchAxisType.XPositive => xAxis1, 
				SwitchAxisType.XNegative => GetReverseSwitchAxisType(xAxis1), 
				SwitchAxisType.YPositive => yAxis1, 
				SwitchAxisType.YNegative => GetReverseSwitchAxisType(yAxis1), 
				SwitchAxisType.ZPositive => zAxis1, 
				SwitchAxisType.ZNegative => GetReverseSwitchAxisType(zAxis1), 
				_ => SwitchAxisType.None, 
			};
		}
	}
}
