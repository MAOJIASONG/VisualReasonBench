using System.Collections.Generic;
using System.Numerics;
using UnityEngine;

namespace LuBanSuo.Core;

public static class OrientationHelper
{
	public static readonly SwitchAxisType[][] RotateAxisTypes;

	private static UnityEngine.Quaternion[] _quaternions;

	private static int[][] _eulerIndices;

	static OrientationHelper()
	{
		RotateAxisTypes = new SwitchAxisType[24][]
		{
			new SwitchAxisType[3]
			{
				SwitchAxisType.XPositive,
				SwitchAxisType.YPositive,
				SwitchAxisType.ZPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.XPositive,
				SwitchAxisType.YNegative,
				SwitchAxisType.ZNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.XNegative,
				SwitchAxisType.YPositive,
				SwitchAxisType.ZNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.XNegative,
				SwitchAxisType.YNegative,
				SwitchAxisType.ZPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.XPositive,
				SwitchAxisType.ZPositive,
				SwitchAxisType.YNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.XPositive,
				SwitchAxisType.ZNegative,
				SwitchAxisType.YPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.XNegative,
				SwitchAxisType.ZPositive,
				SwitchAxisType.YPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.XNegative,
				SwitchAxisType.ZNegative,
				SwitchAxisType.YNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.YPositive,
				SwitchAxisType.ZPositive,
				SwitchAxisType.XPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.YPositive,
				SwitchAxisType.ZNegative,
				SwitchAxisType.XNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.YNegative,
				SwitchAxisType.ZPositive,
				SwitchAxisType.XNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.YNegative,
				SwitchAxisType.ZNegative,
				SwitchAxisType.XPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.YPositive,
				SwitchAxisType.XPositive,
				SwitchAxisType.ZNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.YPositive,
				SwitchAxisType.XNegative,
				SwitchAxisType.ZPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.YNegative,
				SwitchAxisType.XPositive,
				SwitchAxisType.ZPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.YNegative,
				SwitchAxisType.XNegative,
				SwitchAxisType.ZNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.ZPositive,
				SwitchAxisType.XPositive,
				SwitchAxisType.YPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.ZPositive,
				SwitchAxisType.XNegative,
				SwitchAxisType.YNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.ZNegative,
				SwitchAxisType.XPositive,
				SwitchAxisType.YNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.ZNegative,
				SwitchAxisType.XNegative,
				SwitchAxisType.YPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.ZPositive,
				SwitchAxisType.YPositive,
				SwitchAxisType.XNegative
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.ZPositive,
				SwitchAxisType.YNegative,
				SwitchAxisType.XPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.ZNegative,
				SwitchAxisType.YPositive,
				SwitchAxisType.XPositive
			},
			new SwitchAxisType[3]
			{
				SwitchAxisType.ZNegative,
				SwitchAxisType.YNegative,
				SwitchAxisType.XNegative
			}
		};
		_quaternions = new UnityEngine.Quaternion[RotateAxisTypes.Length];
		_eulerIndices = new int[RotateAxisTypes.Length][];
		for (int i = 0; i <= 270; i += 90)
		{
			for (int j = 0; j <= 270; j += 90)
			{
				for (int k = 0; k <= 270; k += 90)
				{
					UnityEngine.Quaternion quaternion = UnityEngine.Quaternion.Euler(new UnityEngine.Vector3(i, j, k));
					UnityEngine.Quaternion quaternion2 = UnityEngine.Quaternion.Inverse(quaternion);
					SwitchAxisType switchAxisType = Parse(quaternion2 * UnityEngine.Vector3.right);
					SwitchAxisType switchAxisType2 = Parse(quaternion2 * UnityEngine.Vector3.up);
					SwitchAxisType switchAxisType3 = Parse(quaternion2 * UnityEngine.Vector3.forward);
					for (int l = 0; l < RotateAxisTypes.Length; l++)
					{
						if ((_quaternions[l] == default(UnityEngine.Quaternion) || _eulerIndices[l] == null) && RotateAxisTypes[l][0] == switchAxisType && RotateAxisTypes[l][1] == switchAxisType2 && RotateAxisTypes[l][2] == switchAxisType3)
						{
							_quaternions[l] = quaternion;
							_eulerIndices[l] = new int[3]
							{
								i / 90,
								j / 90,
								k / 90
							};
						}
					}
				}
			}
		}
	}

	private static SwitchAxisType Parse(UnityEngine.Vector3 v)
	{
		if (UnityEngine.Vector3.Distance(v, UnityEngine.Vector3.right) < 0.125f)
		{
			return SwitchAxisType.XPositive;
		}
		if (UnityEngine.Vector3.Distance(v, UnityEngine.Vector3.left) < 0.125f)
		{
			return SwitchAxisType.XNegative;
		}
		if (UnityEngine.Vector3.Distance(v, UnityEngine.Vector3.up) < 0.125f)
		{
			return SwitchAxisType.YPositive;
		}
		if (UnityEngine.Vector3.Distance(v, UnityEngine.Vector3.down) < 0.125f)
		{
			return SwitchAxisType.YNegative;
		}
		if (UnityEngine.Vector3.Distance(v, UnityEngine.Vector3.forward) < 0.125f)
		{
			return SwitchAxisType.ZPositive;
		}
		return SwitchAxisType.ZNegative;
	}

	public static UnityEngine.Quaternion GetQuaternion(SwitchAxisTypeXYZ xyz)
	{
		for (int i = 0; i < RotateAxisTypes.Length; i++)
		{
			if (RotateAxisTypes[i][0] == xyz.x && RotateAxisTypes[i][1] == xyz.y && RotateAxisTypes[i][2] == xyz.z)
			{
				return _quaternions[i];
			}
		}
		return UnityEngine.Quaternion.identity;
	}

	public static UnityEngine.Vector3 GetEulerAngles(SwitchAxisTypeXYZ xyz)
	{
		for (int i = 0; i < RotateAxisTypes.Length; i++)
		{
			if (RotateAxisTypes[i][0] == xyz.x && RotateAxisTypes[i][1] == xyz.y && RotateAxisTypes[i][2] == xyz.z)
			{
				return new UnityEngine.Vector3(_eulerIndices[i][0] * 90, _eulerIndices[i][1] * 90, _eulerIndices[i][2] * 90);
			}
		}
		return UnityEngine.Vector3.zero;
	}

	public static SwitchAxisTypeXYZ? FindOrientationFromEuler(UnityEngine.Vector3 euler)
	{
		int num = (Mathf.RoundToInt(euler.x / 90f) % 4 + 4) % 4;
		int num2 = (Mathf.RoundToInt(euler.y / 90f) % 4 + 4) % 4;
		int num3 = (Mathf.RoundToInt(euler.z / 90f) % 4 + 4) % 4;
		UnityEngine.Quaternion quaternion = UnityEngine.Quaternion.Inverse(UnityEngine.Quaternion.Euler(num * 90, num2 * 90, num3 * 90));
		SwitchAxisType switchAxisType = Parse(quaternion * UnityEngine.Vector3.right);
		SwitchAxisType switchAxisType2 = Parse(quaternion * UnityEngine.Vector3.up);
		SwitchAxisType switchAxisType3 = Parse(quaternion * UnityEngine.Vector3.forward);
		for (int i = 0; i < RotateAxisTypes.Length; i++)
		{
			if (RotateAxisTypes[i][0] == switchAxisType && RotateAxisTypes[i][1] == switchAxisType2 && RotateAxisTypes[i][2] == switchAxisType3)
			{
				return new SwitchAxisTypeXYZ((int)switchAxisType, (int)switchAxisType2, (int)switchAxisType3);
			}
		}
		return null;
	}

	public static Point RotatePoint(Point p, SwitchAxisTypeXYZ xyz)
	{
		BigInteger coordValue = GetCoordValue(p.X, p.Y, p.Z, xyz.x);
		BigInteger coordValue2 = GetCoordValue(p.X, p.Y, p.Z, xyz.y);
		BigInteger coordValue3 = GetCoordValue(p.X, p.Y, p.Z, xyz.z);
		return new Point(p.Index, coordValue, coordValue2, coordValue3);
	}

	public static void ApplyRotation(List<Point> points, SwitchAxisTypeXYZ xyz)
	{
		for (int i = 0; i < points.Count; i++)
		{
			Point point = points[i];
			BigInteger coordValue = GetCoordValue(point.X, point.Y, point.Z, xyz.x);
			BigInteger coordValue2 = GetCoordValue(point.X, point.Y, point.Z, xyz.y);
			BigInteger coordValue3 = GetCoordValue(point.X, point.Y, point.Z, xyz.z);
			points[i] = new Point(point.Index, coordValue, coordValue2, coordValue3);
		}
	}

	private static BigInteger GetCoordValue(BigInteger x, BigInteger y, BigInteger z, SwitchAxisType type)
	{
		return type switch
		{
			SwitchAxisType.XPositive => x, 
			SwitchAxisType.XNegative => -x, 
			SwitchAxisType.YPositive => y, 
			SwitchAxisType.YNegative => -y, 
			SwitchAxisType.ZPositive => z, 
			SwitchAxisType.ZNegative => -z, 
			_ => x, 
		};
	}
}
