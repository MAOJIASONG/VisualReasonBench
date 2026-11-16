using System;

namespace LuBanSuo.Core;

[Serializable]
public struct SwitchAxisTypeXYZ
{
	public SwitchAxisType x;

	public SwitchAxisType y;

	public SwitchAxisType z;

	public SwitchAxisTypeXYZ(SwitchAxisType x, SwitchAxisType y, SwitchAxisType z)
	{
		this.x = x;
		this.y = y;
		this.z = z;
	}

	public SwitchAxisTypeXYZ(int x, int y, int z)
	{
		this.x = (SwitchAxisType)x;
		this.y = (SwitchAxisType)y;
		this.z = (SwitchAxisType)z;
	}

	public override string ToString()
	{
		return $"{x}_{y}_{z}";
	}
}
