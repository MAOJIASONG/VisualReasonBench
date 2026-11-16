using System;
using System.Collections.Generic;

namespace LuBanSuo.Data;

[Serializable]
public class PuzzleData
{
	public List<PolyhedronData> polyhedrons;

	public List<int> polyhedronTypes;

	public int polyhedronBaseIndex;

	public List<DirectionData> directions;
}
