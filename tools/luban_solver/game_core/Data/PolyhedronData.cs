using System;
using System.Collections.Generic;

namespace LuBanSuo.Data;

[Serializable]
public class PolyhedronData
{
	public List<VertexData> vertexes;

	public List<FaceData> faces;

	public PositionData position1;

	public SwitchAxisTypeXYZData switchAxisTypeXYZ1;

	public PositionData position2;

	public SwitchAxisTypeXYZData switchAxisTypeXYZ2;

	public List<ConvexPolyhedronData> convexPolyhedrons;
}
