namespace BurrPuzzle.DataCore
{
	public class PolyhedronInit
	{
		public readonly AnglePolyhedron AnglePolyhedron;

		public readonly PolyhedronInitAxis X = new PolyhedronInitAxis();

		public readonly PolyhedronInitAxis Y = new PolyhedronInitAxis();

		public readonly PolyhedronInitAxis Z = new PolyhedronInitAxis();

		private int GetAnglePolyhedronIndex(UnityEngine.Quaternion quaternion, IReadOnlyList<UnityEngine.Quaternion> anglePolyhedronsQuaternion)
		{
			for (int i = 0; i < anglePolyhedronsQuaternion.Count; i++)
			{
				if (UnityEngine.Quaternion.Angle(quaternion, anglePolyhedronsQuaternion[i]) < 0.125f)
				{
					return i;
				}
			}
			return -1;
		}

		public PolyhedronInit(List<Point> pointsAll, List<List<int>> polygonsPointsIndex, List<List<List<int>>> convexPolyhedraPointsIndex, UnityEngine.Quaternion quaternion, List<UnityEngine.Quaternion> anglePolyhedronsQuaternion, List<AnglePolyhedron> anglePolyhedrons)
		{
			AnglePolyhedron = new AnglePolyhedron(pointsAll, polygonsPointsIndex, convexPolyhedraPointsIndex);
			anglePolyhedronsQuaternion.Add(quaternion);
			anglePolyhedrons.Add(AnglePolyhedron);
			X.PositiveWise.Add(AnglePolyhedron);
			X.NegativeWise.Add(AnglePolyhedron);
			int i = 1;
			while (i <= 16)
			{
				UnityEngine.Quaternion quaternion2 = UnityEngine.Quaternion.AngleAxis(90f * (float)i / 16f, UnityEngine.Vector3.right) * quaternion;
				int anglePolyhedronIndex = GetAnglePolyhedronIndex(quaternion2, anglePolyhedronsQuaternion);
				AnglePolyhedron item;
				if (anglePolyhedronIndex != -1)
				{
					item = anglePolyhedrons[anglePolyhedronIndex];
				}
				else
				{
					item = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateX(i)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
					anglePolyhedronsQuaternion.Add(quaternion2);
					anglePolyhedrons.Add(item);
				}
				X.PositiveWise.Add(item);
				UnityEngine.Quaternion quaternion3 = UnityEngine.Quaternion.AngleAxis(90f * (float)(-i) / 16f, UnityEngine.Vector3.right) * quaternion;
				int anglePolyhedronIndex2 = GetAnglePolyhedronIndex(quaternion3, anglePolyhedronsQuaternion);
				if (anglePolyhedronIndex2 != -1)
				{
					item = anglePolyhedrons[anglePolyhedronIndex2];
				}
				else
				{
					item = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateX(-i)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
					anglePolyhedronsQuaternion.Add(quaternion3);
					anglePolyhedrons.Add(item);
				}
				X.NegativeWise.Add(item);
				int num = i + 1;
				i = num;
			}
			Y.PositiveWise.Add(AnglePolyhedron);
			Y.NegativeWise.Add(AnglePolyhedron);
			int i2 = 1;
			while (i2 <= 16)
			{
				UnityEngine.Quaternion quaternion4 = UnityEngine.Quaternion.AngleAxis(90f * (float)i2 / 16f, UnityEngine.Vector3.up) * quaternion;
				int anglePolyhedronIndex3 = GetAnglePolyhedronIndex(quaternion4, anglePolyhedronsQuaternion);
				AnglePolyhedron item2;
				if (anglePolyhedronIndex3 != -1)
				{
					item2 = anglePolyhedrons[anglePolyhedronIndex3];
				}
				else
				{
					item2 = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateY(i2)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
					anglePolyhedronsQuaternion.Add(quaternion4);
					anglePolyhedrons.Add(item2);
				}
				Y.PositiveWise.Add(item2);
				UnityEngine.Quaternion quaternion5 = UnityEngine.Quaternion.AngleAxis(90f * (float)(-i2) / 16f, UnityEngine.Vector3.up) * quaternion;
				int anglePolyhedronIndex4 = GetAnglePolyhedronIndex(quaternion5, anglePolyhedronsQuaternion);
				if (anglePolyhedronIndex4 != -1)
				{
					item2 = anglePolyhedrons[anglePolyhedronIndex4];
				}
				else
				{
					item2 = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateY(-i2)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
					anglePolyhedronsQuaternion.Add(quaternion5);
					anglePolyhedrons.Add(item2);
				}
				Y.NegativeWise.Add(item2);
				int num = i2 + 1;
				i2 = num;
			}
			Z.PositiveWise.Add(AnglePolyhedron);
			Z.NegativeWise.Add(AnglePolyhedron);
			int i3 = 1;
			while (i3 <= 16)
			{
				UnityEngine.Quaternion quaternion6 = UnityEngine.Quaternion.AngleAxis(90f * (float)i3 / 16f, UnityEngine.Vector3.forward) * quaternion;
				int anglePolyhedronIndex5 = GetAnglePolyhedronIndex(quaternion6, anglePolyhedronsQuaternion);
				AnglePolyhedron item3;
				if (anglePolyhedronIndex5 != -1)
				{
					item3 = anglePolyhedrons[anglePolyhedronIndex5];
				}
				else
				{
					item3 = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateZ(i3)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
					anglePolyhedronsQuaternion.Add(quaternion6);
					anglePolyhedrons.Add(item3);
				}
				Z.PositiveWise.Add(item3);
				UnityEngine.Quaternion quaternion7 = UnityEngine.Quaternion.AngleAxis(90f * (float)(-i3) / 16f, UnityEngine.Vector3.forward) * quaternion;
				int anglePolyhedronIndex6 = GetAnglePolyhedronIndex(quaternion7, anglePolyhedronsQuaternion);
				if (anglePolyhedronIndex6 != -1)
				{
					item3 = anglePolyhedrons[anglePolyhedronIndex6];
				}
				else
				{
					item3 = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateZ(-i3)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
					anglePolyhedronsQuaternion.Add(quaternion7);
					anglePolyhedrons.Add(item3);
				}
				Z.NegativeWise.Add(item3);
				int num = i3 + 1;
				i3 = num;
			}
		}

		public PolyhedronInit(List<Point> pointsAll, List<List<int>> polygonsPointsIndex, List<List<List<int>>> convexPolyhedraPointsIndex, UnityEngine.Quaternion quaternion, List<UnityEngine.Quaternion> anglePolyhedronsQuaternion, List<AnglePolyhedron> anglePolyhedrons, PolyhedronInit polyhedronInit)
		{
			AnglePolyhedron = new AnglePolyhedron(pointsAll, polygonsPointsIndex, convexPolyhedraPointsIndex);
			anglePolyhedronsQuaternion.Add(quaternion);
			anglePolyhedrons.Add(AnglePolyhedron);
			X.PositiveWise.Add(AnglePolyhedron);
			X.NegativeWise.Add(AnglePolyhedron);
			int i = 1;
			while (i <= 16)
			{
				UnityEngine.Quaternion quaternion2 = UnityEngine.Quaternion.AngleAxis(90f * (float)i / 16f, UnityEngine.Vector3.right) * quaternion;
				AnglePolyhedron item;
				if (i != 8)
				{
					item = polyhedronInit.X.PositiveWise[i];
					anglePolyhedronsQuaternion.Add(quaternion2);
					anglePolyhedrons.Add(item);
				}
				else
				{
					int anglePolyhedronIndex = GetAnglePolyhedronIndex(quaternion2, anglePolyhedronsQuaternion);
					if (anglePolyhedronIndex != -1)
					{
						item = anglePolyhedrons[anglePolyhedronIndex];
					}
					else
					{
						item = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateX(i)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
						anglePolyhedronsQuaternion.Add(quaternion2);
						anglePolyhedrons.Add(item);
					}
				}
				X.PositiveWise.Add(item);
				UnityEngine.Quaternion quaternion3 = UnityEngine.Quaternion.AngleAxis(90f * (float)(-i) / 16f, UnityEngine.Vector3.right) * quaternion;
				if (i != 8)
				{
					item = polyhedronInit.X.NegativeWise[i];
					anglePolyhedronsQuaternion.Add(quaternion3);
					anglePolyhedrons.Add(item);
				}
				else
				{
					int anglePolyhedronIndex2 = GetAnglePolyhedronIndex(quaternion3, anglePolyhedronsQuaternion);
					if (anglePolyhedronIndex2 != -1)
					{
						item = anglePolyhedrons[anglePolyhedronIndex2];
					}
					else
					{
						item = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateX(-i)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
						anglePolyhedronsQuaternion.Add(quaternion3);
						anglePolyhedrons.Add(item);
					}
				}
				X.NegativeWise.Add(item);
				int num = i + 1;
				i = num;
			}
			Y.PositiveWise.Add(AnglePolyhedron);
			Y.NegativeWise.Add(AnglePolyhedron);
			int i2 = 1;
			while (i2 <= 16)
			{
				UnityEngine.Quaternion quaternion4 = UnityEngine.Quaternion.AngleAxis(90f * (float)i2 / 16f, UnityEngine.Vector3.up) * quaternion;
				AnglePolyhedron item2;
				if (i2 != 8)
				{
					item2 = polyhedronInit.Y.PositiveWise[i2];
					anglePolyhedronsQuaternion.Add(quaternion4);
					anglePolyhedrons.Add(item2);
				}
				else
				{
					int anglePolyhedronIndex3 = GetAnglePolyhedronIndex(quaternion4, anglePolyhedronsQuaternion);
					if (anglePolyhedronIndex3 != -1)
					{
						item2 = anglePolyhedrons[anglePolyhedronIndex3];
					}
					else
					{
						item2 = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateY(i2)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
						anglePolyhedronsQuaternion.Add(quaternion4);
						anglePolyhedrons.Add(item2);
					}
				}
				Y.PositiveWise.Add(item2);
				UnityEngine.Quaternion quaternion5 = UnityEngine.Quaternion.AngleAxis(90f * (float)(-i2) / 16f, UnityEngine.Vector3.up) * quaternion;
				if (i2 != 8)
				{
					item2 = polyhedronInit.Y.NegativeWise[i2];
					anglePolyhedronsQuaternion.Add(quaternion5);
					anglePolyhedrons.Add(item2);
				}
				else
				{
					int anglePolyhedronIndex4 = GetAnglePolyhedronIndex(quaternion5, anglePolyhedronsQuaternion);
					if (anglePolyhedronIndex4 != -1)
					{
						item2 = anglePolyhedrons[anglePolyhedronIndex4];
					}
					else
					{
						item2 = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateY(-i2)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
						anglePolyhedronsQuaternion.Add(quaternion5);
						anglePolyhedrons.Add(item2);
					}
				}
				Y.NegativeWise.Add(item2);
				int num = i2 + 1;
				i2 = num;
			}
			Z.PositiveWise.Add(AnglePolyhedron);
			Z.NegativeWise.Add(AnglePolyhedron);
			int i3 = 1;
			while (i3 <= 16)
			{
				UnityEngine.Quaternion quaternion6 = UnityEngine.Quaternion.AngleAxis(90f * (float)i3 / 16f, UnityEngine.Vector3.forward) * quaternion;
				AnglePolyhedron item3;
				if (i3 != 8)
				{
					item3 = polyhedronInit.Z.PositiveWise[i3];
					anglePolyhedronsQuaternion.Add(quaternion6);
					anglePolyhedrons.Add(item3);
				}
				else
				{
					int anglePolyhedronIndex5 = GetAnglePolyhedronIndex(quaternion6, anglePolyhedronsQuaternion);
					if (anglePolyhedronIndex5 != -1)
					{
						item3 = anglePolyhedrons[anglePolyhedronIndex5];
					}
					else
					{
						item3 = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateZ(i3)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
						anglePolyhedronsQuaternion.Add(quaternion6);
						anglePolyhedrons.Add(item3);
					}
				}
				Z.PositiveWise.Add(item3);
				UnityEngine.Quaternion quaternion7 = UnityEngine.Quaternion.AngleAxis(90f * (float)(-i3) / 16f, UnityEngine.Vector3.forward) * quaternion;
				if (i3 != 8)
				{
					item3 = polyhedronInit.Z.NegativeWise[i3];
					anglePolyhedronsQuaternion.Add(quaternion7);
					anglePolyhedrons.Add(item3);
				}
				else
				{
					int anglePolyhedronIndex6 = GetAnglePolyhedronIndex(quaternion7, anglePolyhedronsQuaternion);
					if (anglePolyhedronIndex6 != -1)
					{
						item3 = anglePolyhedrons[anglePolyhedronIndex6];
					}
					else
					{
						item3 = new AnglePolyhedron(pointsAll.Select((Point point) => point.RotateZ(-i3)).ToList(), polygonsPointsIndex, convexPolyhedraPointsIndex);
						anglePolyhedronsQuaternion.Add(quaternion7);
						anglePolyhedrons.Add(item3);
					}
				}
				Z.NegativeWise.Add(item3);
				int num = i3 + 1;
				i3 = num;
			}
		}
	}
}
