namespace BurrPuzzle.DataCore
{
	public class DataTools
	{
		public PuzzleData PuzzleData;

		public int StateId;

		public List<Polyhedron> PolyhedronList;

		private float[] _polyhedronLoadPercents;

		private List<Direction> _directionList;

		public int PolyhedronBaseIndex = -1;

		public int MoveRotatePolyhedronIndex = -1;

		private SwitchAxisTypeXYZ _moveRotateTransformXYZ;

		public RotateAxis MoveRotateAxis;

		private readonly List<Thread> _threads = new List<Thread>();

		private const double DiffPointMax = 1250.0;

		public readonly List<Dictionary<int, int>> MeshVertex2CorePoint = new List<Dictionary<int, int>>();

		public readonly List<Dictionary<int, int>> CorePoint2MeshVertex = new List<Dictionary<int, int>>();

		private const string PuzzleDataFileNameFormat = "puzzle_{0:D3}.json";

		private static string GetPuzzleDataFile(int index)
		{
			return Path.Combine("Assets/Scripts/PuzzlesJson", $"puzzle_{index:D3}.json");
		}

		private bool IsVertexMatchPoint(UnityEngine.Vector3 vertex, Point point)
		{
			if ((double)Math.Abs(vertex.x * 10000f - (float)point.X) < 1250.0 && (double)Math.Abs(vertex.y * 10000f - (float)point.Y) < 1250.0)
			{
				return (double)Math.Abs(vertex.z * 10000f - (float)point.Z) < 1250.0;
			}
			return false;
		}

		private static SwitchAxisType GetRotateAxis(RotateAxis halfRotateAxisBase)
		{
			return halfRotateAxisBase switch
			{
				RotateAxis.XPositiveWise => SwitchAxisType.XPositive, 
				RotateAxis.XNegativeWise => SwitchAxisType.XNegative, 
				RotateAxis.YPositiveWise => SwitchAxisType.YPositive, 
				RotateAxis.YNegativeWise => SwitchAxisType.YNegative, 
				RotateAxis.ZPositiveWise => SwitchAxisType.ZPositive, 
				RotateAxis.ZNegativeWise => SwitchAxisType.ZNegative, 
				_ => SwitchAxisType.None, 
			};
		}

		private static RotateAxis TransformRotateAxis(SwitchAxisTypeXYZ transformXYZ, RotateAxis rotateAxisBase)
		{
			SwitchAxisType rotateAxis = GetRotateAxis(rotateAxisBase);
			if (rotateAxis == transformXYZ.x)
			{
				return RotateAxis.XPositiveWise;
			}
			if (rotateAxis == Polyhedron.ReverseXYZAxis(transformXYZ.x))
			{
				return RotateAxis.XNegativeWise;
			}
			if (rotateAxis == transformXYZ.y)
			{
				return RotateAxis.YPositiveWise;
			}
			if (rotateAxis == Polyhedron.ReverseXYZAxis(transformXYZ.y))
			{
				return RotateAxis.YNegativeWise;
			}
			if (rotateAxis == transformXYZ.z)
			{
				return RotateAxis.ZPositiveWise;
			}
			if (rotateAxis == Polyhedron.ReverseXYZAxis(transformXYZ.z))
			{
				return RotateAxis.ZNegativeWise;
			}
			return RotateAxis.None;
		}

		private bool CheckPolyhedronsSwitchAxisType(Polyhedron polyhedron, SwitchAxisTypeXYZ transformXYZ, SwitchAxisTypeXYZ switchAxisTypeXYZNow, RotateAxis halfRotateAxisNow, SwitchAxisTypeXYZ switchAxisTypeXYZBase, RotateAxis halfRotateAxisBase)
		{
			if (PuzzleData.polyhedronTypes[polyhedron.Index] >= 100)
			{
				return polyhedron.HalfRotateAxis == RotateAxis.None;
			}
			SwitchAxisTypeXYZ switchAxisTypeXYZ = new SwitchAxisTypeXYZ
			{
				x = Position.GetSwitchXYZAxis(switchAxisTypeXYZNow.x, switchAxisTypeXYZNow.y, switchAxisTypeXYZNow.z, transformXYZ.x),
				y = Position.GetSwitchXYZAxis(switchAxisTypeXYZNow.x, switchAxisTypeXYZNow.y, switchAxisTypeXYZNow.z, transformXYZ.y),
				z = Position.GetSwitchXYZAxis(switchAxisTypeXYZNow.x, switchAxisTypeXYZNow.y, switchAxisTypeXYZNow.z, transformXYZ.z)
			};
			if (polyhedron.GetAngleXYZPolyhedronsInitType(switchAxisTypeXYZ) != polyhedron.GetAngleXYZPolyhedronsInitType(switchAxisTypeXYZBase))
			{
				return false;
			}
			SwitchAxisType rotateAxis = GetRotateAxis(halfRotateAxisNow);
			SwitchAxisType rotateAxis2 = GetRotateAxis(halfRotateAxisBase);
			return Position.GetSwitchXYZAxis(transformXYZ.x, transformXYZ.y, transformXYZ.z, rotateAxis) == rotateAxis2;
		}

		private static SwitchAxisTypeXYZ GetTransformAxisTypeXYZ(SwitchAxisTypeXYZ axisTypeXYZ1, SwitchAxisTypeXYZ axisTypeXYZ2)
		{
			SwitchAxisType x = axisTypeXYZ1.x;
			SwitchAxisType y = axisTypeXYZ1.y;
			SwitchAxisType z = axisTypeXYZ1.z;
			SwitchAxisType switchXYZAxisReverse = Position.GetSwitchXYZAxisReverse(SwitchAxisType.XPositive, SwitchAxisType.XNegative, x, y, z);
			SwitchAxisType switchXYZAxisReverse2 = Position.GetSwitchXYZAxisReverse(SwitchAxisType.YPositive, SwitchAxisType.YNegative, x, y, z);
			SwitchAxisType switchXYZAxisReverse3 = Position.GetSwitchXYZAxisReverse(SwitchAxisType.ZPositive, SwitchAxisType.ZNegative, x, y, z);
			SwitchAxisType x2 = axisTypeXYZ2.x;
			SwitchAxisType y2 = axisTypeXYZ2.y;
			SwitchAxisType z2 = axisTypeXYZ2.z;
			SwitchAxisType switchXYZAxis = Position.GetSwitchXYZAxis(switchXYZAxisReverse, switchXYZAxisReverse2, switchXYZAxisReverse3, x2);
			SwitchAxisType switchXYZAxis2 = Position.GetSwitchXYZAxis(switchXYZAxisReverse, switchXYZAxisReverse2, switchXYZAxisReverse3, y2);
			SwitchAxisType switchXYZAxis3 = Position.GetSwitchXYZAxis(switchXYZAxisReverse, switchXYZAxisReverse2, switchXYZAxisReverse3, z2);
			return new SwitchAxisTypeXYZ
			{
				x = switchXYZAxis,
				y = switchXYZAxis2,
				z = switchXYZAxis3
			};
		}

		private bool CheckPolyhedronsPosition(SwitchAxisTypeXYZ transformAxisTypeXYZ, BigFraction diffXBase, BigFraction diffYBase, BigFraction diffZBase, Position positionNow, Position positionsBase)
		{
			Position position = positionNow.SwitchXYZAxis(transformAxisTypeXYZ.x, transformAxisTypeXYZ.y, transformAxisTypeXYZ.z);
			BigFraction bigFraction = position.X - positionsBase.X;
			BigFraction bigFraction2 = position.Y - positionsBase.Y;
			BigFraction bigFraction3 = position.Z - positionsBase.Z;
			if (bigFraction.CompareTo(diffXBase) == 0 && bigFraction2.CompareTo(diffYBase) == 0)
			{
				return bigFraction3.CompareTo(diffZBase) == 0;
			}
			return false;
		}

		private int GetNowIndexByBaseIndex(int baseIndex, IReadOnlyList<PolyhedronCondition> polyhedronConditions)
		{
			return polyhedronConditions?[baseIndex].polyhedronIndex ?? baseIndex;
		}

		public bool CheckPolyhedronsConditionType(int baseType, IReadOnlyList<Position> positionsNow, IReadOnlyList<SwitchAxisTypeXYZ> switchAxisTypeXYZsNow, IReadOnlyList<RotateAxis> halfRotateAxisNow, IReadOnlyList<Position> positionsBase, IReadOnlyList<SwitchAxisTypeXYZ> switchAxisTypeXYZsBase, IReadOnlyList<RotateAxis> halfRotateAxisBase, List<PolyhedronCondition> polyhedronConditions = null, int moveRotatePolyhedronIndex = -1)
		{
			if (positionsNow.Count != switchAxisTypeXYZsNow.Count || switchAxisTypeXYZsNow.Count != halfRotateAxisNow.Count)
			{
				return false;
			}
			if (positionsBase.Count != switchAxisTypeXYZsBase.Count || switchAxisTypeXYZsBase.Count != halfRotateAxisBase.Count)
			{
				return false;
			}
			if (polyhedronConditions != null && polyhedronConditions.Count != positionsBase.Count)
			{
				return false;
			}
			if (polyhedronConditions == null)
			{
				if (positionsNow.Count != positionsBase.Count || positionsNow.Count != PolyhedronList.Count)
				{
					return false;
				}
				for (int i = 0; i < PolyhedronList.Count; i++)
				{
					if (PuzzleData.polyhedronTypes[i] == baseType && CheckPolyhedronsConditionIndex(i, positionsNow, switchAxisTypeXYZsNow, halfRotateAxisNow, positionsBase, switchAxisTypeXYZsBase, halfRotateAxisBase, polyhedronConditions))
					{
						return true;
					}
				}
			}
			else
			{
				for (int j = 0; j < polyhedronConditions.Count; j++)
				{
					PolyhedronCondition polyhedronCondition = polyhedronConditions[j];
					if (PuzzleData.polyhedronTypes[polyhedronCondition.polyhedronIndex] == baseType && CheckPolyhedronsConditionIndex(j, positionsNow, switchAxisTypeXYZsNow, halfRotateAxisNow, positionsBase, switchAxisTypeXYZsBase, halfRotateAxisBase, polyhedronConditions, moveRotatePolyhedronIndex))
					{
						return true;
					}
				}
			}
			return false;
		}

		private bool CheckPolyhedronsConditionIndex(int baseIndex, IReadOnlyList<Position> positionsNow, IReadOnlyList<SwitchAxisTypeXYZ> switchAxisTypeXYZsNow, IReadOnlyList<RotateAxis> halfRotateAxisNow, IReadOnlyList<Position> positionsBase, IReadOnlyList<SwitchAxisTypeXYZ> switchAxisTypeXYZsBase, IReadOnlyList<RotateAxis> halfRotateAxisBase, IReadOnlyList<PolyhedronCondition> polyhedronConditions = null, int moveRotatePolyhedronIndex = -1)
		{
			int nowIndexByBaseIndex = GetNowIndexByBaseIndex(baseIndex, polyhedronConditions);
			Polyhedron polyhedron = PolyhedronList[nowIndexByBaseIndex];
			int angleXYZPolyhedronsInitType = polyhedron.GetAngleXYZPolyhedronsInitType(switchAxisTypeXYZsBase[baseIndex]);
			SwitchAxisType[][] rotateAxisTypes = Polyhedron.RotateAxisTypes;
			foreach (SwitchAxisType[] array in rotateAxisTypes)
			{
				SwitchAxisTypeXYZ switchAxisTypeXYZ = new SwitchAxisTypeXYZ
				{
					x = array[0],
					y = array[1],
					z = array[2]
				};
				int angleXYZPolyhedronsInitType2 = polyhedron.GetAngleXYZPolyhedronsInitType(switchAxisTypeXYZ);
				if (angleXYZPolyhedronsInitType != angleXYZPolyhedronsInitType2)
				{
					continue;
				}
				SwitchAxisTypeXYZ transformAxisTypeXYZ = GetTransformAxisTypeXYZ(switchAxisTypeXYZsNow[nowIndexByBaseIndex], switchAxisTypeXYZ);
				SwitchAxisType x = transformAxisTypeXYZ.x;
				SwitchAxisType y = transformAxisTypeXYZ.y;
				SwitchAxisType z = transformAxisTypeXYZ.z;
				Position position = positionsBase[baseIndex];
				Position position2 = positionsNow[nowIndexByBaseIndex].SwitchXYZAxis(x, y, z);
				BigFraction diffXBase = position2.X - position.X;
				BigFraction diffYBase = position2.Y - position.Y;
				BigFraction diffZBase = position2.Z - position.Z;
				bool[] array2 = new bool[positionsNow.Count];
				bool flag = true;
				for (int j = 0; j < positionsBase.Count; j++)
				{
					int nowIndexByBaseIndex2 = GetNowIndexByBaseIndex(j, polyhedronConditions);
					bool flag2 = false;
					for (int k = 0; k < positionsNow.Count; k++)
					{
						if (!array2[k] && PuzzleData.polyhedronTypes[nowIndexByBaseIndex2] == PuzzleData.polyhedronTypes[k] && CheckPolyhedronsPosition(transformAxisTypeXYZ, diffXBase, diffYBase, diffZBase, positionsNow[k], positionsBase[j]) && CheckPolyhedronsSwitchAxisType(PolyhedronList[k], transformAxisTypeXYZ, switchAxisTypeXYZsNow[k], halfRotateAxisNow[k], switchAxisTypeXYZsBase[j], halfRotateAxisBase[j]))
						{
							flag2 = true;
							array2[k] = true;
							if (nowIndexByBaseIndex2 == moveRotatePolyhedronIndex)
							{
								MoveRotatePolyhedronIndex = k;
							}
							break;
						}
					}
					if (!flag2)
					{
						flag = false;
						break;
					}
				}
				if (flag)
				{
					_moveRotateTransformXYZ = GetTransformAxisTypeXYZ(switchAxisTypeXYZ, switchAxisTypeXYZsNow[nowIndexByBaseIndex]);
					return true;
				}
			}
			return false;
		}

		public bool BuildVertexPointDictionary(IReadOnlyList<UnityEngine.Vector3> vertices, int index)
		{
			MeshVertex2CorePoint[index].Clear();
			CorePoint2MeshVertex[index].Clear();
			List<Point> points = PolyhedronList[index].AngleXYZPolyhedronsInit.First().AnglePolyhedron.Points;
			for (int i = 0; i < vertices.Count; i++)
			{
				bool flag = false;
				for (int j = 0; j < points.Count; j++)
				{
					if (IsVertexMatchPoint(vertices[i], points[j]))
					{
						MeshVertex2CorePoint[index][i] = j;
						CorePoint2MeshVertex[index][j] = i;
						flag = true;
						break;
					}
				}
				if (!flag)
				{
					return false;
				}
			}
			return true;
		}

		public int GetPolygonEdgeIndex(int vertex1, int vertex2, int polyhedronIndex)
		{
			int num = MeshVertex2CorePoint[polyhedronIndex][vertex1];
			int num2 = MeshVertex2CorePoint[polyhedronIndex][vertex2];
			foreach (Polygon polygon in PolyhedronList[polyhedronIndex].AngleXYZPolyhedronsInit.First().AnglePolyhedron.Polygons)
			{
				foreach (LineSegment lineSegment in polygon.LineSegments)
				{
					if ((lineSegment.APoint.Index == num && lineSegment.BPoint.Index == num2) || (lineSegment.APoint.Index == num2 && lineSegment.BPoint.Index == num))
					{
						return lineSegment.Direction.Index;
					}
				}
			}
			return -1;
		}

		public Polygon GetPolygon(int vertex1, int vertex2, int vertex3, int polyhedronIndex)
		{
			int point1 = MeshVertex2CorePoint[polyhedronIndex][vertex1];
			int point2 = MeshVertex2CorePoint[polyhedronIndex][vertex2];
			int point3 = MeshVertex2CorePoint[polyhedronIndex][vertex3];
			return PolyhedronList[polyhedronIndex].AngleXYZPolyhedronsInit.First().AnglePolyhedron.Polygons.First((Polygon polygon) => polygon.ContainTriangle(point1, point2, point3));
		}

		public void LoadPolyhedrons()
		{
			UnityEngine.Debug.Log(DateTime.Now.ToString() + ": load polyhedrons start...");
			PolyhedronList = new List<Polyhedron>();
			_polyhedronLoadPercents = new float[PuzzleData.polyhedrons.Count];
			for (int i = 0; i < PuzzleData.polyhedrons.Count; i++)
			{
				PolyhedronList.Add(null);
				_polyhedronLoadPercents[i] = 0f;
				MeshVertex2CorePoint.Add(new Dictionary<int, int>());
				CorePoint2MeshVertex.Add(new Dictionary<int, int>());
			}
			int num = -1;
			for (int j = 0; j < PuzzleData.polyhedrons.Count; j++)
			{
				if (PuzzleData.polyhedronTypes[j] == num)
				{
					continue;
				}
				num = PuzzleData.polyhedronTypes[j];
				PolyhedronData polyhedronData = PuzzleData.polyhedrons[j];
				List<Point> points = polyhedronData.vertexes.Select((VertexData point, int index) => new Point(index, -point.x, point.y, point.z, fromJson: true)).ToList();
				PositionData position = polyhedronData.position1;
				Position position2 = new Position(position.x, position.y, position.z, fromJson: true);
				PositionData position3 = polyhedronData.position2;
				Position position4 = new Position(position3.x, position3.y, position3.z, fromJson: true);
				List<List<int>> polygonsPointsIndex = polyhedronData.faces.Select((FaceData face) => face.vertexIndexes.ToList()).ToList();
				List<List<List<int>>> convexPolyhedraPointsIndex = new List<List<List<int>>>();
				List<ConvexPolyhedronData> convexPolyhedrons = polyhedronData.convexPolyhedrons;
				for (int num2 = 0; num2 < convexPolyhedrons.Count; num2++)
				{
					List<List<int>> item = convexPolyhedrons[num2].faces.Select((FaceData face) => face.vertexIndexes.ToList()).ToList();
					convexPolyhedraPointsIndex.Add(item);
				}
				int tmpIndex = j;
				Thread thread = new Thread((ThreadStart)delegate
				{
					Polyhedron polyhedron = new Polyhedron(tmpIndex, _polyhedronLoadPercents, points, polygonsPointsIndex, convexPolyhedraPointsIndex, position2, position4, polyhedronData.switchAxisTypeXYZ1, polyhedronData.switchAxisTypeXYZ2);
					PolyhedronList[tmpIndex] = polyhedron;
					_polyhedronLoadPercents[tmpIndex] = 1f;
					int num3 = PuzzleData.polyhedronTypes[tmpIndex];
					for (int k = tmpIndex + 1; k < PuzzleData.polyhedronTypes.Count && num3 == PuzzleData.polyhedronTypes[k]; k++)
					{
						PolyhedronData polyhedronData2 = PuzzleData.polyhedrons[k];
						PositionData position5 = polyhedronData2.position1;
						Position position6 = new Position(position5.x, position5.y, position5.z, fromJson: true);
						PositionData position7 = polyhedronData2.position2;
						Position position8 = new Position(position7.x, position7.y, position7.z, fromJson: true);
						Polyhedron value = new Polyhedron(k, polyhedron.AngleXYZPolyhedronsInit, polyhedron.AngleXYZPolyhedronsInitType, position6, position8, polyhedronData2.switchAxisTypeXYZ1, polyhedronData2.switchAxisTypeXYZ2);
						PolyhedronList[k] = value;
					}
				});
				_threads.Add(thread);
				thread.Start();
			}
			PolyhedronBaseIndex = PuzzleData.polyhedronBaseIndex;
			StateId = 0;
		}

		public bool IsAlreadyLoaded()
		{
			if (PolyhedronBaseIndex >= 0)
			{
				return _threads.All((Thread thread) => !thread.IsAlive);
			}
			return false;
		}

		public float GetPercent()
		{
			int num = 0;
			int lastPolyhedronType = -1;
			foreach (int item in PuzzleData.polyhedronTypes.Where((int t) => t != lastPolyhedronType))
			{
				num++;
				lastPolyhedronType = item;
			}
			if (PolyhedronBaseIndex != -1)
			{
				return Math.Clamp(_polyhedronLoadPercents.Sum() / (float)num, 0f, 1f);
			}
			return 0f;
		}

		public void LoadDirections()
		{
			_directionList = new List<Direction>();
			for (int i = 0; i < PuzzleData.directions.Count; i++)
			{
				DirectionData directionData = PuzzleData.directions[i];
				Direction item = new Direction(i, directionData.x, directionData.y, directionData.z);
				_directionList.Add(item);
			}
		}

		private static BigFraction Move1BeforeCross(Polyhedron polyhedron1, Direction direction, List<Polyhedron> polyhedronList, ICollection<Polyhedron> crossPolyhedronList)
		{
			BigFraction bigFraction = Polyhedron.MoveMaxPositive;
			foreach (Polyhedron polyhedron2 in polyhedronList)
			{
				if (polyhedron2.Index == polyhedron1.Index)
				{
					continue;
				}
				BigFraction bigFraction2 = Polyhedron.MoveBeforeIntersection(polyhedron1, polyhedron2, direction);
				if (bigFraction2 == null)
				{
					continue;
				}
				if (bigFraction.CompareTo(bigFraction2) > 0)
				{
					bigFraction = bigFraction2;
					if (crossPolyhedronList != null)
					{
						crossPolyhedronList.Clear();
						crossPolyhedronList.Add(polyhedron2);
					}
				}
				else if (bigFraction.CompareTo(bigFraction2) == 0)
				{
					crossPolyhedronList?.Add(polyhedron2);
				}
			}
			return bigFraction;
		}

		private static BigFraction Move1SBeforeCross(List<Polyhedron> polyhedron1S, Direction direction, BigFraction defaultDirectionFactor, List<Polyhedron> polyhedronList, ICollection<Polyhedron> crossPolyhedronList)
		{
			BigFraction bigFraction = Polyhedron.MoveMaxPositive;
			List<Polyhedron> items = new List<Polyhedron>();
			foreach (Polyhedron polyhedron in polyhedron1S)
			{
				List<Polyhedron> polyhedronList2 = FilterCrossTestPolyhedronList(polyhedronList, polyhedron, direction, defaultDirectionFactor);
				List<Polyhedron> list = new List<Polyhedron>();
				BigFraction bigFraction2 = Move1BeforeCross(polyhedron, direction, polyhedronList2, list);
				if (bigFraction2.CompareTo(Polyhedron.MoveMaxPositive) != 0 && bigFraction2.CompareTo(bigFraction) < 0)
				{
					bigFraction = bigFraction2;
					items = list;
				}
			}
			crossPolyhedronList?.AddRange(items);
			return bigFraction;
		}

		private static List<Polyhedron> FilterCrossTestPolyhedronList(List<Polyhedron> crossTestPolyhedronList, Polyhedron movePolyhedron, Direction direction, BigFraction defaultDirectionFactor)
		{
			if (defaultDirectionFactor == null)
			{
				return crossTestPolyhedronList;
			}
			List<BoundXYZMove> boundXYZMoveList = movePolyhedron.GetMoveBoundXYZMoves(direction, defaultDirectionFactor);
			return crossTestPolyhedronList.Where((Polyhedron testPolyhedron) => boundXYZMoveList.Any(testPolyhedron.TestBoundXYZMoveIntersection)).ToList();
		}

		public List<Polyhedron> Move1SBeforeCross(List<int> polyhedron1Ids, int directionId, BigFraction defaultDirectionFactor, bool isPlay1 = false)
		{
			return Move1SBeforeCrossVector(polyhedron1Ids, _directionList[directionId], defaultDirectionFactor, isPlay1);
		}

		// Adapter overload: preserve original group-movement logic for any original Direction.
		public List<Polyhedron> Move1SBeforeCrossVector(List<int> polyhedron1Ids, Direction direction, BigFraction defaultDirectionFactor, bool isPlay1 = false)
		{
			List<Polyhedron> polyhedron1S = polyhedron1Ids.Select((int id) => PolyhedronList[id]).ToList();
			List<Polyhedron> polyhedronList = PolyhedronList.Where((Polyhedron polyhedron) => polyhedron1S.All((Polyhedron tmp) => tmp.Index != polyhedron.Index)).ToList();
			BigFraction bigFraction = Move1SBeforeCross(polyhedron1S, direction, defaultDirectionFactor, polyhedronList, null);
			if (bigFraction.CompareTo(Polyhedron.MoveMaxPositive) == 0)
			{
				if (isPlay1 && defaultDirectionFactor.CompareTo(Polyhedron.MoveMaxConditionFactor) > 0 && polyhedron1Ids.Count == 1)
				{
					polyhedron1S.First().Already = true;
					int num = (int)UnityEngine.Vector3.Magnitude(new UnityEngine.Vector3((float)direction.X, (float)direction.Y, (float)direction.Z));
					bigFraction = Polyhedron.MoveMaxFactor / num;
				}
				else
				{
					bigFraction = defaultDirectionFactor ?? new BigFraction((BigInteger)100000);
				}
			}
			else
			{
				if (bigFraction.CompareTo(BigFraction.Zero) == 0)
				{
					return new List<Polyhedron>();
				}
				if (bigFraction.CompareTo(defaultDirectionFactor) > 0)
				{
					bigFraction = defaultDirectionFactor;
				}
			}
			foreach (Polyhedron item in polyhedron1S)
			{
				item.LastPosition = new Position(item.Position);
				item.Position.X = bigFraction * direction.X + item.Position.X;
				item.Position.Y = bigFraction * direction.Y + item.Position.Y;
				item.Position.Z = bigFraction * direction.Z + item.Position.Z;
			}
			StateId++;
			return polyhedron1S;
		}

		public List<Polyhedron> Move1SBeforeCrossAndDistance(List<int> polyhedron1Ids, int directionId, BigFraction directionFactorMax, bool isPlay1 = false)
		{
			List<Polyhedron> polyhedron1S = polyhedron1Ids.Select((int id) => PolyhedronList[id]).ToList();
			return Move1SBeforeCross2SAndDistance(polyhedron1S, _directionList[directionId], null, directionFactorMax, isPlay1);
		}

		private List<Polyhedron> Move1SBeforeCross2SAndDistance(List<Polyhedron> polyhedron1S, Direction direction, Polyhedron polyhedron2, BigFraction directionFactorMax, bool isPlay1 = false)
		{
			List<Polyhedron> movePolyhedronList = new List<Polyhedron>(polyhedron1S);
			List<Polyhedron> list = PolyhedronList.Where((Polyhedron polyhedron4) => movePolyhedronList.All((Polyhedron tmp) => tmp.Index != polyhedron4.Index)).ToList();
			List<Polyhedron> list2 = new List<Polyhedron>(movePolyhedronList);
			for (int num = 0; num < list2.Count; num++)
			{
				Polyhedron polyhedron3 = list2[num];
				List<Polyhedron> list3 = new List<Polyhedron>();
				List<Polyhedron> polyhedronList = FilterCrossTestPolyhedronList(list, polyhedron3, direction, directionFactorMax);
				if (Move1BeforeCross(polyhedron3, direction, polyhedronList, list3).CompareTo(BigFraction.Zero) != 0)
				{
					continue;
				}
				foreach (Polyhedron crossPolyhedron in list3)
				{
					if (!list2.Any((Polyhedron movePolyhedronTest) => movePolyhedronTest.Index == crossPolyhedron.Index))
					{
						list2.Add(crossPolyhedron);
						list.Remove(crossPolyhedron);
						if (polyhedron2 != null && crossPolyhedron.Index == polyhedron2.Index)
						{
							return new List<Polyhedron>();
						}
					}
				}
			}
			BigFraction bigFraction = Polyhedron.MoveMaxPositive;
			movePolyhedronList = list2;
			foreach (Polyhedron item in movePolyhedronList)
			{
				List<Polyhedron> crossPolyhedronList = new List<Polyhedron>();
				List<Polyhedron> polyhedronList2 = FilterCrossTestPolyhedronList(list, item, direction, directionFactorMax);
				BigFraction bigFraction2 = Move1BeforeCross(item, direction, polyhedronList2, crossPolyhedronList);
				if (bigFraction2.CompareTo(Polyhedron.MoveMaxPositive) != 0 && bigFraction2.CompareTo(bigFraction) < 0)
				{
					bigFraction = bigFraction2;
				}
			}
			if (bigFraction.CompareTo(BigFraction.Zero) == 0)
			{
				return new List<Polyhedron>();
			}
			if (isPlay1 && movePolyhedronList.Count == 1 && directionFactorMax.CompareTo(Polyhedron.MoveMaxConditionFactor) > 0 && bigFraction.CompareTo(Polyhedron.MoveMaxPositive) == 0)
			{
				polyhedron1S.First().Already = true;
				int num2 = (int)UnityEngine.Vector3.Magnitude(new UnityEngine.Vector3((float)direction.X, (float)direction.Y, (float)direction.Z));
				bigFraction = Polyhedron.MoveMaxFactor / num2;
			}
			else if (bigFraction.CompareTo(directionFactorMax) > 0)
			{
				bigFraction = directionFactorMax;
			}
			foreach (Polyhedron item2 in movePolyhedronList)
			{
				item2.LastPosition = new Position(item2.Position);
				item2.Position.X = bigFraction * direction.X + item2.Position.X;
				item2.Position.Y = bigFraction * direction.Y + item2.Position.Y;
				item2.Position.Z = bigFraction * direction.Z + item2.Position.Z;
			}
			StateId++;
			return movePolyhedronList;
		}

		public List<Polyhedron> Move1SByVertex2Vertex(IEnumerable<int> polyhedron1Ids, int polyhedron1Id, int point1Id, int polyhedron2Id, int point2Id)
		{
			Polyhedron polyhedron = PolyhedronList[polyhedron1Id];
			Point point = polyhedron.AnglePolyhedron.Points[point1Id];
			Position position = new Position(polyhedron.Position.X + point.X, polyhedron.Position.Y + point.Y, polyhedron.Position.Z + point.Z);
			Polyhedron polyhedron2 = PolyhedronList[polyhedron2Id];
			Point point2 = polyhedron2.AnglePolyhedron.Points[point2Id];
			Position position2 = new Position(polyhedron2.Position.X + point2.X, polyhedron2.Position.Y + point2.Y, polyhedron2.Position.Z + point2.Z);
			Direction direction = new Direction(position2.X - position.X, position2.Y - position.Y, position2.Z - position.Z);
			BigFraction directionFactorMax = new BigFraction(direction.Factor, direction.Factor2);
			if (SettingLogic.QuickMoveNegative)
			{
				direction = -direction;
			}
			List<Polyhedron> polyhedron1S = polyhedron1Ids.Select((int id) => PolyhedronList[id]).ToList();
			return Move1SBeforeCross2SAndDistance(polyhedron1S, direction, polyhedron2, directionFactorMax);
		}

		private static BigFraction GetDirectionFactor(Position vertexPosition1, Direction direction, Position vertexPosition2)
		{
			return direction.GetDirectionFactor(vertexPosition1, vertexPosition2);
		}

		private static Position GetVertexPosition(Polyhedron polyhedron, int vertexId)
		{
			Point point = polyhedron.GetCurrentAnglePolyhedron().Points[vertexId];
			Position position = polyhedron.Position;
			return new Position(position.X + point.X, position.Y + point.Y, position.Z + point.Z);
		}

		public List<Polyhedron> Move1SByVertex2Face(IEnumerable<int> polyhedron1Ids, int polyhedron1Id, int vertex1Id, int polyhedron2Id, int vertex2Id, int direction2Id, int direction = -1)
		{
			Polyhedron polyhedron = PolyhedronList[polyhedron1Id];
			Polyhedron polyhedron2 = PolyhedronList[polyhedron2Id];
			Direction direction2 = ((direction == -1) ? polyhedron2.AnglePolyhedron.SgnDirections[direction2Id] : _directionList[direction]);
			Position vertexPosition = GetVertexPosition(polyhedron, vertex1Id);
			Position vertexPosition2 = GetVertexPosition(polyhedron2, vertex2Id);
			BigFraction bigFraction = GetDirectionFactor(vertexPosition, direction2, vertexPosition2);
			int num = bigFraction.CompareTo(BigFraction.Zero);
			if (num >= 0)
			{
				if (num == 0)
				{
					return new List<Polyhedron>();
				}
			}
			else
			{
				bigFraction = -bigFraction;
				direction2 = -direction2;
			}
			if (SettingLogic.QuickMoveNegative)
			{
				direction2 = -direction2;
			}
			List<Polyhedron> polyhedron1S = polyhedron1Ids.Select((int id) => PolyhedronList[id]).ToList();
			return Move1SBeforeCross2SAndDistance(polyhedron1S, direction2, polyhedron2, bigFraction);
		}

		public List<Polyhedron> Move1SByVertex2Line(IEnumerable<int> polyhedron1Ids, int polyhedron1Id, int vertex1Id, int polyhedron2Id, int vertex2Id, int direction2Id)
		{
			Position vertexPosition = GetVertexPosition(PolyhedronList[polyhedron1Id], vertex1Id);
			Polyhedron polyhedron = PolyhedronList[polyhedron2Id];
			Position vertexPosition2 = GetVertexPosition(polyhedron, vertex2Id);
			Direction direction = polyhedron.AnglePolyhedron.SgnDirections[direction2Id];
			BigFraction directionFactor = GetDirectionFactor(vertexPosition2, direction, vertexPosition);
			BigFraction bigFraction = directionFactor * direction.X;
			BigFraction bigFraction2 = directionFactor * direction.Y;
			BigFraction bigFraction3 = directionFactor * direction.Z;
			BigFraction x = bigFraction + vertexPosition2.X - vertexPosition.X;
			BigFraction y = bigFraction2 + vertexPosition2.Y - vertexPosition.Y;
			BigFraction z = bigFraction3 + vertexPosition2.Z - vertexPosition.Z;
			Direction direction2 = new Direction(x, y, z);
			BigFraction directionFactorMax = new BigFraction(direction2.Factor, direction2.Factor2);
			List<Polyhedron> polyhedron1S = polyhedron1Ids.Select((int id) => PolyhedronList[id]).ToList();
			if (SettingLogic.QuickMoveNegative)
			{
				direction2 = -direction2;
			}
			if (!direction2.IsZero())
			{
				return Move1SBeforeCross2SAndDistance(polyhedron1S, direction2, polyhedron, directionFactorMax);
			}
			return new List<Polyhedron>();
		}

		public Polyhedron DoMoveRotate(MoveRotate moveRotate)
		{
			Polyhedron polyhedron = PolyhedronList[MoveRotatePolyhedronIndex];
			Position lastPosition = new Position(polyhedron.Position);
			Position position = new Position(polyhedron.Position);
			Position position2 = new Position(moveRotate.positionAddScaled);
			position2 = position2.SwitchXYZAxis(_moveRotateTransformXYZ.x, _moveRotateTransformXYZ.y, _moveRotateTransformXYZ.z);
			position.X += position2.X;
			position.Y += position2.Y;
			position.Z += position2.Z;
			MoveRotateAxis = TransformRotateAxis(_moveRotateTransformXYZ, moveRotate.rotateAxis);
			for (int i = 0; i < moveRotate.rotateCounts; i++)
			{
				polyhedron.SwitchXYZAxis(position, MoveRotateAxis, moveRotate.isHalf);
			}
			polyhedron.LastPosition = lastPosition;
			StateId++;
			return polyhedron;
		}

		public List<Polyhedron> RotateXYZ(List<int> polyhedrons, Position position, RotateAxis rotateType, bool isHalf)
		{
			List<Polyhedron> list = new List<Polyhedron>();
			List<Polyhedron> polyhedronsOther = new List<Polyhedron>();
			RotateAxis rotateAxis = Polyhedron.ReverseXYZAxis(rotateType);
			foreach (Polyhedron polyhedron in PolyhedronList)
			{
				if (polyhedrons.Contains(polyhedron.Index))
				{
					if (polyhedron.HalfRotateAxis != RotateAxis.None && polyhedron.HalfRotateAxis != rotateType && polyhedron.HalfRotateAxis != rotateAxis)
					{
						return new List<Polyhedron>();
					}
					list.Add(polyhedron);
				}
				else
				{
					polyhedronsOther.Add(polyhedron);
				}
			}
			List<Position> list2 = new List<Position>();
			foreach (Position item in list.Select((Polyhedron polyhedronRotate) => polyhedronRotate.RotateXYZTest(position, rotateType, isHalf, polyhedronsOther)))
			{
				if (item == null)
				{
					return new List<Polyhedron>();
				}
				list2.Add(item);
			}
			for (int num = 0; num < list2.Count; num++)
			{
				list[num].SwitchXYZAxis(list2[num], rotateType, isHalf);
			}
			StateId++;
			return list;
		}
	}
}
