namespace BurrPuzzle.DataCore
{
	public class Polyhedron
	{
		public float LoadPercent;

		public readonly int Index;

		public readonly List<PolyhedronInit> AngleXYZPolyhedronsInit = new List<PolyhedronInit>();

		public readonly List<int> AngleXYZPolyhedronsInitType = new List<int>();

		public readonly List<UnityEngine.Quaternion> AnglePolyhedronsQuaternion = new List<UnityEngine.Quaternion>();

		public readonly List<AnglePolyhedron> AnglePolyhedrons = new List<AnglePolyhedron>();

		public AnglePolyhedron AnglePolyhedron;

		public Position Position;

		public Position LastPosition;

		public bool Already;

		public Position Position1;

		public Position Position2;

		public UnityEngine.Quaternion Quaternion;

		public SwitchAxisType XSwitchAxisType;

		public SwitchAxisType YSwitchAxisType = SwitchAxisType.YPositive;

		public SwitchAxisType ZSwitchAxisType = SwitchAxisType.ZPositive;

		public RotateAxis HalfRotateAxis;

		private UnityEngine.Quaternion _quaternionBak;

		private SwitchAxisType _xSwitchAxisTypeBak;

		private SwitchAxisType _ySwitchAxisTypeBak = SwitchAxisType.YPositive;

		private SwitchAxisType _zSwitchAxisTypeBak = SwitchAxisType.ZPositive;

		private RotateAxis _halfRotateAxisBak;

		private Position _positionBak;

		private bool _isPlay1;

		public readonly SwitchAxisType XSwitchAxisType1;

		public readonly SwitchAxisType YSwitchAxisType1;

		public readonly SwitchAxisType ZSwitchAxisType1;

		public readonly SwitchAxisType XSwitchAxisType2;

		public readonly SwitchAxisType YSwitchAxisType2;

		public readonly SwitchAxisType ZSwitchAxisType2;

		public static readonly SwitchAxisType[][] RotateAxisTypes;

		public static UnityEngine.Quaternion[] RotateQuaternions;

		public static readonly BigFraction MoveMinNegative;

		public static readonly BigFraction MoveMaxPositive;

		public static readonly BigFraction MoveMaxFactor;

		public static readonly BigFraction MoveMaxConditionFactor;

		static Polyhedron()
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
			RotateQuaternions = new UnityEngine.Quaternion[RotateAxisTypes.Length];
			MoveMinNegative = new BigFraction((BigInteger)int.MinValue);
			MoveMaxPositive = new BigFraction((BigInteger)int.MaxValue);
			MoveMaxFactor = new BigFraction((BigInteger)3000000);
			MoveMaxConditionFactor = new BigFraction((BigInteger)400000);
			for (int i = 0; i <= 270; i += 90)
			{
				for (int j = 0; j <= 270; j += 90)
				{
					for (int k = 0; k <= 270; k += 90)
					{
						UnityEngine.Quaternion quaternion = UnityEngine.Quaternion.Euler(new UnityEngine.Vector3(i, j, k));
						SwitchAxisType switchAxisType = Parse(quaternion * UnityEngine.Vector3.right);
						SwitchAxisType switchAxisType2 = Parse(quaternion * UnityEngine.Vector3.up);
						SwitchAxisType switchAxisType3 = Parse(quaternion * UnityEngine.Vector3.forward);
						for (int l = 0; l < RotateAxisTypes.Length; l++)
						{
							if (RotateAxisTypes[l][0] == switchAxisType && RotateAxisTypes[l][1] == switchAxisType2 && RotateAxisTypes[l][2] == switchAxisType3)
							{
								RotateQuaternions[l] = UnityEngine.Quaternion.Inverse(quaternion);
								break;
							}
						}
					}
				}
			}
		}

		private static SwitchAxisType Parse(UnityEngine.Vector3 vector3)
		{
			if (UnityEngine.Vector3.Distance(vector3, UnityEngine.Vector3.right) < 0.125f)
			{
				return SwitchAxisType.XPositive;
			}
			if (UnityEngine.Vector3.Distance(vector3, UnityEngine.Vector3.left) < 0.125f)
			{
				return SwitchAxisType.XNegative;
			}
			if (UnityEngine.Vector3.Distance(vector3, UnityEngine.Vector3.up) < 0.125f)
			{
				return SwitchAxisType.YPositive;
			}
			if (UnityEngine.Vector3.Distance(vector3, UnityEngine.Vector3.down) < 0.125f)
			{
				return SwitchAxisType.YNegative;
			}
			if (UnityEngine.Vector3.Distance(vector3, UnityEngine.Vector3.forward) < 0.125f)
			{
				return SwitchAxisType.ZPositive;
			}
			return SwitchAxisType.ZNegative;
		}

		private void SwitchPlay(SwitchAxisType x, SwitchAxisType y, SwitchAxisType z)
		{
			XSwitchAxisType = x;
			YSwitchAxisType = y;
			ZSwitchAxisType = z;
			for (int i = 0; i < RotateAxisTypes.Length; i++)
			{
				SwitchAxisType[] array = RotateAxisTypes[i];
				if (array[0] == XSwitchAxisType && array[1] == YSwitchAxisType && array[2] == ZSwitchAxisType)
				{
					Quaternion = RotateQuaternions[i];
					AnglePolyhedron = AngleXYZPolyhedronsInit[i].AnglePolyhedron;
					break;
				}
			}
		}

		public void EnterHelpView(bool isPlay1, UnityEngine.Quaternion quaternion)
		{
			_xSwitchAxisTypeBak = XSwitchAxisType;
			_ySwitchAxisTypeBak = YSwitchAxisType;
			_zSwitchAxisTypeBak = ZSwitchAxisType;
			_halfRotateAxisBak = HalfRotateAxis;
			_positionBak = new Position(Position);
			_quaternionBak = quaternion;
			if (isPlay1)
			{
				SwitchPlay(XSwitchAxisType2, YSwitchAxisType2, ZSwitchAxisType2);
				HalfRotateAxis = RotateAxis.None;
				SwitchAxisType[] array = ComputeSwitchXYZAxis(RotateAxis.XNegativeWise);
				XSwitchAxisType = array[0];
				YSwitchAxisType = array[1];
				ZSwitchAxisType = array[2];
				AnglePolyhedron = AdjustAnglePolyhedron();
				Position = new Position(Position2);
				Quaternion = UnityEngine.Quaternion.Euler(-90f, 0f, 0f);
			}
			else
			{
				SwitchPlay(XSwitchAxisType1, YSwitchAxisType1, ZSwitchAxisType1);
				AnglePolyhedron = AdjustAnglePolyhedron();
				HalfRotateAxis = RotateAxis.None;
				Position = new Position(Position1);
			}
		}

		public void ExitHelpView()
		{
			XSwitchAxisType = _xSwitchAxisTypeBak;
			YSwitchAxisType = _ySwitchAxisTypeBak;
			ZSwitchAxisType = _zSwitchAxisTypeBak;
			HalfRotateAxis = _halfRotateAxisBak;
			AnglePolyhedron = AdjustAnglePolyhedron();
			Quaternion = _quaternionBak;
			Position = _positionBak;
		}

		public void SwitchPlay1()
		{
			SwitchPlay(XSwitchAxisType1, YSwitchAxisType1, ZSwitchAxisType1);
			Position = new Position(Position1);
			Already = false;
		}

		public void SwitchPlayOnly2()
		{
			SwitchPlay(XSwitchAxisType2, YSwitchAxisType2, ZSwitchAxisType2);
			Position = new Position(Position2);
			Already = false;
		}

		public void SwitchPlay2()
		{
			SwitchPlay(XSwitchAxisType2, YSwitchAxisType2, ZSwitchAxisType2);
			HalfRotateAxis = RotateAxis.None;
			SwitchXYZAxisAlign(Position2, RotateAxis.XNegativeWise);
			Quaternion = UnityEngine.Quaternion.Euler(-90f, 0f, 0f);
			Already = false;
		}

		public Polyhedron(int index, IList<float> polyhedronLoadPercents, IReadOnlyCollection<Point> pointsAll, List<List<int>> polygonsPointsIndex, List<List<List<int>>> convexPolyhedraPointsIndex, Position position1, Position position2, SwitchAxisTypeXYZ switchAxisTypeXYZ1, SwitchAxisTypeXYZ switchAxisTypeXYZ2)
		{
			Index = index;
			Position1 = position1;
			Position2 = position2;
			Position = new Position(position1);
			LastPosition = new Position(position1);
			XSwitchAxisType1 = switchAxisTypeXYZ1.x;
			YSwitchAxisType1 = switchAxisTypeXYZ1.y;
			ZSwitchAxisType1 = switchAxisTypeXYZ1.z;
			XSwitchAxisType2 = switchAxisTypeXYZ2.x;
			YSwitchAxisType2 = switchAxisTypeXYZ2.y;
			ZSwitchAxisType2 = switchAxisTypeXYZ2.z;
			XSwitchAxisType = XSwitchAxisType1;
			YSwitchAxisType = YSwitchAxisType1;
			ZSwitchAxisType = ZSwitchAxisType1;
			for (int i = 0; i < RotateAxisTypes.Length; i++)
			{
				SwitchAxisType[] rotateAxisType = RotateAxisTypes[i];
				List<Point> list = pointsAll.Select((Point point) => point.SwitchXYZAxis(rotateAxisType[0], rotateAxisType[1], rotateAxisType[2])).ToList();
				bool flag = false;
				HashSet<int> hashSet = new HashSet<int>();
				int num = i;
				for (int num2 = 0; num2 < AngleXYZPolyhedronsInit.Count; num2++)
				{
					if (!hashSet.Contains(AngleXYZPolyhedronsInitType[num2]))
					{
						hashSet.Add(AngleXYZPolyhedronsInitType[num2]);
						if (ComparePolyhedronInit(list, polygonsPointsIndex, AngleXYZPolyhedronsInit[num2]))
						{
							flag = true;
							num = num2;
							if (Luban.Original.EngineDiagnostics.VerboseGeometry) UnityEngine.Debug.Log("Polyhedron[" + index + "] type pair[" + i + ", " + num2 + "] ComparePolyhedronInit return true");
							break;
						}
					}
				}
				AngleXYZPolyhedronsInitType.Add(num);
				PolyhedronInit item = (flag ? new PolyhedronInit(list, polygonsPointsIndex, convexPolyhedraPointsIndex, RotateQuaternions[i], AnglePolyhedronsQuaternion, AnglePolyhedrons, AngleXYZPolyhedronsInit[num]) : new PolyhedronInit(list, polygonsPointsIndex, convexPolyhedraPointsIndex, RotateQuaternions[i], AnglePolyhedronsQuaternion, AnglePolyhedrons));
				AngleXYZPolyhedronsInit.Add(item);
				polyhedronLoadPercents[index] = ((float)i + 1f) / (float)RotateAxisTypes.Length;
			}
		}

		public int GetAngleXYZPolyhedronsInitType(SwitchAxisTypeXYZ switchAxisTypeXYZ)
		{
			for (int i = 0; i < RotateAxisTypes.Length; i++)
			{
				if (RotateAxisTypes[i][0] == switchAxisTypeXYZ.x && RotateAxisTypes[i][1] == switchAxisTypeXYZ.y && RotateAxisTypes[i][2] == switchAxisTypeXYZ.z)
				{
					return AngleXYZPolyhedronsInitType[i];
				}
			}
			return -1;
		}

		private bool ComparePolyhedronInit(IReadOnlyList<Point> points1, List<List<int>> polygonsPointsIndex, PolyhedronInit polyhedronInit2)
		{
			List<Point> points2 = polyhedronInit2.AnglePolyhedron.Points;
			int[] array = new int[points1.Count];
			for (int i = 0; i < points1.Count; i++)
			{
				bool flag = false;
				for (int j = 0; j < points2.Count; j++)
				{
					if (points1[i].X.Equals(points2[j].X) && points1[i].Y.Equals(points2[j].Y) && points1[i].Z.Equals(points2[j].Z))
					{
						flag = true;
						array[i] = j;
						break;
					}
				}
				if (!flag)
				{
					array[i] = -1;
				}
			}
			if (!array.Contains(-1))
			{
				if (Luban.Original.EngineDiagnostics.VerboseGeometry) UnityEngine.Debug.Log("ComparePolyhedronInit may return true");
			}
			foreach (List<int> item in polygonsPointsIndex)
			{
				HashSet<int> polygonPointsIndex2 = new HashSet<int>();
				foreach (int item2 in item)
				{
					polygonPointsIndex2.Add(array[item2]);
				}
				if (!polygonsPointsIndex.Any((List<int> polygonPointsIndexTmp) => Enumerable.ToHashSet(polygonPointsIndexTmp).SetEquals(polygonPointsIndex2)))
				{
					return false;
				}
			}
			return true;
		}

		public Polyhedron(int index, List<PolyhedronInit> angleXYZPolyhedronsInit, List<int> angleXYZPolyhedronsInitType, Position position1, Position position2, SwitchAxisTypeXYZ switchAxisTypeXYZ1, SwitchAxisTypeXYZ switchAxisTypeXYZ2)
		{
			Index = index;
			Position1 = position1;
			Position2 = position2;
			Position = new Position(position1);
			LastPosition = new Position(position1);
			XSwitchAxisType1 = switchAxisTypeXYZ1.x;
			YSwitchAxisType1 = switchAxisTypeXYZ1.y;
			ZSwitchAxisType1 = switchAxisTypeXYZ1.z;
			XSwitchAxisType2 = switchAxisTypeXYZ2.x;
			YSwitchAxisType2 = switchAxisTypeXYZ2.y;
			ZSwitchAxisType2 = switchAxisTypeXYZ2.z;
			XSwitchAxisType = XSwitchAxisType1;
			YSwitchAxisType = YSwitchAxisType1;
			ZSwitchAxisType = ZSwitchAxisType1;
			AngleXYZPolyhedronsInit = angleXYZPolyhedronsInit;
			AngleXYZPolyhedronsInitType = angleXYZPolyhedronsInitType;
		}

		public AnglePolyhedron GetCurrentAnglePolyhedron()
		{
			return AnglePolyhedron;
		}

		public static bool TestIntersection(Polyhedron polyhedron1, Polyhedron polyhedron2)
		{
			return AnglePolyhedron.TestIntersection(polyhedron1.GetCurrentAnglePolyhedron(), anglePolyhedron2: polyhedron2.GetCurrentAnglePolyhedron(), position1: polyhedron1.Position, position2: polyhedron2.Position);
		}

		public List<BoundXYZMove> GetMoveBoundXYZMoves(Direction direction, BigFraction defaultDirectionFactor)
		{
			BigFraction moveX = defaultDirectionFactor * direction.X;
			BigFraction moveY = defaultDirectionFactor * direction.Y;
			BigFraction moveZ = defaultDirectionFactor * direction.Z;
			return AnglePolyhedron.ConvexPolyhedronList.Select((ConvexPolyhedron convexPolyhedron) => convexPolyhedron.GetMoveBounds(Position, moveX, moveY, moveZ)).ToList();
		}

		public bool TestBoundXYZMoveIntersection(BoundXYZMove boundXYZMove)
		{
			return AnglePolyhedron.ConvexPolyhedronList.Any((ConvexPolyhedron convexPolyhedron) => convexPolyhedron.TestBoundXYZMoveIntersection(Position, boundXYZMove));
		}

		public static BigFraction MoveBeforeIntersection(Polyhedron polyhedron1, Polyhedron polyhedron2, Direction move1To2)
		{
			return AnglePolyhedron.MoveBeforeIntersection(polyhedron1.GetCurrentAnglePolyhedron(), anglePolyhedron2: polyhedron2.GetCurrentAnglePolyhedron(), position1: polyhedron1.Position, position2: polyhedron2.Position, move1To2: move1To2);
		}

		private static Position ComputeRotatePosition(int angleSteps, BigFraction positionU, BigFraction positionV, BigFraction centerU, BigFraction centerV, RotateAxis angleType, Position position)
		{
			BigFraction bigFraction = positionU - centerU;
			BigFraction bigFraction2 = positionV - centerV;
			Point.CosSinByDegree csDegree = Point.GetCsDegree(angleSteps);
			BigFraction sinDegree = csDegree.SinDegree;
			BigFraction cosDegree = csDegree.CosDegree;
			BigFraction bigFraction3 = bigFraction * cosDegree - bigFraction2 * sinDegree;
			BigFraction bigFraction4 = bigFraction * sinDegree + bigFraction2 * cosDegree;
			bigFraction = bigFraction3 + centerU;
			bigFraction2 = bigFraction4 + centerV;
			switch (angleType)
			{
			case RotateAxis.XPositiveWise:
			case RotateAxis.XNegativeWise:
				return new Position(position.X, bigFraction, bigFraction2);
			case RotateAxis.YPositiveWise:
			case RotateAxis.YNegativeWise:
				return new Position(bigFraction2, position.Y, bigFraction);
			default:
				return new Position(bigFraction, bigFraction2, position.Z);
			}
		}

		public static SwitchAxisType ReverseXYZAxis(SwitchAxisType axisType)
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

		public static RotateAxis ReverseXYZAxis(RotateAxis axisType)
		{
			return axisType switch
			{
				RotateAxis.XPositiveWise => RotateAxis.XNegativeWise, 
				RotateAxis.XNegativeWise => RotateAxis.XPositiveWise, 
				RotateAxis.YPositiveWise => RotateAxis.YNegativeWise, 
				RotateAxis.YNegativeWise => RotateAxis.YPositiveWise, 
				RotateAxis.ZPositiveWise => RotateAxis.ZNegativeWise, 
				RotateAxis.ZNegativeWise => RotateAxis.ZPositiveWise, 
				_ => RotateAxis.None, 
			};
		}

		public void SwitchXYZAxis(Position position, RotateAxis rotateAxis, bool isHalf)
		{
			if (isHalf)
			{
				if (HalfRotateAxis == RotateAxis.None)
				{
					HalfRotateAxis = rotateAxis;
					AnglePolyhedron = AdjustAnglePolyhedron();
					LastPosition = new Position(Position);
					Position = new Position(position);
				}
				else if (HalfRotateAxis == rotateAxis)
				{
					HalfRotateAxis = RotateAxis.None;
					SwitchXYZAxisAlign(position, rotateAxis);
				}
				else if (HalfRotateAxis == ReverseXYZAxis(rotateAxis))
				{
					HalfRotateAxis = RotateAxis.None;
					AnglePolyhedron = AdjustAnglePolyhedron();
					LastPosition = new Position(Position);
					Position = new Position(position);
				}
			}
			else
			{
				SwitchXYZAxisAlign(position, rotateAxis);
			}
		}

		public SwitchAxisType[] ComputeSwitchXYZAxis(RotateAxis rotateAxis)
		{
			SwitchAxisType switchAxisType;
			SwitchAxisType switchAxisType2;
			SwitchAxisType switchAxisType3;
			switch (rotateAxis)
			{
			case RotateAxis.XPositiveWise:
				switchAxisType = XSwitchAxisType;
				switchAxisType2 = ReverseXYZAxis(ZSwitchAxisType);
				switchAxisType3 = YSwitchAxisType;
				break;
			case RotateAxis.XNegativeWise:
				switchAxisType = XSwitchAxisType;
				switchAxisType2 = ZSwitchAxisType;
				switchAxisType3 = ReverseXYZAxis(YSwitchAxisType);
				break;
			case RotateAxis.YPositiveWise:
				switchAxisType2 = YSwitchAxisType;
				switchAxisType3 = ReverseXYZAxis(XSwitchAxisType);
				switchAxisType = ZSwitchAxisType;
				break;
			case RotateAxis.YNegativeWise:
				switchAxisType2 = YSwitchAxisType;
				switchAxisType3 = XSwitchAxisType;
				switchAxisType = ReverseXYZAxis(ZSwitchAxisType);
				break;
			case RotateAxis.ZPositiveWise:
				switchAxisType3 = ZSwitchAxisType;
				switchAxisType = ReverseXYZAxis(YSwitchAxisType);
				switchAxisType2 = XSwitchAxisType;
				break;
			case RotateAxis.ZNegativeWise:
				switchAxisType3 = ZSwitchAxisType;
				switchAxisType = YSwitchAxisType;
				switchAxisType2 = ReverseXYZAxis(XSwitchAxisType);
				break;
			default:
				switchAxisType = XSwitchAxisType;
				switchAxisType2 = YSwitchAxisType;
				switchAxisType3 = ZSwitchAxisType;
				break;
			}
			return new SwitchAxisType[3] { switchAxisType, switchAxisType2, switchAxisType3 };
		}

		private void SwitchXYZAxisAlign(Position position, RotateAxis rotateAxis)
		{
			SwitchAxisType[] array = ComputeSwitchXYZAxis(rotateAxis);
			XSwitchAxisType = array[0];
			YSwitchAxisType = array[1];
			ZSwitchAxisType = array[2];
			AnglePolyhedron = AdjustAnglePolyhedron();
			LastPosition = new Position(Position);
			Position = new Position(position);
		}

		public AnglePolyhedron AdjustAnglePolyhedron()
		{
			for (int i = 0; i < RotateAxisTypes.Length; i++)
			{
				if (RotateAxisTypes[i][0] == XSwitchAxisType && RotateAxisTypes[i][1] == YSwitchAxisType && RotateAxisTypes[i][2] == ZSwitchAxisType)
				{
					int index = 8;
					return HalfRotateAxis switch
					{
						RotateAxis.XPositiveWise => AngleXYZPolyhedronsInit[i].X.PositiveWise[index], 
						RotateAxis.XNegativeWise => AngleXYZPolyhedronsInit[i].X.NegativeWise[index], 
						RotateAxis.YPositiveWise => AngleXYZPolyhedronsInit[i].Y.PositiveWise[index], 
						RotateAxis.YNegativeWise => AngleXYZPolyhedronsInit[i].Y.NegativeWise[index], 
						RotateAxis.ZPositiveWise => AngleXYZPolyhedronsInit[i].Z.PositiveWise[index], 
						RotateAxis.ZNegativeWise => AngleXYZPolyhedronsInit[i].Z.NegativeWise[index], 
						_ => AngleXYZPolyhedronsInit[i].AnglePolyhedron, 
					};
				}
			}
			return null;
		}

		private List<AnglePolyhedron> CollectRotateXYZTestAnglePolyhedrons(RotateAxis angleType, bool isHalf)
		{
			if (HalfRotateAxis == RotateAxis.None)
			{
				int count = (isHalf ? 8 : 16);
				PolyhedronInit angleXYZPolyhedronsInit = GetAngleXYZPolyhedronsInit(XSwitchAxisType, YSwitchAxisType, ZSwitchAxisType);
				return angleType switch
				{
					RotateAxis.XPositiveWise => angleXYZPolyhedronsInit.X.PositiveWise.GetRange(1, count), 
					RotateAxis.XNegativeWise => angleXYZPolyhedronsInit.X.NegativeWise.GetRange(1, count), 
					RotateAxis.YPositiveWise => angleXYZPolyhedronsInit.Y.PositiveWise.GetRange(1, count), 
					RotateAxis.YNegativeWise => angleXYZPolyhedronsInit.Y.NegativeWise.GetRange(1, count), 
					RotateAxis.ZPositiveWise => angleXYZPolyhedronsInit.Z.PositiveWise.GetRange(1, count), 
					RotateAxis.ZNegativeWise => angleXYZPolyhedronsInit.Z.NegativeWise.GetRange(1, count), 
					_ => null, 
				};
			}
			if (isHalf)
			{
				int index = ((HalfRotateAxis == angleType) ? 9 : 0);
				PolyhedronInit angleXYZPolyhedronsInit2 = GetAngleXYZPolyhedronsInit(XSwitchAxisType, YSwitchAxisType, ZSwitchAxisType);
				List<AnglePolyhedron> list = HalfRotateAxis switch
				{
					RotateAxis.XPositiveWise => angleType switch
					{
						RotateAxis.XPositiveWise => angleXYZPolyhedronsInit2.X.PositiveWise.GetRange(index, 8), 
						RotateAxis.XNegativeWise => angleXYZPolyhedronsInit2.X.PositiveWise.GetRange(index, 8), 
						_ => null, 
					}, 
					RotateAxis.XNegativeWise => angleType switch
					{
						RotateAxis.XPositiveWise => angleXYZPolyhedronsInit2.X.NegativeWise.GetRange(index, 8), 
						RotateAxis.XNegativeWise => angleXYZPolyhedronsInit2.X.NegativeWise.GetRange(index, 8), 
						_ => null, 
					}, 
					RotateAxis.YPositiveWise => angleType switch
					{
						RotateAxis.YPositiveWise => angleXYZPolyhedronsInit2.Y.PositiveWise.GetRange(index, 8), 
						RotateAxis.YNegativeWise => angleXYZPolyhedronsInit2.Y.PositiveWise.GetRange(index, 8), 
						_ => null, 
					}, 
					RotateAxis.YNegativeWise => angleType switch
					{
						RotateAxis.YPositiveWise => angleXYZPolyhedronsInit2.Y.NegativeWise.GetRange(index, 8), 
						RotateAxis.YNegativeWise => angleXYZPolyhedronsInit2.Y.NegativeWise.GetRange(index, 8), 
						_ => null, 
					}, 
					RotateAxis.ZPositiveWise => angleType switch
					{
						RotateAxis.ZPositiveWise => angleXYZPolyhedronsInit2.Z.PositiveWise.GetRange(index, 8), 
						RotateAxis.ZNegativeWise => angleXYZPolyhedronsInit2.Z.PositiveWise.GetRange(index, 8), 
						_ => null, 
					}, 
					RotateAxis.ZNegativeWise => angleType switch
					{
						RotateAxis.ZPositiveWise => angleXYZPolyhedronsInit2.Z.NegativeWise.GetRange(index, 8), 
						RotateAxis.ZNegativeWise => angleXYZPolyhedronsInit2.Z.NegativeWise.GetRange(index, 8), 
						_ => null, 
					}, 
					_ => null, 
				};
				if (HalfRotateAxis != angleType)
				{
					list?.Reverse();
				}
				return list;
			}
			if (HalfRotateAxis == angleType)
			{
				SwitchAxisType[] array = ComputeSwitchXYZAxis(angleType);
				SwitchAxisType xRotateAxis = array[0];
				SwitchAxisType yRotateAxis = array[1];
				SwitchAxisType zRotateAxis = array[2];
				return CollectRotateXYZTestAnglePolyhedrons(xRotateAxis, yRotateAxis, zRotateAxis, ReverseXYZAxis(HalfRotateAxis));
			}
			return CollectRotateXYZTestAnglePolyhedrons(XSwitchAxisType, YSwitchAxisType, ZSwitchAxisType, HalfRotateAxis);
		}

		private List<AnglePolyhedron> CollectRotateXYZTestAnglePolyhedrons(SwitchAxisType xRotateAxis, SwitchAxisType yRotateAxis, SwitchAxisType zRotateAxis, RotateAxis halfAngleType)
		{
			List<AnglePolyhedron> list = new List<AnglePolyhedron>();
			PolyhedronInit angleXYZPolyhedronsInit = GetAngleXYZPolyhedronsInit(xRotateAxis, yRotateAxis, zRotateAxis);
			list.AddRange(halfAngleType switch
			{
				RotateAxis.XPositiveWise => angleXYZPolyhedronsInit.X.PositiveWise.GetRange(0, 8), 
				RotateAxis.XNegativeWise => angleXYZPolyhedronsInit.X.NegativeWise.GetRange(0, 8), 
				RotateAxis.YPositiveWise => angleXYZPolyhedronsInit.Y.PositiveWise.GetRange(0, 8), 
				RotateAxis.YNegativeWise => angleXYZPolyhedronsInit.Y.NegativeWise.GetRange(0, 8), 
				RotateAxis.ZPositiveWise => angleXYZPolyhedronsInit.Z.PositiveWise.GetRange(0, 8), 
				RotateAxis.ZNegativeWise => angleXYZPolyhedronsInit.Z.NegativeWise.GetRange(0, 8), 
				_ => new List<AnglePolyhedron>(), 
			});
			list.Reverse();
			list.AddRange(halfAngleType switch
			{
				RotateAxis.XPositiveWise => angleXYZPolyhedronsInit.X.NegativeWise.GetRange(1, 8), 
				RotateAxis.XNegativeWise => angleXYZPolyhedronsInit.X.PositiveWise.GetRange(1, 8), 
				RotateAxis.YPositiveWise => angleXYZPolyhedronsInit.Y.NegativeWise.GetRange(1, 8), 
				RotateAxis.YNegativeWise => angleXYZPolyhedronsInit.Y.PositiveWise.GetRange(1, 8), 
				RotateAxis.ZPositiveWise => angleXYZPolyhedronsInit.Z.NegativeWise.GetRange(1, 8), 
				RotateAxis.ZNegativeWise => angleXYZPolyhedronsInit.Z.PositiveWise.GetRange(1, 8), 
				_ => new List<AnglePolyhedron>(), 
			});
			return list;
		}

		private PolyhedronInit GetAngleXYZPolyhedronsInit(SwitchAxisType xRotateAxis, SwitchAxisType yRotateAxis, SwitchAxisType zRotateAxis)
		{
			for (int i = 0; i < RotateAxisTypes.Length; i++)
			{
				if (RotateAxisTypes[i][0] == xRotateAxis && RotateAxisTypes[i][1] == yRotateAxis && RotateAxisTypes[i][2] == zRotateAxis)
				{
					return AngleXYZPolyhedronsInit[i];
				}
			}
			return null;
		}

		private bool IsPositiveWise(RotateAxis angleType)
		{
			if (angleType != RotateAxis.XPositiveWise && angleType != RotateAxis.YPositiveWise)
			{
				return angleType == RotateAxis.ZPositiveWise;
			}
			return true;
		}

		public Position RotateXYZTest(Position rotateCenter, RotateAxis angleType, bool isHalf, List<Polyhedron> polyhedrons)
		{
			PolyhedronInit polyhedronInit = null;
			for (int i = 0; i < RotateAxisTypes.Length; i++)
			{
				if (RotateAxisTypes[i][0] == XSwitchAxisType && RotateAxisTypes[i][1] == YSwitchAxisType && RotateAxisTypes[i][2] == ZSwitchAxisType)
				{
					polyhedronInit = AngleXYZPolyhedronsInit[i];
					break;
				}
			}
			if (polyhedronInit == null)
			{
				return null;
			}
			List<AnglePolyhedron> anglePolyhedrons = CollectRotateXYZTestAnglePolyhedrons(angleType, isHalf);
			bool positiveWise = IsPositiveWise(angleType);
			BigFraction rotateCenterU;
			BigFraction rotateCenterV;
			BigFraction polyhedronU;
			BigFraction polyhedronV;
			switch (angleType)
			{
			case RotateAxis.XPositiveWise:
			case RotateAxis.XNegativeWise:
				rotateCenterU = rotateCenter.Y;
				rotateCenterV = rotateCenter.Z;
				polyhedronU = Position.Y;
				polyhedronV = Position.Z;
				break;
			case RotateAxis.YPositiveWise:
			case RotateAxis.YNegativeWise:
				rotateCenterU = rotateCenter.Z;
				rotateCenterV = rotateCenter.X;
				polyhedronU = Position.Z;
				polyhedronV = Position.X;
				break;
			default:
				rotateCenterU = rotateCenter.X;
				rotateCenterV = rotateCenter.Y;
				polyhedronU = Position.X;
				polyhedronV = Position.Y;
				break;
			}
			int num = (isHalf ? 8 : 16);
			int angleStepStart = ((HalfRotateAxis != RotateAxis.None) ? (IsPositiveWise(HalfRotateAxis) ? 8 : (-8)) : 0);
			Position[] positionsNew = new Position[num];
			List<Thread> list = new List<Thread>();
			int num2 = num / 4;
			for (int j = 0; j < num; j += num2)
			{
				int startIndex = j;
				int endIndex = startIndex + num2;
				if (endIndex > num)
				{
					endIndex = num;
				}
				Thread thread = new Thread((ThreadStart)delegate
				{
					for (int k = startIndex; k < endIndex; k++)
					{
						Position position = Position;
						if (rotateCenterU.CompareTo(polyhedronU) != 0 || rotateCenterV.CompareTo(polyhedronV) != 0)
						{
							if (isHalf || HalfRotateAxis != RotateAxis.None)
							{
								positionsNew[k] = null;
								break;
							}
							position = ComputeRotatePosition(positiveWise ? (angleStepStart + k + 1) : (angleStepStart - k - 1), polyhedronU, polyhedronV, rotateCenterU, rotateCenterV, angleType, Position);
						}
						else
						{
							positionsNew[k] = position;
							k++;
							if (k >= endIndex)
							{
								break;
							}
						}
						positionsNew[k] = position;
						foreach (Polyhedron polyhedron in polyhedrons)
						{
							if (AnglePolyhedron.TestIntersection(anglePolyhedrons[k], position, polyhedron.GetCurrentAnglePolyhedron(), polyhedron.Position))
							{
								positionsNew[k] = null;
								break;
							}
						}
						if (positionsNew[k] == null)
						{
							break;
						}
					}
				});
				list.Add(thread);
				thread.Start();
			}
			foreach (Thread item in list)
			{
				item.Join();
			}
			if (!positionsNew.Any((Position positionNew) => positionNew == null))
			{
				return positionsNew.Last();
			}
			return null;
		}
	}
}
