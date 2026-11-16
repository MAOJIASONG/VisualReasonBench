using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using LuBanSuo.Core;
using UnityEngine;

namespace LuBanSuo.Logic;

public class Polyhedron
{
	private AnglePolyhedron[] _cachedHulls;

	private AnglePolyhedron _currentHull;

	public static readonly BigFraction MoveMinNegative = new BigFraction((BigInteger)int.MinValue);

	public static readonly BigFraction MoveMaxPositive = new BigFraction((BigInteger)int.MaxValue);

	public static readonly BigFraction MoveMaxFactor = new BigFraction((BigInteger)100000);

	public static readonly BigFraction MoveMaxConditionFactor = new BigFraction((BigInteger)250);

	public int Index { get; private set; }

	public Position Position { get; set; }

	public Position LastPosition { get; set; }

	public Position Position1 { get; private set; }

	public Position Position2 { get; private set; }

	public SwitchAxisTypeXYZ Orientation1 { get; private set; }

	public SwitchAxisTypeXYZ Orientation2 { get; private set; }

	public SwitchAxisTypeXYZ CurrentOrientation { get; private set; }

	public UnityEngine.Vector3 VisualEuler { get; set; }

	public AnglePolyhedron AnglePolyhedron => _currentHull;

	public AnglePolyhedron OriginalAnglePolyhedron
	{
		get
		{
			if (_cachedHulls == null || _cachedHulls.Length == 0)
			{
				return null;
			}
			return _cachedHulls[0];
		}
	}

	public Polyhedron(int index, AnglePolyhedron[] hulls, Position pos1, Position pos2, SwitchAxisTypeXYZ rot1, SwitchAxisTypeXYZ rot2)
	{
		Index = index;
		_cachedHulls = hulls;
		Position1 = pos1;
		Position2 = pos2;
		Orientation1 = rot1;
		Orientation2 = rot2;
		Position = new Position(pos1);
		LastPosition = new Position(pos1);
		SetOrientation(rot1);
	}

	public void SetOrientation(SwitchAxisTypeXYZ orientation)
	{
		CurrentOrientation = orientation;
		for (int i = 0; i < OrientationHelper.RotateAxisTypes.Length; i++)
		{
			SwitchAxisType[] array = OrientationHelper.RotateAxisTypes[i];
			if (array[0] == orientation.x && array[1] == orientation.y && array[2] == orientation.z)
			{
				_currentHull = _cachedHulls[i];
				return;
			}
		}
		_currentHull = _cachedHulls[0];
	}

	public Polyhedron(int index, AnglePolyhedron ap, Position pos)
	{
		Index = index;
		_cachedHulls = new AnglePolyhedron[1] { ap };
		_currentHull = ap;
		Position = new Position(pos);
		LastPosition = new Position(pos);
	}

	public List<BoundXYZMove> GetMoveBoundXYZMoves(Direction direction, BigFraction defaultDirectionFactor)
	{
		return AnglePolyhedron.ConvexPolyhedronList.Select((ConvexPolyhedron cp) => cp.GetMoveBounds(Position, new BigFraction(direction.X) * defaultDirectionFactor, new BigFraction(direction.Y) * defaultDirectionFactor, new BigFraction(direction.Z) * defaultDirectionFactor)).ToList();
	}

	public bool TestBoundXYZMoveIntersection(BoundXYZMove boundMove)
	{
		foreach (ConvexPolyhedron convexPolyhedron in AnglePolyhedron.ConvexPolyhedronList)
		{
			if (convexPolyhedron.TestBoundXYZMoveIntersection(boundMove, Position))
			{
				return true;
			}
		}
		return false;
	}

	public static bool TestIntersection(Polyhedron p1, Polyhedron p2)
	{
		return AnglePolyhedron.TestIntersection(p1.AnglePolyhedron, p1.Position, p2.AnglePolyhedron, p2.Position);
	}

	public static BigFraction MoveBeforeIntersection(Polyhedron p1, Polyhedron p2, Direction direction)
	{
		return AnglePolyhedron.MoveBeforeIntersection(p1.AnglePolyhedron, p1.Position, p2.AnglePolyhedron, p2.Position, direction);
	}
}
