"""
测试用例 - 验证核心功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_core import Vec3, PieceDef, normalize_points, points_to_signature
from rotation import ROTATION_MATRICES
from loader import load_puzzle_by_name, create_game_state
from placement import place_piece_by_cells, place_piece_by_transform, pickup_piece


def test_vec3():
    """测试Vec3基本功能"""
    print("\n=== Testing Vec3 ===")

    v1 = Vec3(1, 2, 3)
    v2 = Vec3(1, 2, 3)
    v3 = Vec3(2, 3, 4)

    assert v1 == v2, "Vec3 equality failed"
    assert v1 != v3, "Vec3 inequality failed"
    assert v1.to_key() == "1,2,3", "Vec3.to_key() failed"
    assert v1.to_tuple() == (1, 2, 3), "Vec3.to_tuple() failed"

    v4 = Vec3.from_key("4,5,6")
    assert v4.to_tuple() == (4, 5, 6), "Vec3.from_key() failed"

    print("✓ Vec3 tests passed")


def test_normalization():
    """测试点集规范化"""
    print("\n=== Testing Normalization ===")

    # 测试平移到(0,0,0)
    points = [Vec3(2, 3, 4), Vec3(3, 4, 5), Vec3(2, 3, 5)]
    normalized = normalize_points(points)

    assert normalized[0].x == 0, "Normalization failed: min x should be 0"
    assert normalized[0].y == 0, "Normalization failed: min y should be 0"
    assert normalized[0].z == 0, "Normalization failed: min z should be 0"

    # 测试排序
    sig1 = points_to_signature(points)
    # 相同点但顺序不同应该得到相同签名
    points2 = [Vec3(3, 4, 5), Vec3(2, 3, 5), Vec3(2, 3, 4)]
    sig2 = points_to_signature(points2)

    assert sig1 == sig2, f"Signatures should be equal: {sig1} != {sig2}"

    print(f"✓ Normalization tests passed")


def test_rotations():
    """测试旋转矩阵"""
    print("\n=== Testing Rotations ===")

    assert len(ROTATION_MATRICES) == 24, f"Expected 24 rotations, got {len(ROTATION_MATRICES)}"

    # 测试所有旋转都是正交矩阵且行列式为1
    for i, R in enumerate(ROTATION_MATRICES):
        import numpy as np

        # 正交性
        product = R @ R.T
        assert np.allclose(product, np.eye(3)), f"Rotation {i} is not orthogonal"

        # 行列式为1
        det = np.linalg.det(R)
        assert np.isclose(det, 1.0), f"Rotation {i} has determinant {det}, expected 1"

    # 测试旋转唯一性
    unique_rots = set()
    for R in ROTATION_MATRICES:
        unique_rots.add(tuple(map(tuple, R)))

    assert len(unique_rots) == 24, f"Expected 24 unique rotations, got {len(unique_rots)}"

    print("✓ Rotation tests passed")


def test_piece_preprocessing():
    """测试piece预处理"""
    print("\n=== Testing Piece Preprocessing ===")

    # 创建一个简单的L形piece
    local_voxels = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
    piece = PieceDef(id="test", local_voxels=local_voxels)

    # 预处理
    from loader import preprocess_piece
    piece = preprocess_piece(piece)

    assert len(piece.rotation_signatures) > 0, "No rotation signatures generated"
    assert len(piece.rotation_signatures) <= 24, "Too many rotation signatures"

    print(f"✓ Piece preprocessing tests passed (generated {len(piece.rotation_signatures)} unique rotations)")


def test_simple_placement():
    """测试简单的放置操作"""
    print("\n=== Testing Simple Placement ===")

    # 创建一个2x2x2的简单puzzle
    piece1 = PieceDef(
        id="0",
        local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0)]
    )
    piece2 = PieceDef(
        id="1",
        local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0)]
    )

    from loader import preprocess_piece
    from game_core import LevelSpec, GameState

    piece1 = preprocess_piece(piece1)
    piece2 = preprocess_piece(piece2)

    spec = LevelSpec(box=(2, 2, 2), pieces=[piece1, piece2])
    state = GameState(spec=spec)

    # 尝试放置piece0到底层
    target_cells = [Vec3(1, 1, 1), Vec3(2, 1, 1), Vec3(1, 2, 1), Vec3(2, 2, 1)]
    result = place_piece_by_cells(state, "0", target_cells)

    assert result.success, f"Placement failed: {result.message}"
    assert "0" in state.placed, "Piece not in placed dict"
    assert len(state.occupied) == 4, f"Expected 4 occupied cells, got {len(state.occupied)}"

    # 尝试放置piece1到顶层
    target_cells2 = [Vec3(1, 1, 2), Vec3(2, 1, 2), Vec3(1, 2, 2), Vec3(2, 2, 2)]
    result2 = place_piece_by_cells(state, "1", target_cells2)

    assert result2.success, f"Placement failed: {result2.message}"
    assert state.is_complete(), "Puzzle should be complete"

    print("✓ Simple placement tests passed")


def test_collision_detection():
    """测试碰撞检测"""
    print("\n=== Testing Collision Detection ===")

    from loader import preprocess_piece
    from game_core import LevelSpec, GameState, ErrorCode

    piece = PieceDef(id="0", local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0)])
    piece = preprocess_piece(piece)

    spec = LevelSpec(box=(3, 3, 3), pieces=[piece])
    state = GameState(spec=spec)

    # 放置第一个
    result1 = place_piece_by_cells(state, "0", [Vec3(1, 1, 1), Vec3(2, 1, 1)])
    assert result1.success, "First placement should succeed"

    # 手动添加另一个piece来测试碰撞
    piece2 = PieceDef(id="1", local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0)])
    piece2 = preprocess_piece(piece2)
    state.spec.pieces.append(piece2)
    state.unplaced.add("1")

    # 尝试放置到相同位置
    result2 = place_piece_by_cells(state, "1", [Vec3(1, 1, 1), Vec3(2, 1, 1)])
    assert not result2.success, "Second placement should fail (collision)"
    assert result2.error == ErrorCode.COLLISION, f"Expected COLLISION, got {result2.error}"

    print("✓ Collision detection tests passed")


def test_support_check():
    """测试支撑检查"""
    print("\n=== Testing Support Check ===")

    from loader import preprocess_piece
    from game_core import LevelSpec, GameState, ErrorCode

    piece = PieceDef(id="0", local_voxels=[Vec3(0, 0, 0)])
    piece = preprocess_piece(piece)

    spec = LevelSpec(box=(3, 3, 3), pieces=[piece])
    state = GameState(spec=spec)

    # 在底部放置应该成功 (z=1)
    result1 = place_piece_by_cells(state, "0", [Vec3(1, 1, 1)])
    assert result1.success, "Bottom placement should succeed"

    # 取出
    pickup_piece(state, "0")

    # 在空中放置应该失败 (z=2, no support)
    result2 = place_piece_by_cells(state, "0", [Vec3(1, 1, 2)])
    assert not result2.success, "Floating placement should fail"
    assert result2.error == ErrorCode.FLOATING, f"Expected FLOATING, got {result2.error}"

    print("✓ Support check tests passed")


def test_bounds_check():
    """测试边界检查"""
    print("\n=== Testing Bounds Check ===")

    from loader import preprocess_piece
    from game_core import LevelSpec, GameState, ErrorCode

    piece = PieceDef(id="0", local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0)])
    piece = preprocess_piece(piece)

    spec = LevelSpec(box=(2, 2, 2), pieces=[piece])
    state = GameState(spec=spec)

    # 超出边界
    result = place_piece_by_cells(state, "0", [Vec3(2, 1, 1), Vec3(3, 1, 1)])
    assert not result.success, "Out of bounds placement should fail"
    assert result.error == ErrorCode.OUT_OF_BOUNDS, f"Expected OUT_OF_BOUNDS, got {result.error}"

    print("✓ Bounds check tests passed")


def test_real_puzzle():
    """测试加载真实puzzle"""
    print("\n=== Testing Real Puzzle ===")

    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"
    spec = load_puzzle_by_name(puzzles_dir, "2x2x2", "puzzle_001")

    if not spec:
        print("⚠ Skipping real puzzle test (puzzle not found)")
        return

    assert spec.box == (2, 2, 2), f"Expected box (2,2,2), got {spec.box}"
    assert len(spec.pieces) == 2, f"Expected 2 pieces, got {len(spec.pieces)}"

    # 检查piece预处理
    for piece in spec.pieces:
        assert len(piece.rotation_signatures) > 0, f"Piece {piece.id} has no rotation signatures"
        assert len(piece.local_voxels) > 0, f"Piece {piece.id} has no voxels"

    print(f"✓ Real puzzle test passed (loaded 2x2x2/puzzle_001)")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("Running Stacking Game Tests")
    print("=" * 50)

    tests = [
        test_vec3,
        test_normalization,
        test_rotations,
        test_piece_preprocessing,
        test_simple_placement,
        test_collision_detection,
        test_support_check,
        test_bounds_check,
        test_real_puzzle,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed.append(test.__name__)

    print("\n" + "=" * 50)
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
    else:
        print("✅ All tests passed!")
    print("=" * 50)

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
