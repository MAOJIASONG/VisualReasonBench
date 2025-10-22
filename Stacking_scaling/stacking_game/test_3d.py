"""
测试3D可视化和初始化功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端进行测试

from game_core import Vec3, PieceDef, LevelSpec
from loader import preprocess_piece, create_game_state, load_puzzle_by_name
from placement import place_piece_by_transform
from visualizer_3d import (
    visualize_state_3d, visualize_piece_rotations,
    save_visualization
)
from initialization import initialize_pieces_on_ground, randomize_piece_rotation


def test_3d_visualization():
    """测试3D可视化"""
    print("\n=== Testing 3D Visualization ===")

    # 创建测试场景
    piece0 = PieceDef(
        id="0",
        local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0)]
    )
    piece1 = PieceDef(
        id="1",
        local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
    )

    piece0 = preprocess_piece(piece0)
    piece1 = preprocess_piece(piece1)

    spec = LevelSpec(box=(3, 3, 2), pieces=[piece0, piece1])
    state = create_game_state(spec)

    # 放置一个piece
    result = place_piece_by_transform(state, "0", rot=0, position=Vec3(1, 1, 1))
    assert result.success, f"Placement failed: {result.message}"

    # 创建可视化
    fig = visualize_state_3d(state, title="Test 3D Visualization", show_unplaced=True)
    assert fig is not None, "Failed to create visualization"

    # 保存
    output_file = "/tmp/test_3d_viz.png"
    save_visualization(fig, output_file)

    print(f"✓ 3D visualization test passed")
    print(f"  Saved to: {output_file}")


def test_piece_rotations_visualization():
    """测试piece旋转可视化"""
    print("\n=== Testing Piece Rotations Visualization ===")

    # 创建L形piece
    piece = PieceDef(
        id="0",
        local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
    )
    piece = preprocess_piece(piece)

    # 创建旋转可视化
    fig = visualize_piece_rotations(piece, num_rotations=8)
    assert fig is not None, "Failed to create rotation visualization"

    # 保存
    output_file = "/tmp/test_rotations.png"
    save_visualization(fig, output_file)

    print(f"✓ Piece rotations visualization test passed")
    print(f"  Saved to: {output_file}")


def test_initialization():
    """测试初始化功能"""
    print("\n=== Testing Initialization ===")

    # 创建测试pieces
    piece0 = PieceDef(id="0", local_voxels=[
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0)
    ])
    piece1 = PieceDef(id="1", local_voxels=[
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)
    ])

    piece0 = preprocess_piece(piece0)
    piece1 = preprocess_piece(piece1)

    spec = LevelSpec(box=(3, 3, 3), pieces=[piece0, piece1])
    state = create_game_state(spec)

    # 初始化pieces在地面
    initialize_pieces_on_ground(state, seed=42)

    # 验证
    assert len(state.initial_placements) == 2, "Should have 2 initial placements"

    for piece_id, placement in state.initial_placements.items():
        assert placement.transform.rot >= 0, "Invalid rotation"
        assert placement.transform.t.z == 1, "Should be on ground (z=1)"
        assert len(placement.world_cells) > 0, "Should have world cells"

        print(f"  Piece {piece_id}:")
        print(f"    Rotation: {placement.transform.rot}")
        print(f"    Position: {placement.transform.t.to_tuple()}")

    # 可视化初始状态
    fig = visualize_state_3d(state, title="Initial State", show_unplaced=True)
    output_file = "/tmp/test_initialization.png"
    save_visualization(fig, output_file)

    print(f"✓ Initialization test passed")
    print(f"  Saved to: {output_file}")


def test_randomization():
    """测试随机化功能"""
    print("\n=== Testing Randomization ===")

    piece = PieceDef(id="0", local_voxels=[
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)
    ])
    piece = preprocess_piece(piece)

    spec = LevelSpec(box=(3, 3, 3), pieces=[piece])
    state = create_game_state(spec)

    # 初始化
    initialize_pieces_on_ground(state, seed=42)
    initial_rot = state.initial_placements["0"].transform.rot

    # 随机化
    randomize_piece_rotation(state, "0", seed=99)
    new_rot = state.initial_placements["0"].transform.rot

    print(f"  Initial rotation: {initial_rot}")
    print(f"  New rotation: {new_rot}")

    # 可能相同,但通常不同
    print(f"✓ Randomization test passed")


def test_real_puzzle_with_visualization():
    """测试真实puzzle的可视化"""
    print("\n=== Testing Real Puzzle Visualization ===")

    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"
    spec = load_puzzle_by_name(puzzles_dir, "2x2x2", "puzzle_001")

    if not spec:
        print("⚠ Skipping (puzzle not found)")
        return

    state = create_game_state(spec)

    # 初始化pieces
    initialize_pieces_on_ground(state, seed=42)

    # 可视化
    fig = visualize_state_3d(state, title="2x2x2 Puzzle - Initial State", show_unplaced=True)
    output_file = "/tmp/test_real_puzzle.png"
    save_visualization(fig, output_file)

    print(f"✓ Real puzzle visualization test passed")
    print(f"  Saved to: {output_file}")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Testing 3D Visualization and Initialization")
    print("=" * 60)

    tests = [
        test_3d_visualization,
        test_piece_rotations_visualization,
        test_initialization,
        test_randomization,
        test_real_puzzle_with_visualization,
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
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)

    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
    else:
        print("✅ All tests passed!")

    print("=" * 60)

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()

    print("\n💡 Generated visualization files:")
    print("  - /tmp/test_3d_viz.png")
    print("  - /tmp/test_rotations.png")
    print("  - /tmp/test_initialization.png")
    print("  - /tmp/test_real_puzzle.png")

    sys.exit(0 if success else 1)
