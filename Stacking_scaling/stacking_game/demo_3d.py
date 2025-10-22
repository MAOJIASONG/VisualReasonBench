"""
3D可视化演示 - 展示所有新功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
# 使用Agg后端生成图片,不显示窗口
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from game_core import Vec3
from loader import load_puzzle_by_name, create_game_state
from placement import place_piece_by_transform
from visualizer_3d import (
    visualize_state_3d,
    visualize_piece_rotations,
    save_visualization
)
from initialization import initialize_pieces_on_ground


def demo_3d_visualization():
    """演示3D可视化功能"""
    print("=" * 70)
    print("3D Visualization Demo")
    print("=" * 70)

    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"

    # 加载2x2x2 puzzle
    print("\n📦 Loading 2x2x2 puzzle...")
    spec = load_puzzle_by_name(puzzles_dir, "2x2x2", "puzzle_001")

    if not spec:
        print("❌ Failed to load puzzle")
        return False

    state = create_game_state(spec)
    print(f"✓ Loaded puzzle with {len(spec.pieces)} pieces")

    # 初始化pieces在地面外
    print("\n🎲 Initializing pieces with random rotations...")
    initialize_pieces_on_ground(state, seed=42)

    for piece_id, placement in state.initial_placements.items():
        print(f"  Piece {piece_id}: rotation={placement.transform.rot}, " +
              f"position={placement.transform.t.to_tuple()}")

    # 可视化初始状态
    print("\n📊 Creating initial state visualization...")
    fig1 = visualize_state_3d(
        state,
        title="2x2x2 Puzzle - Initial State (Pieces Outside Box)",
        show_unplaced=True
    )
    output1 = "/tmp/demo_initial_state.png"
    save_visualization(fig1, output1, dpi=200)
    print(f"✓ Saved: {output1}")

    # 放置第一个piece
    print("\n▶ Placing piece 0 at bottom...")
    result = place_piece_by_transform(state, "0", rot=0, position=Vec3(1, 1, 1))

    if result.success:
        print(f"✓ Piece 0 placed successfully")

        # 可视化部分完成状态
        fig2 = visualize_state_3d(
            state,
            title="2x2x2 Puzzle - Piece 0 Placed",
            show_unplaced=True
        )
        output2 = "/tmp/demo_partial.png"
        save_visualization(fig2, output2, dpi=200)
        print(f"✓ Saved: {output2}")
    else:
        print(f"✗ Failed: {result.message}")

    # 放置第二个piece
    print("\n▶ Placing piece 1 at top...")
    result = place_piece_by_transform(state, "1", rot=0, position=Vec3(1, 1, 2))

    if result.success:
        print(f"✓ Piece 1 placed successfully")

        # 可视化完成状态
        fig3 = visualize_state_3d(
            state,
            title="2x2x2 Puzzle - COMPLETE!",
            show_unplaced=False
        )
        output3 = "/tmp/demo_complete.png"
        save_visualization(fig3, output3, dpi=200)
        print(f"✓ Saved: {output3}")

        if state.is_complete():
            print("\n🎉 Puzzle completed!")
    else:
        print(f"✗ Failed: {result.message}")

    # 显示piece的不同旋转
    print("\n🔄 Creating piece rotations visualization...")
    piece = state.get_piece_def("0")
    fig4 = visualize_piece_rotations(piece, num_rotations=8)
    output4 = "/tmp/demo_piece_rotations.png"
    save_visualization(fig4, output4, dpi=150)
    print(f"✓ Saved: {output4}")

    return True


def demo_3x3x3_initial_state():
    """演示3x3x3 puzzle的初始状态"""
    print("\n" + "=" * 70)
    print("3x3x3 Puzzle Initial State Demo")
    print("=" * 70)

    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"

    print("\n📦 Loading 3x3x3 puzzle...")
    spec = load_puzzle_by_name(puzzles_dir, "3x3x3", "puzzle_001")

    if not spec:
        print("❌ Failed to load puzzle")
        return False

    state = create_game_state(spec)
    print(f"✓ Loaded puzzle with {len(spec.pieces)} pieces")

    # 初始化
    print("\n🎲 Initializing pieces...")
    initialize_pieces_on_ground(state, seed=123, spacing=2)

    # 可视化
    print("\n📊 Creating visualization...")
    fig = visualize_state_3d(
        state,
        title=f"3x3x3 Puzzle - Initial State ({len(spec.pieces)} pieces)",
        show_unplaced=True
    )
    output = "/tmp/demo_3x3x3_initial.png"
    save_visualization(fig, output, dpi=200)
    print(f"✓ Saved: {output}")

    # 显示每个piece的信息
    print("\n📋 Piece information:")
    for piece in spec.pieces:
        print(f"  Piece {piece.id}:")
        print(f"    Voxels: {len(piece.local_voxels)}")
        print(f"    Unique rotations: {len(piece.rotation_signatures)}")

        if piece.id in state.initial_placements:
            placement = state.initial_placements[piece.id]
            print(f"    Initial rotation: {placement.transform.rot}")
            print(f"    Initial position: {placement.transform.t.to_tuple()}")

    return True


def main():
    """主函数"""
    print("\n" + "🎮" * 35)
    print("3D Polycube Stacking Game - Visualization Demo")
    print("🎮" * 35 + "\n")

    # Demo 1: 2x2x2 完整流程
    success1 = demo_3d_visualization()

    # Demo 2: 3x3x3 初始状态
    success2 = demo_3x3x3_initial_state()

    # 总结
    print("\n" + "=" * 70)
    if success1 and success2:
        print("✅ All demos completed successfully!")
    else:
        print("⚠ Some demos failed")
    print("=" * 70)

    print("\n📁 Generated visualizations:")
    print("  1. /tmp/demo_initial_state.png    - Initial state with pieces outside")
    print("  2. /tmp/demo_partial.png          - Partial completion (1 piece placed)")
    print("  3. /tmp/demo_complete.png         - Complete puzzle")
    print("  4. /tmp/demo_piece_rotations.png  - Piece rotations showcase")
    print("  5. /tmp/demo_3x3x3_initial.png    - 3x3x3 initial state")

    print("\n💡 Key features demonstrated:")
    print("  ✓ 3D visualization with matplotlib")
    print("  ✓ Pieces initialized outside the box on the ground")
    print("  ✓ Random rotations for each piece")
    print("  ✓ Side-by-side view of box and unplaced pieces")
    print("  ✓ Color-coded pieces")
    print("  ✓ Multiple viewing angles")

    print("\n🚀 Try the interactive game:")
    print("  python game_3d.py")

    return success1 and success2


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
