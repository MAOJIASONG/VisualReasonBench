"""
完整演示脚本 - 展示3D Polycube Stacking Game的所有功能

这个脚本演示:
1. 加载不同大小的puzzle (2x2x2, 3x3x3)
2. 初始化pieces在地面上（随机旋转）
3. 生成两种可视化:
   - 完整状态图（box + pieces）
   - Pieces网格图（每个piece单独显示）
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from loader import load_puzzle_by_name, create_game_state
from initialization import initialize_pieces_on_ground
from visualizer_3d import visualize_state_3d, save_visualization
from visualizer_pieces import save_pieces_visualization

def demo_puzzle(puzzles_dir: str, size: str, puzzle_id: str, seed: int = 42):
    """演示一个puzzle的完整可视化"""

    print(f"\n{'='*60}")
    print(f"  Demo: {size} {puzzle_id}")
    print(f"{'='*60}")

    # 加载puzzle
    spec = load_puzzle_by_name(puzzles_dir, size, puzzle_id)
    if not spec:
        print(f"✗ Failed to load puzzle {size}/{puzzle_id}")
        return

    state = create_game_state(spec)

    print(f"✓ Loaded puzzle: {size}/{puzzle_id}")
    print(f"  Box size: {spec.box[0]}x{spec.box[1]}x{spec.box[2]}")
    print(f"  Pieces: {len(spec.pieces)}")
    print(f"  Total voxels: {sum(len(p.local_voxels) for p in spec.pieces)}")

    # 初始化pieces在地面上
    initialize_pieces_on_ground(state, spacing=2, seed=seed)

    print(f"\n✓ Initialized pieces on ground:")
    for piece_id in sorted(state.unplaced):
        if piece_id in state.initial_placements:
            placement = state.initial_placements[piece_id]
            cells = placement.world_cells
            min_z = min(c.z for c in cells)
            max_z = max(c.z for c in cells)
            print(f"  Piece {piece_id}: rotation={placement.transform.rot}, "
                  f"z=[{min_z},{max_z}], voxels={len(cells)}")

    # 生成完整状态图
    print(f"\n📊 Generating visualizations...")

    state_file = f"tmp/demo_{size}_{puzzle_id}_state.png"
    fig = visualize_state_3d(
        state,
        title=f"{size} {puzzle_id} - Initial State",
        show_unplaced=True
    )
    save_visualization(fig, state_file, dpi=200)
    print(f"  ✓ State visualization: {state_file}")

    # 生成pieces网格图
    pieces_file = f"tmp/demo_{size}_{puzzle_id}_pieces.png"
    save_pieces_visualization(
        state,
        pieces_file,
        title=f"{size} {puzzle_id} - Pieces",
        dpi=200
    )
    print(f"  ✓ Pieces grid: {pieces_file}")

    print(f"\n✓ Demo completed for {size}/{puzzle_id}")


def main():
    """主函数 - 演示多个puzzle"""

    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"

    print("\n" + "="*60)
    print("  3D Polycube Stacking Game - Complete Demo")
    print("="*60)
    print("\n这个演示将生成以下可视化:")
    print("  1. 完整状态图: 显示空box和所有未放置的pieces")
    print("  2. Pieces网格图: 每个piece单独显示在自己的子图中")
    print("\n所有图片将保存在 tmp/ 目录")

    # 演示2x2x2 puzzle
    demo_puzzle(puzzles_dir, "2x2x2", "puzzle_001", seed=42)

    # 演示3x3x3 puzzle
    demo_puzzle(puzzles_dir, "3x3x3", "puzzle_001", seed=123)

    print("\n" + "="*60)
    print("  All demos completed!")
    print("="*60)
    print("\n生成的文件:")
    print("  tmp/demo_2x2x2_puzzle_001_state.png")
    print("  tmp/demo_2x2x2_puzzle_001_pieces.png")
    print("  tmp/demo_3x3x3_puzzle_001_state.png")
    print("  tmp/demo_3x3x3_puzzle_001_pieces.png")
    print("\n使用 game_3d_file.py 来交互式游玩puzzle:")
    print("  python game_3d_file.py")
    print()


if __name__ == "__main__":
    main()
