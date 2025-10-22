"""
演示脚本 - 自动解决一个简单的2x2x2 puzzle
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_core import Vec3
from loader import load_puzzle_by_name, create_game_state
from placement import place_piece_by_cells
import json


def visualize_state(state):
    """可视化当前状态"""
    A, B, C = state.spec.box
    print("\n=== 3D View (Layer by Layer) ===")

    for z in range(C, 0, -1):
        print(f"\nLayer z={z}:")
        for y in range(B, 0, -1):
            row = ""
            for x in range(1, A + 1):
                key = Vec3(x, y, z).to_key()
                if key in state.by_cell:
                    piece_id = state.by_cell[key]
                    row += f"[{piece_id}]"
                else:
                    row += " · "
            print(f"  {row}")


def demo_2x2x2():
    """演示解决2x2x2 puzzle"""
    print("=" * 60)
    print("3D Polycube Stacking Game - Demo")
    print("=" * 60)

    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"

    # 加载puzzle
    print("\n📦 Loading puzzle: 2x2x2/puzzle_001")
    spec = load_puzzle_by_name(puzzles_dir, "2x2x2", "puzzle_001")

    if not spec:
        print("❌ Failed to load puzzle")
        return False

    state = create_game_state(spec)

    print(f"✓ Box size: {spec.box[0]}x{spec.box[1]}x{spec.box[2]}")
    print(f"✓ Number of pieces: {len(spec.pieces)}")

    # 显示pieces信息
    print("\n📋 Pieces:")
    for piece in spec.pieces:
        print(f"  Piece {piece.id}: {len(piece.local_voxels)} cells")
        print(f"    Coordinates: {[v.to_tuple() for v in piece.local_voxels]}")
        print(f"    Unique rotations: {len(piece.rotation_signatures)}")

    visualize_state(state)

    # 加载solution
    json_path = f"{puzzles_dir}/2x2x2/puzzle_001/puzzle_001_2x2x2.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    solution = data.get('solution', {})

    if not solution:
        print("\n⚠ No solution found in puzzle data")
        return False

    print(f"\n🎯 Found solution with {len(solution)} pieces")

    # 按solution放置pieces
    for piece_id, target_coords in solution.items():
        print(f"\n▶ Placing piece {piece_id}...")

        # 转换坐标 (JSON是0-based, 我们需要1-based)
        target_cells = [Vec3(c[0] + 1, c[1] + 1, c[2] + 1) for c in target_coords]

        print(f"  Target cells: {[c.to_tuple() for c in target_cells]}")

        result = place_piece_by_cells(state, piece_id, target_cells)

        if result.success:
            print(f"  ✓ {result.message}")
            visualize_state(state)
        else:
            print(f"  ✗ {result.error.value}: {result.message}")
            return False

    # 检查完成
    if state.is_complete():
        print("\n" + "=" * 60)
        print("🎉 PUZZLE SOLVED! 🎉")
        print("=" * 60)
        visualize_state(state)

        # 统计
        print("\n📊 Statistics:")
        print(f"  Total cells: {state.spec.box[0] * state.spec.box[1] * state.spec.box[2]}")
        print(f"  Occupied cells: {len(state.occupied)}")
        print(f"  Placed pieces: {len(state.placed)}")
        return True
    else:
        print("\n⚠ Puzzle not complete")
        print(f"  Occupied: {len(state.occupied)}/{state.spec.box[0] * state.spec.box[1] * state.spec.box[2]}")
        return False


def demo_3x3x3():
    """演示解决3x3x3 puzzle"""
    print("\n" + "=" * 60)
    print("Demo: 3x3x3 Puzzle")
    print("=" * 60)

    puzzles_dir = "/mnt/moonfs/wuyuhao-m2/wyh/VisualReasonBench/Stacking_scaling/puzzles_full_v9"

    print("\n📦 Loading puzzle: 3x3x3/puzzle_001")
    spec = load_puzzle_by_name(puzzles_dir, "3x3x3", "puzzle_001")

    if not spec:
        print("❌ Failed to load puzzle")
        return False

    state = create_game_state(spec)

    print(f"✓ Box size: {spec.box[0]}x{spec.box[1]}x{spec.box[2]}")
    print(f"✓ Number of pieces: {len(spec.pieces)}")

    # 显示初始状态
    visualize_state(state)

    # 加载solution
    json_path = f"{puzzles_dir}/3x3x3/puzzle_001/puzzle_001_3x3x3.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    solution = data.get('solution', {})
    assembly_order = data.get('assembly_order', [])

    print(f"\n🎯 Solving with {len(solution)} pieces...")

    # assembly_order可能是拆卸顺序,我们需要逆序来装配
    if assembly_order:
        # assembly_order是 [[piece_id, direction], ...]
        piece_order = [str(item[0]) for item in reversed(assembly_order)]
        print(f"   Using reversed assembly order")
    else:
        piece_order = sorted(solution.keys())

    # 按顺序放置pieces
    for i, piece_id in enumerate(piece_order):
        if piece_id not in solution:
            continue

        target_coords = solution[piece_id]
        # 转换坐标
        target_cells = [Vec3(c[0] + 1, c[1] + 1, c[2] + 1) for c in target_coords]

        result = place_piece_by_cells(state, piece_id, target_cells)

        if result.success:
            print(f"  ✓ Piece {piece_id} placed ({i+1}/{len(solution)})")
        else:
            print(f"  ✗ Piece {piece_id} failed: {result.error.value}")
            print(f"    Target cells: {[c.to_tuple() for c in target_cells]}")
            return False

    # 显示最终状态
    if state.is_complete():
        print("\n🎉 3x3x3 PUZZLE SOLVED! 🎉")
        visualize_state(state)
        return True
    else:
        print(f"\n⚠ Puzzle not complete: {len(state.occupied)}/27 cells")
        visualize_state(state)
        return False


def main():
    """主函数"""
    # Demo 1: 2x2x2 puzzle (详细演示)
    success1 = demo_2x2x2()

    # Demo 2: 3x3x3 puzzle (快速演示)
    success2 = demo_3x3x3()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All demos completed successfully!")
    else:
        print("⚠ Some demos failed")
    print("=" * 60)

    print("\n💡 To play interactively, run:")
    print("   python game_cli.py")


if __name__ == "__main__":
    main()
