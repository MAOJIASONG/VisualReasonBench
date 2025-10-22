"""
简单示例 - 展示如何使用游戏系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_core import Vec3, PieceDef, LevelSpec, GameState
from loader import preprocess_piece
from placement import place_piece_by_transform, pickup_piece


def simple_example():
    """一个简单的手工示例"""
    print("=" * 60)
    print("Simple Example: 2x2x2 Box with Two Pieces")
    print("=" * 60)

    # 创建两个简单的piece
    # Piece 0: 一个2x2的方块
    piece0 = PieceDef(
        id="0",
        local_voxels=[
            Vec3(0, 0, 0), Vec3(1, 0, 0),
            Vec3(0, 1, 0), Vec3(1, 1, 0)
        ]
    )

    # Piece 1: 另一个2x2的方块
    piece1 = PieceDef(
        id="1",
        local_voxels=[
            Vec3(0, 0, 0), Vec3(1, 0, 0),
            Vec3(0, 1, 0), Vec3(1, 1, 0)
        ]
    )

    # 预处理
    piece0 = preprocess_piece(piece0)
    piece1 = preprocess_piece(piece1)

    # 创建关卡
    spec = LevelSpec(box=(2, 2, 2), pieces=[piece0, piece1])
    state = GameState(spec=spec)

    print("\n📋 Setup:")
    print(f"  Box: 2x2x2")
    print(f"  Piece 0: 4 cells (2x2 square)")
    print(f"  Piece 1: 4 cells (2x2 square)")

    # 可视化函数
    def show_state():
        A, B, C = state.spec.box
        print("\n=== 3D View ===")
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

    show_state()

    # 步骤1: 放置piece 0在底层
    print("\n▶ Step 1: Place piece 0 at bottom (z=1)")
    result = place_piece_by_transform(state, "0", rot=0, position=Vec3(1, 1, 1))

    if result.success:
        print(f"  ✓ Success!")
        show_state()
    else:
        print(f"  ✗ Failed: {result.message}")
        return False

    # 步骤2: 放置piece 1在顶层
    print("\n▶ Step 2: Place piece 1 at top (z=2)")
    result = place_piece_by_transform(state, "1", rot=0, position=Vec3(1, 1, 2))

    if result.success:
        print(f"  ✓ Success!")
        show_state()
    else:
        print(f"  ✗ Failed: {result.message}")
        return False

    # 检查完成
    if state.is_complete():
        print("\n" + "=" * 60)
        print("🎉 PUZZLE COMPLETE! 🎉")
        print("=" * 60)
        return True
    else:
        print(f"\n⚠ Not complete: {len(state.occupied)}/8 cells occupied")
        return False


def l_shaped_example():
    """L形piece示例"""
    print("\n" + "=" * 60)
    print("L-Shaped Piece Example")
    print("=" * 60)

    # 创建一个L形piece
    piece = PieceDef(
        id="0",
        local_voxels=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
    )
    piece = preprocess_piece(piece)

    print(f"\n📋 Piece has {len(piece.local_voxels)} cells")
    print(f"   Coordinates: {[v.to_tuple() for v in piece.local_voxels]}")
    print(f"   Unique rotations: {len(piece.rotation_signatures)}")

    # 创建一个3x3x1的盒子
    spec = LevelSpec(box=(3, 3, 1), pieces=[piece])
    state = GameState(spec=spec)

    def show_2d():
        print("\n=== Top View ===")
        for y in range(3, 0, -1):
            row = ""
            for x in range(1, 4):
                key = Vec3(x, y, 1).to_key()
                if key in state.by_cell:
                    row += "[0]"
                else:
                    row += " · "
            print(f"  {row}")

    show_2d()

    # 尝试不同的旋转
    print("\n▶ Testing different rotations:")

    for rot in range(min(4, len(piece.rotation_signatures))):
        print(f"\n  Rotation {rot}:")
        result = place_piece_by_transform(state, "0", rot=rot, position=Vec3(1, 1, 1))

        if result.success:
            print(f"    ✓ Placed successfully")
            show_2d()
            pickup_piece(state, "0")
        else:
            print(f"    ✗ Failed: {result.error.value}")

    return True


def main():
    """主函数"""
    success1 = simple_example()
    success2 = l_shaped_example()

    print("\n" + "=" * 60)
    if success1:
        print("✅ Examples completed!")
    else:
        print("⚠ Some examples failed")
    print("=" * 60)

    print("\n💡 Next steps:")
    print("  1. Run tests: python test_game.py")
    print("  2. Try demo with real puzzles: python demo.py")
    print("  3. Play interactively: python game_cli.py")


if __name__ == "__main__":
    main()
