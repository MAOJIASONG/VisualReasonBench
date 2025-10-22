"""
3D可视化模块 - 使用matplotlib绘制3D视图
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from typing import List, Dict, Optional, Tuple
from matplotlib.colors import to_rgba

from game_core import Vec3, GameState, PieceDef


# 预定义的颜色方案
PIECE_COLORS = [
    '#FF6B6B',  # 红色
    '#4ECDC4',  # 青色
    '#45B7D1',  # 蓝色
    '#96CEB4',  # 绿色
    '#FFEAA7',  # 黄色
    '#DFE6E9',  # 灰色
    '#FD79A8',  # 粉色
    '#A29BFE',  # 紫色
    '#74B9FF',  # 浅蓝
    '#55EFC4',  # 薄荷绿
    '#FDCB6E',  # 橙色
    '#E17055',  # 橙红
]


def get_piece_color(piece_id: str) -> str:
    """获取piece的颜色"""
    try:
        idx = int(piece_id)
        return PIECE_COLORS[idx % len(PIECE_COLORS)]
    except:
        return PIECE_COLORS[0]


def create_cube_vertices(pos: Vec3, size: float = 0.9) -> np.ndarray:
    """
    创建单位立方体的顶点

    Args:
        pos: 立方体中心位置 (1-based)
        size: 立方体大小 (0-1之间,留出间隙)

    Returns:
        8x3的顶点数组
    """
    # 将1-based坐标转换为实际坐标
    cx, cy, cz = pos.x - 0.5, pos.y - 0.5, pos.z - 0.5
    d = size / 2.0

    vertices = np.array([
        [cx - d, cy - d, cz - d],  # 0: 左下前
        [cx + d, cy - d, cz - d],  # 1: 右下前
        [cx + d, cy + d, cz - d],  # 2: 右上前
        [cx - d, cy + d, cz - d],  # 3: 左上前
        [cx - d, cy - d, cz + d],  # 4: 左下后
        [cx + d, cy - d, cz + d],  # 5: 右下后
        [cx + d, cy + d, cz + d],  # 6: 右上后
        [cx - d, cy + d, cz + d],  # 7: 左上后
    ])

    return vertices


def create_cube_faces(vertices: np.ndarray) -> List[np.ndarray]:
    """
    创建立方体的6个面

    Returns:
        6个面,每个面是4个顶点的数组
    """
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 前面 (z min)
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 后面 (z max)
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 底面 (y min)
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 顶面 (y max)
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面 (x min)
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面 (x max)
    ]

    return faces


def draw_voxel(ax: Axes3D, pos: Vec3, color: str, alpha: float = 0.8,
               edge_color: str = 'black', linewidth: float = 0.5):
    """
    绘制一个体素(单位立方体)

    Args:
        ax: matplotlib 3D axes
        pos: 体素位置 (1-based)
        color: 颜色
        alpha: 透明度
        edge_color: 边缘颜色
        linewidth: 边缘线宽
    """
    vertices = create_cube_vertices(pos)
    faces = create_cube_faces(vertices)

    # 创建3D多边形集合
    face_collection = Poly3DCollection(
        faces,
        facecolors=to_rgba(color, alpha),
        edgecolors=edge_color,
        linewidths=linewidth
    )

    ax.add_collection3d(face_collection)


def draw_box_frame(ax: Axes3D, box_size: Tuple[int, int, int],
                   color: str = 'gray', linewidth: float = 2.0):
    """
    绘制盒子的线框

    Args:
        ax: matplotlib 3D axes
        box_size: (A, B, C) 盒子尺寸
        color: 线框颜色
        linewidth: 线宽
    """
    A, B, C = box_size

    # 定义盒子的8个顶点 (0-based坐标系统)
    vertices = np.array([
        [0, 0, 0],  # 0
        [A, 0, 0],  # 1
        [A, B, 0],  # 2
        [0, B, 0],  # 3
        [0, 0, C],  # 4
        [A, 0, C],  # 5
        [A, B, C],  # 6
        [0, B, C],  # 7
    ])

    # 定义12条边
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],  # 竖边
    ]

    for edge in edges:
        points = vertices[edge]
        ax.plot3D(*points.T, color=color, linewidth=linewidth, alpha=0.3)


def visualize_state_3d(state: GameState,
                       title: str = "3D Polycube Puzzle",
                       show_unplaced: bool = True,
                       figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    可视化游戏状态(3D视图)

    Args:
        state: 游戏状态
        title: 图表标题
        show_unplaced: 是否显示未放置的piece
        figsize: 图表大小

    Returns:
        matplotlib Figure对象
    """
    A, B, C = state.spec.box

    if show_unplaced and state.unplaced:
        # 创建两个子图:盒子 + 未放置的pieces
        fig = plt.figure(figsize=figsize)

        # 主盒子视图
        ax1 = fig.add_subplot(121, projection='3d')
        _draw_box_view(ax1, state, title)

        # 未放置pieces视图
        ax2 = fig.add_subplot(122, projection='3d')
        _draw_unplaced_pieces(ax2, state)

    else:
        # 只显示盒子
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        _draw_box_view(ax, state, title)

    plt.tight_layout()
    return fig


def _draw_box_view(ax: Axes3D, state: GameState, title: str):
    """绘制盒子视图"""
    A, B, C = state.spec.box

    # 绘制盒子线框
    draw_box_frame(ax, (A, B, C))

    # 绘制已放置的pieces
    for piece_id, placed in state.placed.items():
        color = get_piece_color(piece_id)
        for cell in placed.world_cells:
            draw_voxel(ax, cell, color, alpha=0.7)

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.5, A + 0.5])
    ax.set_ylim([-0.5, B + 0.5])
    ax.set_zlim([-0.5, C + 0.5])

    # 设置视角
    ax.view_init(elev=20, azim=45)

    # 标题
    occupied = len(state.occupied)
    total = A * B * C
    status = "COMPLETE!" if state.is_complete() else f"{occupied}/{total} cells"
    ax.set_title(f"{title}\n{status}")


def _draw_unplaced_pieces(ax: Axes3D, state: GameState):
    """绘制未放置的pieces（使用initial_placements中的位置和旋转）"""
    ax.set_title("Unplaced Pieces")

    if not state.initial_placements:
        ax.text(0.5, 0.5, 0.5, "No pieces", ha='center', va='center')
        return

    # 找到所有初始放置的边界
    all_cells = []
    for piece_id in state.unplaced:
        if piece_id in state.initial_placements:
            all_cells.extend(state.initial_placements[piece_id].world_cells)

    if not all_cells:
        return

    # 计算边界
    min_x = min(c.x for c in all_cells)
    max_x = max(c.x for c in all_cells)
    min_y = min(c.y for c in all_cells)
    max_y = max(c.y for c in all_cells)
    min_z = min(c.z for c in all_cells)
    max_z = max(c.z for c in all_cells)

    # 绘制每个未放置的piece
    for piece_id in sorted(state.unplaced):
        if piece_id not in state.initial_placements:
            continue

        placement = state.initial_placements[piece_id]
        color = get_piece_color(piece_id)

        # 绘制piece的每个体素
        for cell in placement.world_cells:
            draw_voxel(ax, cell, color, alpha=0.7)

    # 绘制地面网格（帮助理解pieces在地上）
    ground_z = 0.5  # 稍微低于z=1
    ground_x = [min_x - 1, max_x + 1]
    ground_y = [min_y - 1, max_y + 1]

    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches

    # 画一个地面矩形
    xx, yy = np.meshgrid(ground_x, ground_y)
    zz = np.ones_like(xx) * ground_z
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([min_x - 1, max_x + 1])
    ax.set_ylim([min_y - 1, max_y + 1])
    ax.set_zlim([0, max_z + 1])

    ax.view_init(elev=20, azim=45)


def visualize_piece_rotations(piece: PieceDef,
                              num_rotations: int = 8,
                              title: Optional[str] = None) -> plt.Figure:
    """
    可视化piece的多个旋转

    Args:
        piece: piece定义
        num_rotations: 显示的旋转数量
        title: 图表标题

    Returns:
        matplotlib Figure对象
    """
    from rotation import ROTATION_MATRICES
    import numpy as np

    num_to_show = min(num_rotations, len(piece.rotation_signatures))

    # 计算子图布局
    cols = 4
    rows = (num_to_show + cols - 1) // cols

    fig = plt.figure(figsize=(3 * cols, 3 * rows))

    color = get_piece_color(piece.id)

    for i in range(num_to_show):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        # 应用旋转
        rot_matrix = ROTATION_MATRICES[i]
        rotated_voxels = []

        for v in piece.local_voxels:
            vec = np.array([v.x, v.y, v.z])
            rotated = rot_matrix @ vec
            rotated_voxels.append(Vec3(int(rotated[0]), int(rotated[1]), int(rotated[2])))

        # 规范化到(1,1,1)起点
        min_x = min(v.x for v in rotated_voxels)
        min_y = min(v.y for v in rotated_voxels)
        min_z = min(v.z for v in rotated_voxels)

        for v in rotated_voxels:
            pos = Vec3(v.x - min_x + 1, v.y - min_y + 1, v.z - min_z + 1)
            draw_voxel(ax, pos, color, alpha=0.7)

        # 设置坐标轴
        max_coord = max(
            max(v.x - min_x for v in rotated_voxels),
            max(v.y - min_y for v in rotated_voxels),
            max(v.z - min_z for v in rotated_voxels)
        ) + 1

        ax.set_xlim([0, max_coord + 1])
        ax.set_ylim([0, max_coord + 1])
        ax.set_zlim([0, max_coord + 1])
        ax.set_title(f"Rotation {i}")
        ax.view_init(elev=20, azim=45)

        # 隐藏坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f"Piece {piece.id} - {len(piece.rotation_signatures)} Unique Rotations",
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def animate_placement(state: GameState,
                     piece_id: str,
                     target_cells: List[Vec3],
                     num_frames: int = 20) -> List[plt.Figure]:
    """
    创建放置动画的帧序列

    Args:
        state: 游戏状态
        piece_id: 要放置的piece ID
        target_cells: 目标位置
        num_frames: 动画帧数

    Returns:
        图表列表
    """
    from placement import place_piece_by_cells

    # 先尝试放置(不提交)
    result = place_piece_by_cells(state, piece_id, target_cells)

    if not result.success:
        print(f"Cannot animate: {result.message}")
        return []

    # TODO: 实现动画逻辑
    # 这里可以添加piece从初始位置移动到目标位置的动画

    return []


def save_visualization(fig: plt.Figure, filename: str, dpi: int = 150):
    """
    保存可视化图表

    Args:
        fig: matplotlib Figure对象
        filename: 输出文件名
        dpi: 分辨率
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved visualization to {filename}")


def show_visualization(fig: plt.Figure):
    """显示可视化图表"""
    plt.show()


# 测试代码
if __name__ == "__main__":
    print("Testing 3D Visualizer...")

    # 创建一个简单的测试场景
    from game_core import PieceDef, LevelSpec, GameState
    from loader import preprocess_piece
    from placement import place_piece_by_transform

    # 创建piece
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

    # 创建关卡
    spec = LevelSpec(box=(3, 3, 2), pieces=[piece0, piece1])
    state = GameState(spec=spec)

    # 放置一些pieces
    place_piece_by_transform(state, "0", rot=0, position=Vec3(1, 1, 1))

    # 可视化
    print("Creating visualization...")
    fig = visualize_state_3d(state, title="Test Scene", show_unplaced=True)

    print("Displaying... (close window to continue)")
    show_visualization(fig)

    # 显示piece的旋转
    print("\nShowing piece rotations...")
    fig2 = visualize_piece_rotations(piece1, num_rotations=8)
    show_visualization(fig2)

    print("Done!")
