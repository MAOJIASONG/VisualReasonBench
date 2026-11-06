"""
3D可视化模块 - 使用matplotlib绘制3D视图
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from typing import List, Dict, Optional, Tuple
from matplotlib.colors import to_rgba

from game_core import Vec3, GameState, PieceDef


# 预定义的颜色方案 —— 高对比度、互相可区分（不含接近白色的颜色，
# 否则在浅色背景上几乎看不见）
PIECE_COLORS = [
    '#E6194B',  # 红
    '#3CB44B',  # 绿
    '#4363D8',  # 蓝
    '#F58231',  # 橙
    '#911EB4',  # 紫
    '#42D4F4',  # 青
    '#F032E6',  # 品红
    '#9A6324',  # 棕
    '#808000',  # 橄榄
    '#469990',  # 蓝绿
    '#FFE119',  # 黄
    '#000075',  # 深蓝
    '#800000',  # 栗色
    '#BFEF45',  # 黄绿
    '#FABED4',  # 粉
    '#A9A9A9',  # 灰
]
# 16 种颜色：4x4x4 关卡最多 13 块，每块颜色都不同（原先只有 12 种，#12 与 #0 同色）


def piece_sort_key(piece_id: str):
    """按数字排序 piece id（'2' 在 '10' 之前），非数字 id 排在后面。"""
    return (0, int(piece_id), '') if str(piece_id).isdigit() else (1, 0, str(piece_id))


def label_color(color: str) -> str:
    """标题文字颜色：浅色（黄、黄绿、粉）在白底上看不清，改用深灰。"""
    r, g, b, _ = to_rgba(color)
    return '#333333' if 0.299 * r + 0.587 * g + 0.114 * b > 0.72 else color


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
               edge_color: str = 'black', linewidth: float = 0.8):
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


# 方块绘制。所有面放进同一个 Poly3DCollection，由 matplotlib 统一按深度排序
# （每个体素一个 collection 时只能整块排序，会出现后面的方块盖住前面的情况）。
#
# CUBE_STYLE:
#   "glass"（默认）：略透明（alpha 0.9）、单色、方块之间留 0.1 的缝、画全部 6 个面。
#       透过去能隐约看到被挡住的棱，深度不会读反。
#   "solid"：不透明、按方向着色（顶亮侧暗）、无缝、只画外表面。看起来更干净，但
#       实测 gpt-6.1-sol（effort low，one-shot，15 关 x 两个随机姿态）把 piece 读成
#       镜像的次数：glass 3-5 次，solid 14-15 次（加淡色坐标面板也没改善）；解出率
#       glass 0.67、solid 0.53，所以不作默认。
CUBE_STYLE = "glass"
FACE_SHADE = {(0, 0, 1): 1.0, (0, 0, -1): 0.5,
              (1, 0, 0): 0.80, (-1, 0, 0): 0.80,
              (0, 1, 0): 0.64, (0, -1, 0): 0.64}


def _face_quad(c: Tuple[int, int, int], n: Tuple[int, int, int],
               size: float = 1.0) -> List[Tuple[float, float, float]]:
    """1-based 格子 c（占据 [c-1, c]^3）在法向 n 一侧的面；size < 1 时向格子中心收缩。"""
    h = size / 2.0
    ctr = [c[0] - 0.5, c[1] - 0.5, c[2] - 0.5]
    ax = next(i for i in range(3) if n[i] != 0)
    u, v = [i for i in range(3) if i != ax]
    quad = []
    for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = [0.0, 0.0, 0.0]
        p[ax] = ctr[ax] + (h if n[ax] > 0 else -h)
        p[u] = ctr[u] + du * h
        p[v] = ctr[v] + dv * h
        quad.append(tuple(p))
    return quad


def draw_cells_solid(ax: Axes3D, groups, linewidth: float = 0.9,
                     style: Optional[str] = None):
    """
    groups: [(cells, color, alpha), ...]，cells 为 Vec3 或 (x, y, z)（1-based）。
    所有组的格子一起深度排序；style 默认取 CUBE_STYLE。
    """
    style = style or CUBE_STYLE
    glass = style == "glass"
    norm = []
    occupied = set()
    for cells, color, alpha in groups:
        cs = [(c.x, c.y, c.z) if isinstance(c, Vec3) else tuple(c) for c in cells]
        norm.append((cs, color, min(alpha, 0.9) if glass else alpha))
        occupied.update(cs)
    polys, faces, edges = [], [], []
    for cs, color, alpha in norm:
        r, g, b, _ = to_rgba(color)
        # 棱线：深色（深蓝、栗色）用浅色描边，否则看不出格子
        dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.3
        if dark:
            er, eg, eb = 0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b
        elif glass:
            er, eg, eb = 0.0, 0.0, 0.0
        else:
            er, eg, eb = r * 0.35, g * 0.35, b * 0.35
        for c in cs:
            for n, k in FACE_SHADE.items():
                if not glass and alpha >= 0.99 and \
                        (c[0] + n[0], c[1] + n[1], c[2] + n[2]) in occupied:
                    continue
                polys.append(_face_quad(c, n, 0.9 if glass else 1.0))
                k = 1.0 if glass else k
                faces.append((r * k, g * k, b * k, alpha))
                edges.append((er, eg, eb, min(1.0, alpha + 0.25)))
    if polys:
        ax.add_collection3d(Poly3DCollection(polys, facecolors=faces, edgecolors=edges,
                                             linewidths=linewidth))


def draw_floor_grid(ax: Axes3D, A: int, B: int, color=(0.75, 0.75, 0.78), lw: float = 0.6):
    """z=0 平面上的格线，每格一条，帮助读出 x / y。"""
    for i in range(A + 1):
        ax.plot3D([i, i], [0, B], [0, 0], color=color, linewidth=lw)
    for j in range(B + 1):
        ax.plot3D([0, A], [j, j], [0, 0], color=color, linewidth=lw)


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


# 两个相机：从相对的两个角往下看，盒子里每个格子至少在一个视角里可见；
# 单个 piece 的两个小视角同样来自这两个角，被挡住的方块在另一个视角里能看到。
BOX_VIEWS = (("front", 25, 45), ("back", 25, 225))
THUMB_ZOOM = 1.25


def visualize_state_3d(state: GameState,
                       title: str = "3D Polycube Puzzle",
                       show_unplaced: bool = True,
                       figsize: Optional[Tuple[float, float]] = None,
                       keep_unplaced_initial_pose: bool = True,
                       normalize_unplaced_view: bool = False,
                       pad_unplaced_view: int = 1,
                       two_views: bool = True) -> plt.Figure:
    """
    可视化游戏状态(3D视图)

    左侧：盒子（two_views=True 时为 front + back 两个相机，上下排列）；
    右侧：每个 piece 一格，按 id 数字顺序排列，每格两个小视角（两个相对的角）。

    Args:
        state: 游戏状态
        title: 图表标题
        show_unplaced: 是否显示右侧 piece 面板
        figsize: 图表大小，None 时按 piece 数自动决定
        keep_unplaced_initial_pose: 未放置piece保持加载时姿态与位置
        normalize_unplaced_view: 右侧小视角是否归一化到局部坐标
        pad_unplaced_view: 右侧小视角边界留白
        two_views: 盒子和 piece 都画两个相对的视角（False 为旧版单视角）

    Returns:
        matplotlib Figure对象
    """
    box_views = BOX_VIEWS if two_views else BOX_VIEWS[:1]
    piece_views = BOX_VIEWS if two_views else BOX_VIEWS[:1]

    if not (show_unplaced and state.initial_placements):
        if figsize is None:
            figsize = (8, 8 * len(box_views))
        fig = plt.figure(figsize=figsize)
        for i, (name, elev, azim) in enumerate(box_views):
            ax = fig.add_subplot(len(box_views), 1, i + 1, projection='3d')
            _draw_box_view(ax, state, title if i == 0 else "", elev=elev, azim=azim,
                           view_label=name if two_views else None)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.95)
        return fig

    piece_ids = sorted(state.initial_placements.keys(), key=piece_sort_key)
    n_pieces = len(piece_ids)
    right_cols = 2 if n_pieces <= 6 else 3
    right_rows = max(1, math.ceil(n_pieces / right_cols))
    if figsize is None:
        if two_views:
            figsize = (8 + 4.0 * right_cols, max(10.0, 2.6 * right_rows + 0.8))
        else:
            figsize = (14, 8)
    fig = plt.figure(figsize=figsize)
    # 先定边距：下面按 gridspec 位置摆的标题要用最终坐标
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.93)

    # 计算所有piece的最大尺寸，用于统一缩放比例
    max_span = 1.0
    for pid in piece_ids:
        placement = state.initial_placements.get(pid)
        if placement and placement.world_cells:
            cells = placement.world_cells
            for axis in ('x', 'y', 'z'):
                vals = [getattr(c, axis) for c in cells]
                max_span = max(max_span, max(vals) - min(vals) + 1)
    view_size = max_span + 0.5   # 紧凑取景：所有 piece 同一比例

    # 左右宽度比例与 figsize 无关（调用方传小 figsize 时也不会出错）
    gs = fig.add_gridspec(nrows=1, ncols=2,
                          width_ratios=[8.0, 4.0 * right_cols], wspace=0.06)
    gs_left = gs[0, 0].subgridspec(len(box_views), 1, hspace=0.16)
    for i, (name, elev, azim) in enumerate(box_views):
        ax = fig.add_subplot(gs_left[i, 0], projection='3d')
        _draw_box_view(ax, state, "", elev=elev, azim=azim,
                       view_label=name if two_views else None)
    A, B, C = state.spec.box
    status = "COMPLETE!" if state.is_complete() else f"{len(state.occupied)}/{A * B * C} cells"
    left_box = gs[0, 0].get_position(fig)
    fig.text((left_box.x0 + left_box.x1) / 2, 0.985, f"{title}   ({status})",
             ha="center", va="top", fontsize=13)

    gs_right = gs[0, 1].subgridspec(right_rows, right_cols, wspace=0.10, hspace=0.25)
    for idx, pid in enumerate(piece_ids):
        cell = gs_right[idx // right_cols, idx % right_cols]
        sub = cell.subgridspec(1, len(piece_views), wspace=0.0)
        is_placed = pid not in state.unplaced
        for j, (name, elev, azim) in enumerate(piece_views):
            ax = fig.add_subplot(sub[0, j], projection='3d')
            _draw_single_unplaced_piece(
                ax, state, pid,
                is_placed=is_placed,
                keep_initial_pose=keep_unplaced_initial_pose,
                normalize_view=normalize_unplaced_view,
                pad=pad_unplaced_view,
                unified_view_size=view_size,
                compact=True,
                elev=elev, azim=azim,
                show_title=False,
            )
        # 每个 piece 一个标题，居中放在它的两个小视角上方
        bbox = cell.get_position(fig)
        color = get_piece_color(pid)
        n_cells = len(state.initial_placements[pid].world_cells)
        text = f"#{pid} placed" if is_placed else f"#{pid}  ({n_cells} cells)"
        fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.004, text,
                 ha='center', va='bottom', fontsize=11,
                 fontweight='normal' if is_placed else 'bold',
                 color='#888888' if is_placed else label_color(color))

    header = "Pieces (each from the front and back corner)" if two_views \
        else "Pieces (initial pose)"
    right_box = gs[0, 1].get_position(fig)
    fig.text((right_box.x0 + right_box.x1) / 2, 0.985, header, ha="center",
             va="top", fontsize=12, color='#444444')
    return fig


def _draw_box_view(ax: Axes3D, state: GameState, title: str,
                   elev: float = 20, azim: float = 45,
                   view_label: Optional[str] = None):
    """绘制盒子视图"""
    A, B, C = state.spec.box

    # 绘制盒子线框
    draw_box_frame(ax, (A, B, C))

    draw_floor_grid(ax, A, B)

    # 绘制已放置的pieces（实心、按方向着色、统一深度排序）
    draw_cells_solid(ax, [(placed.world_cells, get_piece_color(pid), 1.0)
                          for pid, placed in state.placed.items()])

    # 已放置piece的编号：标在该piece露顶、且从当前相机看得见的体素上（文字总是
    # 画在最上层，标在被挡住的体素上会落到别的piece上，误导）。这个视角里看不见
    # 顶面的piece不标——另一个相机会标出它。
    occupied = {(c.x, c.y, c.z) for p in state.placed.values()
                for c in p.world_cells}
    e, az = math.radians(elev), math.radians(azim)
    to_cam = (math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e))

    def top_visible(c: Vec3) -> bool:
        # 从顶面中心朝相机走，碰到别的已占格子就是被挡住
        p0 = (c.x - 0.5, c.y - 0.5, c.z + 0.02)
        for k in range(1, 200):
            t = 0.05 * k
            q = tuple(p0[i] + t * to_cam[i] for i in range(3))
            cell = (math.floor(q[0]) + 1, math.floor(q[1]) + 1, math.floor(q[2]) + 1)
            if cell[2] > C:
                return True
            if cell in occupied:
                return False
        return True

    sx = 1 if to_cam[0] >= 0 else -1
    sy = 1 if to_cam[1] >= 0 else -1
    for piece_id, placed in state.placed.items():
        exposed = [c for c in placed.world_cells
                   if (c.x, c.y, c.z + 1) not in occupied and top_visible(c)]
        if not exposed:
            continue
        top = max(exposed, key=lambda c: (c.z, sx * c.x + sy * c.y, c.x))
        ax.text(top.x - 0.5, top.y - 0.5, top.z + 0.15, str(piece_id),
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='black', zorder=100,
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec=get_piece_color(piece_id), lw=1.5, alpha=0.9))

    # 坐标轴：刻度位于单元格中心，标签为 1-based，与 place 工具的坐标一致
    ax.set_xlabel('x', labelpad=6, fontsize=11, fontweight='bold')
    ax.set_ylabel('y', labelpad=6, fontsize=11, fontweight='bold')
    ax.set_zlabel('z', labelpad=6, fontsize=11, fontweight='bold')
    ax.set_xlim([0, A])
    ax.set_ylim([0, B])
    ax.set_zlim([0, C])
    ax.grid(False)
    ax.set_xticks([i + 0.5 for i in range(A)])
    ax.set_yticks([i + 0.5 for i in range(B)])
    ax.set_zticks([i + 0.5 for i in range(C)])
    ax.set_xticklabels([str(i + 1) for i in range(A)])
    ax.set_yticklabels([str(i + 1) for i in range(B)])
    ax.set_zticklabels([str(i + 1) for i in range(C)])
    for axis in ('x', 'y', 'z'):
        ax.tick_params(axis=axis, which='both', length=0, pad=1, labelsize=10)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.96, 0.96, 0.97, 1.0))
        pane.set_edgecolor((0.8, 0.8, 0.82, 1.0))
    try:
        ax.set_box_aspect((A, B, C))
    except Exception:
        pass

    # 设置视角
    ax.view_init(elev=elev, azim=azim)

    # 标题
    occupied = len(state.occupied)
    total = A * B * C
    status = "COMPLETE!" if state.is_complete() else f"{occupied}/{total} cells"
    head = f"{title}\n{status}" if title else ""
    if view_label:
        head = (head + "\n" if head else "") + f"{view_label} view"
    ax.set_title(head)


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
    for piece_id in sorted(state.unplaced, key=piece_sort_key):
        if piece_id not in state.initial_placements:
            continue

        placement = state.initial_placements[piece_id]
        color = get_piece_color(piece_id)

        # 绘制piece的每个体素
        for cell in placement.world_cells:
            draw_voxel(ax, cell, color, alpha=0.7)

    # 绘制地面网格（帮助理解pieces在地上）
    ground_z = 0.5  # 稍微低于z=1
    ground_x = [max(0, min_x - 1), max_x + 1]
    ground_y = [max(0, min_y - 1), max_y + 1]

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
    x0 = max(0, min_x - 1)
    y0 = max(0, min_y - 1)
    z0 = 0
    x1 = max_x + 1
    y1 = max_y + 1
    z1 = max_z + 1
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    ax.set_zlim([z0, z1])
    ax.set_xticks(list(range(int(x0), int(x1) + 1)))
    ax.set_yticks(list(range(int(y0), int(y1) + 1)))
    ax.set_zticks(list(range(int(z0), int(z1) + 1)))

    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=20, azim=45)


def _draw_single_unplaced_piece(ax: Axes3D, state: GameState, piece_id: str,
                                is_placed: bool = False,
                                keep_initial_pose: bool = True,
                                normalize_view: bool = False,
                                pad: int = 1,
                                unified_view_size: Optional[float] = None,
                                compact: bool = False,
                                elev: float = 20, azim: float = 45,
                                show_title: bool = True):
    """绘制单个piece的小视角（网格排布用）"""
    # keep_initial_pose: 保持初始加载姿态与位置
    # normalize_view: 归一化到局部坐标（与初始位置无关）
    # pad: 视角边界留白（单位为格）
    # unified_view_size: 统一的视图尺寸（用于保持所有piece比例一致）
    placement = state.initial_placements.get(piece_id)
    if not placement or not placement.world_cells:
        ax.text(0.5, 0.5, 0.5, "No data", ha='center', va='center')
        ax.set_axis_off()
        return

    color = get_piece_color(piece_id)
    min_x = min(c.x for c in placement.world_cells)
    max_x = max(c.x for c in placement.world_cells)
    min_y = min(c.y for c in placement.world_cells)
    max_y = max(c.y for c in placement.world_cells)
    min_z = min(c.z for c in placement.world_cells)
    max_z = max(c.z for c in placement.world_cells)

    if keep_initial_pose and not normalize_view:
        # 保持与加载时一致的朝向与位置
        draw_cells = placement.world_cells
        
        # 计算当前piece的中心（格子 c 占据 [c-1, c]，所以中心要减 0.5；
        # 以前直接用 (min+max)/2，取景偏了半格，底部方块会被裁掉）
        cx = (min_x + max_x) / 2 - 0.5
        cy = (min_y + max_y) / 2 - 0.5
        cz = (min_z + max_z) / 2 - 0.5
        
        if unified_view_size:
            # 使用统一的视图尺寸，居中显示
            d = unified_view_size / 2
            x0, x1 = cx - d, cx + d
            y0, y1 = cy - d, cy + d
            z0, z1 = cz - d, cz + d
        else:
            x0 = min_x - pad
            y0 = min_y - pad
            z0 = min_z - pad
            x1 = max_x + pad
            y1 = max_y + pad
            z1 = max_z + pad
    else:
        # 规范化到小视角坐标系，避免受全局摆放影响
        draw_cells = [
            Vec3(c.x - min_x + 1, c.y - min_y + 1, c.z - min_z + 1)
            for c in placement.world_cells
        ]
        
        if unified_view_size:
            # 规范化模式下如果也要求统一比例（虽然这里通常不需要）
            size_x = max_x - min_x + 1
            size_y = max_y - min_y + 1
            size_z = max_z - min_z + 1
            cx, cy, cz = size_x / 2, size_y / 2, size_z / 2   # 格子占 [0, size]
            d = unified_view_size / 2
            x0, x1 = cx - d, cx + d
            y0, y1 = cy - d, cy + d
            z0, z1 = cz - d, cz + d
        else:
            size_x = max_x - min_x + 1
            size_y = max_y - min_y + 1
            size_z = max_z - min_z + 1
            x0 = 0
            y0 = 0
            z0 = 0
            x1 = size_x + pad
            y1 = size_y + pad
            z1 = size_z + pad

    if is_placed:
        # 已放置：淡化显示（原来用的 ❌ emoji 在默认字体里会渲染成方块）
        draw_cells_solid(ax, [(draw_cells, color, 0.22)], linewidth=0.5)
    else:
        draw_cells_solid(ax, [(draw_cells, color, 1.0)])

    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    ax.set_zlim([z0, z1])
    # 设置正确的空间比例，防止变形；小图放大（3D 轴默认留白很大）。THUMB_ZOOM
    # 1.25 在全部 56 关（两个随机姿态）下都不裁掉方块；1.3 起开始有裁切
    try:
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=THUMB_ZOOM if compact else 1.0)
    except TypeError:  # matplotlib < 3.6 没有 zoom 参数
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))

    n_cells = len(placement.world_cells)
    if show_title and is_placed:
        ax.set_title(f"#{piece_id} placed", fontsize=9 if compact else 10,
                     color='#888888')
    elif show_title:
        label = f"#{piece_id} ({n_cells})" if compact else f"#{piece_id}  ({n_cells} cells)"
        ax.set_title(label, fontsize=9 if compact else 10,
                     fontweight='bold', color=label_color(color))
    ax.view_init(elev=elev, azim=azim)
    # 保留三块淡色坐标面板（不要刻度），给出地面和墙角的参照。
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.97, 0.97, 0.98, 1.0))
        pane.set_edgecolor((0.72, 0.72, 0.75, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((0.62, 0.62, 0.66, 1.0))


def visualize_piece_rotations(piece: PieceDef,
                              num_rotations: int = 8,
                              title: Optional[str] = None,
                              rotate_piece: bool = False) -> plt.Figure:
    """
    可视化piece的多个旋转

    Args:
        piece: piece定义
        num_rotations: 显示的旋转数量
        title: 图表标题
        rotate_piece: 是否对物体进行旋转

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
        if rotate_piece:
            rot_matrix = ROTATION_MATRICES[i]
            rotated_voxels = []

            for v in piece.local_voxels:
                vec = np.array([v.x, v.y, v.z])
                rotated = rot_matrix @ vec
                rotated_voxels.append(Vec3(int(rotated[0]), int(rotated[1]), int(rotated[2])))
        else:
            rotated_voxels = list(piece.local_voxels)

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
        # 强制为正方体比例，因为坐标轴范围一致
        ax.set_box_aspect((1, 1, 1))
        
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
    fig2 = visualize_piece_rotations(piece1, num_rotations=8, rotate_piece=False)
    show_visualization(fig2)

    print("Done!")
