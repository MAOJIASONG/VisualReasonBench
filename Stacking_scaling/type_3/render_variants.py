"""
FluidNet 3D - Rendering Style Bakeoff

Produces the SAME game state in several visual styles so we can judge
which one communicates the task (pipe topology, cell roles, fluid state,
coordinates) most clearly.

Styles:
    A. voxel_clean    - simple shaded cubes, bold flat colors, axis ticks
                        on every integer.  Maximally legible at small size.
    B. labeled        - plotly pipes + spheres, plus 3D text labels at
                        every valve/source/target/protected showing (x,y,z).
    C. schematic      - flat 2D layer slices (one subplot per z) with
                        connectivity arrows between layers.  Pure matplotlib.
    D. triview        - composite: perspective 3D + top-down + side-view
                        all in one figure.  matplotlib subplots.
    E. pipes_plotly   - previous best plotly renderer (for reference).

Output: /tmp/fluidnet_variants/<style>/seed70_<frame>.png
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# Package-relative imports (run as `python -m type_3.render_variants`)
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from type_3.game_core import CellType, GameState, face_neighbors
    from type_3.generator import generate_level
    from type_3.loader import create_game_state
    from type_3.propagation import step_open_valve
    from type_3.visualizer_plotly import render_state as render_plotly
else:
    from .game_core import CellType, GameState, face_neighbors
    from .generator import generate_level
    from .loader import create_game_state
    from .propagation import step_open_valve
    from .visualizer_plotly import render_state as render_plotly


# ---------------------------------------------------------------------------
# Shared palette
# ---------------------------------------------------------------------------

FLUID = {
    "blue":   "#2E86DE", "red": "#EE5253", "green": "#10AC84",
    "orange": "#FF9F43", "purple": "#8854D0", "cyan": "#00D2D3",
}
FLUID_SOFT = {k: to_rgba(v, 0.35) for k, v in FLUID.items()}
PIPE     = "#6C7A89"
VALVE    = "#F5C043"
HANDLE   = "#2D3436"
PROTECT  = "#D94A4A"
PROTECT_DK = "#6E1212"
FRAME    = "#8E9199"
GRID_GHOST = "#D6D8DD"
BG = "#F6F7F9"


def _is_conductive(state: GameState, pos) -> bool:
    c = state.spec.cells.get(pos)
    if c is None:
        return False
    if c.type in (CellType.PIPE, CellType.SOURCE, CellType.TARGET):
        return True
    if c.type == CellType.VALVE:
        return pos in state.opened_valves
    return False


# ---------------------------------------------------------------------------
# STYLE A  voxel_clean  -- minimal cubes + axis ticks every unit
# ---------------------------------------------------------------------------

def _cube_faces(pos, size=0.88):
    cx, cy, cz = pos[0] - 0.5, pos[1] - 0.5, pos[2] - 0.5
    d = size / 2
    v = np.array([
        [cx-d, cy-d, cz-d], [cx+d, cy-d, cz-d],
        [cx+d, cy+d, cz-d], [cx-d, cy+d, cz-d],
        [cx-d, cy-d, cz+d], [cx+d, cy-d, cz+d],
        [cx+d, cy+d, cz+d], [cx-d, cy+d, cz+d],
    ])
    return [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
        [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]],
    ]


def _add_cube(ax, pos, color, alpha, edge="black", linewidth=0.8, size=0.88):
    coll = Poly3DCollection(_cube_faces(pos, size=size),
                            facecolors=to_rgba(color, alpha),
                            edgecolors=edge, linewidths=linewidth)
    ax.add_collection3d(coll)


def style_voxel_clean(state: GameState, save_path: str, title: str = "") -> None:
    spec = state.spec
    A, B, C = spec.grid_size
    fig = plt.figure(figsize=(9, 8), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)

    # Box frame
    verts = [(0,0,0),(A,0,0),(A,B,0),(0,B,0),
             (0,0,C),(A,0,C),(A,B,C),(0,B,C)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    segs = [np.array([verts[a], verts[b]]) for a, b in edges]
    ax.add_collection3d(Line3DCollection(segs, colors=FRAME, linewidths=1.6))

    # Floor grid + integer tick marker cubes at ground for coord reference
    for pos, cell in spec.cells.items():
        fluid = state.fluid_cells.get(pos)
        t = cell.type
        if t == CellType.PIPE:
            col = FLUID[fluid] if fluid else PIPE
            _add_cube(ax, pos, col, alpha=0.85 if fluid else 0.65, size=0.55)
        elif t == CellType.SOURCE:
            col = FLUID[cell.color]
            _add_cube(ax, pos, col, alpha=0.95, size=0.9, linewidth=1.3)
        elif t == CellType.TARGET:
            col = FLUID[cell.color]
            filled = fluid == cell.color
            _add_cube(ax, pos, col,
                      alpha=0.85 if filled else 0.18,
                      size=0.9, linewidth=1.8, edge=col)
        elif t == CellType.VALVE:
            opened = pos in state.opened_valves
            if opened:
                col = FLUID[fluid] if fluid else PIPE
                _add_cube(ax, pos, col, alpha=0.85, size=0.75,
                          edge=VALVE, linewidth=1.5)
            else:
                _add_cube(ax, pos, VALVE, alpha=0.95, size=0.8,
                          edge=HANDLE, linewidth=1.2)
        elif t == CellType.PROTECTED:
            breached = fluid is not None
            _add_cube(ax, pos, PROTECT_DK if breached else PROTECT,
                      alpha=0.95, size=0.88, edge="#2B0606",
                      linewidth=1.6 if breached else 1.0)

    ax.set_xlim(0, A); ax.set_ylim(0, B); ax.set_zlim(0, C)
    ax.set_xticks(range(0, A + 1)); ax.set_yticks(range(0, B + 1))
    ax.set_zticks(range(0, C + 1))
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect((A, B, C))
    ax.view_init(elev=24, azim=-55)
    ax.grid(True, color=GRID_GHOST, linewidth=0.5)
    ax.set_title(f"[A · voxel_clean] {title}", fontsize=11)
    _add_compact_legend(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _add_compact_legend(ax):
    handles = [
        mpatches.Patch(color="#2E86DE", label="source / fluid"),
        mpatches.Patch(color=VALVE, label="valve (closed)"),
        mpatches.Patch(color=PIPE, label="pipe"),
        mpatches.Patch(color=PROTECT, label="protected"),
        mpatches.Patch(facecolor="none", edgecolor="#2E86DE",
                       linewidth=1.8, label="target (empty)"),
        mpatches.Patch(color=to_rgba("#2E86DE", 0.6), label="target (filled)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# STYLE B  labeled  -- plotly render + 3D text labels at every key cell
# ---------------------------------------------------------------------------

def style_labeled(state: GameState, save_path: str, title: str = "") -> None:
    import plotly.graph_objects as go
    fig = render_plotly(state, save_path="/tmp/_tmp_plotly.png",
                        title=f"[B · labeled] {title}")
    # Overlay text labels on another render by editing the figure directly.
    # Re-render via the underlying figure then write to save_path.
    spec = state.spec
    labels_x, labels_y, labels_z, labels_txt = [], [], [], []
    for s in spec.sources:
        labels_x.append(s[0] - 0.5); labels_y.append(s[1] - 0.5); labels_z.append(s[2] - 0.5 + 0.6)
        labels_txt.append(f"S {s}")
    for t in spec.targets:
        labels_x.append(t[0] - 0.5); labels_y.append(t[1] - 0.5); labels_z.append(t[2] - 0.5 + 0.55)
        labels_txt.append(f"T {t}")
    for i, v in enumerate(spec.valves):
        opened = v in state.opened_valves
        tag = "open" if opened else "V"
        labels_x.append(v[0] - 0.5); labels_y.append(v[1] - 0.5); labels_z.append(v[2] - 0.5 + 0.55)
        labels_txt.append(f"{tag} {v}")
    for p in spec.protected:
        labels_x.append(p[0] - 0.5); labels_y.append(p[1] - 0.5); labels_z.append(p[2] - 0.5 + 0.55)
        labels_txt.append(f"X {p}")
    fig.add_trace(go.Scatter3d(
        x=labels_x, y=labels_y, z=labels_z,
        mode="text",
        text=labels_txt,
        textfont=dict(color="#1A1A1A", size=11, family="monospace"),
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(title=dict(text=f"[B · labeled] {title}",
                                  x=0.5, xanchor="center"))
    fig.write_image(save_path, width=1000, height=850, scale=1.4)


# ---------------------------------------------------------------------------
# STYLE C  schematic  -- layered 2D slices with inter-layer arrows
# ---------------------------------------------------------------------------

def style_schematic(state: GameState, save_path: str, title: str = "") -> None:
    spec = state.spec
    A, B, C = spec.grid_size
    fig, axes = plt.subplots(1, C, figsize=(4.5 * C, 5.5), facecolor=BG)
    if C == 1:
        axes = [axes]

    for z_idx, ax in enumerate(axes):
        z = z_idx + 1
        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(0.3, A + 0.7); ax.set_ylim(0.3, B + 0.7)
        ax.set_aspect("equal")
        ax.set_xticks(range(1, A + 1)); ax.set_yticks(range(1, B + 1))
        ax.grid(True, color="#E5E7EC", linewidth=0.6)
        ax.set_title(f"layer z={z}", fontsize=11, pad=6)

        # Draw intra-layer connections between conductive cells
        drawn: set = set()
        for pos in spec.cells:
            if pos[2] != z:
                continue
            cell = spec.cells[pos]
            if cell.type == CellType.PROTECTED:
                continue
            for nb in face_neighbors(pos, spec.grid_size):
                if nb[2] != z:
                    continue
                nb_cell = spec.cells.get(nb)
                if nb_cell is None or nb_cell.type == CellType.PROTECTED:
                    continue
                if not (_is_conductive(state, pos) or _is_conductive(state, nb)):
                    continue
                edge = tuple(sorted((pos, nb)))
                if edge in drawn:
                    continue
                drawn.add(edge)
                fluid = state.fluid_cells.get(pos) or state.fluid_cells.get(nb)
                if nb_cell.type == CellType.VALVE and nb not in state.opened_valves:
                    fluid = None
                if cell.type == CellType.VALVE and pos not in state.opened_valves:
                    fluid = None
                col = FLUID[fluid] if fluid else PIPE
                lw = 4 if fluid else 2
                ax.plot([pos[0], nb[0]], [pos[1], nb[1]],
                        color=col, linewidth=lw, solid_capstyle="round",
                        zorder=2)

        # Draw cells as markers
        for pos, cell in spec.cells.items():
            if pos[2] != z:
                continue
            x, y = pos[0], pos[1]
            fluid = state.fluid_cells.get(pos)
            t = cell.type
            if t == CellType.SOURCE:
                ax.scatter(x, y, s=600, c=FLUID[cell.color],
                           marker="o", edgecolors="#1A3D5C",
                           linewidths=1.8, zorder=3)
                ax.text(x, y, "S", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold",
                        zorder=4)
            elif t == CellType.TARGET:
                col = FLUID[cell.color]
                filled = fluid == cell.color
                ax.scatter(x, y, s=600, c=col if filled else "#FFFFFF",
                           marker="s", edgecolors=col, linewidths=2.5, zorder=3)
                ax.text(x, y, "T", ha="center", va="center",
                        color="white" if filled else col,
                        fontsize=11, fontweight="bold", zorder=4)
            elif t == CellType.VALVE:
                opened = pos in state.opened_valves
                col_bg = FLUID[fluid] if (opened and fluid) else VALVE
                ax.scatter(x, y, s=500,
                           c=col_bg if opened else VALVE,
                           marker="D", edgecolors=HANDLE,
                           linewidths=1.8, zorder=3)
                label = "v" if opened else "V"
                ax.text(x, y, label, ha="center", va="center",
                        color="#2D3436", fontsize=11, fontweight="bold",
                        zorder=4)
            elif t == CellType.PROTECTED:
                breached = fluid is not None
                ax.scatter(x, y, s=560, c=PROTECT_DK if breached else PROTECT,
                           marker="X", edgecolors="#1A1A1A",
                           linewidths=1.6, zorder=3)
            elif t == CellType.PIPE and fluid is None:
                ax.scatter(x, y, s=80, c=PIPE, marker="o",
                           edgecolors="none", zorder=3)

        # Inter-layer indicator: a tiny arrow for vertical links
        for pos, cell in spec.cells.items():
            if pos[2] != z:
                continue
            below = (pos[0], pos[1], pos[2] - 1)
            above = (pos[0], pos[1], pos[2] + 1)
            for nb, direction in [(below, "↓"), (above, "↑")]:
                nb_cell = spec.cells.get(nb)
                if nb_cell is None or nb_cell.type == CellType.PROTECTED:
                    continue
                if not (_is_conductive(state, pos) or _is_conductive(state, nb)):
                    continue
                fluid = state.fluid_cells.get(pos) or state.fluid_cells.get(nb)
                if nb_cell.type == CellType.VALVE and nb not in state.opened_valves:
                    fluid = None
                col = FLUID[fluid] if fluid else PIPE
                ax.text(pos[0] + 0.32, pos[1] - 0.32, direction,
                        color=col, fontsize=12, fontweight="bold",
                        zorder=5)

    fig.suptitle(f"[C · schematic] {title}", fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# STYLE D  triview  -- 3D + top-down + side-view composite
# ---------------------------------------------------------------------------

def style_triview(state: GameState, save_path: str, title: str = "") -> None:
    spec = state.spec
    A, B, C = spec.grid_size
    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_top = fig.add_subplot(1, 3, 2)
    ax_side = fig.add_subplot(1, 3, 3)

    # --- 3D panel (voxel_clean style) ---
    ax3d.set_facecolor(BG)
    verts = [(0,0,0),(A,0,0),(A,B,0),(0,B,0),
             (0,0,C),(A,0,C),(A,B,C),(0,B,C)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    ax3d.add_collection3d(Line3DCollection(
        [np.array([verts[a], verts[b]]) for a, b in edges],
        colors=FRAME, linewidths=1.5))
    for pos, cell in spec.cells.items():
        fluid = state.fluid_cells.get(pos)
        _draw_voxel_in_3d(ax3d, pos, cell, fluid, state)
    ax3d.set_xlim(0, A); ax3d.set_ylim(0, B); ax3d.set_zlim(0, C)
    ax3d.set_box_aspect((A, B, C))
    ax3d.view_init(elev=22, azim=-55)
    ax3d.set_title("perspective", fontsize=10)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

    # --- Top view (x vs y, collapsed over z) ---
    ax_top.set_facecolor("#FFFFFF")
    ax_top.set_xlim(0.4, A + 0.6); ax_top.set_ylim(0.4, B + 0.6)
    ax_top.set_aspect("equal")
    ax_top.set_xticks(range(1, A + 1)); ax_top.set_yticks(range(1, B + 1))
    ax_top.grid(True, color="#E5E7EC", linewidth=0.6)
    ax_top.set_title("top view  (X, Y — highest cell per column)", fontsize=10)
    _draw_projection(ax_top, state, axis_x=0, axis_y=1, collapse_over=2)

    # --- Side view (x vs z, collapsed over y) ---
    ax_side.set_facecolor("#FFFFFF")
    ax_side.set_xlim(0.4, A + 0.6); ax_side.set_ylim(0.4, C + 0.6)
    ax_side.set_aspect("equal")
    ax_side.set_xticks(range(1, A + 1)); ax_side.set_yticks(range(1, C + 1))
    ax_side.grid(True, color="#E5E7EC", linewidth=0.6)
    ax_side.set_title("side view  (X, Z — deepest cell per column)", fontsize=10)
    _draw_projection(ax_side, state, axis_x=0, axis_y=2, collapse_over=1)

    fig.suptitle(f"[D · triview] {title}", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_voxel_in_3d(ax, pos, cell, fluid, state):
    t = cell.type
    if t == CellType.PIPE:
        col = FLUID[fluid] if fluid else PIPE
        _add_cube(ax, pos, col, alpha=0.85 if fluid else 0.65, size=0.55)
    elif t == CellType.SOURCE:
        _add_cube(ax, pos, FLUID[cell.color], alpha=0.95, size=0.9, linewidth=1.3)
    elif t == CellType.TARGET:
        col = FLUID[cell.color]
        filled = fluid == cell.color
        _add_cube(ax, pos, col,
                  alpha=0.85 if filled else 0.18,
                  size=0.9, linewidth=1.8, edge=col)
    elif t == CellType.VALVE:
        opened = pos in state.opened_valves
        if opened:
            col = FLUID[fluid] if fluid else PIPE
            _add_cube(ax, pos, col, alpha=0.85, size=0.75,
                      edge=VALVE, linewidth=1.5)
        else:
            _add_cube(ax, pos, VALVE, alpha=0.95, size=0.8,
                      edge=HANDLE, linewidth=1.2)
    elif t == CellType.PROTECTED:
        breached = fluid is not None
        _add_cube(ax, pos, PROTECT_DK if breached else PROTECT,
                  alpha=0.95, size=0.88, edge="#2B0606",
                  linewidth=1.6 if breached else 1.0)


def _draw_projection(ax, state, axis_x: int, axis_y: int, collapse_over: int):
    """
    Project 3D cells onto a 2D plane.  For each (x, y) bucket we pick the
    most important cell to display based on type priority.
    """
    spec = state.spec
    grid = spec.grid_size

    priority = {
        CellType.PROTECTED: 7, CellType.SOURCE: 6, CellType.TARGET: 5,
        CellType.VALVE: 4, CellType.PIPE: 2,
    }

    # Bucket cells by projected position
    buckets: Dict[Tuple[int, int], List] = {}
    for pos, cell in spec.cells.items():
        key = (pos[axis_x], pos[axis_y])
        buckets.setdefault(key, []).append((pos, cell))

    # Draw connectivity lines (within the projected plane)
    drawn: set = set()
    for pos, cell in spec.cells.items():
        if cell.type == CellType.PROTECTED:
            continue
        for nb in face_neighbors(pos, grid):
            if nb[collapse_over] != pos[collapse_over]:
                continue
            nb_cell = spec.cells.get(nb)
            if nb_cell is None or nb_cell.type == CellType.PROTECTED:
                continue
            if not (_is_conductive(state, pos) or _is_conductive(state, nb)):
                continue
            edge = tuple(sorted((pos, nb)))
            if edge in drawn:
                continue
            drawn.add(edge)
            fluid = state.fluid_cells.get(pos) or state.fluid_cells.get(nb)
            if nb_cell.type == CellType.VALVE and nb not in state.opened_valves:
                fluid = None
            if cell.type == CellType.VALVE and pos not in state.opened_valves:
                fluid = None
            col = FLUID[fluid] if fluid else PIPE
            lw = 3.5 if fluid else 1.8
            ax.plot([pos[axis_x], nb[axis_x]],
                    [pos[axis_y], nb[axis_y]],
                    color=col, linewidth=lw, solid_capstyle="round",
                    alpha=0.9, zorder=2)

    # Draw the top-priority cell per bucket
    for key, items in buckets.items():
        items.sort(key=lambda pc: -priority.get(pc[1].type, 0))
        pos, cell = items[0]
        x, y = key
        fluid = state.fluid_cells.get(pos)
        t = cell.type
        if t == CellType.SOURCE:
            ax.scatter(x, y, s=260, c=FLUID[cell.color],
                       marker="o", edgecolors="#1A3D5C",
                       linewidths=1.5, zorder=3)
        elif t == CellType.TARGET:
            col = FLUID[cell.color]
            filled = fluid == cell.color
            ax.scatter(x, y, s=260, c=col if filled else "#FFFFFF",
                       marker="s", edgecolors=col, linewidths=2.2, zorder=3)
        elif t == CellType.VALVE:
            opened = pos in state.opened_valves
            col_bg = FLUID[fluid] if (opened and fluid) else VALVE
            ax.scatter(x, y, s=240, c=col_bg if opened else VALVE,
                       marker="D", edgecolors=HANDLE,
                       linewidths=1.4, zorder=3)
        elif t == CellType.PROTECTED:
            breached = fluid is not None
            ax.scatter(x, y, s=260, c=PROTECT_DK if breached else PROTECT,
                       marker="X", edgecolors="#1A1A1A",
                       linewidths=1.4, zorder=3)
        elif t == CellType.PIPE and fluid is None:
            ax.scatter(x, y, s=40, c=PIPE, marker="o",
                       edgecolors="none", zorder=2)


# ---------------------------------------------------------------------------
# STYLE E  pipes_plotly  (reference baseline)
# ---------------------------------------------------------------------------

def style_pipes_plotly(state: GameState, save_path: str, title: str = "") -> None:
    render_plotly(state, save_path=save_path, title=f"[E · pipes_plotly] {title}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STYLES = {
    "A_voxel_clean":   style_voxel_clean,
    "B_labeled":       style_labeled,
    "C_schematic":     style_schematic,
    "D_triview":       style_triview,
    "E_pipes_plotly":  style_pipes_plotly,
}


def main():
    out_dir = "/tmp/fluidnet_variants"
    os.makedirs(out_dir, exist_ok=True)

    spec, meta = generate_level(
        grid_size=(5, 5, 5), n_targets=2, n_valves=6, n_protected=3, seed=70,
    )
    # Make a few interesting snapshots: initial, mid-play, win, breach.
    snapshots: List[Tuple[str, GameState]] = []
    snapshots.append(("initial", create_game_state(spec)))

    mid = create_game_state(spec)
    step_open_valve(mid, (3, 3, 3))
    step_open_valve(mid, (3, 4, 3))
    snapshots.append(("mid_step2", mid))

    win = create_game_state(spec)
    for v in [(3, 3, 3), (3, 4, 3), (3, 5, 4)]:
        step_open_valve(win, v)
    snapshots.append(("win", win))

    breach = create_game_state(spec)
    step_open_valve(breach, (3, 4, 1))
    snapshots.append(("breach", breach))

    for style, func in STYLES.items():
        sdir = os.path.join(out_dir, style)
        os.makedirs(sdir, exist_ok=True)
        for name, st in snapshots:
            out = os.path.join(sdir, f"seed70_{name}.png")
            title = f"seed70 · {name}"
            print(f"  rendering {style:20s} / {name:10s} -> {out}")
            func(st, out, title=title)

    print(f"\nall variants under {out_dir}/")


if __name__ == "__main__":
    main()
