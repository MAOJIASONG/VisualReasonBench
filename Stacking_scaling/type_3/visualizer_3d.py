"""
FluidNet 3D - Aesthetic Renderer

Replaces the earlier "stack of cubes" look with a pipe-network aesthetic:

  * PIPE / open-VALVE cells -> cylinders connecting cell centers to every
    conductive neighbor (actual pipes, not floating cubes).
  * SOURCE                  -> faceted sphere (colored droplet).
  * TARGET (empty)          -> glass frame (wireframe + translucent faces).
  * TARGET (filled)         -> same glass frame + coloured fluid fill inside.
  * VALVE (closed)          -> yellow body + grey handwheel on top.
  * VALVE (open)            -> ring collar fused to the pipe tube passing through.
  * PROTECTED (safe)        -> red warning block with diagonal hatch.
  * PROTECTED (breached)    -> same block with cracked-glass overlay + splash.

Everything is drawn via Poly3DCollection so we still render into a pure
matplotlib figure with no external dependencies.

The module exposes two public functions:
    render_state(state, save_path=None, title=None, ...)
    render_sequence(state_iter, out_dir, prefix, ...)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .game_core import CellType, GameState, face_neighbors


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

COLOR_FLUID: Dict[str, str] = {
    "blue":   "#2E86DE",
    "red":    "#EE5253",
    "green":  "#10AC84",
    "orange": "#FF9F43",
    "purple": "#8854D0",
    "cyan":   "#00D2D3",
}
COLOR_FLUID_LIGHT: Dict[str, str] = {
    "blue":   "#A8D5F7",
    "red":    "#F6B5B5",
    "green":  "#9DEFC9",
    "orange": "#FFD7A8",
    "purple": "#D4BEEF",
    "cyan":   "#A0EEEE",
}

PIPE_BODY       = "#8E9AAF"          # dry pipe cylinder
PIPE_RING       = "#555E6B"          # pipe flange dark accent
VALVE_BODY      = "#F7C948"          # closed-valve body (warm yellow)
VALVE_BODY_OPEN = "#FFE382"          # open-valve body (lighter)
VALVE_HANDLE    = "#2D3436"          # handwheel spokes
PROTECTED_FACE  = "#E54B4B"          # forbidden block
PROTECTED_DARK  = "#8B1E1E"          # breached shade
FRAME_COLOR     = "#8C9199"          # box outline
FRAME_FAINT     = "#D6D8DD"          # grid ghost
BG_COLOR        = "#F5F6F8"


# ---------------------------------------------------------------------------
# Primitive mesh helpers
# ---------------------------------------------------------------------------

def _cell_center(pos: Tuple[int, int, int]) -> np.ndarray:
    """Return the continuous-coord center of a 1-based grid cell."""
    return np.array([pos[0] - 0.5, pos[1] - 0.5, pos[2] - 0.5])


def _cylinder_mesh(p1: np.ndarray, p2: np.ndarray,
                   radius: float, n_seg: int = 18) -> List[np.ndarray]:
    """Quad faces for a capped cylinder between p1 and p2."""
    axis = p2 - p1
    L = np.linalg.norm(axis)
    if L < 1e-9:
        return []
    axis = axis / L
    # Perpendicular basis
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    thetas = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    c1 = np.stack([p1 + radius * (np.cos(t) * u + np.sin(t) * v) for t in thetas])
    c2 = np.stack([p2 + radius * (np.cos(t) * u + np.sin(t) * v) for t in thetas])
    faces: List[np.ndarray] = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append(np.array([c1[i], c1[j], c2[j], c2[i]]))
    # End caps (as fans) so the tube does not look hollow at elbows.
    faces.append(np.array(list(c1)))
    faces.append(np.array(list(c2)))
    return faces


def _sphere_mesh(center: np.ndarray, radius: float,
                 n_u: int = 14, n_v: int = 9) -> List[np.ndarray]:
    """Quad faces for a UV sphere."""
    thetas = np.linspace(0.0, 2 * np.pi, n_u, endpoint=False)
    phis   = np.linspace(-np.pi / 2, np.pi / 2, n_v)
    pts = np.zeros((n_v, n_u, 3))
    for i, phi in enumerate(phis):
        for j, theta in enumerate(thetas):
            pts[i, j] = center + radius * np.array([
                np.cos(phi) * np.cos(theta),
                np.cos(phi) * np.sin(theta),
                np.sin(phi),
            ])
    faces: List[np.ndarray] = []
    for i in range(n_v - 1):
        for j in range(n_u):
            k = (j + 1) % n_u
            faces.append(np.array([pts[i, j], pts[i, k],
                                    pts[i + 1, k], pts[i + 1, j]]))
    return faces


def _disc_mesh(center: np.ndarray, normal: np.ndarray, radius: float,
               n_seg: int = 20) -> np.ndarray:
    """Single polygon approximating a disc."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, ref); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    thetas = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    return np.stack([center + radius * (np.cos(t) * u + np.sin(t) * v) for t in thetas])


def _cube_edges(center: np.ndarray, size: float) -> List[np.ndarray]:
    d = size / 2.0
    cx, cy, cz = center
    v = np.array([
        [cx - d, cy - d, cz - d], [cx + d, cy - d, cz - d],
        [cx + d, cy + d, cz - d], [cx - d, cy + d, cz - d],
        [cx - d, cy - d, cz + d], [cx + d, cy - d, cz + d],
        [cx + d, cy + d, cz + d], [cx - d, cy + d, cz + d],
    ])
    pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    return [np.array([v[a], v[b]]) for a, b in pairs]


def _cube_faces(center: np.ndarray, size: float) -> List[np.ndarray]:
    d = size / 2.0
    cx, cy, cz = center
    v = np.array([
        [cx - d, cy - d, cz - d], [cx + d, cy - d, cz - d],
        [cx + d, cy + d, cz - d], [cx - d, cy + d, cz - d],
        [cx - d, cy - d, cz + d], [cx + d, cy - d, cz + d],
        [cx + d, cy + d, cz + d], [cx - d, cy + d, cz + d],
    ])
    return [
        np.array([v[0], v[1], v[2], v[3]]),
        np.array([v[4], v[5], v[6], v[7]]),
        np.array([v[0], v[1], v[5], v[4]]),
        np.array([v[2], v[3], v[7], v[6]]),
        np.array([v[0], v[3], v[7], v[4]]),
        np.array([v[1], v[2], v[6], v[5]]),
    ]


def _add_faces(ax, faces: Iterable[np.ndarray], color: str,
               alpha: float = 0.9, edge: Optional[str] = None,
               linewidth: float = 0.4, zorder: int = 10) -> None:
    coll = Poly3DCollection(
        list(faces),
        facecolors=to_rgba(color, alpha),
        edgecolors=to_rgba(edge, alpha) if edge else "none",
        linewidths=linewidth,
    )
    coll.set_zsort("average")
    ax.add_collection3d(coll)


def _add_edges(ax, segments: Iterable[np.ndarray], color: str,
               alpha: float = 0.9, linewidth: float = 1.0) -> None:
    coll = Line3DCollection(list(segments),
                            colors=to_rgba(color, alpha),
                            linewidths=linewidth)
    ax.add_collection3d(coll)


# ---------------------------------------------------------------------------
# Domain draw routines
# ---------------------------------------------------------------------------

def _is_conductive(state: GameState, pos: Tuple[int, int, int]) -> bool:
    cell = state.spec.cells.get(pos)
    if cell is None:
        return False
    if cell.type in (CellType.PIPE, CellType.SOURCE, CellType.TARGET):
        return True
    if cell.type == CellType.VALVE:
        return pos in state.opened_valves
    return False


def _draw_pipe_tubes(ax, state: GameState) -> None:
    """
    For every conductive cell, draw a small tube to the centre of every
    face-adjacent conductive neighbour.  Each edge is drawn once.
    """
    grid = state.spec.grid_size
    seen: set = set()
    for pos, cell in state.spec.cells.items():
        if cell.type not in (CellType.PIPE, CellType.SOURCE,
                             CellType.TARGET, CellType.VALVE):
            continue
        for nb in face_neighbors(pos, grid):
            edge = tuple(sorted((pos, nb)))
            if edge in seen:
                continue
            nb_cell = state.spec.cells.get(nb)
            if nb_cell is None:
                continue
            if nb_cell.type == CellType.PROTECTED:
                continue
            # Decide if either end is conductive so a tube makes sense.
            conductive_a = _is_conductive(state, pos)
            conductive_b = _is_conductive(state, nb)
            # Draw tube only when at least one end is a PIPE/SOURCE/TARGET/open-VALVE
            if not (conductive_a or conductive_b):
                continue
            seen.add(edge)

            p1 = _cell_center(pos)
            p2 = _cell_center(nb)

            fluid_a = state.fluid_cells.get(pos)
            fluid_b = state.fluid_cells.get(nb)
            fluid = fluid_a if fluid_a is not None else fluid_b

            # Don't let fluid paint a tube into an unopened valve.
            if nb_cell.type == CellType.VALVE and nb not in state.opened_valves:
                fluid = None
            if cell.type == CellType.VALVE and pos not in state.opened_valves:
                fluid = None

            if fluid:
                body = COLOR_FLUID.get(fluid, "#2E86DE")
                faces = _cylinder_mesh(p1, p2, radius=0.17)
                _add_faces(ax, faces, body, alpha=0.95,
                           edge=PIPE_RING, linewidth=0.25)
            else:
                body = PIPE_BODY
                faces = _cylinder_mesh(p1, p2, radius=0.11)
                _add_faces(ax, faces, body, alpha=0.85,
                           edge=PIPE_RING, linewidth=0.25)


def _draw_pipe_joints(ax, state: GameState) -> None:
    """Small spheres at each PIPE cell to cover elbows prettily."""
    for pos, cell in state.spec.cells.items():
        if cell.type != CellType.PIPE:
            continue
        fluid = state.fluid_cells.get(pos)
        color = COLOR_FLUID.get(fluid, PIPE_BODY) if fluid else PIPE_BODY
        r = 0.19 if fluid else 0.13
        faces = _sphere_mesh(_cell_center(pos), radius=r, n_u=12, n_v=8)
        _add_faces(ax, faces, color, alpha=0.95, edge=PIPE_RING, linewidth=0.2)


def _draw_source(ax, pos: Tuple[int, int, int], color: str) -> None:
    c = _cell_center(pos)
    base = COLOR_FLUID.get(color, "#2E86DE")
    halo = COLOR_FLUID_LIGHT.get(color, "#A8D5F7")
    # Outer halo
    _add_faces(ax, _sphere_mesh(c, radius=0.42, n_u=18, n_v=11),
               halo, alpha=0.28, edge=None, linewidth=0.0)
    # Core droplet
    _add_faces(ax, _sphere_mesh(c, radius=0.32, n_u=18, n_v=11),
               base, alpha=0.98, edge="#1A3D5C", linewidth=0.3)
    # Bright highlight cap for a "liquid" feel
    highlight = c + np.array([-0.1, -0.1, 0.18])
    _add_faces(ax, _sphere_mesh(highlight, radius=0.07, n_u=10, n_v=6),
               "#FFFFFF", alpha=0.55, edge=None, linewidth=0.0)


def _draw_target(ax, pos: Tuple[int, int, int], color: str,
                 fluid_color: Optional[str]) -> None:
    c = _cell_center(pos)
    filled = fluid_color == color
    base = COLOR_FLUID.get(color, "#2E86DE")

    # Outer glass shell
    _add_faces(ax, _cube_faces(c, size=0.88),
               base if filled else "#FFFFFF",
               alpha=0.18 if filled else 0.10,
               edge=base, linewidth=1.8)
    # Frame edges highlighted thicker
    _add_edges(ax, _cube_edges(c, size=0.88),
               color=base, alpha=0.95, linewidth=1.8)

    if filled:
        # Fluid level inside: a slab 70% filled along +z.
        level = 0.60
        d_xy = 0.34
        dz   = 0.30
        bottom_z = c[2] - dz
        top_z    = c[2] - dz + 2 * dz * level
        v = np.array([
            [c[0]-d_xy, c[1]-d_xy, bottom_z], [c[0]+d_xy, c[1]-d_xy, bottom_z],
            [c[0]+d_xy, c[1]+d_xy, bottom_z], [c[0]-d_xy, c[1]+d_xy, bottom_z],
            [c[0]-d_xy, c[1]-d_xy, top_z],    [c[0]+d_xy, c[1]-d_xy, top_z],
            [c[0]+d_xy, c[1]+d_xy, top_z],    [c[0]-d_xy, c[1]+d_xy, top_z],
        ])
        fluid_faces = [
            np.array([v[0], v[1], v[2], v[3]]),
            np.array([v[4], v[5], v[6], v[7]]),
            np.array([v[0], v[1], v[5], v[4]]),
            np.array([v[2], v[3], v[7], v[6]]),
            np.array([v[0], v[3], v[7], v[4]]),
            np.array([v[1], v[2], v[6], v[5]]),
        ]
        _add_faces(ax, fluid_faces, base, alpha=0.85,
                   edge=base, linewidth=0.6)
        # Little white ripple on the top surface
        top_disc = _disc_mesh(np.array([c[0], c[1], top_z + 0.01]),
                              np.array([0.0, 0.0, 1.0]), radius=d_xy * 0.9)
        _add_faces(ax, [top_disc], "#FFFFFF", alpha=0.32,
                   edge=base, linewidth=0.6)


def _draw_valve(ax, pos: Tuple[int, int, int], opened: bool,
                fluid_color: Optional[str]) -> None:
    c = _cell_center(pos)
    if opened:
        # Collar around the tube + colored body
        ring_color = COLOR_FLUID.get(fluid_color, "#2E86DE") if fluid_color else PIPE_BODY
        faces = _sphere_mesh(c, radius=0.26, n_u=14, n_v=9)
        _add_faces(ax, faces, ring_color, alpha=0.92,
                   edge=VALVE_HANDLE, linewidth=0.4)
        # Bright brass ring to signal "open"
        ring_top = _disc_mesh(c + np.array([0, 0, 0.26]),
                              np.array([0, 0, 1.0]), radius=0.22)
        _add_faces(ax, [ring_top], VALVE_BODY_OPEN, alpha=0.9,
                   edge=VALVE_HANDLE, linewidth=0.6)
    else:
        # Chunky yellow body
        _add_faces(ax, _cube_faces(c, size=0.58), VALVE_BODY,
                   alpha=0.97, edge=VALVE_HANDLE, linewidth=0.9)
        # Handwheel (torus approximation: ring + 4 spokes)
        wheel_center = c + np.array([0, 0, 0.38])
        outer = _disc_mesh(wheel_center, np.array([0, 0, 1.0]), radius=0.22)
        inner = _disc_mesh(wheel_center, np.array([0, 0, 1.0]), radius=0.10)
        _add_faces(ax, [outer], VALVE_HANDLE, alpha=0.95,
                   edge=VALVE_HANDLE, linewidth=0.5)
        _add_faces(ax, [inner], "#F5F6F8", alpha=1.0,
                   edge=VALVE_HANDLE, linewidth=0.5)
        # Spokes
        spokes = []
        for k in range(4):
            ang = k * np.pi / 2
            p1 = wheel_center + np.array([0.22 * np.cos(ang), 0.22 * np.sin(ang), 0])
            p2 = wheel_center - np.array([0.22 * np.cos(ang), 0.22 * np.sin(ang), 0])
            spokes.append(np.array([p1, p2]))
        _add_edges(ax, spokes, VALVE_HANDLE, alpha=1.0, linewidth=1.4)
        # Valve stem going into the body
        stem_faces = _cylinder_mesh(c + np.array([0, 0, 0.05]),
                                    wheel_center - np.array([0, 0, 0.02]),
                                    radius=0.04, n_seg=10)
        _add_faces(ax, stem_faces, VALVE_HANDLE, alpha=1.0,
                   edge=VALVE_HANDLE, linewidth=0.3)


def _draw_protected(ax, pos: Tuple[int, int, int], breached: bool) -> None:
    c = _cell_center(pos)
    face_color = PROTECTED_DARK if breached else PROTECTED_FACE
    _add_faces(ax, _cube_faces(c, size=0.82),
               face_color, alpha=0.94,
               edge="#38070E", linewidth=1.2)
    # Diagonal hatch on top face
    top_z = c[2] + 0.41
    stripes = []
    for k in np.linspace(-0.35, 0.35, 7):
        stripes.append(np.array([
            [c[0] - 0.4, c[1] + k, top_z + 0.005],
            [c[0] + 0.4, c[1] + k + 0.15, top_z + 0.005],
        ]))
    _add_edges(ax, stripes, "#FFD7D7" if not breached else "#FFAAAA",
               alpha=0.85, linewidth=1.2)
    if breached:
        # Cracked-glass starburst on top
        cracks = []
        for k in range(8):
            ang = k * np.pi / 4
            p1 = np.array([c[0], c[1], top_z + 0.02])
            p2 = np.array([c[0] + 0.42 * np.cos(ang),
                           c[1] + 0.42 * np.sin(ang),
                           top_z + 0.02])
            cracks.append(np.array([p1, p2]))
        _add_edges(ax, cracks, "#FFFFFF", alpha=0.95, linewidth=1.6)


def _draw_box_frame(ax, grid_size: Tuple[int, int, int]) -> None:
    A, B, C = grid_size
    verts = np.array([
        [0, 0, 0], [A, 0, 0], [A, B, 0], [0, B, 0],
        [0, 0, C], [A, 0, C], [A, B, C], [0, B, C],
    ])
    edges_idx = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
    _add_edges(ax, [np.array([verts[a], verts[b]]) for a, b in edges_idx],
               FRAME_COLOR, alpha=0.9, linewidth=1.6)

    # Floor grid (faint lines on z=0)
    lines = []
    for x in range(A + 1):
        lines.append(np.array([[x, 0, 0], [x, B, 0]]))
    for y in range(B + 1):
        lines.append(np.array([[0, y, 0], [A, y, 0]]))
    _add_edges(ax, lines, FRAME_FAINT, alpha=0.8, linewidth=0.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_state(state: GameState,
                 save_path: Optional[str] = None,
                 title: Optional[str] = None,
                 figsize: Tuple[float, float] = (10, 9),
                 elev: float = 22.0,
                 azim: float = -55.0,
                 show_legend: bool = True) -> plt.Figure:
    """Render one frame.  Returns the Figure (caller responsible for close)."""
    spec = state.spec
    A, B, C = spec.grid_size

    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COLOR)

    # --- Layout pass: draw back-to-front so alpha-blending works out. ---
    _draw_box_frame(ax, spec.grid_size)

    # Protected cells in back so fluid tubes land on top
    for p in spec.protected:
        breached = p in state.fluid_cells
        _draw_protected(ax, p, breached=breached)

    # Pipe tubes (empty pipes first so fluid overwrites)
    _draw_pipe_tubes(ax, state)
    _draw_pipe_joints(ax, state)

    # Valves on top of tubes
    for v in spec.valves:
        opened = v in state.opened_valves
        fluid_here = state.fluid_cells.get(v) if opened else None
        _draw_valve(ax, v, opened=opened, fluid_color=fluid_here)

    # Targets
    for t in spec.targets:
        col = spec.cells[t].color
        fluid = state.fluid_cells.get(t)
        _draw_target(ax, t, col, fluid)

    # Sources last
    for s in spec.sources:
        _draw_source(ax, s, spec.cells[s].color)

    # --- Cosmetics ---
    ax.set_xlim(0, A); ax.set_ylim(0, B); ax.set_zlim(0, C)
    ax.set_box_aspect((A, B, C))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X", color="#444"); ax.set_ylabel("Y", color="#444")
    ax.set_zlabel("Z", color="#444")
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)

    # Header
    if title is None:
        filled = sum(1 for t in spec.targets
                     if state.fluid_cells.get(t) == spec.cells[t].color)
        term = state.terminal_code.value if state.terminal_code else "playing"
        title = (f"FluidNet  |  targets {filled}/{len(spec.targets)}   "
                 f"valves open {len(state.opened_valves)}/{len(spec.valves)}   "
                 f"[{term}]")
    ax.set_title(title, fontsize=12, pad=16, color="#222")

    if show_legend:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#2E86DE", markersize=12,
                       markeredgecolor="#1A3D5C", label="source / fluid"),
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=VALVE_BODY, markersize=11,
                       markeredgecolor=VALVE_HANDLE, label="valve (closed)"),
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=VALVE_BODY_OPEN, markersize=11,
                       markeredgecolor=VALVE_HANDLE, label="valve (open)"),
            plt.Line2D([0], [0], marker="_", color=PIPE_BODY,
                       markersize=14, linewidth=3, label="pipe"),
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor="none", markersize=12,
                       markeredgecolor="#2E86DE", markeredgewidth=2,
                       label="target (empty)"),
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor="#A8D5F7", markersize=12,
                       markeredgecolor="#2E86DE", markeredgewidth=2,
                       label="target (filled)"),
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=PROTECTED_FACE, markersize=11,
                       markeredgecolor="#38070E", label="protected"),
        ]
        leg = ax.legend(handles=handles, loc="upper left",
                        fontsize=8, frameon=True, framealpha=0.9,
                        edgecolor="#CCCCCC")
        leg.set_zorder(100)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def render_sequence(states: Sequence[GameState],
                    out_dir: str,
                    prefix: str = "frame",
                    titles: Optional[Sequence[str]] = None,
                    **kwargs) -> List[str]:
    """Render a list of states to `{out_dir}/{prefix}_{i:02d}.png`."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for i, st in enumerate(states):
        title = titles[i] if titles is not None else None
        path = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
        fig = render_state(st, save_path=path, title=title, **kwargs)
        plt.close(fig)
        paths.append(path)
    return paths
