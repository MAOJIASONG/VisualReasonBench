"""
Shared plotly scene helpers for the CHAIN 3D grid tasks
(type_2_3 PipeFit, type_3 FluidNet, type_3_1 Domino).

Every task uses the same world convention: grid cell (x, y, z) is 1-based and
occupies the unit cube [x-1, x] x [y-1, y] x [z-1, z], so its centre is
(x-0.5, y-0.5, z-0.5) and the floor is the plane z = 0.

What lives here (and was previously copy-pasted, slightly differently, into
each visualizer):
  * the floor grid + bounding box wireframe
  * coordinate labels (x=1.., y=1.., z=1..) on the edges nearest the camera,
    so a model can read cell coordinates straight off the image
  * "drop lines": a floor footprint + dashed vertical guide under floating
    objects, so their (x, y) and height can be read despite perspective
  * the two canonical cameras and a scene layout that always frames the full
    grid (no content-dependent cropping, so framing never leaks the solution)
  * STATUS / INVENTORY / legend panel styling and PNG export
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

Vec3 = Tuple[int, int, int]

FRAME_COLOR = "#3D4148"
FRAME_FAINT = "#D3D6DB"
LABEL_COLOR = "#4A5059"
AXIS_TITLE_COLOR = "#1A1F26"
DROP_COLOR = "#A3A8B0"
TEXT_DARK = "#1A1F26"
BG = "#FFFFFF"
FONT_FAMILY = "Arial"

PANEL_FONT_SIZE = 13
TITLE_FONT_SIZE = 17
TICK_FONT_SIZE = 14

# Eye positions are in normalised scene units (the longest axis spans 1).
# Both views look slightly down at the grid from opposite corners, so every
# cell is visible from at least one of them.
CAMERAS: Dict[str, dict] = {
    "front": dict(eye=dict(x=1.30, y=-1.30, z=0.95),
                  center=dict(x=0, y=0, z=-0.08),
                  up=dict(x=0, y=0, z=1),
                  projection=dict(type="perspective")),
    "back":  dict(eye=dict(x=-1.30, y=1.30, z=0.95),
                  center=dict(x=0, y=0, z=-0.08),
                  up=dict(x=0, y=0, z=1),
                  projection=dict(type="perspective")),
    "top":   dict(eye=dict(x=0.0, y=-0.001, z=2.1),
                  center=dict(x=0, y=0, z=0),
                  up=dict(x=0, y=1, z=0),
                  projection=dict(type="orthographic")),
}


def line_trace(segs: Iterable[Tuple[Sequence[float], Sequence[float]]],
               color: str, width: float, name: str = "",
               dash: Optional[str] = None) -> go.Scatter3d:
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    for a, b in segs:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    line = dict(color=color, width=width)
    if dash:
        line["dash"] = dash
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=line,
                        hoverinfo="skip", name=name, showlegend=False)


def text_trace(points: Sequence[Sequence[float]], texts: Sequence[str],
               color: str, size: int, name: str = "",
               family: str = FONT_FAMILY) -> go.Scatter3d:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="text",
                        text=list(texts),
                        textfont=dict(color=color, size=size, family=family),
                        textposition="middle center",
                        hoverinfo="skip", name=name, showlegend=False)


# ---------------------------------------------------------------------------
# Frame, coordinates, drop lines
# ---------------------------------------------------------------------------

def frame_traces(grid_size: Vec3) -> List[go.Scatter3d]:
    """Floor grid (every cell) + the grid's bounding box."""
    A, B, C = grid_size
    floor = [((x, 0, 0), (x, B, 0)) for x in range(A + 1)]
    floor += [((0, y, 0), (A, y, 0)) for y in range(B + 1)]
    v = [(0, 0, 0), (A, 0, 0), (A, B, 0), (0, B, 0),
         (0, 0, C), (A, 0, C), (A, B, C), (0, B, C)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return [line_trace(floor, FRAME_FAINT, 1.5, "floor-grid"),
            line_trace([(v[a], v[b]) for a, b in edges], FRAME_COLOR, 2.5,
                       "box")]


def coordinate_traces(grid_size: Vec3, view: str = "front"
                      ) -> List[go.Scatter3d]:
    """Cell-index labels along the floor edges and one vertical edge.

    Labels go on the edges facing the camera so they are never hidden behind
    the scene: the front view (eye at +x, -y) sees the y=0 and x=A edges, the
    back view the opposite pair.  z labels sit on the (0, 0) vertical edge,
    which is a silhouette edge in both views (left in front, right in back)
    and away from where the "+x" / "+y" names land.
    """
    A, B, C = grid_size
    off = 0.42
    back = view == "back"
    y_edge = B + off if back else -off           # where the x labels live
    x_edge = -off if back else A + off           # where the y labels live
    z_corner = (-off, -off)

    traces: List[go.Scatter3d] = []
    xs = [(x - 0.5, y_edge, 0.0) for x in range(1, A + 1)]
    traces.append(text_trace(xs, [f"x={x}" for x in range(1, A + 1)],
                             LABEL_COLOR, TICK_FONT_SIZE, "x-ticks"))
    ys = [(x_edge, y - 0.5, 0.0) for y in range(1, B + 1)]
    traces.append(text_trace(ys, [f"y={y}" for y in range(1, B + 1)],
                             LABEL_COLOR, TICK_FONT_SIZE, "y-ticks"))
    zs = [(z_corner[0], z_corner[1], z - 0.5) for z in range(1, C + 1)]
    traces.append(text_trace(zs, [f"z={z}" for z in range(1, C + 1)],
                             LABEL_COLOR, TICK_FONT_SIZE, "z-ticks"))

    # Axis direction arrows ("x →") just past the last tick on each edge.
    ax_pts = [(A + 0.75, y_edge, 0.0), (x_edge, B + 0.75, 0.0)]
    traces.append(text_trace(ax_pts, ["<b>+x</b>", "<b>+y</b>"],
                             AXIS_TITLE_COLOR, TICK_FONT_SIZE + 2, "axis-names"))
    return traces


def drop_line_traces(cells: Iterable[Vec3], color: str = DROP_COLOR,
                     footprint: float = 0.9, name: str = "drop"
                     ) -> List[go.Scatter3d]:
    """Floor footprint square + dashed vertical guide for every cell at z>1.

    Without these, a cell hanging in mid-air is ambiguous under perspective
    (it could be further away at a lower height).
    """
    verticals = []
    squares = []
    h = footprint / 2
    seen = set()
    for pos in cells:
        x, y, z = pos
        if z <= 1 or (x, y) in seen:
            continue
        seen.add((x, y))
        cx, cy = x - 0.5, y - 0.5
        verticals.append(((cx, cy, 0.0), (cx, cy, z - 1.0)))
        corners = [(cx - h, cy - h, 0.01), (cx + h, cy - h, 0.01),
                   (cx + h, cy + h, 0.01), (cx - h, cy + h, 0.01)]
        squares += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    if not verticals:
        return []
    return [line_trace(verticals, color, 2.5, f"{name}-v", dash="dash"),
            line_trace(squares, color, 3, f"{name}-fp")]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def scene_layout(grid_size: Vec3, view: str = "front",
                 pad: float = 0.9) -> dict:
    """Scene dict that frames the whole grid (+ room for the labels)."""
    A, B, C = grid_size
    x_lo, x_hi = -pad, A + pad
    y_lo, y_hi = -pad, B + pad
    z_lo, z_hi = -0.25, C + 0.35
    sx, sy, sz = x_hi - x_lo, y_hi - y_lo, z_hi - z_lo
    m = max(sx, sy, sz)
    hidden = dict(visible=False, showbackground=False, backgroundcolor=BG)
    return dict(
        xaxis=dict(range=[x_lo, x_hi], **hidden),
        yaxis=dict(range=[y_lo, y_hi], **hidden),
        zaxis=dict(range=[z_lo, z_hi], **hidden),
        aspectmode="manual",
        aspectratio=dict(x=sx / m, y=sy / m, z=sz / m),
        camera=CAMERAS.get(view, CAMERAS["front"]),
        domain=dict(x=[0.0, 1.0], y=[0.0, 0.95]),
    )


def panel(text: str, x: float, y: float, xanchor: str, yanchor: str,
          border: str = FRAME_COLOR, color: str = TEXT_DARK,
          align: str = "left", size: int = PANEL_FONT_SIZE) -> dict:
    return dict(text=text, xref="paper", yref="paper", x=x, y=y,
                xanchor=xanchor, yanchor=yanchor, showarrow=False,
                align=align, font=dict(size=size, color=color,
                                       family=FONT_FAMILY),
                bgcolor="rgba(255,255,255,0.93)", bordercolor=border,
                borderwidth=1.5, borderpad=8)


def finish_figure(fig: go.Figure, grid_size: Vec3, view: str, title: str,
                  annotations: List[dict], save_path: Optional[str],
                  width: int = 1400, height: int = 1100,
                  scale: float = 1.4) -> go.Figure:
    """Apply the shared layout and (optionally) write the PNG."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.985,
                   yref="paper",
                   font=dict(size=TITLE_FONT_SIZE, color=TEXT_DARK,
                             family=FONT_FAMILY)),
        paper_bgcolor=BG,
        scene=scene_layout(grid_size, view),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        annotations=annotations,
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_image(save_path, width=width, height=height, scale=scale)
    return fig


def state_color(terminal: Optional[str]) -> str:
    """Panel colour for a terminal code: green on WIN, red on any loss."""
    if terminal is None:
        return TEXT_DARK
    return "#0E7D45" if terminal.upper() == "WIN" else "#C81818"
