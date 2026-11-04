"""
Shared plotly scene helpers for the CHAIN 3D grid tasks
(type_2_3 PipeFit, type_3 FluidNet, type_3_1 Domino).

Every task uses the same world convention: grid cell (x, y, z) is 1-based and
occupies the unit cube [x-1, x] x [y-1, y] x [z-1, z], so its centre is
(x-0.5, y-0.5, z-0.5) and the floor is the plane z = 0.

What lives here:
  * the game board: a solid two-tone checkered floor with a raised rim, plus
    a light wireframe for the playable volume
  * coordinate labels (x=1.., y=1.., z=1..) on the edges nearest the camera,
    so a model can read cell coordinates straight off the image
  * "drop lines": a floor footprint + dashed vertical guide under floating
    objects, so their (x, y) and height can be read despite perspective
  * the two canonical cameras and a scene layout that always frames the full
    grid (no content-dependent cropping, so framing never leaks the solution)
  * render_game(): renders the 3D scene with a transparent background and
    hands it to chain_hud for the game-style frame (title bar, HUD cards,
    legend icons)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

import chain_hud as hud_mod

Vec3 = Tuple[int, int, int]

FONT_FAMILY = "Arial"
TICK_FONT_SIZE = 15

# Final frame: 1960 x 1540 px.  The 3D scene fills the area left of the HUD
# column and below the title bar.
FRAME_SIZE = (1960, 1540)
SCENE_BOX = (0, 96, 1490, 1466)


@dataclass
class BoardStyle:
    tile_a: str
    tile_b: str
    rim: str
    frame: str            # playable-volume wireframe
    label: str            # coordinate labels
    axis: str             # "+x" / "+y" names
    drop: str             # drop lines / footprints


# Per-game look: (HUD theme, board style).
THEMES: Dict[str, Tuple[hud_mod.Theme, BoardStyle]] = {
    "domino": (
        hud_mod.Theme("domino", bg_top="#1B1530", bg_bottom="#3A2450",
                      glow="#8E5CFF", accent="#F2C63A"),
        BoardStyle(tile_a="#2E8B57", tile_b="#277A4C", rim="#5A3A22",
                   frame="rgba(255,255,255,0.28)", label="#E9E3FF",
                   axis="#FFD866", drop="rgba(255,255,255,0.55)"),
    ),
    "pipefit": (
        hud_mod.Theme("pipefit", bg_top="#0E1E26", bg_bottom="#1E3F48",
                      glow="#2FD3C6", accent="#2FD3C6"),
        BoardStyle(tile_a="#C9D2DA", tile_b="#B8C2CC", rim="#59636E",
                   frame="rgba(255,255,255,0.30)", label="#DDF7F5",
                   axis="#7DF2E6", drop="rgba(230,255,252,0.6)"),
    ),
    "fluidnet": (
        hud_mod.Theme("fluidnet", bg_top="#07142B", bg_bottom="#123A63",
                      glow="#3D8BFF", accent="#4DA3FF"),
        BoardStyle(tile_a="#D5DFEA", tile_b="#C3D0DE", rim="#3E5670",
                   frame="rgba(255,255,255,0.30)", label="#DCEBFF",
                   axis="#8FC2FF", drop="rgba(220,235,255,0.6)"),
    ),
}

# Eye positions are in normalised scene units (the longest axis spans 1).
# Both views look slightly down at the grid from opposite corners, so every
# cell is visible from at least one of them.
CAMERAS: Dict[str, dict] = {
    "front": dict(eye=dict(x=1.02, y=-1.02, z=0.78),
                  center=dict(x=0, y=0, z=-0.14),
                  up=dict(x=0, y=0, z=1),
                  projection=dict(type="perspective")),
    "back":  dict(eye=dict(x=-1.02, y=1.02, z=0.78),
                  center=dict(x=0, y=0, z=-0.14),
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
# Board, frame, coordinates, drop lines
# ---------------------------------------------------------------------------

_BOX_TRIS = np.array([
    (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
    (0, 5, 1), (0, 4, 5), (2, 7, 3), (2, 6, 7),
    (0, 3, 7), (0, 7, 4), (1, 6, 2), (1, 5, 6),
])


def _boxes_mesh(boxes: List[Tuple[Sequence[float], Sequence[float]]], color: str,
                ambient: float = 0.7) -> go.Mesh3d:
    vs, ts = [], []
    for k, ((x0, y0, z0), (x1, y1, z1)) in enumerate(boxes):
        vs.append([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                   [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
        ts.append(_BOX_TRIS + 8 * k)
    v = np.concatenate(vs)
    t = np.concatenate(ts)
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=t[:, 0], j=t[:, 1],
                     k=t[:, 2], color=color, flatshading=True,
                     lighting=dict(ambient=ambient, diffuse=0.45, specular=0.1,
                                   roughness=0.8),
                     lightposition=dict(x=1000, y=-2000, z=3000),
                     hoverinfo="skip", showlegend=False)


def board_traces(grid_size: Vec3, style: BoardStyle, view: str = "front") -> List:
    """Checkered tiles (top at z=0), a raised rim, the playable-volume
    wireframe.  The vertical edge at the corner nearest the camera is left
    out: it would cut straight through the pieces in the foreground."""
    A, B, C = grid_size
    t = 0.10
    tiles = {0: [], 1: []}
    for x in range(A):
        for y in range(B):
            tiles[(x + y) % 2].append(((x, y, -t), (x + 1, y + 1, 0.0)))
    r = 0.22
    rim = [((-r, -r, -t - 0.06), (A + r, 0, 0.04)),
           ((-r, B, -t - 0.06), (A + r, B + r, 0.04)),
           ((-r, 0, -t - 0.06), (0, B, 0.04)),
           ((A, 0, -t - 0.06), (A + r, B, 0.04))]
    v = [(0, 0, 0), (A, 0, 0), (A, B, 0), (0, B, 0),
         (0, 0, C), (A, 0, C), (A, B, C), (0, B, C)]
    near = 3 if view == "back" else 1          # (0,B) for back, (A,0) for front
    edges = [(4, 5), (5, 6), (6, 7), (7, 4)] + [(i, i + 4) for i in range(4) if i != near]
    return [_boxes_mesh(tiles[0], style.tile_a),
            _boxes_mesh(tiles[1], style.tile_b),
            _boxes_mesh(rim, style.rim, ambient=0.6),
            line_trace([(v[a], v[b]) for a, b in edges], style.frame, 2.5, "volume")]


def coordinate_traces(grid_size: Vec3, view: str = "front",
                      style: Optional[BoardStyle] = None) -> List[go.Scatter3d]:
    """Cell-index labels along the floor edges and one vertical edge.

    Labels go on the edges facing the camera so they are never hidden behind
    the scene: the front view (eye at +x, -y) sees the y=0 and x=A edges, the
    back view the opposite pair.  z labels sit on the (0, 0) vertical edge,
    which is a silhouette edge in both views (left in front, right in back)
    and away from where the "+x" / "+y" names land.
    """
    label_color = style.label if style else "#4A5059"
    axis_color = style.axis if style else "#1A1F26"
    A, B, C = grid_size
    if view == "top":
        return _top_coordinate_traces(grid_size, label_color, axis_color)
    off = 0.62
    back = view == "back"
    y_edge = B + off if back else -off           # where the x labels live
    x_edge = -off if back else A + off           # where the y labels live
    z_corner = (-off, -off)

    traces: List[go.Scatter3d] = []
    xs = [(x - 0.5, y_edge, 0.0) for x in range(1, A + 1)]
    traces.append(text_trace(xs, [f"<b>x={x}</b>" for x in range(1, A + 1)],
                             label_color, TICK_FONT_SIZE, "x-ticks"))
    ys = [(x_edge, y - 0.5, 0.0) for y in range(1, B + 1)]
    traces.append(text_trace(ys, [f"<b>y={y}</b>" for y in range(1, B + 1)],
                             label_color, TICK_FONT_SIZE, "y-ticks"))
    zs = [(z_corner[0], z_corner[1], z - 0.5) for z in range(1, C + 1)]
    traces.append(text_trace(zs, [f"<b>z={z}</b>" for z in range(1, C + 1)],
                             label_color, TICK_FONT_SIZE, "z-ticks"))

    ax_pts = [(A + 1.0, y_edge, 0.0), (x_edge, B + 1.0, 0.0)]
    traces.append(text_trace(ax_pts, ["<b>+x</b>", "<b>+y</b>"],
                             axis_color, TICK_FONT_SIZE + 3, "axis-names"))
    return traces


def _top_coordinate_traces(grid_size: Vec3, label_color: str,
                           axis_color: str) -> List[go.Scatter3d]:
    """Top view: bare cell numbers outside the rim (an "x=10" label is wider
    than a cell seen from above), raised above the rim so it cannot hide
    them; no z labels, heights are not readable from above."""
    A, B, _ = grid_size
    off, lift = 0.95, 0.4
    xs = [(x - 0.5, -off, lift) for x in range(1, A + 1)]
    ys = [(-off, y - 0.5, lift) for y in range(1, B + 1)]
    return [text_trace(xs, [f"<b>{x}</b>" for x in range(1, A + 1)],
                       label_color, TICK_FONT_SIZE, "x-ticks"),
            text_trace(ys, [f"<b>{y}</b>" for y in range(1, B + 1)],
                       label_color, TICK_FONT_SIZE, "y-ticks"),
            text_trace([(A + 0.9, -off, lift), (-off, B + 0.9, lift)],
                       ["<b>+x</b>", "<b>+y</b>"], axis_color, TICK_FONT_SIZE + 3,
                       "axis-names")]


def drop_line_traces(cells: Iterable[Vec3], color: str = "rgba(255,255,255,0.55)",
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
        corners = [(cx - h, cy - h, 0.012), (cx + h, cy - h, 0.012),
                   (cx + h, cy + h, 0.012), (cx - h, cy + h, 0.012)]
        squares += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    if not verticals:
        return []
    return [line_trace(verticals, color, 3, f"{name}-v", dash="dash"),
            line_trace(squares, color, 3.5, f"{name}-fp")]


# ---------------------------------------------------------------------------
# Layout + render
# ---------------------------------------------------------------------------

def scene_layout(grid_size: Vec3, view: str = "front",
                 pad: float = 1.2) -> dict:
    """Scene dict that frames the whole grid (+ room for the labels)."""
    A, B, C = grid_size
    x_lo, x_hi = -pad, A + pad
    y_lo, y_hi = -pad, B + pad
    z_lo, z_hi = -0.35, C + 0.3
    sx, sy, sz = x_hi - x_lo, y_hi - y_lo, z_hi - z_lo
    m = max(sx, sy, sz)
    hidden = dict(visible=False, showbackground=False)
    return dict(
        xaxis=dict(range=[x_lo, x_hi], **hidden),
        yaxis=dict(range=[y_lo, y_hi], **hidden),
        zaxis=dict(range=[z_lo, z_hi], **hidden),
        aspectmode="manual",
        aspectratio=dict(x=sx / m, y=sy / m, z=sz / m),
        camera=CAMERAS.get(view, CAMERAS["front"]),
        bgcolor="rgba(0,0,0,0)",
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
    )


def scene_traces(grid_size: Vec3, view: str, game: str) -> List:
    """Board + coordinates for a game's theme (put these first)."""
    style = THEMES[game][1]
    return board_traces(grid_size, style, view) + coordinate_traces(grid_size, view, style)


def drop_color(game: str) -> str:
    return THEMES[game][1].drop


def render_game(fig: go.Figure, grid_size: Vec3, view: str, game: str,
                hud: "hud_mod.Hud", save_path: Optional[str]) -> go.Figure:
    """Render the scene (transparent) and compose the game frame around it."""
    theme, _ = THEMES[game]
    sx0, sy0, sx1, sy1 = SCENE_BOX
    scale = 1.4
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=scene_layout(grid_size, view),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    if save_path:
        png = fig.to_image(format="png", width=int((sx1 - sx0) / scale),
                           height=int((sy1 - sy0) / scale), scale=scale)
        frame = hud_mod.compose(png, hud, theme, FRAME_SIZE, SCENE_BOX)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        frame.save(save_path)
    return fig


def result_label(terminal: Optional[str]) -> str:
    """Badge text for a terminal code: WIN, LOSE (any failure) or PLAYING."""
    if terminal is None:
        return "PLAYING"
    return "WIN" if terminal.upper() == "WIN" else "LOSE"
