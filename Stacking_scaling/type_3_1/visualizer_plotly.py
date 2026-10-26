"""
Domino Chain Reaction 3D — Plotly renderer (headless, kaleido 0.2.1).

Renders:
  - DOMINO (D)     : ivory standing slab with pips on every face
  - SPLITTER (T)   : ivory slab + brass cross cap on top
  - GOAL (G)       : gold slab + 5-point star on top
  - PROTECTED (P)  : red-lacquered slab + white warning X on top
  - RAMP (R)       : direction-free stepped block until activated; once the
                     cascade has used it, a wedge rising in that direction
  - Toppled pieces : lie flat inside their own cell in a darker shade of the
                     same colour, with a white arrow showing the fall
                     direction (so neighbouring toppled pieces never overlap)
  - Cascade trail  : one arrow per pulse edge (from -> hit piece), so
                     splitter branches are drawn as real branches
  - Grid coordinates, floor guides under raised pieces, STATUS / INVENTORY /
    legend panels (shared with the other CHAIN tasks via chain_viz)
  - Two cameras via render_state_pair
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import chain_viz as cv  # noqa: E402

from .game_core import (  # noqa: E402
    DIR_NAME_TO_VEC, GameState, PieceType, PlacedPiece, Vec3,
)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

DOMINO_BODY    = "#F2E7D2"    # warm ivory
DOMINO_EDGE    = "#3B3228"
DOMINO_PIPS    = "#1C1611"
SPLITTER_BODY  = "#E8DCB8"    # slightly yellower
SPLITTER_CAP   = "#9C7A1E"    # brass
GOAL_BODY      = "#F2C63A"    # gold
GOAL_STAR      = "#8A5A00"    # dark amber star so it reads against gold
GOAL_EDGE      = "#7A5A0A"
PROT_BODY      = "#C0241A"    # red lacquer
PROT_WARN      = "#FFFFFF"
PROT_EDGE      = "#4F0807"
RAMP_BODY      = "#B98858"    # warm tan wood
RAMP_EDGE      = "#5C3E20"
RAMP_TREAD     = "#F0D6A8"
CASCADE_LINE   = "#2F6FB0"    # steel-blue cascade arrows
PROT_HIT_HALO  = "#E53935"    # bright red halo for toppled P
FALL_ARROW     = "#FFFFFF"

# Toppled pieces keep their identity colour, just darker.
TOPPLED_BODY: Dict[PieceType, str] = {
    PieceType.DOMINO:    "#A39274",
    PieceType.SPLITTER:  "#9C8E68",
    PieceType.GOAL:      "#B8901C",
    PieceType.PROTECTED: "#7A1410",
}
STANDING_BODY: Dict[PieceType, str] = {
    PieceType.DOMINO:    DOMINO_BODY,
    PieceType.SPLITTER:  SPLITTER_BODY,
    PieceType.GOAL:      GOAL_BODY,
    PieceType.PROTECTED: PROT_BODY,
}

SLAB_W = 0.46          # footprint of a standing domino
TOPPLED_LEN = 0.92     # toppled slab stays inside its own cell
TOPPLED_H = 0.26


# ---------------------------------------------------------------------------
# Mesh primitives
# ---------------------------------------------------------------------------

class MeshBuilder:
    """Accumulates triangles per colour so each colour is a single trace."""

    def __init__(self) -> None:
        self._parts: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    def add(self, color: str, verts: np.ndarray, tris: np.ndarray) -> None:
        self._parts.setdefault(color, []).append((verts, tris))

    def traces(self) -> List[go.Mesh3d]:
        out = []
        for color, parts in self._parts.items():
            vs, ts, off = [], [], 0
            for v, t in parts:
                vs.append(v)
                ts.append(t + off)
                off += len(v)
            v = np.concatenate(vs)
            t = np.concatenate(ts)
            out.append(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=t[:, 0], j=t[:, 1], k=t[:, 2],
                color=color, opacity=1.0, flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.6, specular=0.2,
                              roughness=0.6, fresnel=0.1),
                lightposition=dict(x=1000, y=-2000, z=3000),
                hoverinfo="skip", showlegend=False))
        return out


_BOX_TRIS = np.array([
    (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
    (0, 5, 1), (0, 4, 5), (2, 7, 3), (2, 6, 7),
    (0, 3, 7), (0, 7, 4), (1, 6, 2), (1, 5, 6),
])


def _box(lo: Sequence[float], hi: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=float)
    return v, _BOX_TRIS


def _box_c(center: Sequence[float], sx: float, sy: float, sz: float):
    c = np.asarray(center, dtype=float)
    h = np.array([sx, sy, sz]) / 2
    return _box(c - h, c + h)


def _star(cx: float, cy: float, z: float, r_out: float, r_in: float,
          n: int = 5, rot: float = np.pi / 2):
    """Flat star polygon (triangle fan) lying in the plane z."""
    pts = [[cx, cy, z]]
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        a = rot + k * np.pi / n
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a), z])
    v = np.array(pts)
    m = 2 * n
    tris = np.array([(0, 1 + k, 1 + (k + 1) % m) for k in range(m)])
    return v, tris


def _wedge(cell_lo: Sequence[float], rise_dir: str):
    """Triangular prism filling a unit cell, tall side at `rise_dir`."""
    x0, y0, z0 = cell_lo
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    # Build for +x then rotate the footprint for the other directions.
    base = np.array([
        [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1],
    ], dtype=float)
    if rise_dir == "-x":
        base[:, 0] = 1 - base[:, 0]
    elif rise_dir == "+y":
        base[:, [0, 1]] = base[:, [1, 0]]
    elif rise_dir == "-y":
        base[:, [0, 1]] = base[:, [1, 0]]
        base[:, 1] = 1 - base[:, 1]
    inset = 0.04
    v = base.copy()
    v[:, 0] = x0 + inset + v[:, 0] * (1 - 2 * inset)
    v[:, 1] = y0 + inset + v[:, 1] * (1 - 2 * inset)
    v[:, 2] = z0 + v[:, 2] * (z1 - z0)
    tris = np.array([(0, 1, 3), (0, 3, 2), (0, 4, 5), (0, 5, 1),
                     (2, 3, 5), (2, 5, 4), (0, 2, 4), (1, 5, 3)])
    return v, tris


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def _cell_lo(pos: Vec3) -> np.ndarray:
    return np.array([pos[0] - 1.0, pos[1] - 1.0, pos[2] - 1.0])


def _standing(mb: MeshBuilder, segs: Dict[str, list], p: PlacedPiece) -> None:
    x, y, z = p.pos
    cx, cy = x - 0.5, y - 0.5
    z0 = z - 1.0
    top = z0 + 2.0
    mb.add(STANDING_BODY[p.type], *_box_c((cx, cy, z0 + 1.0),
                                          SLAB_W, SLAB_W, 2.0))
    if p.type == PieceType.DOMINO:
        # Two pips per half on every vertical face, so both cameras see them.
        h = SLAB_W / 2 + 0.006
        for dz in (0.5, 1.5):
            for s in (-0.1, 0.1):
                for (fx, fy, along_x) in ((0, -h, True), (0, h, True),
                                          (-h, 0, False), (h, 0, False)):
                    px = cx + fx + (s if along_x else 0)
                    py = cy + fy + (0 if along_x else s)
                    mb.add(DOMINO_PIPS, *_box_c(
                        (px, py, z0 + dz),
                        0.05 if along_x else 0.012,
                        0.012 if along_x else 0.05, 0.05))
    elif p.type == PieceType.SPLITTER:
        mb.add(SPLITTER_CAP, *_box_c((cx, cy, top + 0.05), 0.66, 0.11, 0.1))
        mb.add(SPLITTER_CAP, *_box_c((cx, cy, top + 0.05), 0.11, 0.66, 0.1))
    elif p.type == PieceType.GOAL:
        mb.add(GOAL_STAR, *_star(cx, cy, top + 0.012, 0.22, 0.09))
    elif p.type == PieceType.PROTECTED:
        d = 0.17
        segs.setdefault(PROT_WARN, []).extend([
            ((cx - d, cy - d, top + 0.015), (cx + d, cy + d, top + 0.015)),
            ((cx - d, cy + d, top + 0.015), (cx + d, cy - d, top + 0.015)),
        ])


def _toppled(mb: MeshBuilder, segs: Dict[str, list], p: PlacedPiece) -> None:
    x, y, z = p.pos
    cx, cy = x - 0.5, y - 0.5
    z0 = z - 1.0
    fd = p.fall_dir or "+x"
    dx, dy, _ = DIR_NAME_TO_VEC[fd]
    sx, sy = (TOPPLED_LEN, SLAB_W) if dx else (SLAB_W, TOPPLED_LEN)
    mb.add(TOPPLED_BODY[p.type], *_box_c((cx, cy, z0 + TOPPLED_H / 2),
                                         sx, sy, TOPPLED_H))
    top = z0 + TOPPLED_H + 0.012
    if p.type == PieceType.GOAL:
        mb.add(GOAL_STAR, *_star(cx, cy, top, 0.17, 0.07))
        return
    # Arrow on the top face pointing in the fall direction.
    tip = np.array([cx + dx * 0.36, cy + dy * 0.36, top])
    tail = np.array([cx - dx * 0.30, cy - dy * 0.30, top])
    side = np.array([-dy, dx, 0.0]) * 0.14
    back = np.array([dx, dy, 0.0]) * 0.16
    segs.setdefault(FALL_ARROW, []).extend([
        (tail, tip), (tip, tip - back + side), (tip, tip - back - side)])
    if p.type == PieceType.PROTECTED:
        # LOSE indicator: halo ring on the floor around the fallen piece.
        n, r = 36, 0.62
        ring = [((cx + r * np.cos(2 * np.pi * i / n),
                  cy + r * np.sin(2 * np.pi * i / n), z0 + 0.02),
                 (cx + r * np.cos(2 * np.pi * (i + 1) / n),
                  cy + r * np.sin(2 * np.pi * (i + 1) / n), z0 + 0.02))
                for i in range(n)]
        segs.setdefault(PROT_HIT_HALO + "|8", []).extend(ring)


def _ramp_dir(state: GameState, pos: Vec3) -> Optional[str]:
    for evs in state.all_cascades:
        for e in evs:
            if e.get("kind") == "ramp" and tuple(e.get("anchor", ())) == pos:
                return e.get("dir")
    return None


def _ramp(mb: MeshBuilder, segs: Dict[str, list], state: GameState,
          p: PlacedPiece) -> None:
    lo = _cell_lo(p.pos)
    rise = _ramp_dir(state, p.pos)
    if rise is None:
        # Direction-free until the cascade uses it: a three-step block.
        for k, (s, h) in enumerate(((0.92, 0.34), (0.66, 0.67), (0.40, 1.0))):
            c = lo + np.array([0.5, 0.5, 0.0])
            z_lo = 0.0 if k == 0 else (0.34, 0.67)[k - 1]
            mb.add(RAMP_BODY if k % 2 == 0 else "#A57447",
                   *_box((c[0] - s / 2, c[1] - s / 2, lo[2] + z_lo),
                         (c[0] + s / 2, c[1] + s / 2, lo[2] + h)))
        return
    mb.add(RAMP_BODY, *_wedge(lo, rise))
    dx, dy, _ = DIR_NAME_TO_VEC[rise]
    treads = []
    for f in (0.2, 0.4, 0.6, 0.8):
        # Point on the slope at fraction f from the low edge.
        a = 0.04 + f * 0.92
        px = lo[0] + (a if dx > 0 else 1 - a if dx < 0 else 0.5)
        py = lo[1] + (a if dy > 0 else 1 - a if dy < 0 else 0.5)
        pz = lo[2] + f + 0.01
        if dx:
            treads.append(((px, lo[1] + 0.1, pz), (px, lo[1] + 0.9, pz)))
        else:
            treads.append(((lo[0] + 0.1, py, pz), (lo[0] + 0.9, py, pz)))
    segs.setdefault(RAMP_TREAD + "|4", []).extend(treads)


def _cascade_traces(state: GameState) -> List:
    """One arrow per pulse edge, drawn just above the fallen slabs."""
    segs, cones_at, cones_dir = [], [], []
    for cascade in state.all_cascades:
        for ev in cascade:
            src = ev.get("from")
            if ev.get("kind") not in ("topple", "ramp") or src is None:
                continue
            dst = tuple(ev["anchor"])
            z_from = ev.get("z", ev.get("z_in", dst[2]))
            a = np.array([src[0] - 0.5, src[1] - 0.5, z_from - 1 + 0.42])
            b = np.array([dst[0] - 0.5, dst[1] - 0.5, dst[2] - 1 + 0.42])
            if ev.get("kind") == "ramp":
                b[2] = dst[2] - 1 + 0.55
            segs.append((a, b))
            d = b - a
            cones_at.append(a + 0.62 * d)
            cones_dir.append(d / (np.linalg.norm(d) + 1e-9))
    if not segs:
        return []
    pts = np.array(cones_at)
    dirs = np.array(cones_dir)
    return [
        cv.line_trace(segs, CASCADE_LINE, 6, "cascade"),
        go.Cone(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                u=dirs[:, 0], v=dirs[:, 1], w=dirs[:, 2],
                sizemode="absolute", sizeref=0.32, anchor="center",
                colorscale=[[0, CASCADE_LINE], [1, CASCADE_LINE]],
                showscale=False, hoverinfo="skip"),
    ]


def _scene_traces(state: GameState, view: str) -> List:
    grid = state.spec.grid_size
    mb = MeshBuilder()
    segs: Dict[str, list] = {}
    raised = []
    for p in state.anchors():
        if p.type == PieceType.RAMP:
            _ramp(mb, segs, state, p)
        elif p.toppled:
            _toppled(mb, segs, p)
        else:
            _standing(mb, segs, p)
        raised.append(p.pos)
    traces: List = cv.frame_traces(grid) + cv.coordinate_traces(grid, view)
    traces += cv.drop_line_traces(raised)
    traces += mb.traces()
    for key, s in segs.items():
        color, _, width = key.partition("|")
        traces.append(cv.line_trace(s, color, float(width or 5)))
    traces += _cascade_traces(state)
    return traces


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _inventory_html(state: GameState) -> str:
    rem = state.remaining
    start = state.spec.inventory
    lines = ["<b>INVENTORY (patches)</b>"]
    for pt, label, col in ((PieceType.DOMINO, "D domino", DOMINO_EDGE),
                           (PieceType.RAMP, "R ramp", RAMP_EDGE)):
        tot = start.get(pt, 0)
        left = rem.get(pt, 0)
        if tot == 0 and left == 0:
            continue
        lines.append(f"<span style='color:{col}'>■</span> <b>{label}</b>"
                     f"   left {left}/{tot}")
    if len(lines) == 1:
        lines.append("<i>(no patches to place)</i>")
    return "<br>".join(lines)


def _status_html(state: GameState) -> Tuple[str, str]:
    anchors = state.anchors()
    goals = [p for p in anchors if p.type == PieceType.GOAL]
    prot = [p for p in anchors if p.type == PieceType.PROTECTED]
    term = state.terminal_code.value if state.terminal_code else None
    color = cv.state_color(term)
    return (f"<b>STATUS</b><br>"
            f"phase: {state.phase.value}<br>"
            f"pushes: {state.push_count} / {state.spec.push_budget}<br>"
            f"goals toppled: {sum(p.toppled for p in goals)} / {len(goals)}<br>"
            f"protected toppled: {sum(p.toppled for p in prot)} / {len(prot)}<br>"
            f"<b>result: {term or '-'}</b>"), color


def _legend_html() -> str:
    return (
        "<b>Domino Chain Reaction 3D</b><br>"
        f"<span style='color:{DOMINO_EDGE}'>▮</span> D domino: falls straight on<br>"
        f"<span style='color:{SPLITTER_CAP}'>✚</span> T splitter: sends 2 sideways pulses<br>"
        f"<span style='color:{RAMP_EDGE}'>◢</span> R ramp: cascade continues one level up<br>"
        f"<span style='color:{GOAL_EDGE}'>★</span> G gold goal: must be toppled<br>"
        f"<span style='color:{PROT_BODY}'>✖</span> P red protected: must NOT be toppled<br>"
        f"<span style='color:{CASCADE_LINE}'>➜</span> cascade path · flat piece = toppled (arrow = fall dir)<br>"
        "<i>Phase 1: place inventory patches.  Phase 2: push one piece.</i>"
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_state(state: GameState, save_path: Optional[str],
                 title: Optional[str] = None,
                 view: str = "front",
                 width: int = 1400, height: int = 1100) -> go.Figure:
    if title is None:
        term = state.terminal_code.value if state.terminal_code else state.phase.value
        title = f"Domino Chain 3D  |  {term}  |  view: {view}"
    status, status_color = _status_html(state)
    annotations = [
        cv.panel(_legend_html(), 0.005, 0.005, "left", "bottom"),
        cv.panel(status, 0.995, 0.995, "right", "top",
                 border=status_color, color=status_color),
        cv.panel(_inventory_html(state), 0.995, 0.80, "right", "top"),
    ]
    fig = go.Figure(data=_scene_traces(state, view))
    return cv.finish_figure(fig, state.spec.grid_size, view, title,
                            annotations, save_path, width, height)


def render_state_pair(state: GameState, save_dir: str, base_name: str,
                      title: Optional[str] = None,
                      views: Tuple[str, ...] = ("front", "back"),
                      **kwargs) -> List[str]:
    os.makedirs(save_dir, exist_ok=True)
    paths: List[str] = []
    for v in views:
        path = os.path.join(save_dir, f"{base_name}_{v}.png")
        render_state(state, save_path=path, title=title, view=v, **kwargs)
        paths.append(path)
    return paths
