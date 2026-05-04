"""
Domino Chain Reaction 3D — Plotly renderer (headless, kaleido 0.2.1).

Renders:
  - DOMINO (D)       : ivory standing slab, toppled = flat slab in fall dir
  - SPLITTER (T)     : ivory slab + brass cross cap on top
  - GOAL (G)         : gold slab + gold 4-point star on top
  - PROTECTED (P)    : red-lacquered slab + white warning X on top
  - RAMP (R)         : wedge / triangular prism in warm tan
  - Cascade arrows   : faint ghost chain from push origin → toppled pieces
  - STATUS / INVENTORY / legend panels
  - Two cameras via render_state_pair

Canvas 1400x1100; rectangular bbox with aspectmode=manual.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import plotly.graph_objects as go

from .game_core import (
    DIR_NAME_TO_VEC, GameState, LevelSpec, Phase, PieceType, PlacedPiece,
    Vec3, piece_is_standing, piece_occupied_cells,
)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

DOMINO_BODY    = "#F2E7D2"    # warm ivory
DOMINO_EDGE    = "#3B3228"
DOMINO_PIPS    = "#1C1611"
SPLITTER_BODY  = "#E8DCB8"    # slightly yellower
SPLITTER_CAP   = "#B8922A"    # brass
GOAL_BODY      = "#F2C63A"    # gold
GOAL_STAR      = "#F7E28A"
GOAL_EDGE      = "#7A5A0A"
PROT_BODY      = "#C0241A"    # red lacquer
PROT_WARN      = "#FFFFFF"
PROT_EDGE      = "#4F0807"
RAMP_BODY      = "#B98858"    # warm tan wood
RAMP_EDGE      = "#5C3E20"
RAMP_TREAD     = "#E5C290"
TOPPLED_DARK   = "#7F6F57"    # dusty shadow for toppled dominos
CASCADE_LINE   = "#4682B4"    # steel-blue cascade arrows
PROT_HIT_HALO  = "#E53935"    # bright red halo for toppled P

FRAME_COLOR    = "#3D4148"
FRAME_FAINT    = "#C9CCD2"
BG_PAPER       = "#FFFFFF"
BG_SCENE       = "#FFFFFF"


# ---------------------------------------------------------------------------
# Mesh primitives
# ---------------------------------------------------------------------------

def _empty_mesh() -> Dict[str, np.ndarray]:
    return {"x": np.array([]), "y": np.array([]), "z": np.array([]),
            "i": np.array([]), "j": np.array([]), "k": np.array([])}


def _concat_meshes(*parts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    xs, ys, zs, ii, jj, kk = [], [], [], [], [], []
    offset = 0
    for m in parts:
        if len(m["x"]) == 0:
            continue
        xs.append(m["x"]); ys.append(m["y"]); zs.append(m["z"])
        ii.append(m["i"] + offset); jj.append(m["j"] + offset); kk.append(m["k"] + offset)
        offset += len(m["x"])
    if not xs:
        return _empty_mesh()
    return {
        "x": np.concatenate(xs), "y": np.concatenate(ys), "z": np.concatenate(zs),
        "i": np.concatenate(ii), "j": np.concatenate(jj), "k": np.concatenate(kk),
    }


def _mesh_trace(m: Dict[str, np.ndarray], color: str, opacity: float,
                name: str) -> go.Mesh3d:
    return go.Mesh3d(
        x=m["x"], y=m["y"], z=m["z"],
        i=m["i"], j=m["j"], k=m["k"],
        color=color, opacity=opacity, flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.55, specular=0.25,
                      roughness=0.55, fresnel=0.10),
        lightposition=dict(x=1000, y=-2000, z=3000),
        hoverinfo="skip", name=name, showlegend=False,
    )


def _line_trace(segs, color: str, width: float, name: str) -> go.Scatter3d:
    xs, ys, zs = [], [], []
    for a, b in segs:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=color, width=width),
                        hoverinfo="skip", name=name, showlegend=False)


def _box(center: np.ndarray, sx: float, sy: float, sz: float
         ) -> Dict[str, np.ndarray]:
    cx, cy, cz = center
    dx, dy, dz = sx / 2, sy / 2, sz / 2
    v = np.array([
        [cx-dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz-dz],
        [cx+dx, cy+dy, cz-dz], [cx-dx, cy+dy, cz-dz],
        [cx-dx, cy-dy, cz+dz], [cx+dx, cy-dy, cz+dz],
        [cx+dx, cy+dy, cz+dz], [cx-dx, cy+dy, cz+dz],
    ])
    tris = np.array([
        (0,1,2),(0,2,3), (4,6,5),(4,7,6),
        (0,5,1),(0,4,5), (2,7,3),(2,6,7),
        (0,3,7),(0,7,4), (1,6,2),(1,5,6),
    ])
    return {"x": v[:,0], "y": v[:,1], "z": v[:,2],
            "i": tris[:,0], "j": tris[:,1], "k": tris[:,2]}


def _wedge(base_center: np.ndarray, sx: float, sy: float, sz: float,
           rise_dir: str) -> Dict[str, np.ndarray]:
    """A triangular prism (ramp) sitting inside a 1x1x1 cell.

    rise_dir determines which side is the tall end (where the wedge peaks):
      '+x' : tall end at +x side; '-x' at -x side; '+y' at +y; '-y' at -y.
    The low end is at sz=0; the tall end peaks at sz.
    """
    cx, cy, cz = base_center
    dx, dy, dz = sx/2, sy/2, sz/2
    # Low side at cz-dz, high peak (one ridge) on the rise_dir face.
    if rise_dir == "+x":
        # 6 verts: low at -x (both y), peak at +x (both y along ridge)
        v = np.array([
            [cx-dx, cy-dy, cz-dz], [cx-dx, cy+dy, cz-dz],
            [cx+dx, cy-dy, cz-dz], [cx+dx, cy+dy, cz-dz],
            [cx+dx, cy-dy, cz+dz], [cx+dx, cy+dy, cz+dz],
        ])
    elif rise_dir == "-x":
        v = np.array([
            [cx+dx, cy-dy, cz-dz], [cx+dx, cy+dy, cz-dz],
            [cx-dx, cy-dy, cz-dz], [cx-dx, cy+dy, cz-dz],
            [cx-dx, cy-dy, cz+dz], [cx-dx, cy+dy, cz+dz],
        ])
    elif rise_dir == "+y":
        v = np.array([
            [cx-dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz-dz],
            [cx-dx, cy+dy, cz-dz], [cx+dx, cy+dy, cz-dz],
            [cx-dx, cy+dy, cz+dz], [cx+dx, cy+dy, cz+dz],
        ])
    else:  # '-y'
        v = np.array([
            [cx-dx, cy+dy, cz-dz], [cx+dx, cy+dy, cz-dz],
            [cx-dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz-dz],
            [cx-dx, cy-dy, cz+dz], [cx+dx, cy-dy, cz+dz],
        ])
    # Verts layout: (0,1) low-far, (2,3) high-base, (4,5) ridge
    tris = np.array([
        (0,1,3),(0,3,2),       # base
        (0,4,5),(0,5,1),       # slope (slanted face from low-far to ridge)
        (2,3,5),(2,5,4),       # back vertical face at the rise_dir wall
        (0,2,4),               # side triangle 1
        (1,5,3),               # side triangle 2
    ])
    return {"x": v[:,0], "y": v[:,1], "z": v[:,2],
            "i": tris[:,0], "j": tris[:,1], "k": tris[:,2]}


# ---------------------------------------------------------------------------
# Per-piece render
# ---------------------------------------------------------------------------

def _standing_piece_mesh(piece: PlacedPiece
                         ) -> Tuple[Dict[str, np.ndarray], List[tuple]]:
    """A standing 1x1x2 domino occupies cells pos=(x,y,z) and (x,y,z+1).
    We render it as a slim cuboid centered at (x, y, z + 0.5)  (base cell)
    through (x, y, z+1.5) (upper cell).  Footprint 0.45 x 0.45.
    """
    x, y, z = piece.pos
    cx = x - 0.5
    cy = y - 0.5
    if not piece.toppled:
        # Standing: centered z = z, spanning z-0.5 to z+1.5 height range
        # i.e. the cell pair (z, z+1) corresponds to world z in [z-1, z+1]
        # But our coordinate convention: cell z=1 center is at world z = 0.5.
        # So a standing piece with base=z occupies world-z from (z-1) to (z+1).
        center_z = (z - 1) + 1.0   # midpoint of two cells = z
        mesh = _box(np.array([cx, cy, center_z]), 0.45, 0.45, 2.0)
        return mesh, []
    # Toppled: lies flat in fall_dir.  Length 2 along fall_dir, thickness 0.45.
    # Lies at world z ≈ (z-1) + 0.22 (half thickness above ground).
    fd = piece.fall_dir or "+x"
    # Cell center from which it fell
    base_z = (z - 1) + 0.22
    # It fell from its base-cell into the next cell in fall_dir; the
    # long axis lies from base-cell center + 0 to +2 along fall_dir.
    # Place the toppled slab centered at (cx + 1*dir_vec[0], cy + 1*dir_vec[1], base_z)
    # — this is the next cell's center (the toppled slab's head lies there).
    dx, dy, _ = DIR_NAME_TO_VEC[fd]
    center = np.array([cx + dx * 1.0, cy + dy * 1.0, base_z])
    if fd in ("+x", "-x"):
        mesh = _box(center, 2.0, 0.45, 0.35)
    else:
        mesh = _box(center, 0.45, 2.0, 0.35)
    return mesh, []


def _build_dominos(state: GameState) -> List:
    traces: List = []
    for p in state.anchors():
        if p.type != PieceType.DOMINO:
            continue
        body, _ = _standing_piece_mesh(p)
        color = DOMINO_BODY if not p.toppled else TOPPLED_DARK
        traces.append(_mesh_trace(body, color, 1.0, f"D-{p.pos}"))
        # Pips (decorative): tiny black dots on the "front" (standing only)
        if not p.toppled:
            x, y, z = p.pos
            cx = x - 0.5
            cy = y - 0.5
            cz = (z - 1) + 1.0  # center of tall slab
            # Two pip rows on the front face (y = cy - 0.225)
            y_front = cy - 0.228
            for (dx, dz) in [(-0.11, 0.45), (0.11, 0.45),
                             (-0.11, -0.45), (0.11, -0.45)]:
                pip = _box(np.array([cx + dx, y_front, cz + dz]),
                            0.045, 0.02, 0.045)
                traces.append(_mesh_trace(pip, DOMINO_PIPS, 1.0, f"pip-{p.pos}-{dx}-{dz}"))
    return traces


def _build_splitters(state: GameState) -> List:
    traces: List = []
    for p in state.anchors():
        if p.type != PieceType.SPLITTER:
            continue
        body, _ = _standing_piece_mesh(p)
        color = SPLITTER_BODY if not p.toppled else TOPPLED_DARK
        traces.append(_mesh_trace(body, color, 1.0, f"T-body-{p.pos}"))
        if not p.toppled:
            x, y, z = p.pos
            cx = x - 0.5; cy = y - 0.5
            top_z = (z - 1) + 2.0  # top face world-z
            # Cross cap: two thin bars on top forming +
            cap_thick = 0.08
            cap_h = 0.08
            bar_x = _box(np.array([cx, cy, top_z + cap_h/2]),
                          0.60, cap_thick, cap_h)
            bar_y = _box(np.array([cx, cy, top_z + cap_h/2]),
                          cap_thick, 0.60, cap_h)
            traces.append(_mesh_trace(bar_x, SPLITTER_CAP, 1.0, f"T-bar1-{p.pos}"))
            traces.append(_mesh_trace(bar_y, SPLITTER_CAP, 1.0, f"T-bar2-{p.pos}"))
    return traces


def _build_goals(state: GameState, push_count_hint: int = 0) -> List:
    traces: List = []
    for p in state.anchors():
        if p.type != PieceType.GOAL:
            continue
        body, _ = _standing_piece_mesh(p)
        color = GOAL_BODY if not p.toppled else GOAL_STAR  # toppled: paler gold
        traces.append(_mesh_trace(body, color, 1.0, f"G-body-{p.pos}"))
        if not p.toppled:
            x, y, z = p.pos
            cx = x - 0.5; cy = y - 0.5
            top_z = (z - 1) + 2.0
            # Four thin bars meeting at center -> 4-point star silhouette
            star_h = 0.05
            bar1 = _box(np.array([cx, cy, top_z + star_h/2]), 0.40, 0.06, star_h)
            bar2 = _box(np.array([cx, cy, top_z + star_h/2]), 0.06, 0.40, star_h)
            traces.append(_mesh_trace(bar1, GOAL_STAR, 1.0, f"G-star1-{p.pos}"))
            traces.append(_mesh_trace(bar2, GOAL_STAR, 1.0, f"G-star2-{p.pos}"))
            # Tiny text "G" via edge outline (lines)
    return traces


def _build_protected(state: GameState) -> List:
    traces: List = []
    line_segs = []
    for p in state.anchors():
        if p.type != PieceType.PROTECTED:
            continue
        body, _ = _standing_piece_mesh(p)
        color = PROT_BODY if not p.toppled else TOPPLED_DARK
        traces.append(_mesh_trace(body, color, 1.0, f"P-body-{p.pos}"))
        if not p.toppled:
            x, y, z = p.pos
            cx = x - 0.5; cy = y - 0.5
            top_z = (z - 1) + 2.0
            # White warning X on top
            d = 0.22
            line_segs.append((np.array([cx - d, cy - d, top_z + 0.015]),
                              np.array([cx + d, cy + d, top_z + 0.015])))
            line_segs.append((np.array([cx - d, cy + d, top_z + 0.015]),
                              np.array([cx + d, cy - d, top_z + 0.015])))
    if line_segs:
        traces.append(_line_trace(line_segs, PROT_WARN, 6, "P-warn"))
    # Halo for toppled P
    for p in state.anchors():
        if p.type == PieceType.PROTECTED and p.toppled:
            x, y, z = p.pos
            cx = x - 0.5; cy = y - 0.5
            center_z = (z - 1) + 0.5
            # Ring at ground level
            ring_segs = []
            n = 30
            r = 0.55
            for i in range(n):
                t1 = 2*np.pi*i/n; t2 = 2*np.pi*(i+1)/n
                p1 = np.array([cx + r*np.cos(t1), cy + r*np.sin(t1), center_z])
                p2 = np.array([cx + r*np.cos(t2), cy + r*np.sin(t2), center_z])
                ring_segs.append((p1, p2))
            traces.append(_line_trace(ring_segs, PROT_HIT_HALO, 8,
                                        f"P-halo-{p.pos}"))
    return traces


def _build_ramps(state: GameState) -> List:
    traces: List = []
    for p in state.anchors():
        if p.type != PieceType.RAMP:
            continue
        x, y, z = p.pos
        cx = x - 0.5; cy = y - 0.5
        cz = (z - 1) + 0.5  # cell center world z
        # Determine rise direction: if ramp was activated, use its fall_dir
        # history?  R doesn't store fall_dir; we look at meta-free info.
        # For unactivated ramps, pick rise_dir = +x by default.  For
        # activated ramps (toppled=True), we also just pick +x; cascade
        # direction is faintly shown via arrows.
        # IMPROVEMENT: we can infer rise direction from skeleton-meta if
        # the caller provides it, but for now visual is symbolic.
        # Try to find direction from any cascade event log where this
        # piece appears as a "ramp" event.
        rise_dir = "+x"
        for evs in state.all_cascades:
            for e in evs:
                if (e.get("kind") == "ramp"
                        and tuple(e.get("anchor", ())) == p.pos):
                    rise_dir = e.get("dir", "+x")
                    break
        wedge = _wedge(np.array([cx, cy, cz]), 1.0, 1.0, 1.0, rise_dir)
        color = RAMP_BODY if not p.toppled else TOPPLED_DARK
        traces.append(_mesh_trace(wedge, color, 1.0, f"R-{p.pos}"))
        # Tread line on the slope for visual readability
        dx, dy, _ = DIR_NAME_TO_VEC[rise_dir]
        # Tread segments across the slope (perpendicular to rise_dir)
        tread_segs = []
        n_tread = 5
        for ti in range(1, n_tread):
            frac = ti / n_tread
            # Point on the slope at this fraction from low to high
            # Low point at (cx - dx*0.5, cy - dy*0.5, cz - 0.5)
            # High ridge at (cx + dx*0.5, cy + dy*0.5, cz + 0.5)
            xs = cx + dx * (0.5 - frac * 1.0) * -1  # frac=0 -> low
            ys = cy + dy * (0.5 - frac * 1.0) * -1
            zs = (cz - 0.5) + frac * 1.0
            # Perpendicular direction
            if rise_dir in ("+x", "-x"):
                p1 = np.array([xs, ys - 0.45, zs + 0.005])
                p2 = np.array([xs, ys + 0.45, zs + 0.005])
            else:
                p1 = np.array([xs - 0.45, ys, zs + 0.005])
                p2 = np.array([xs + 0.45, ys, zs + 0.005])
            tread_segs.append((p1, p2))
        traces.append(_line_trace(tread_segs, RAMP_TREAD, 3, f"R-tread-{p.pos}"))
    return traces


def _build_cascade_arrows(state: GameState) -> List:
    """Draw faint arrows along the cascade from push origin through each
       toppled piece, for all pushes so far."""
    segs = []
    for cascade in state.all_cascades:
        prev: Optional[Vec3] = None
        for ev in cascade:
            if ev.get("kind") == "push":
                prev = tuple(ev["anchor"])
                continue
            anchor = ev.get("anchor")
            if anchor is None:
                continue
            anchor = tuple(anchor)
            if prev is not None:
                # world-space arrow from prev center to anchor center
                p1 = np.array([prev[0] - 0.5, prev[1] - 0.5,
                                (prev[2] - 1) + 1.0])
                p2 = np.array([anchor[0] - 0.5, anchor[1] - 0.5,
                                (anchor[2] - 1) + 1.0])
                # raise slightly above the dominoes for visibility
                p1[2] += 0.15; p2[2] += 0.15
                segs.append((p1, p2))
            prev = anchor
    if not segs:
        return []
    return [_line_trace(segs, CASCADE_LINE, 4, "cascade")]


# ---------------------------------------------------------------------------
# Box frame
# ---------------------------------------------------------------------------

def _build_box_frame(state: GameState) -> List[go.Scatter3d]:
    A, B, C = state.spec.grid_size
    verts = np.array([
        [0, 0, 0], [A, 0, 0], [A, B, 0], [0, B, 0],
        [0, 0, C], [A, 0, C], [A, B, C], [0, B, C],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    box = _line_trace([(verts[a], verts[b]) for a, b in edges],
                      FRAME_COLOR, 3, "box")
    floor = []
    for x in range(A + 1):
        floor.append((np.array([x, 0, 0]), np.array([x, B, 0])))
    for y in range(B + 1):
        floor.append((np.array([0, y, 0]), np.array([A, y, 0])))
    grid = _line_trace(floor, FRAME_FAINT, 1.5, "grid")
    return [grid, box]


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

CAMERAS: Dict[str, dict] = {
    "front": dict(eye=dict(x=1.15, y=-1.15, z=0.80),
                  center=dict(x=0, y=0, z=-0.05),
                  up=dict(x=0, y=0, z=1),
                  projection=dict(type="perspective")),
    "back":  dict(eye=dict(x=-1.15, y=1.15, z=0.80),
                  center=dict(x=0, y=0, z=-0.05),
                  up=dict(x=0, y=0, z=1),
                  projection=dict(type="perspective")),
}


# ---------------------------------------------------------------------------
# Content bbox
# ---------------------------------------------------------------------------

def _content_bbox(state: GameState, pad: float = 0.3):
    spec = state.spec
    A, B, C = spec.grid_size
    all_cells: Set[Vec3] = set()
    for pt, pos in spec.fixed_pieces:
        for c in piece_occupied_cells(pt, pos):
            all_cells.add(c)
    for c in state.pieces:
        all_cells.add(c)
    # patches in meta
    for _, pos in spec.meta.get("patch_pieces", []) if spec.meta else []:
        all_cells.add(tuple(pos))
    if not all_cells:
        return (-0.5, A + 0.5), (-0.5, B + 0.5), (-0.5, C + 0.5)
    xs = [p[0] for p in all_cells]
    ys = [p[1] for p in all_cells]
    zs = [p[2] for p in all_cells]
    x_lo = min(xs) - 1 - pad
    x_hi = max(xs) + pad
    y_lo = min(ys) - 1 - pad
    y_hi = max(ys) + pad
    z_lo = min(zs) - 1 - pad
    z_hi = max(zs) + pad
    z_lo = max(z_lo, -0.3)
    return (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _inventory_html(state: GameState) -> str:
    rem = state.remaining
    start = state.spec.inventory
    lines = ["<b>INVENTORY (patches)</b>"]
    order = [
        (PieceType.DOMINO, "D (domino)", DOMINO_EDGE),
        (PieceType.RAMP,   "R (ramp)",   RAMP_EDGE),
    ]
    for pt, label, col in order:
        tot = start.get(pt, 0)
        left = rem.get(pt, 0)
        if tot == 0 and left == 0:
            continue
        lines.append(
            f"<span style='color:{col}'>■</span> "
            f"<b>{label}</b>  left:{left}/{tot}  used:{tot-left}")
    if len(lines) == 1:
        lines.append("<i>(no patches to place)</i>")
    return "<br>".join(lines)


def _status_html(state: GameState):
    goals_total = sum(1 for p in state.anchors() if p.type == PieceType.GOAL)
    goals_toppled = sum(1 for p in state.anchors()
                         if p.type == PieceType.GOAL and p.toppled)
    prot_total = sum(1 for p in state.anchors() if p.type == PieceType.PROTECTED)
    prot_toppled = sum(1 for p in state.anchors()
                        if p.type == PieceType.PROTECTED and p.toppled)
    code = state.terminal_code.value if state.terminal_code else state.phase.value
    color = "#1A1F26"
    if state.terminal_code is not None:
        color = "#0E7D45" if code == "WIN" else "#C81818"
    return (f"<b>STATUS</b><br>"
            f"phase : {state.phase.value}<br>"
            f"pushes : {state.push_count} / {state.spec.push_budget}<br>"
            f"goals : {goals_toppled}/{goals_total}<br>"
            f"protected : {prot_toppled}/{prot_total}<br>"
            f"<span style='color:{color}'>state : {code}</span>"), color


def _legend_html() -> str:
    return (
        "<b>Domino Chain Reaction 3D</b><br>"
        f"<span style='color:{DOMINO_EDGE}'>▇ DOMINO (D)</span> basic<br>"
        f"<span style='color:{SPLITTER_CAP}'>✚ SPLITTER (T)</span> emits 2 orthogonal<br>"
        f"<span style='color:{RAMP_EDGE}'>◢ RAMP (R)</span> cascade climbs z+1<br>"
        f"<span style='color:{GOAL_EDGE}'>★ GOAL (G)</span> must be toppled<br>"
        f"<span style='color:{PROT_EDGE}'>✖ PROTECTED (P)</span> must NOT be toppled<br>"
        "<i>Phase 1: place inventory patches in legal cells.</i><br>"
        "<i>Phase 2: push one piece; cascade propagates.</i>"
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_state(state: GameState, save_path: str,
                 title: Optional[str] = None,
                 view: str = "front",
                 width: int = 1400, height: int = 1100,
                 tight_bbox: bool = True) -> go.Figure:
    traces: List = []
    traces.extend(_build_box_frame(state))
    traces.extend(_build_ramps(state))
    traces.extend(_build_dominos(state))
    traces.extend(_build_splitters(state))
    traces.extend(_build_goals(state))
    traces.extend(_build_protected(state))
    traces.extend(_build_cascade_arrows(state))

    fig = go.Figure(data=traces)

    if title is None:
        term = state.terminal_code.value if state.terminal_code else state.phase.value
        title = f"Domino Chain 3D  |  {term}  |  view: {view}"

    cam = CAMERAS.get(view, CAMERAS["front"])

    if tight_bbox:
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = _content_bbox(state)
    else:
        A, B, C = state.spec.grid_size
        x_lo, x_hi = -0.5, A + 0.5
        y_lo, y_hi = -0.5, B + 0.5
        z_lo, z_hi = -0.5, C + 0.5

    ax_common = dict(backgroundcolor=BG_SCENE, showbackground=False, visible=False)
    xaxis = dict(range=[x_lo, x_hi], **ax_common)
    yaxis = dict(range=[y_lo, y_hi], **ax_common)
    zaxis = dict(range=[z_lo, z_hi], **ax_common)

    annotations = [
        dict(text=_legend_html(), xref="paper", yref="paper",
             x=0.005, y=0.005, xanchor="left", yanchor="bottom",
             showarrow=False, align="left",
             font=dict(size=11, color="#1A1F26", family="Arial"),
             bgcolor="rgba(255,255,255,0.92)",
             bordercolor="#3D4148", borderwidth=1, borderpad=8),
    ]
    status_text, status_color = _status_html(state)
    annotations.append(dict(
        text=status_text, xref="paper", yref="paper",
        x=0.995, y=0.995, xanchor="right", yanchor="top",
        showarrow=False, align="right",
        font=dict(size=11, color=status_color, family="Arial"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor=status_color, borderwidth=1.5, borderpad=8,
    ))
    annotations.append(dict(
        text=_inventory_html(state), xref="paper", yref="paper",
        x=0.995, y=0.80, xanchor="right", yanchor="top",
        showarrow=False, align="left",
        font=dict(size=11, color="#1A1F26", family="Arial"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#3D4148", borderwidth=1, borderpad=8,
    ))

    span_x = max(0.001, x_hi - x_lo)
    span_y = max(0.001, y_hi - y_lo)
    span_z = max(0.001, z_hi - z_lo)
    m = max(span_x, span_y, span_z)
    ar_x, ar_y, ar_z = span_x / m, span_y / m, span_z / m

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=15, color="#1A1F26"),
                   yref="paper", y=0.985),
        paper_bgcolor=BG_PAPER,
        scene=dict(
            xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
            aspectmode="manual",
            aspectratio=dict(x=ar_x, y=ar_y, z=ar_z),
            camera=cam,
            domain=dict(x=[0.0, 1.0], y=[0.02, 0.96]),
        ),
        margin=dict(l=0, r=0, t=28, b=0),
        showlegend=False,
        annotations=annotations,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.write_image(save_path, width=width, height=height, scale=1.4)
    return fig


def render_state_pair(state: GameState, save_dir: str, base_name: str,
                      title: Optional[str] = None,
                      views: Tuple[str, str] = ("front", "back"),
                      **kwargs) -> List[str]:
    os.makedirs(save_dir, exist_ok=True)
    paths: List[str] = []
    for v in views:
        path = os.path.join(save_dir, f"{base_name}_{v}.png")
        render_state(state, save_path=path, title=title, view=v, **kwargs)
        paths.append(path)
    return paths
