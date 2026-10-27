"""
PipeFit 3D — Plotly renderer (headless, kaleido 0.2.1).

Aesthetic decisions adopted from T3 lessons on day one:
  * One color per role — pipes share one graphite tone regardless of I/L/T;
    supports share one warm-grey tone regardless of S1/S2.
  * Connector-accurate pipe rendering: cylindrical stubs from cell center to
    each connector face.
  * Embedded legend + STATUS panel + INVENTORY panel per render.
  * 1400 × 1100 canvas, rectangular bbox, manual aspect ratio normalised to
    max = 1, so empty axis planes crop out and the camera distance stays
    predictable.
  * Two camera angles (front / back) via render_state_pair.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import plotly.graph_objects as go

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import chain_hud as hud  # noqa: E402
import chain_viz as cv  # noqa: E402

from .game_core import (  # noqa: E402
    DIRS, GameState, LevelSpec, PieceType, PlacedPiece, Vec3,
    connector_dirs, face_neighbors, piece_is_pipe, piece_is_support,
    portal_inward,
)
from .mechanics import (  # noqa: E402
    _support_cells, double_unanchored_violations, unanchored_pipe_cells,
)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

PIPE_BODY       = "#3A4556"     # slightly bluer, more "metal pipe"
PIPE_JOINT      = "#2A3240"     # darker joints for contrast
PIPE_FLANGE     = "#B8922A"     # brass flange at pipe ends
SUPPORT_BODY    = "#8B7355"     # warm grey/tan
SUPPORT_ACCENT  = "#5C4B38"
ENTRANCE_COLOR  = "#1E8F3E"     # leaf green
ENTRANCE_ACCENT = "#A6E4B5"     # lighter green for water
EXIT_COLOR      = "#E0751A"     # bright orange
EXIT_ACCENT     = "#FFB873"     # lighter orange for funnel
WATER_BLUE      = "#3FA9F5"     # droplet / stream colour
PORTAL_FLANGE   = "#D9C280"     # gold/brass flange on portals
OBSTACLE_BODY   = "#7A1A1A"
OBSTACLE_DARK   = "#4F0F0F"
VIOLATION_COLOR = "#E53935"     # bright red edge for failed anchor edges ONLY


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


def _sphere(center: np.ndarray, radius: float,
            n_u: int = 20, n_v: int = 12) -> Dict[str, np.ndarray]:
    thetas = np.linspace(0.0, 2 * np.pi, n_u, endpoint=False)
    phis   = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, n_v)
    pts = np.zeros((n_v, n_u, 3))
    for a, phi in enumerate(phis):
        for b, theta in enumerate(thetas):
            pts[a, b] = center + radius * np.array([
                np.cos(phi) * np.cos(theta),
                np.cos(phi) * np.sin(theta),
                np.sin(phi),
            ])
    xs = pts[..., 0].ravel(); ys = pts[..., 1].ravel(); zs = pts[..., 2].ravel()
    ii, jj, kk = [], [], []
    for a in range(n_v - 1):
        for b in range(n_u):
            c = (b + 1) % n_u
            idx = lambda aa, bb: aa * n_u + bb
            ii.append(idx(a, b));     jj.append(idx(a, c));     kk.append(idx(a + 1, c))
            ii.append(idx(a, b));     jj.append(idx(a + 1, c)); kk.append(idx(a + 1, b))
    return {"x": xs, "y": ys, "z": zs,
            "i": np.array(ii), "j": np.array(jj), "k": np.array(kk)}


def _cylinder(p1: np.ndarray, p2: np.ndarray, radius: float,
              n_seg: int = 20, cap: bool = True) -> Dict[str, np.ndarray]:
    axis = p2 - p1
    L = float(np.linalg.norm(axis))
    if L < 1e-9:
        return _empty_mesh()
    axis = axis / L
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    thetas = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    c1 = np.stack([p1 + radius * (np.cos(t) * u + np.sin(t) * v) for t in thetas])
    c2 = np.stack([p2 + radius * (np.cos(t) * u + np.sin(t) * v) for t in thetas])
    pts = np.concatenate([c1, c2], axis=0)
    ii, jj, kk = [], [], []
    for a in range(n_seg):
        b = (a + 1) % n_seg
        ii.append(a); jj.append(b); kk.append(n_seg + b)
        ii.append(a); jj.append(n_seg + b); kk.append(n_seg + a)
    if cap:
        pts = np.concatenate([pts, p1[None, :], p2[None, :]], axis=0)
        base_idx = 2 * n_seg
        top_idx  = 2 * n_seg + 1
        for a in range(n_seg):
            b = (a + 1) % n_seg
            ii.append(base_idx); jj.append(b);         kk.append(a)
            ii.append(top_idx);  jj.append(n_seg + a); kk.append(n_seg + b)
    return {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
            "i": np.array(ii), "j": np.array(jj), "k": np.array(kk)}


def _box_asym(center: np.ndarray, sx: float, sy: float, sz: float) -> Dict[str, np.ndarray]:
    dx, dy, dz = sx / 2, sy / 2, sz / 2
    cx, cy, cz = center
    v = np.array([
        [cx - dx, cy - dy, cz - dz], [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz], [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz], [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz], [cx - dx, cy + dy, cz + dz],
    ])
    tris = np.array([
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 5, 1), (0, 4, 5), (2, 7, 3), (2, 6, 7),
        (0, 3, 7), (0, 7, 4), (1, 6, 2), (1, 5, 6),
    ])
    return {"x": v[:, 0], "y": v[:, 1], "z": v[:, 2],
            "i": tris[:, 0], "j": tris[:, 1], "k": tris[:, 2]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_center(pos: Vec3) -> np.ndarray:
    return np.array([pos[0] - 0.5, pos[1] - 0.5, pos[2] - 0.5])


def _mesh_trace(mesh: Dict[str, np.ndarray], color: str, opacity: float,
                name: str) -> go.Mesh3d:
    if len(mesh["x"]) == 0:
        return go.Mesh3d(x=[], y=[], z=[], showlegend=False, hoverinfo="skip")
    return go.Mesh3d(
        x=mesh["x"], y=mesh["y"], z=mesh["z"],
        i=mesh["i"], j=mesh["j"], k=mesh["k"],
        color=color, opacity=opacity, flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.85, specular=0.25,
                      roughness=0.55, fresnel=0.12),
        lightposition=dict(x=100, y=200, z=300),
        name=name, hoverinfo="skip", showlegend=False,
    )


def _line_trace(lines: List[Tuple[np.ndarray, np.ndarray]],
                color: str, width: float, name: str) -> go.Scatter3d:
    xs, ys, zs = [], [], []
    for a, b in lines:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        name=name, hoverinfo="skip", showlegend=False,
    )


# ---------------------------------------------------------------------------
# Pipe + joint geometry
# ---------------------------------------------------------------------------

def _build_pipes(state: GameState) -> Tuple[List[go.Mesh3d], List[go.Scatter3d]]:
    """Render pipes as metallic tubes:
       * thicker cylindrical body
       * brass flange ring at the connector tip (where pipe meets cell face)
       * slightly darker joint sphere at cell center
       * highlight ring at the top of each cylinder for glint
    ONLY violated cells get a red halo (double-unanchored edges).  Legal
    unanchored pipes get no halo — they're normal play state.
    """
    body_parts: List[Dict[str, np.ndarray]] = []
    joint_parts: List[Dict[str, np.ndarray]] = []
    flange_parts: List[Dict[str, np.ndarray]] = []
    glint_parts: List[Dict[str, np.ndarray]] = []

    violated_edges = double_unanchored_violations(state)
    violated_cells: Set[Vec3] = set()
    for u, v in violated_edges:
        violated_cells.add(u)
        violated_cells.add(v)

    for piece in state.piece_anchors():
        if not piece_is_pipe(piece.type):
            continue
        center = _cell_center(piece.pos)
        # Joint sphere at center (slightly bigger than tube for clean junction)
        joint_parts.append(_sphere(center, radius=0.20, n_u=16, n_v=10))
        # Cylinder stubs toward each connector face + brass flange at tip
        for d_idx in connector_dirs(piece.type, piece.orientation):
            d = np.array(DIRS[d_idx][1], dtype=float)
            tip_outer = center + d * 0.50       # at cell face
            tip_inner = center + d * 0.42       # just inside the face
            body_parts.append(_cylinder(center, tip_outer,
                                        radius=0.17, n_seg=22, cap=True))
            # brass flange: short fat ring near the tip
            flange_parts.append(_cylinder(tip_inner, tip_outer,
                                          radius=0.22, n_seg=24, cap=True))
            # white glint: very thin disc just below the flange (in tube direction)
            glint_center = center + d * 0.22
            glint_parts.append(_cylinder(glint_center - d * 0.01,
                                         glint_center + d * 0.01,
                                         radius=0.175, n_seg=20, cap=False))

    mesh_traces: List[go.Mesh3d] = []
    if body_parts:
        mesh_traces.append(_mesh_trace(_concat_meshes(*body_parts),
                                       color=PIPE_BODY, opacity=1.0, name="pipe-body"))
    if joint_parts:
        mesh_traces.append(_mesh_trace(_concat_meshes(*joint_parts),
                                       color=PIPE_JOINT, opacity=1.0, name="pipe-joint"))
    if flange_parts:
        mesh_traces.append(_mesh_trace(_concat_meshes(*flange_parts),
                                       color=PIPE_FLANGE, opacity=1.0, name="pipe-flange"))
    if glint_parts:
        mesh_traces.append(_mesh_trace(_concat_meshes(*glint_parts),
                                       color="#E8EEF5", opacity=0.55, name="pipe-glint"))

    # ONLY violation halos — never legal-unanchored markers
    line_traces: List[go.Scatter3d] = []
    if violated_cells:
        rings = []
        for c in violated_cells:
            cc = _cell_center(c)
            n = 30
            for t in range(n):
                t0 = 2 * np.pi * t / n
                t1 = 2 * np.pi * (t + 1) / n
                r = 0.32
                p0 = cc + np.array([r * np.cos(t0), r * np.sin(t0), -0.32])
                p1 = cc + np.array([r * np.cos(t1), r * np.sin(t1), -0.32])
                rings.append((p0, p1))
        line_traces.append(_line_trace(
            rings, VIOLATION_COLOR, 7, "violation-halo"))
    return mesh_traces, line_traces


# ---------------------------------------------------------------------------
# Supports
# ---------------------------------------------------------------------------

def _build_supports(state: GameState) -> Tuple[List[go.Mesh3d], List[go.Scatter3d]]:
    """Render supports as slimmer columns with a visible equator line so S2 reads as a single piece."""
    s1_parts: List[Dict[str, np.ndarray]] = []
    s2_parts: List[Dict[str, np.ndarray]] = []
    accent_parts: List[Dict[str, np.ndarray]] = []
    s2_divider_lines: List[Tuple[np.ndarray, np.ndarray]] = []

    for piece in state.piece_anchors():
        if not piece_is_support(piece.type):
            continue
        if piece.type == PieceType.SUPPORT_S1:
            c = _cell_center(piece.pos)
            s1_parts.append(_box_asym(c, sx=0.46, sy=0.46, sz=0.82))
            # Single top cap
            top = c + np.array([0, 0, 0.42])
            accent_parts.append(_box_asym(top, sx=0.54, sy=0.54, sz=0.04))
        else:  # S2
            base = _cell_center(piece.pos)
            tall_center = base + np.array([0.0, 0.0, 0.5])
            s2_parts.append(_box_asym(tall_center,
                                      sx=0.46, sy=0.46, sz=1.82))
            # Top cap
            accent_parts.append(_box_asym(
                base + np.array([0, 0, 1.42]), sx=0.54, sy=0.54, sz=0.04))
            # Equator line (y-frame) to show the piece is one continuous bar
            # at the joining z-face between its two occupied cells
            equator_z = base[2] + 0.5  # which is the top of lower cell
            d = 0.25
            for (a, b) in [
                ([-d, -d, 0], [+d, -d, 0]),
                ([+d, -d, 0], [+d, +d, 0]),
                ([+d, +d, 0], [-d, +d, 0]),
                ([-d, +d, 0], [-d, -d, 0]),
            ]:
                p1 = np.array([base[0] + a[0], base[1] + a[1], equator_z])
                p2 = np.array([base[0] + b[0], base[1] + b[1], equator_z])
                s2_divider_lines.append((p1, p2))

    traces: List[go.Mesh3d] = []
    if s1_parts:
        traces.append(_mesh_trace(_concat_meshes(*s1_parts),
                                  color=SUPPORT_BODY, opacity=0.95, name="s1"))
    if s2_parts:
        traces.append(_mesh_trace(_concat_meshes(*s2_parts),
                                  color=SUPPORT_BODY, opacity=0.95, name="s2"))
    if accent_parts:
        traces.append(_mesh_trace(_concat_meshes(*accent_parts),
                                  color=SUPPORT_ACCENT, opacity=1.0, name="support-accent"))
    line_traces: List[go.Scatter3d] = []
    if s2_divider_lines:
        line_traces.append(_line_trace(s2_divider_lines, "#3B2F22", 3,
                                       "s2-equator"))
    return traces, line_traces


# ---------------------------------------------------------------------------
# Entrance / exit / obstacle
# ---------------------------------------------------------------------------

def _portal_connect_dir(pos: Vec3, grid_size: Vec3) -> int:
    """Side a portal is joined from (shared with the rules)."""
    return portal_inward(pos, grid_size)


def _build_portals(state: GameState) -> Tuple[List[go.Mesh3d], List[go.Scatter3d]]:
    """Direction-agnostic portal: decorative ring flange around the cell face
       + gem core + star emblem.  No directional "faucet/funnel" bias.
       Entrance = green gem, exit = orange gem; same shape otherwise."""
    mesh_traces: List[go.Mesh3d] = []
    line_traces: List[go.Scatter3d] = []

    for role, cells, color, accent in [
        ("E", state.spec.entrances, ENTRANCE_COLOR, ENTRANCE_ACCENT),
        ("X", state.spec.exits,     EXIT_COLOR,     EXIT_ACCENT),
    ]:
        for p in cells:
            cc = _cell_center(p)
            conn_d = _portal_connect_dir(p, state.spec.grid_size)
            d = np.array(DIRS[conn_d][1], dtype=float)
            # Basis in the plane perpendicular to d
            ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(d, ref); u /= np.linalg.norm(u)
            v = np.cross(d, u)

            # Gem core — centred, slightly offset toward the outer face so it
            # "pokes out" a little through the face for visibility
            core = _sphere(cc + d * 0.08, radius=0.26, n_u=20, n_v=13)
            mesh_traces.append(_mesh_trace(core, color, 1.0, f"{role}-core-{p}"))

            # Two short brass flange rings at the face (shallow tubes,
            # concentric, different radii)
            face_mid = cc + d * 0.42
            face_out = cc + d * 0.50
            flange_outer = _cylinder(face_mid, face_out,
                                      radius=0.36, n_seg=32, cap=True)
            flange_inner = _cylinder(face_mid, face_out,
                                      radius=0.28, n_seg=28, cap=True)
            mesh_traces.append(_mesh_trace(flange_outer, PORTAL_FLANGE, 1.0,
                                           f"{role}-flange-outer-{p}"))
            mesh_traces.append(_mesh_trace(flange_inner, color, 1.0,
                                           f"{role}-flange-inner-{p}"))

            # Ring of accent dots spaced around the flange (4 small spheres)
            for t in range(4):
                ang = t * np.pi / 2 + np.pi / 4
                dot_center = face_mid + 0.32 * (np.cos(ang) * u + np.sin(ang) * v)
                dot = _sphere(dot_center, radius=0.045, n_u=10, n_v=7)
                mesh_traces.append(_mesh_trace(dot, accent, 1.0,
                                               f"{role}-dot-{p}-{t}"))

            # Star / cross emblem on the outward face (drawn as lines)
            emblem: List[Tuple[np.ndarray, np.ndarray]] = []
            for t in range(4):
                ang = t * np.pi / 2
                a1 = face_out + 0.18 * (np.cos(ang) * u + np.sin(ang) * v) + d * 0.005
                a2 = face_out + 0.02 * (np.cos(ang) * u + np.sin(ang) * v) + d * 0.005
                emblem.append((a1, a2))
            emb_color = "#FFFFFF" if role == "E" else "#2A1808"
            line_traces.append(_line_trace(emblem, emb_color, 7,
                                            f"{role}-emblem-{p}"))

            # Decorative ring at the gem equator for extra interest
            ring_lines: List[Tuple[np.ndarray, np.ndarray]] = []
            for t in range(24):
                t0 = 2 * np.pi * t / 24
                t1 = 2 * np.pi * (t + 1) / 24
                r = 0.27
                p0 = cc + d * 0.08 + r * (np.cos(t0) * u + np.sin(t0) * v)
                p1 = cc + d * 0.08 + r * (np.cos(t1) * u + np.sin(t1) * v)
                ring_lines.append((p0, p1))
            line_traces.append(_line_trace(ring_lines, accent, 2,
                                            f"{role}-gem-ring-{p}"))

            # Role tag letter above the cell (tiny)
            tag_pos = cc + d * 0.55 + np.array([0, 0, 0.35])
            # Use a text scatter3d for the role letter
            line_traces.append(go.Scatter3d(
                x=[tag_pos[0]], y=[tag_pos[1]], z=[tag_pos[2]],
                mode="text",
                text=[f"<b>{role}</b>"],
                textfont=dict(color=color, size=18, family="Arial Black"),
                textposition="middle center",
                hoverinfo="skip", showlegend=False,
            ))

    return mesh_traces, line_traces


def _build_obstacles(state: GameState) -> Tuple[List[go.Mesh3d], List[go.Scatter3d]]:
    """Render each OBSTACLE SHAPE as one merged rectangular block with an X
       on the largest face.  Shapes can span 1, 2, or 3 cells in any axis."""
    mesh_traces: List[go.Mesh3d] = []
    line_traces: List[go.Scatter3d] = []

    shapes = state.spec.obstacle_shapes
    if not shapes:
        # Back-compat: treat each cell as its own 1-cell shape
        shapes = [[c] for c in state.spec.obstacles]

    for shape in shapes:
        if not shape:
            continue
        xs = [c[0] for c in shape]; ys = [c[1] for c in shape]; zs = [c[2] for c in shape]
        x_lo, x_hi = min(xs), max(xs)
        y_lo, y_hi = min(ys), max(ys)
        z_lo, z_hi = min(zs), max(zs)
        cx = (x_lo - 0.5 + x_hi - 0.5) / 2
        cy = (y_lo - 0.5 + y_hi - 0.5) / 2
        cz = (z_lo - 0.5 + z_hi - 0.5) / 2
        sx = (x_hi - x_lo + 1) * 0.76
        sy = (y_hi - y_lo + 1) * 0.76
        sz = (z_hi - z_lo + 1) * 0.76
        body = _box_asym(np.array([cx, cy, cz]), sx, sy, sz)
        mesh_traces.append(_mesh_trace(body, OBSTACLE_BODY, 1.0,
                                       f"obs-{x_lo}-{y_lo}-{z_lo}"))
        # Add a thin darker cap at top
        top_z = cz + sz / 2
        cap = _box_asym(np.array([cx, cy, top_z + 0.01]),
                        sx + 0.03, sy + 0.03, 0.04)
        mesh_traces.append(_mesh_trace(cap, OBSTACLE_DARK, 1.0,
                                       f"obs-cap-{x_lo}-{y_lo}-{z_lo}"))
        # Big white X on top — scale with shape footprint
        hx = (sx / 2) * 0.75
        hy = (sy / 2) * 0.75
        x_lines = [
            (np.array([cx - hx, cy - hy, top_z + 0.04]),
             np.array([cx + hx, cy + hy, top_z + 0.04])),
            (np.array([cx - hx, cy + hy, top_z + 0.04]),
             np.array([cx + hx, cy - hy, top_z + 0.04])),
        ]
        line_traces.append(_line_trace(x_lines, "#FFFFFF", 10,
                                       f"obs-x-{x_lo}-{y_lo}-{z_lo}"))
        # Warning stripes on the front face (perpendicular to y_lo face)
        front_y = cy - sy / 2
        stripe_lines = []
        for k in np.linspace(-hx + 0.05, hx - 0.05, 4):
            stripe_lines.append((
                np.array([cx + k - 0.06, front_y - 0.003, cz - sz / 2 + 0.10]),
                np.array([cx + k + 0.06, front_y - 0.003, cz + sz / 2 - 0.10]),
            ))
        line_traces.append(_line_trace(stripe_lines, "#F2E2A8", 3,
                                        f"obs-stripes-{x_lo}-{y_lo}-{z_lo}"))
    return mesh_traces, line_traces


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

_PIECE_ICONS = {
    PieceType.PIPE_I: ("Pipe  I  straight", lambda: hud.icon_pipe("I", PIPE_BODY, PIPE_FLANGE)),
    PieceType.PIPE_L: ("Pipe  L  corner", lambda: hud.icon_pipe("L", PIPE_BODY, PIPE_FLANGE)),
    PieceType.PIPE_T: ("Pipe  T  fork", lambda: hud.icon_pipe("T", PIPE_BODY, PIPE_FLANGE)),
    PieceType.SUPPORT_S1: ("Support  S1", lambda: hud.icon_pillar(SUPPORT_BODY, 1)),
    PieceType.SUPPORT_S2: ("Support  S2", lambda: hud.icon_pillar(SUPPORT_BODY, 2)),
}


def _hud(state: GameState, caption: str) -> "hud.Hud":
    n_placed = len({p.pos for p in state.piece_anchors()})
    n_total = sum(state.spec.inventory.values())
    n_prob = len(double_unanchored_violations(state))
    term = state.terminal_code.value if state.terminal_code else None
    stats = [
        hud.Stat(hud.icon_pipe("L", PIPE_BODY, PIPE_FLANGE), "Pieces placed",
                 f"{n_placed} / {n_total}", n_placed / max(1, n_total), "#2FD3C6"),
        hud.Stat(hud.icon_arrow("#8FD8FF"), "Actions used",
                 f"{state.action_count} / {state.max_actions}",
                 state.action_count / max(1, state.max_actions), "#8FD8FF"),
        hud.Stat(hud.icon_ring(VIOLATION_COLOR), "Problems (after submit)",
                 str(n_prob), None, "#FF4D5E" if n_prob else "#2BD07A"),
    ]
    inventory = []
    for pt, (label, make_icon) in _PIECE_ICONS.items():
        tot = state.spec.inventory.get(pt, 0)
        if tot:
            inventory.append(hud.Item(make_icon(), label, state.remaining.get(pt, 0), tot))
    legend = [
        hud.LegendEntry(hud.icon_gem(ENTRANCE_COLOR, PORTAL_FLANGE), "Entrance",
                        "water in; takes one pipe from inside"),
        hud.LegendEntry(hud.icon_gem(EXIT_COLOR, PORTAL_FLANGE), "Exit",
                        "every exit must be joined too"),
        hud.LegendEntry(hud.icon_pipe("T", PIPE_BODY, PIPE_FLANGE), "Pipe  I / L / T",
                        "brass rings = open ends; ends must meet"),
        hud.LegendEntry(hud.icon_pillar(SUPPORT_BODY, 2), "Support  S1 / S2",
                        "1- or 2-cell block that holds pipes up"),
        hud.LegendEntry(hud.icon_block_x(OBSTACLE_BODY), "Obstacle",
                        "nothing can be built here"),
        hud.LegendEntry(hud.icon_ring(VIOLATION_COLOR), "Red ring",
                        "leak or missing support (after submit)"),
    ]
    return hud.Hud(
        title="PIPEFIT",
        caption=caption,
        result=cv.result_label(term),
        stats=stats,
        inventory=inventory,
        inventory_title="PARTS BIN",
        legend=legend,
        hint="Join every entrance and exit in one leak-free network  \u2022  every pipe rests on the floor, a support or a supported pipe",
    )


def render_state(state: GameState,
                 save_path: Optional[str],
                 title: Optional[str] = None,
                 view: str = "front") -> go.Figure:
    spec = state.spec
    grid = spec.grid_size

    traces: List = cv.scene_traces(grid, view, "pipefit")
    raised = (list(spec.entrances) + list(spec.exits) + list(spec.obstacles)
              + [p.pos for p in state.piece_anchors()
                 if piece_is_pipe(p.type)])
    traces.extend(cv.drop_line_traces(raised, cv.drop_color("pipefit")))

    obs_m, obs_l = _build_obstacles(state)
    traces.extend(obs_m); traces.extend(obs_l)

    supp_m, supp_l = _build_supports(state)
    traces.extend(supp_m); traces.extend(supp_l)

    pipe_m, pipe_l = _build_pipes(state)
    traces.extend(pipe_m); traces.extend(pipe_l)

    port_m, port_l = _build_portals(state)
    traces.extend(port_m); traces.extend(port_l)

    term = state.terminal_code.value if state.terminal_code else "playing"
    caption = f"{title or term}   \u00b7   view: {view}"
    fig = go.Figure(data=traces)
    return cv.render_game(fig, grid, view, "pipefit", _hud(state, caption), save_path)


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
