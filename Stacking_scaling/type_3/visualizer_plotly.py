"""
FluidNet 3D - Plotly Renderer (headless, high-quality)

Uses plotly + kaleido to produce WebGL-grade 3D renders with smooth shading,
proper depth-buffer occlusion, and anti-aliased edges.  Works without X11.

Aesthetic choices:
  * PIPE / open-VALVE / SOURCE / TARGET connectivity is drawn as actual
    cylindrical tubes between cell centers (Mesh3d with triangulated
    cylinders).  When the segment carries fluid it thickens and takes the
    fluid color; empty it stays a slim graphite tube.
  * SOURCE is a UV-faceted sphere with a tiny white specular cap.
  * TARGET (empty) is a translucent colored frame; (filled) is a frame with
    a visible colored volume inside and a bright water-line on top.
  * VALVE (closed) is a brass yellow rounded body with a dark hand-wheel;
    (open) is a slim ring collar flush with the conducting tube.
  * PROTECTED is a matte red block with a spiky top icon; on breach, a
    darker tone + starburst crack lines appear.
  * The box frame is a cool-gray wireframe; a faint floor grid sits at z=0.

Public API:
    render_state(state, save_path, title=None, view="front")
    render_state_pair(state, save_dir, base_name, titles=None)
        -> writes {base}_A.png (front-right) and {base}_B.png (reverse-left)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import chain_hud as hud  # noqa: E402
import chain_viz as cv  # noqa: E402

from .game_core import CellType, GameState, face_neighbors  # noqa: E402


def _clash_marker(state: GameState) -> List[go.Scatter3d]:
    """Red box + cross where two colours met (or the wrong colour reached a
    tank): a colour loss otherwise shows no visible culprit."""
    from .game_core import ActionCode
    if state.terminal_code not in (ActionCode.COLOR_CONTAMINATION,
                                   ActionCode.WRONG_COLOR_TARGET):
        return []
    if state.violation_cell is None:
        return []
    c = _cell_center(state.violation_cell)
    h = 0.42
    corners = [c + np.array([sx, sy, sz]) for sz in (-h, h)
               for sx, sy in ((-h, -h), (h, -h), (h, h), (-h, h))]
    edges = [(i, (i + 1) % 4) for i in range(4)]
    edges += [(4 + i, 4 + (i + 1) % 4) for i in range(4)]
    edges += [(i, i + 4) for i in range(4)]
    segs = [(corners[a], corners[b]) for a, b in edges]
    segs += [(corners[4], corners[6]), (corners[5], corners[7])]
    return [_line_trace(segs, "#FF1744", 8, "colour-clash")]


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

COLOR_FLUID: Dict[str, str] = {
    "blue":   "#1E5FB8",          # deeper saturated blue
    "red":    "#D62E2E",
    "green":  "#0F8F5A",
    "orange": "#E07F1A",
    "purple": "#6E3CB8",
    "cyan":   "#0F9CA8",
}

PIPE_BODY         = "#2F3640"     # ONE pipe gray for everything dry
PIPE_BODY_LIGHT   = "#2F3640"     # (alias, kept for backward compat)
VALVE_BODY        = "#F2B315"     # vivid amber
VALVE_BODY_OPEN   = "#FFD75A"
VALVE_HANDLE      = "#1A1F26"     # near-black
PROTECTED_FACE    = "#C81818"     # vivid danger red
PROTECTED_DARK    = "#560808"


# ---------------------------------------------------------------------------
# Mesh primitives (return x, y, z flat arrays + i, j, k triangle indices)
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


def _sphere(center: np.ndarray, radius: float, n_u: int = 20, n_v: int = 12) -> Dict[str, np.ndarray]:
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
        # two tris per quad (c1[a],c1[b],c2[b]) + (c1[a],c2[b],c2[a])
        ii.append(a);           jj.append(b);           kk.append(n_seg + b)
        ii.append(a);           jj.append(n_seg + b);   kk.append(n_seg + a)
    if cap:
        # Bottom cap: p1 index = 2n, top cap: p2 index = 2n + 1
        pts = np.concatenate([pts, p1[None, :], p2[None, :]], axis=0)
        base_idx = 2 * n_seg
        top_idx  = 2 * n_seg + 1
        for a in range(n_seg):
            b = (a + 1) % n_seg
            ii.append(base_idx); jj.append(b);         kk.append(a)
            ii.append(top_idx);  jj.append(n_seg + a); kk.append(n_seg + b)
    return {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
            "i": np.array(ii), "j": np.array(jj), "k": np.array(kk)}


def _cube(center: np.ndarray, size: float) -> Dict[str, np.ndarray]:
    d = size / 2.0
    cx, cy, cz = center
    v = np.array([
        [cx - d, cy - d, cz - d], [cx + d, cy - d, cz - d],
        [cx + d, cy + d, cz - d], [cx - d, cy + d, cz - d],
        [cx - d, cy - d, cz + d], [cx + d, cy - d, cz + d],
        [cx + d, cy + d, cz + d], [cx - d, cy + d, cz + d],
    ])
    # 12 triangles (two per face)
    tris = [
        (0, 1, 2), (0, 2, 3),       # -z
        (4, 6, 5), (4, 7, 6),       # +z
        (0, 5, 1), (0, 4, 5),       # -y
        (2, 7, 3), (2, 6, 7),       # +y
        (0, 3, 7), (0, 7, 4),       # -x
        (1, 6, 2), (1, 5, 6),       # +x
    ]
    tris = np.array(tris)
    return {"x": v[:, 0], "y": v[:, 1], "z": v[:, 2],
            "i": tris[:, 0], "j": tris[:, 1], "k": tris[:, 2]}


def _disc(center: np.ndarray, radius: float,
          normal: np.ndarray, n_seg: int = 26) -> Dict[str, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, ref); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    thetas = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    ring = np.stack([center + radius * (np.cos(t) * u + np.sin(t) * v) for t in thetas])
    pts = np.concatenate([ring, center[None, :]], axis=0)
    hub = n_seg
    ii, jj, kk = [], [], []
    for a in range(n_seg):
        b = (a + 1) % n_seg
        ii.append(hub); jj.append(a); kk.append(b)
    return {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
            "i": np.array(ii), "j": np.array(jj), "k": np.array(kk)}


# ---------------------------------------------------------------------------
# Build traces
# ---------------------------------------------------------------------------

def _cell_center(pos: Tuple[int, int, int]) -> np.ndarray:
    return np.array([pos[0] - 0.5, pos[1] - 0.5, pos[2] - 0.5])


def _mesh_trace(mesh: Dict[str, np.ndarray], color: str, opacity: float,
                name: str, flatshading: bool = True) -> go.Mesh3d:
    return go.Mesh3d(
        x=mesh["x"], y=mesh["y"], z=mesh["z"],
        i=mesh["i"], j=mesh["j"], k=mesh["k"],
        color=color, opacity=opacity, flatshading=flatshading,
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.3,
                      roughness=0.55, fresnel=0.12),
        lightposition=dict(x=100, y=200, z=300),
        name=name, hoverinfo="skip",
        showlegend=False,
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


def _is_conductive(state: GameState, pos: Tuple[int, int, int]) -> bool:
    cell = state.spec.cells.get(pos)
    if cell is None:
        return False
    if cell.type in (CellType.PIPE, CellType.SOURCE, CellType.TARGET):
        return True
    if cell.type == CellType.VALVE:
        return pos in state.opened_valves
    return False


def _build_tubes(state: GameState) -> Tuple[List[go.Mesh3d], List[go.Scatter3d]]:
    """Return [fluid_mesh, dry_mesh], and a list of dark accent rings."""
    grid = state.spec.grid_size
    dry_parts: List[Dict[str, np.ndarray]] = []
    fluid_by_color: Dict[str, List[Dict[str, np.ndarray]]] = {}
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
                # Short stub into the red block: shows that this pipe feeds
                # it (fluid in this cell = breach).
                seen.add(edge)
                p1 = _cell_center(pos)
                p2 = p1 + (_cell_center(nb) - p1) * 0.72
                fluid = state.fluid_cells.get(pos)
                if fluid and not (cell.type == CellType.VALVE
                                  and pos not in state.opened_valves):
                    fluid_by_color.setdefault(fluid, []).append(
                        _cylinder(p1, p2, radius=0.13, n_seg=20))
                else:
                    dry_parts.append(_cylinder(p1, p2, radius=0.08, n_seg=16))
                continue
            # skip segments where both ends are closed valves / empty
            if not (_is_conductive(state, pos) or _is_conductive(state, nb)):
                continue
            seen.add(edge)
            p1 = _cell_center(pos)
            p2 = _cell_center(nb)
            fluid = state.fluid_cells.get(pos)
            if fluid is None:
                fluid = state.fluid_cells.get(nb)
            # No paint into closed valve
            if nb_cell.type == CellType.VALVE and nb not in state.opened_valves:
                fluid = None
            if cell.type == CellType.VALVE and pos not in state.opened_valves:
                fluid = None
            if fluid:
                fluid_by_color.setdefault(fluid, []).append(
                    _cylinder(p1, p2, radius=0.13, n_seg=20))
            else:
                dry_parts.append(_cylinder(p1, p2, radius=0.08, n_seg=16))
    traces: List[go.Mesh3d] = []
    if dry_parts:
        traces.append(_mesh_trace(_concat_meshes(*dry_parts),
                                  color=PIPE_BODY, opacity=0.92, name="dry-pipe"))
    for col, parts in fluid_by_color.items():
        traces.append(_mesh_trace(_concat_meshes(*parts),
                                  color=COLOR_FLUID.get(col, "#2E86DE"),
                                  opacity=0.98, name=f"fluid-{col}"))
    return traces, []


def _build_joints(state: GameState) -> List[go.Mesh3d]:
    """
    Joint spheres at every PIPE cell.

    Kept the same color as the cylinder body so the pipe network reads as
    ONE continuous color (was previously two shades of gray, confusing).
    """
    dry_parts: List[Dict[str, np.ndarray]] = []
    fluid_by_color: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for pos, cell in state.spec.cells.items():
        if cell.type != CellType.PIPE:
            continue
        fluid = state.fluid_cells.get(pos)
        c = _cell_center(pos)
        if fluid:
            fluid_by_color.setdefault(fluid, []).append(
                _sphere(c, radius=0.13, n_u=12, n_v=8))
        else:
            dry_parts.append(_sphere(c, radius=0.09, n_u=10, n_v=7))
    traces: List[go.Mesh3d] = []
    if dry_parts:
        # Same color as cylinders — unified pipe palette.
        traces.append(_mesh_trace(_concat_meshes(*dry_parts),
                                  color=PIPE_BODY, opacity=1.0, name="joints"))
    for col, parts in fluid_by_color.items():
        traces.append(_mesh_trace(_concat_meshes(*parts),
                                  color=COLOR_FLUID.get(col, "#1E5FB8"),
                                  opacity=1.0, name=f"fluid-joints-{col}"))
    return traces


def _build_sources(state: GameState) -> List:
    """The flow's starting point must be the most obvious object on the
    board: a large glossy sphere on a pump pedestal with a halo ring (the
    old 0.28-radius ball read as just another pipe joint)."""
    traces: List = []
    for s in state.spec.sources:
        col = COLOR_FLUID.get(state.spec.cells[s].color, "#1E5FB8")
        c = _cell_center(s)
        base = _cylinder(c + np.array([0.0, 0.0, -0.5]),
                         c + np.array([0.0, 0.0, -0.32]), radius=0.42, n_seg=28,
                         cap=True)
        traces.append(_mesh_trace(base, "#2B3A4F", 1.0, "src-pump"))
        core = _sphere(c + np.array([0, 0, 0.04]), radius=0.38, n_u=26, n_v=16)
        traces.append(_mesh_trace(core, col, 1.0, "src-core", flatshading=False))
        highlight = _sphere(c + np.array([-0.12, -0.12, 0.25]),
                            radius=0.08, n_u=10, n_v=7)
        traces.append(_mesh_trace(highlight, "#FFFFFF", 0.9, "src-spec"))
        ring = [(c + np.array([0.5 * np.cos(a), 0.5 * np.sin(a), -0.3]),
                 c + np.array([0.5 * np.cos(b), 0.5 * np.sin(b), -0.3]))
                for a, b in zip(np.linspace(0, 2 * np.pi, 40)[:-1],
                                np.linspace(0, 2 * np.pi, 40)[1:])]
        traces.append(_line_trace(ring, "#7FD3FF", 6, "src-halo"))
    return traces


def _build_targets(state: GameState) -> Tuple[List[go.Mesh3d], List[go.Scatter3d]]:
    mesh_traces: List[go.Mesh3d] = []
    line_traces: List[go.Scatter3d] = []
    for t in state.spec.targets:
        col = state.spec.cells[t].color
        base = COLOR_FLUID.get(col, "#1E5FB8")
        filled = state.fluid_cells.get(t) == col
        c = _cell_center(t)
        # Translucent glass shell — slightly smaller for clearance
        shell_size = 0.78
        shell = _cube(c, size=shell_size)
        mesh_traces.append(_mesh_trace(shell, base,
                                       0.30 if filled else 0.10,
                                       f"target-shell-{t}"))
        # Accent edges
        d = shell_size / 2
        edges = []
        corners = np.array([
            [c[0] - d, c[1] - d, c[2] - d], [c[0] + d, c[1] - d, c[2] - d],
            [c[0] + d, c[1] + d, c[2] - d], [c[0] - d, c[1] + d, c[2] - d],
            [c[0] - d, c[1] - d, c[2] + d], [c[0] + d, c[1] - d, c[2] + d],
            [c[0] + d, c[1] + d, c[2] + d], [c[0] - d, c[1] + d, c[2] + d],
        ])
        for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]:
            edges.append((corners[a], corners[b]))
        line_traces.append(_line_trace(edges, base, 5, f"target-edge-{t}"))
        if filled:
            # Fluid volume (60% fill)
            level = 0.55
            d_xy = 0.30
            dz = 0.28
            bottom_z = c[2] - dz
            top_z = c[2] - dz + 2 * dz * level
            fluid_center = np.array([c[0], c[1], (bottom_z + top_z) / 2])
            fluid_box = _cube(fluid_center, size=1.0)
            # scale to asymmetric by manual construction
            v = np.array([
                [c[0]-d_xy, c[1]-d_xy, bottom_z], [c[0]+d_xy, c[1]-d_xy, bottom_z],
                [c[0]+d_xy, c[1]+d_xy, bottom_z], [c[0]-d_xy, c[1]+d_xy, bottom_z],
                [c[0]-d_xy, c[1]-d_xy, top_z],    [c[0]+d_xy, c[1]-d_xy, top_z],
                [c[0]+d_xy, c[1]+d_xy, top_z],    [c[0]-d_xy, c[1]+d_xy, top_z],
            ])
            tris = np.array([
                (0,1,2),(0,2,3),(4,6,5),(4,7,6),
                (0,5,1),(0,4,5),(2,7,3),(2,6,7),
                (0,3,7),(0,7,4),(1,6,2),(1,5,6),
            ])
            mesh_traces.append(_mesh_trace(
                {"x": v[:,0], "y": v[:,1], "z": v[:,2],
                 "i": tris[:,0], "j": tris[:,1], "k": tris[:,2]},
                base, 0.85, f"target-fluid-{t}",
            ))
            # Water surface highlight
            top_disc = _disc(np.array([c[0], c[1], top_z + 0.005]),
                             radius=d_xy * 0.95,
                             normal=np.array([0.0, 0.0, 1.0]))
            mesh_traces.append(_mesh_trace(top_disc, "#FFFFFF", 0.45,
                                           f"target-surface-{t}"))
    return mesh_traces, line_traces


def _build_valves(state: GameState) -> Tuple[List[go.Mesh3d], List[go.Scatter3d]]:
    mesh_traces: List[go.Mesh3d] = []
    line_traces: List[go.Scatter3d] = []
    for v in state.spec.valves:
        c = _cell_center(v)
        if v in state.opened_valves:
            fluid_here = state.fluid_cells.get(v)
            ring_color = COLOR_FLUID.get(fluid_here, "#1E5FB8") if fluid_here else PIPE_BODY
            collar = _sphere(c, radius=0.22, n_u=18, n_v=11)
            mesh_traces.append(_mesh_trace(collar, ring_color, 1.0, f"open-valve-{v}"))
            ring = _disc(c + np.array([0, 0, 0.22]), radius=0.17,
                         normal=np.array([0.0, 0.0, 1.0]))
            mesh_traces.append(_mesh_trace(ring, VALVE_BODY_OPEN, 1.0,
                                           f"open-valve-ring-{v}"))
        else:
            body = _cube(c, size=0.45)              # was 0.6 — more clearance
            mesh_traces.append(_mesh_trace(body, VALVE_BODY, 1.0, f"valve-body-{v}"))
            wheel_center = c + np.array([0, 0, 0.30])
            outer = _disc(wheel_center, radius=0.16,
                          normal=np.array([0.0, 0.0, 1.0]))
            inner = _disc(wheel_center, radius=0.06,
                          normal=np.array([0.0, 0.0, 1.0]))
            mesh_traces.append(_mesh_trace(outer, VALVE_HANDLE, 1.0, f"valve-wheel-{v}"))
            mesh_traces.append(_mesh_trace(inner, "#FFFFFF", 1.0, f"valve-hub-{v}"))
            spokes = []
            for k in range(4):
                ang = k * np.pi / 2
                p1 = wheel_center + np.array([0.16 * np.cos(ang),
                                              0.16 * np.sin(ang), 0])
                p2 = wheel_center - np.array([0.16 * np.cos(ang),
                                              0.16 * np.sin(ang), 0])
                spokes.append((p1, p2))
            line_traces.append(_line_trace(spokes, VALVE_HANDLE, 3, f"valve-spokes-{v}"))
            # Stem
            stem = _cylinder(c + np.array([0, 0, 0.0]),
                             wheel_center - np.array([0, 0, 0.02]),
                             radius=0.03, n_seg=10)
            mesh_traces.append(_mesh_trace(stem, VALVE_HANDLE, 1.0, f"valve-stem-{v}"))
    return mesh_traces, line_traces


def _build_protected(state: GameState) -> Tuple[List[go.Mesh3d], List[go.Scatter3d]]:
    mesh_traces: List[go.Mesh3d] = []
    line_traces: List[go.Scatter3d] = []
    for p in state.spec.protected:
        breached = p in state.fluid_cells
        c = _cell_center(p)
        body = _cube(c, size=0.65)               # was 0.82 — more clearance
        mesh_traces.append(_mesh_trace(
            body,
            PROTECTED_DARK if breached else PROTECTED_FACE,
            1.0,
            f"prot-{p}",
        ))
        # Bold white "⛔" X on top to make "no entry" obvious
        top_z = c[2] + 0.33
        x_lines = [
            (np.array([c[0] - 0.22, c[1] - 0.22, top_z + 0.01]),
             np.array([c[0] + 0.22, c[1] + 0.22, top_z + 0.01])),
            (np.array([c[0] - 0.22, c[1] + 0.22, top_z + 0.01]),
             np.array([c[0] + 0.22, c[1] - 0.22, top_z + 0.01])),
        ]
        line_traces.append(_line_trace(
            x_lines, "#FFFFFF", 8, f"prot-x-{p}",
        ))
        # Faint warning stripes underneath the X
        stripes = []
        for kk in np.linspace(-0.22, 0.22, 5):
            stripes.append((
                np.array([c[0] - 0.30, c[1] + kk, top_z + 0.005]),
                np.array([c[0] + 0.30, c[1] + kk + 0.07, top_z + 0.005]),
            ))
        line_traces.append(_line_trace(
            stripes, "#FFD9D9" if not breached else "#FFFFFF", 4,
            f"prot-stripes-{p}",
        ))
        if breached:
            cracks = []
            for k in range(8):
                ang = k * np.pi / 4
                p1 = np.array([c[0], c[1], top_z + 0.03])
                p2 = np.array([c[0] + 0.45 * np.cos(ang),
                               c[1] + 0.45 * np.sin(ang),
                               top_z + 0.03])
                cracks.append((p1, p2))
            line_traces.append(_line_trace(cracks, "#FFFFFF", 6, f"cracks-{p}"))
            # Bright outline so the breach is visible even when the dark
            # block is half hidden behind pipes.
            h = 0.45
            corners = [c + np.array([sx, sy, sz]) for sz in (-h, h)
                       for sx, sy in ((-h, -h), (h, -h), (h, h), (-h, h))]
            edges = [(i, (i + 1) % 4) for i in range(4)]
            edges += [(4 + i, 4 + (i + 1) % 4) for i in range(4)]
            edges += [(i, i + 4) for i in range(4)]
            line_traces.append(_line_trace(
                [(corners[a], corners[b]) for a, b in edges],
                "#FF1744", 7, f"breach-outline-{p}"))
    return mesh_traces, line_traces


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------

def _build_labels(state: GameState, view: str) -> List[go.Scatter3d]:
    """
    Smart 3D text labels for source / target / valve / protected cells.

    Each label is offset OUTWARD from the box center on every axis and
    additionally spread within its z-layer using the cell's relative
    position so labels do not stack on the same screen pixel.  Pipes are
    not labeled (too noisy and not actionable).
    """
    spec = state.spec
    A, B, C = spec.grid_size
    cx_box, cy_box, cz_box = A / 2.0, B / 2.0, C / 2.0

    role_color = {
        "S": "#0E3D6E",
        "T": "#0A5E4D",
        "V": "#6E4F08",
        "X": "#7A1414",
    }
    # Z-direction bias by role: source labels go below cell, target above,
    # valves slightly above with horizontal spread, protected sideways out.
    role_dz = {"S": -0.65, "T": 0.55, "V": 0.45, "X": 0.45}

    items: List[Tuple[str, Tuple[int, int, int]]] = []
    for s in spec.sources:
        items.append(("S", s))
    for t in spec.targets:
        items.append(("T", t))
    for v in spec.valves:
        tag = "v" if v in state.opened_valves else "V"
        items.append((tag, v))
    for p in spec.protected:
        items.append(("X", p))

    traces: List[go.Scatter3d] = []
    for tag, pos in items:
        cx, cy, cz = pos[0] - 0.5, pos[1] - 0.5, pos[2] - 0.5
        # Outward direction in xy: away from box center
        sx = 1.0 if cx >= cx_box else -1.0
        sy = 1.0 if cy >= cy_box else -1.0
        dz = role_dz.get(tag.upper(), 0.45)

        # Outward-radial offset to spread labels out into open space.
        if tag.upper() in ("X", "V", "T"):
            offset = np.array([sx * 0.65, sy * 0.65, dz])
        else:
            offset = np.array([sx * 0.40, sy * 0.40, dz])

        # Allow labels to live slightly OUTSIDE the box so corner cells don't
        # collide with their own geometry (or with each other inside the cube).
        lx = float(np.clip(cx + offset[0], -0.4, A + 0.4))
        ly = float(np.clip(cy + offset[1], -0.4, B + 0.4))
        lz = float(np.clip(cz + offset[2], -0.3, C + 0.3))

        col = role_color.get(tag.upper(), "#2D3436")
        text = f"<b>{tag}</b>·({pos[0]},{pos[1]},{pos[2]})"

        traces.append(go.Scatter3d(
            x=[lx], y=[ly], z=[lz],
            mode="text",
            text=[text],
            textfont=dict(color=col, size=14, family="Arial Black"),
            textposition="middle center",
            showlegend=False,
            hoverinfo="skip",
        ))
    return traces


def _hud(state: GameState, caption: str) -> "hud.Hud":
    spec = state.spec
    n_filled = sum(1 for t in spec.targets
                   if state.fluid_cells.get(t) == spec.cells[t].color)
    n_open = len(state.opened_valves)
    n_breach = sum(1 for p in spec.protected if p in state.fluid_cells)
    from .game_core import ActionCode
    clash = state.terminal_code in (ActionCode.COLOR_CONTAMINATION,
                                    ActionCode.WRONG_COLOR_TARGET)
    term = state.terminal_code.value if state.terminal_code else None
    fluid = COLOR_FLUID["blue"]
    two = len(spec.colors) > 1
    stats = [
        hud.Stat(hud.icon_target(fluid), "Targets filled",
                 f"{n_filled} / {len(spec.targets)}",
                 n_filled / max(1, len(spec.targets)), "#4DA3FF"),
        hud.Stat(hud.icon_valve(VALVE_BODY), "Valves opened",
                 f"{n_open} / {len(spec.valves)}",
                 n_open / max(1, len(spec.valves)), "#F2B315"),
        hud.Stat(hud.icon_block_x(PROTECTED_FACE), "Protected breached",
                 str(n_breach), None, "#FF4D5E" if n_breach else "#2BD07A"),
    ]
    if len(spec.colors) > 1:
        stats.append(hud.Stat(hud.icon_sphere(COLOR_FLUID["orange"]),
                              "Colours mixed", "yes" if clash else "no", None,
                              "#FF4D5E" if clash else "#2BD07A"))
    legend = [
        hud.LegendEntry(hud.icon_sphere(fluid), "Source",
                        "start of the flow (pump on a ring)"),
    ]
    if two:
        legend.append(hud.LegendEntry(hud.icon_sphere(COLOR_FLUID["orange"]),
                                      "Second pump",
                                      "its colour must never meet the other"))
    legend += [
        hud.LegendEntry(hud.icon_target(fluid), "Target tank",
                        "needs the colour of its frame" if two
                        else "fill every tank to win"),
        hud.LegendEntry(hud.icon_pipe("bar", PIPE_BODY, None), "Pipe",
                        "always open; coloured = carrying fluid"),
        hud.LegendEntry(hud.icon_valve(VALVE_BODY), "Valve (closed)",
                        "amber with a hand-wheel: blocks flow"),
        hud.LegendEntry(hud.icon_valve(open_=True), "Valve (opened)",
                        "opening is permanent"),
        hud.LegendEntry(hud.icon_block_x(PROTECTED_FACE), "Protected",
                        "fluid touching it = lose; a stub = its feed"),
    ]
    return hud.Hud(
        title="FLUIDNET",
        caption=caption,
        result=cv.result_label(term),
        stats=stats,
        legend=legend,
        hint=("Open one valve per turn  \u2022  each tank wants its frame's colour  \u2022  colours never meet  \u2022  spare the red cells"
              if two else
              "Each turn open one valve  \u2022  fluid floods everything connected  \u2022  fill all tanks, spare the red cells"),
    )


def render_state(state: GameState,
                 save_path: Optional[str],
                 title: Optional[str] = None,
                 view: str = "front",
                 show_labels: bool = False) -> go.Figure:
    """Render one state.  `show_labels` additionally prints the (x,y,z) of
    every source / target / valve / protected cell next to it; the grid
    coordinates on the floor edges are always drawn."""
    spec = state.spec
    grid = spec.grid_size

    traces: List = cv.scene_traces(grid, view, "fluidnet")
    key_cells = (list(spec.sources) + list(spec.targets)
                 + list(spec.valves) + list(spec.protected))
    traces.extend(cv.drop_line_traces(key_cells, cv.drop_color("fluidnet")))

    prot_mesh, prot_lines = _build_protected(state)
    traces.extend(prot_mesh); traces.extend(prot_lines)

    tube_traces, _ = _build_tubes(state)
    traces.extend(tube_traces)
    traces.extend(_build_joints(state))

    valve_mesh, valve_lines = _build_valves(state)
    traces.extend(valve_mesh); traces.extend(valve_lines)

    target_mesh, target_lines = _build_targets(state)
    traces.extend(target_mesh); traces.extend(target_lines)

    traces.extend(_build_sources(state))
    traces.extend(_clash_marker(state))

    if show_labels:
        traces.extend(_build_labels(state, view))

    term = state.terminal_code.value if state.terminal_code else "playing"
    caption = f"{title or term}   \u00b7   view: {view}"
    fig = go.Figure(data=traces)
    return cv.render_game(fig, grid, view, "fluidnet", _hud(state, caption), save_path)


def render_state_pair(state: GameState, save_dir: str, base_name: str,
                      title: Optional[str] = None,
                      views: Tuple[str, str] = ("front", "back"),
                      **kwargs) -> List[str]:
    """Render the same state from two different cameras.  Returns paths."""
    os.makedirs(save_dir, exist_ok=True)
    paths: List[str] = []
    for v in views:
        path = os.path.join(save_dir, f"{base_name}_{v}.png")
        render_state(state, save_path=path, title=title, view=v, **kwargs)
        paths.append(path)
    return paths
