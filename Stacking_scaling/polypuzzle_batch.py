# polypuzzle_batch_v8_planC.py
# Plan C: globally require linear assembly, but for 2x2x2 also generate a
# "non-linear-assembly" lane (e.g., L+L) and mark it in JSON as linear_assembly:false.
# Other features: MP, difficulty tuning, min-piece fallback (4->3), true-cube rendering,
# per-puzzle subfolders, strong dedup by multiset of piece shapes (up to rigid motion).

import os, json, math, random, argparse, time, hashlib
from typing import List, Tuple, Set, Dict
from collections import defaultdict, deque
from functools import lru_cache
from itertools import permutations
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

Vec = Tuple[int,int,int]

# ================= Geometry =================

def neighbors6(p: Vec):
    x,y,z = p
    return [(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)]

def normalize_shape(cells: Set[Vec]):
    minx = min(x for x,_,_ in cells)
    miny = min(y for _,y,_ in cells)
    minz = min(z for *_,z in cells)
    return tuple(sorted((x-minx, y-miny, z-minz) for x,y,z in cells))

@lru_cache(maxsize=None)
def rotation_matrices_24():
    mats = []
    for perm in permutations([0,1,2]):
        for s0 in (-1,1):
            for s1 in (-1,1):
                for s2 in (-1,1):
                    M = [[0,0,0] for _ in range(3)]
                    M[0][perm[0]] = s0
                    M[1][perm[1]] = s1
                    M[2][perm[2]] = s2
                    det = (
                        M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) -
                        M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) +
                        M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0])
                    )
                    if det == 1:
                        mats.append(tuple(tuple(r) for r in M))
    uniq, seen = [], set()
    for M in mats:
        key = tuple(M[0]+M[1]+M[2])
        if key not in seen:
            uniq.append(M); seen.add(key)
    return tuple(uniq)

def apply_rot(M, p: Vec) -> Vec:
    x,y,z = p
    return (
        M[0][0]*x + M[0][1]*y + M[0][2]*z,
        M[1][0]*x + M[1][1]*y + M[1][2]*z,
        M[2][0]*x + M[2][1]*y + M[2][2]*z,
    )

@lru_cache(maxsize=20000)
def all_rotations_of(norm_shape: Tuple[Vec,...]) -> Tuple[Tuple[Vec,...], ...]:
    base = set(norm_shape)
    rots = set()
    for M in rotation_matrices_24():
        rotated = {apply_rot(M, p) for p in base}
        rots.add(normalize_shape(rotated))
    return tuple(sorted(rots))

def voxel_box(a,b,c):
    return [(x,y,z) for x in range(a) for y in range(b) for z in range(c)]

# ============== Constraints & piece ops ==============

def enforce_min_piece(pieces: List[Set[Vec]], min_piece: int) -> List[Set[Vec]]:
    pieces = [set(p) for p in pieces]
    changed = True
    while changed:
        changed = False
        for i in range(len(pieces)):
            if i >= len(pieces): break
            if len(pieces[i]) < min_piece:
                best_j, best_score = None, -1
                for j in range(len(pieces)):
                    if j==i: continue
                    score = 0
                    for x,y,z in pieces[i]:
                        if (x+1,y,z) in pieces[j]: score += 1
                        if (x-1,y,z) in pieces[j]: score += 1
                        if (x,y+1,z) in pieces[j]: score += 1
                        if (x,y-1,z) in pieces[j]: score += 1
                        if (x,y,z+1) in pieces[j]: score += 1
                        if (x,y,z-1) in pieces[j]: score += 1
                    if score > best_score:
                        best_score, best_j = score, j
                if best_j is None:
                    best_j = max(range(len(pieces)), key=lambda k: (k!=i, len(pieces[k])))
                pieces[best_j] |= pieces[i]
                pieces.pop(i)
                changed = True
                break
    return pieces

def reduce_piece_count(pieces: List[Set[Vec]], target_max: int) -> List[Set[Vec]]:
    pieces = [set(p) for p in pieces]
    while len(pieces) > target_max:
        i = min(range(len(pieces)), key=lambda k: len(pieces[k]))
        best_j, best_score = None, -1
        for j in range(len(pieces)):
            if j==i: continue
            score = 0
            for x,y,z in pieces[i]:
                if (x+1,y,z) in pieces[j]: score += 1
                if (x-1,y,z) in pieces[j]: score += 1
                if (x,y+1,z) in pieces[j]: score += 1
                if (x,y-1,z) in pieces[j]: score += 1
                if (x,y,z+1) in pieces[j]: score += 1
                if (x,y,z-1) in pieces[j]: score += 1
            if score > best_score:
                best_score, best_j = score, j
        if best_j is None:
            best_j = max(range(len(pieces)), key=lambda k: (k!=i, len(pieces[k])))
        pieces[best_j] |= pieces[i]
        pieces.pop(i)
    return pieces

def has_forbidden_2x3_plane(piece: Set[Vec]) -> bool:
    S = set(piece)
    xs = [x for x,_,_ in S]; ys = [y for _,y,_ in S]; zs = [z for *_,z in S]
    minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys); minz,maxz=min(zs),max(zs)
    # XY
    for z in range(minz, maxz+1):
        for x0 in range(minx, maxx):
            for y0 in range(miny, maxy-1):
                rect={(x0,y0,z),(x0+1,y0,z),(x0,y0+1,z),(x0+1,y0+1,z),(x0,y0+2,z),(x0+1,y0+2,z)}
                if rect.issubset(S): return True
        for x0 in range(minx, maxx-1):
            for y0 in range(miny, maxy):
                rect={(x0,y0,z),(x0+1,y0,z),(x0+2,y0,z),(x0,y0+1,z),(x0+1,y0+1,z),(x0+2,y0+1,z)}
                if rect.issubset(S): return True
    # YZ
    for x in range(minx, maxx+1):
        for y0 in range(miny, maxy):
            for z0 in range(minz, maxz-1):
                rect={(x,y0,z0),(x,y0+1,z0),(x,y0+2,z0),(x,y0,z0+1),(x,y0+1,z0+1),(x,y0+2,z0+1)}
                if rect.issubset(S): return True
        for y0 in range(miny, maxy-1):
            for z0 in range(minz, maxz):
                rect={(x,y0,z0),(x,y0+1,z0),(x,y0+2,z0),(x,y0,z0+1),(x,y0+1,z0+1),(x,y0+2,z0+1)}
                if rect.issubset(S): return True
    # XZ
    for y in range(miny, maxy+1):
        for x0 in range(minx, maxx):
            for z0 in range(minz, maxz-1):
                rect={(x0,y,z0),(x0+1,y,z0),(x0,y,z0+1),(x0+1,y,z0+1),(x0,y,z0+2),(x0+1,y,z0+2)}
                if rect.issubset(S): return True
        for x0 in range(minx, maxx-1):
            for z0 in range(minz, maxz):
                rect={(x0,y,z0),(x0+1,y,z0),(x0+2,y,z0),(x0,y,z0+1),(x0+1,y,z0+1),(x0+2,y,z0+1)}
                if rect.issubset(S): return True
    return False

def pieces_pass_plane_rule(pieces: List[Set[Vec]]) -> bool:
    return all(not has_forbidden_2x3_plane(p) for p in pieces)

# ============== Guided partition ==============

def guided_partition(a,b,c, target_k:int, min_sz:int, max_sz:int, seed:int) -> List[Set[Vec]]:
    rng = random.Random(seed)
    V = set(voxel_box(a,b,c))
    if target_k*min_sz > len(V):
        target_k = max(1, len(V)//min_sz)
    seeds = rng.sample(list(V), target_k)
    pieces = [set([s]) for s in seeds]
    frontiers = [deque([s]) for s in seeds]
    assigned = set(seeds)
    sizes = [1]*target_k
    while len(assigned) < a*b*c:
        progressed = False
        for i in range(target_k):
            if sizes[i] >= max_sz: continue
            if not frontiers[i]:
                border = {nb for v in pieces[i] for nb in neighbors6(v)
                          if 0<=nb[0]<a and 0<=nb[1]<b and 0<=nb[2]<c and nb not in assigned}
                if not border: continue
                v = rng.choice(tuple(border))
                pieces[i].add(v); assigned.add(v); frontiers[i].append(v); sizes[i]+=1; progressed=True
            else:
                v0 = frontiers[i].popleft()
                for nb in neighbors6(v0):
                    if 0<=nb[0]<a and 0<=nb[1]<b and 0<=nb[2]<c and nb not in assigned:
                        pieces[i].add(nb); assigned.add(nb); frontiers[i].append(nb); sizes[i]+=1; progressed=True
                        break
        if not progressed:
            rest = [v for v in voxel_box(a,b,c) if v not in assigned]
            for v in rest:
                best_i, best_len = None, 10**9
                for i,p in enumerate(pieces):
                    if any(nb in p for nb in neighbors6(v)):
                        if sizes[i] < best_len:
                            best_len = sizes[i]; best_i = i
                if best_i is None:
                    best_i = rng.randrange(target_k)
                pieces[best_i].add(v); sizes[best_i]+=1; assigned.add(v)
            break
    pieces = enforce_min_piece(pieces, min_sz)
    return pieces

def random_connected_partition_capped(a,b,c, min_piece, max_pieces, max_piece_cells, seed=None):
    rng = random.Random(seed)
    vol = a*b*c
    k = min(max_pieces, max(math.ceil(vol/max_piece_cells), vol // ((min_piece+max_piece_cells)//2)))
    for _ in range(150):
        pieces = guided_partition(a,b,c, target_k=k, min_sz=min_piece, max_sz=max_piece_cells, seed=rng.randrange(10**9))
        pieces = enforce_min_piece(pieces, min_piece)
        pieces = reduce_piece_count(pieces, target_max=max_pieces)
        if all(min_piece <= len(p) <= max_piece_cells for p in pieces):
            return pieces
    return None

# ============== Placements cache ==============

@lru_cache(maxsize=60000)
def placements_cache(norm_shape: Tuple[Vec,...], dims: Tuple[int,int,int]) -> Tuple[Tuple[Vec,...], ...]:
    a,b,c = dims
    rots = all_rotations_of(norm_shape)
    placements = set()
    for rot in rots:
        xs = [x for x,_,_ in rot]; ys = [y for _,y,_ in rot]; zs = [z for *_,z in rot]
        maxx,maxy,maxz = max(xs), max(ys), max(zs)
        if maxx >= a or maxy >= b or maxz >= c:
            continue
        for tx in range(a - maxx):
            for ty in range(b - maxy):
                for tz in range(c - maxz):
                    placed = tuple(sorted((x+tx,y+ty,z+tz) for x,y,z in rot))
                    placements.add(placed)
    return tuple(placements)

# ============== DLX (anchor + piece-use priority) ==============

class DLX:
    def __init__(self, n_cols: int):
        self.n_cols = n_cols
        self.col_to_rows: Dict[int, List[int]] = defaultdict(list)
        self.row_to_cols: List[List[int]] = []
        self.solution_rows: List[int] = []
        self.visited_nodes = 0
        self.piece_use_start = None
        self._initial_cols_order = None

    def add_row(self, cols: List[int]):
        idx = len(self.row_to_cols); self.row_to_cols.append(cols)
        for c in cols: self.col_to_rows[c].append(idx)

    def _choose_col_mrv(self, active_cols, active_rows):
        best = None
        best_cnt = 10**9
        best_piece_use = -1
        for col in active_cols:
            cnt = 0
            for r in self.col_to_rows[col]:
                if r in active_rows:
                    cnt += 1
                    if cnt >= best_cnt and best is not None:
                        break
            is_piece_use = 1 if (self.piece_use_start is not None and col >= self.piece_use_start) else 0
            if cnt < best_cnt or (cnt == best_cnt and is_piece_use > best_piece_use):
                best = col; best_cnt = cnt; best_piece_use = is_piece_use
        return best

    def _solve(self, active_cols, active_rows) -> bool:
        if not active_cols:
            return True
        c = self._choose_col_mrv(active_cols, active_rows)
        if c is None: return False
        rows = [r for r in self.col_to_rows[c] if r in active_rows]
        if not rows: return False
        rows.sort(key=lambda r: len(self.row_to_cols[r]))
        idx = active_cols.index(c)
        base_cols = active_cols[:idx] + active_cols[idx+1:]
        for r in rows:
            self.visited_nodes += 1
            self.solution_rows.append(r)
            used_cols = set(self.row_to_cols[r])
            new_cols = [x for x in base_cols if x not in used_cols]
            removed = set()
            for cc in used_cols:
                for rr in self.col_to_rows[cc]:
                    if rr in active_rows: removed.add(rr)
            new_rows = active_rows - removed
            if self._solve(new_cols, new_rows):
                return True
            self.solution_rows.pop()
        return False

    def solve_one(self) -> bool:
        self.solution_rows.clear(); self.visited_nodes = 0
        init = self._initial_cols_order if self._initial_cols_order is not None else list(range(self.n_cols))
        return self._solve(init, set(range(len(self.row_to_cols))))

def build_exact_cover(pieces: List[Set[Vec]], a,b,c):
    dims = (a,b,c)
    norm_shapes = [normalize_shape(p) for p in pieces]
    all_places = [placements_cache(ns, dims) for ns in norm_shapes]
    place_counts = [len(pls) for pls in all_places]
    anchor_idx = min(range(len(pieces)), key=lambda i: place_counts[i])

    voxel_to_col = {}
    col = 0
    for x in range(a):
        for y in range(b):
            for z in range(c):
                voxel_to_col[(x,y,z)] = col; col += 1
    piece_use_cols = [col+i for i in range(len(pieces))]
    n_cols = col + len(pieces)

    order = sorted(range(len(pieces)), key=lambda i: (i!=anchor_idx, place_counts[i]))

    dlx = DLX(n_cols)
    dlx.piece_use_start = col
    dlx._initial_cols_order = list(range(col, n_cols)) + list(range(col))  # piece-use first

    row_meta = []
    origin = (0,0,0)
    for i in order:
        use_col = piece_use_cols[i]
        if i == anchor_idx:
            places = [pl for pl in all_places[i] if origin in pl]
        else:
            places = all_places[i]
        for place in places:
            cols = [voxel_to_col[v] for v in place] + [use_col]
            dlx.add_row(cols)
            row_meta.append((i, place))

    return dlx, row_meta

def solve_puzzle(pieces: List[Set[Vec]], a,b,c):
    dlx, row_meta = build_exact_cover(pieces, a,b,c)
    if not dlx.solve_one():
        return None, {"visited_nodes": dlx.visited_nodes, "solutions": 0}
    placement_by_piece = {}
    for r in dlx.solution_rows:
        i, place = row_meta[r]
        placement_by_piece[i] = place
    return placement_by_piece, {"visited_nodes": dlx.visited_nodes, "solutions": 1}

# ============== Physical feasibility ==============

DIRS = {
    '+x': (1,0,0), '-x': (-1,0,0),
    '+y': (0,1,0), '-y': (0,-1,0),
    '+z': (0,0,1), '-z': (0,0,-1),
}

def piece_removable_in_dir(piece_cells, other_cells_set, a,b,c, direction):
    dx,dy,dz = DIRS[direction]
    for (x,y,z) in piece_cells:
        nx,ny,nz = x+dx, y+dy, z+dz
        if 0 <= nx < a and 0 <= ny < b and 0 <= nz < c:
            if (nx,ny,nz) in other_cells_set:
                return False
    return True

def has_isolated_piece_in_solution(placement_by_piece: Dict[int, Tuple[Vec,...]]) -> bool:
    pieces = list(placement_by_piece.keys())
    cells = {i: set(placement_by_piece[i]) for i in pieces}
    for i in pieces:
        touching_other = False
        for (x,y,z) in cells[i]:
            for j in pieces:
                if j==i: continue
                if (x+1,y,z) in cells[j] or (x-1,y,z) in cells[j] or (x,y+1,z) in cells[j] or (x,y-1,z) in cells[j] or (x,y,z+1) in cells[j] or (x,y,z-1) in cells[j]:
                    touching_other = True; break
            if touching_other: break
        if not touching_other:
            return True
    return False

def find_assembly_order(placement_by_piece, a,b,c):
    remaining = set(placement_by_piece.keys())
    cells_by_piece = {i:set(placement_by_piece[i]) for i in remaining}
    order = []
    while remaining:
        found = None
        others_union = set().union(*[cells_by_piece[i] for i in remaining])
        for i in list(remaining):
            other = others_union - cells_by_piece[i]
            for d in DIRS.keys():
                if piece_removable_in_dir(cells_by_piece[i], other, a,b,c, d):
                    found = (i, d); break
            if found: break
        if not found:
            return None
        i,d = found
        order.append((i,d))
        remaining.remove(i)
    return list(reversed(order))

# ============== Staged generator (min 4 -> min 3), with/without linear =================

def pieces_pass_plane_rule(pieces: List[Set[Vec]]) -> bool:
    return all(not has_forbidden_2x3_plane(p) for p in pieces)

def generate_one_constrained_staged(a,b,c, *, max_pieces:int, max_piece_cells:int,
                                    difficulty_nodes:int, seed:int, tries:int,
                                    require_linear_assembly: bool, allow_min3: bool):
    vol = a*b*c
    rng = random.Random(seed)
    stage_plan = [(4, tries//2), (3, tries - tries//2 if allow_min3 else 0)]
    avg_sz = vol / max_pieces
    if allow_min3 and avg_sz < 4.0:
        # If average piece size is small, don't waste time enforcing min=4
        stage_plan = [(3, tries)]

    for min_piece_cells, budget in stage_plan:
        if budget <= 0: continue
        min_needed = math.ceil(vol / max_piece_cells)
        if min_needed > max_pieces:
            return None
        for _ in range(budget):
            pieces = random_connected_partition_capped(
                a,b,c,
                min_piece=min_piece_cells,
                max_pieces=max_pieces,
                max_piece_cells=max_piece_cells,
                seed=rng.randrange(10**9)
            )
            if pieces is None: continue
            if not pieces_pass_plane_rule(pieces): continue
            sol, stats = solve_puzzle(pieces, a,b,c)
            if sol is None: continue
            if has_isolated_piece_in_solution(sol): continue
            if require_linear_assembly:
                order = find_assembly_order(sol, a,b,c)
                if order is None: continue
                linear_flag = True
            else:
                order = [(i, None) for i in sorted(sol.keys())]  # placeholder
                linear_flag = False
            if stats.get("visited_nodes", 0) < difficulty_nodes: continue
            return {
                "min_piece_used": min_piece_cells,
                "pieces": pieces,
                "solution": sol,
                "stats": stats,
                "order": order,
                "linear_assembly": linear_flag
            }
    return None

# ============== Dedup signature ==============

def canonical_piece_tuple(piece: Set[Vec]) -> Tuple[Vec,...]:
    norm = normalize_shape(piece)
    rots = all_rotations_of(norm)
    return min(rots)

def puzzle_signature(pieces: List[Set[Vec]]) -> str:
    canon_list = [canonical_piece_tuple(p) for p in pieces]
    canon_list_sorted = tuple(sorted(canon_list))
    return hashlib.sha1(repr(canon_list_sorted).encode("utf-8")).hexdigest()

def try_reserve_signature(size_dir: str, sig_hex: str) -> bool:
    sig_dir = os.path.join(size_dir, "_signatures")
    os.makedirs(sig_dir, exist_ok=True)
    marker = os.path.join(sig_dir, f"{sig_hex}.marker")
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

# ============== Visualization (true cubes) & saving ==============

def _style_axis_as_cube(ax, a,b,c):
    ax.set_box_aspect((a,b,c))
    ax.set_xlim(0, a); ax.set_ylim(0, b); ax.set_zlim(0, c)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.view_init(elev=20, azim=45)

def render_voxels_colored(a,b,c, placement_by_piece, title, save_path):
    filled = np.zeros((a,b,c), dtype=bool)
    facecolors = np.zeros((a,b,c,4), dtype=float)
    cmap = plt.cm.get_cmap('tab20', len(placement_by_piece))
    for pid, cells in placement_by_piece.items():
        color = list(cmap(pid % cmap.N)); color[3] = 0.98
        for x,y,z in cells:
            filled[x,y,z] = True; facecolors[x,y,z] = color
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=facecolors, edgecolor='k', linewidth=0.6, shade=True)
    _style_axis_as_cube(ax, a,b,c)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(save_path, dpi=240, bbox_inches='tight'); plt.close(fig)

def render_pieces_sheet(a,b,c, placement_by_piece, save_path):
    n = len(placement_by_piece)
    cols = min(3, n); rows = (n + cols - 1)//cols
    fig = plt.figure(figsize=(4.5*cols, 4.5*rows))
    cmap = plt.cm.get_cmap('tab20', n)
    for k,pid in enumerate(sorted(placement_by_piece.keys()), start=1):
        ax = fig.add_subplot(rows, cols, k, projection='3d')
        cells = placement_by_piece[pid]
        filled = np.zeros((a,b,c), dtype=bool)
        facecolors = np.zeros((a,b,c,4), dtype=float)
        color = list(cmap(pid % cmap.N)); color[3] = 0.98
        for x,y,z in cells:
            filled[x,y,z] = True; facecolors[x,y,z] = color
        ax.voxels(filled, facecolors=facecolors, edgecolor='k', linewidth=0.6, shade=True)
        _style_axis_as_cube(ax, a,b,c)
        ax.set_title(f"Piece {pid} (|V|={len(cells)})")
    fig.tight_layout(); fig.savefig(save_path, dpi=240, bbox_inches='tight'); plt.close(fig)

def save_one_per_puzzle_folder(root_dir, dims, idx, gen_result, sig_hex):
    a,b,c = dims
    sub = os.path.join(root_dir, f"puzzle_{idx:03d}")
    os.makedirs(sub, exist_ok=True)
    manifest = {
        "box": [a,b,c],
        "min_piece": gen_result["min_piece_used"],
        "max_piece_cells": 5,
        "no_2x3_plane": True,
        "no_isolated_piece": True,
        "linear_assembly": gen_result["linear_assembly"],
        "pieces": [sorted(list(p)) for p in gen_result["pieces"]],
        "solution": {str(i): [list(p) for p in gen_result["solution"][i]] for i in range(len(gen_result["pieces"]))},
        "assembly_order": gen_result["order"],
        "stats": gen_result["stats"],
        "signature": sig_hex
    }
    jpath = os.path.join(sub, f"puzzle_{idx:03d}_{a}x{b}x{c}.json")
    with open(jpath, "w") as f: json.dump(manifest, f, indent=2)
    asm = os.path.join(sub, f"assembly_{a}x{b}x{c}.png")
    render_voxels_colored(a,b,c, gen_result["solution"], f"Assembly {a}x{b}x{c} (nodes={gen_result['stats']['visited_nodes']})", asm)
    sheet = os.path.join(sub, f"pieces_{a}x{b}x{c}.png")
    render_pieces_sheet(a,b,c, gen_result["solution"], sheet)
    return {"json": jpath, "assembly": asm, "sheet": sheet, "stats": gen_result["stats"], "folder": sub, "signature": sig_hex}

# ============== Per-size configuration (Plan C) ==============
# 2x2x2: 线性装配 1 个 + 非线性装配 1 个（用于 L+L）
# 其他尺寸：只生成线性装配（nonlinear_count=0）

MIN_PIECE_FALLBACK = 3
MAX_PIECE_CELLS = 5

PER_SIZE_CFG = {
    #(2,2,2):  {"linear_count": 1, "nonlinear_count": 1, "difficulty_nodes": 0,   "max_pieces": 2},
    (2,2,3):  {"linear_count":10, "nonlinear_count": 0, "difficulty_nodes": 10,  "max_pieces": 4},
    (2,3,3):  {"linear_count":10, "nonlinear_count": 0, "difficulty_nodes": 20,  "max_pieces": 5},
    # (3,3,3):  {"linear_count":10, "nonlinear_count": 0, "difficulty_nodes": 140, "max_pieces": 6},
    # (2,2,4):  {"linear_count":10, "nonlinear_count": 0, "difficulty_nodes": 80,  "max_pieces": 4},
    # (2,3,4):  {"linear_count":10, "nonlinear_count": 0, "difficulty_nodes": 120, "max_pieces": 6},
    # (3,3,4):  {"linear_count":10, "nonlinear_count": 0, "difficulty_nodes": 160, "max_pieces": 8},
    # (2,4,4):  {"linear_count":10, "nonlinear_count": 0, "difficulty_nodes": 150, "max_pieces": 7},
    # (3,4,4):  {"linear_count":10, "nonlinear_count": 0, "difficulty_nodes": 200, "max_pieces":10},
    # (4,4,4):  {"linear_count":10, "nonlinear_count": 0, "difficulty_nodes": 260, "max_pieces":13},
}

# ============== MP task ==============

def _task_generate(args):
    dims, idx, out_dir, base_seed, difficulty_nodes, tries_per_puzzle, max_pieces, require_linear, allow_min3 = args
    a,b,c = dims
    vol = a*b*c
    min_needed = math.ceil(vol / MAX_PIECE_CELLS)
    if min_needed > max_pieces:
        return {"ok": False, "reason": f"impossible: need >= {min_needed} pieces<=5, max_pieces={max_pieces}"}
    seed = base_seed + 37*(idx+1) + (0 if require_linear else 1000000)
    gen = generate_one_constrained_staged(
        a,b,c,
        max_pieces=max_pieces,
        max_piece_cells=MAX_PIECE_CELLS,
        difficulty_nodes=difficulty_nodes,
        seed=seed,
        tries=tries_per_puzzle,
        require_linear_assembly=require_linear,
        allow_min3=allow_min3
    )
    if gen is None:
        return {"ok": False, "reason": "not-found-in-tries", "require_linear": require_linear}
    sig_hex = puzzle_signature(gen["pieces"])
    # 原子占位去重（同一尺寸目录）
    if not try_reserve_signature(out_dir, sig_hex):
        return {"ok": False, "reason": "duplicate-signature", "require_linear": require_linear}
    res = save_one_per_puzzle_folder(out_dir, dims, idx, gen, sig_hex)
    res["ok"] = True
    res["require_linear"] = require_linear
    return res

# ============== Batch runner (MP) ==============

def run_batch_mp(
    output_root="./puzzles_full_v9",
    workers: int = max(1, cpu_count()//2),
    seed: int = 8001,
    tries_per_puzzle: int = 1800,
    node_scale: float = 1.0,
    node_offset: int = 0,
    allow_min3_fallback: bool = True
):
    os.makedirs(output_root, exist_ok=True)
    tasks = []
    for dims, cfg in PER_SIZE_CFG.items():
        a,b,c = dims
        size_dir = os.path.join(output_root, f"{a}x{b}x{c}")
        os.makedirs(size_dir, exist_ok=True)
        diff = max(0, int(cfg["difficulty_nodes"] * node_scale) + int(node_offset))
        # 线性装配任务
        for i in range(1, cfg["linear_count"]+1):
            tasks.append((dims, i, size_dir, seed, diff, tries_per_puzzle, cfg["max_pieces"], True, allow_min3_fallback))
        # 非线性装配任务（仅 2x2x2 等需要的尺寸有）
        for i in range(cfg["linear_count"]+1, cfg["linear_count"]+cfg["nonlinear_count"]+1):
            tasks.append((dims, i, size_dir, seed, diff, tries_per_puzzle, cfg["max_pieces"], False, allow_min3_fallback))

    print(f"[INFO] Tasks={len(tasks)} workers={workers} tries/quiz={tries_per_puzzle} allow_min3={allow_min3_fallback}")
    t0 = time.time()
    summary = []
    reasons = defaultdict(int)

    with Pool(processes=workers) as pool:
        for res in pool.imap_unordered(_task_generate, tasks):
            if res.get("ok"):
                summary.append({
                    "dims": res["folder"].split(os.sep)[-2],
                    "folder": res["folder"],
                    "json": res["json"],
                    "assembly": res["assembly"],
                    "sheet": res["sheet"],
                    "stats": res["stats"],
                    "signature": res["signature"],
                    "linear_required": res["require_linear"]
                })
                label = "LIN" if res["require_linear"] else "NONLIN"
                print(f"[OK-{label}] {res['folder']} (nodes={res['stats']['visited_nodes']})")
            else:
                reasons[(res.get("reason"), res.get("require_linear"))] += 1
                label = "LIN" if res.get("require_linear") else "NONLIN"
                if str(res.get("reason","")).startswith("impossible"):
                    print(f"[SKIP-{label}] {res['reason']}")
                elif res.get("reason") == "duplicate-signature":
                    print(f"[DUP-{label}] duplicate shape multiset (ignored).")
                else:
                    print(f"[MISS-{label}] No puzzle within tries.")

    with open(os.path.join(output_root, "SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)

    dt = time.time() - t0
    print(f"[DONE] saved={len(summary)} time={dt:.1f}s reasons={ {str(k):v for k,v in reasons.items()} }")

# ============== CLI ==============

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Polycube puzzle batch generator (v8, Plan C: 2x2x2 also outputs non-linear)")
    ap.add_argument("--out", type=str, default="./puzzles_full_10", help="Output root")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()//2), help="Processes")
    ap.add_argument("--seed", type=int, default=8001, help="Base seed")
    ap.add_argument("--tries", type=int, default=5000, help="Max tries per puzzle (shared by stages)")
    ap.add_argument("--node-scale", type=float, default=1.0, help="Multiply difficulty_nodes")
    ap.add_argument("--node-offset", type=int, default=0, help="Add to difficulty_nodes")
    ap.add_argument("--no-min3-fallback", action="store_true", help="Disable fallback to min-piece=3")
    args = ap.parse_args()

    run_batch_mp(
        output_root=args.out,
        workers=max(1, args.workers),
        seed=args.seed,
        tries_per_puzzle=args.tries,
        node_scale=args.node_scale,
        node_offset=args.node_offset,
        allow_min3_fallback=not args.no_min3_fallback
    )
