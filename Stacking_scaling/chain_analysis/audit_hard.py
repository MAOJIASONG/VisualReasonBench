"""
Logic audit of the hard levels of puzzles_50 (beyond verify_levels, which
only replays the reference plan).

FluidNet, per level:
  unique   the minimal winning valve set is unique (analyze: n_min_sets == 1)
  filler   every valve is in that set or loses when added to it (filler == 0)
  start    no tank is full before any move (prefilled == 0)
  spaced   pumps / tanks / valves / red cells never touch (concepts.spaced)
  joined   every pipe cell is connected to a pump through the network
  reach    with every valve open, fluid reaches every tank and every red cell
           (no red cell is decoration)
  plan     the stored solution has the minimal size and wins
Domino, per level (k = patch count):
  unique   exactly one winning (patch set, push) with <= k patches, of size k
  fewer    no win with k - 1 patches
  bearing  removing any one patch from the solution loses
  stand    every solution patch stands on the floor or a step
  outcome  the solution topples every goal and no red piece
  reds     every red piece ends a decoy line: for some push direction the
           solution plus one extra patch topples it (no red is decoration)
  raised   every fixed piece above the floor stands on a grey step
PipeFit, per level:
  plan     the stored plan replays to a win through the game rules
  few      1 to 12 leak-free networks fit the inventory (difficulty.solutions)
  stairs   the obvious step-and-climb route does not solve the level
  idle     every obstacle changes the solution count or the staircase verdict
  stock    the inventory is the plan's pieces plus at most one spare
  portals  every portal sits on a side wall with a free cell inside it

    python -m chain_analysis.audit_hard             # audit every hard level
    python -m chain_analysis.audit_hard --selftest  # plant defects, expect failures
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def audit_fluidnet(level_dir: str) -> dict:
    from type_3.concepts import spaced
    from type_3.difficulty import analyze, outcome
    from type_3.game_core import CellType, face_neighbors
    from type_3.loader import load_level_from_json
    spec = load_level_from_json(os.path.join(level_dir, "level.json"))
    m = analyze(spec)
    cells = spec.cells
    seen = set(spec.sources)
    q = deque(spec.sources)
    while q:
        c = q.popleft()
        for nb in face_neighbors(c, spec.grid_size):
            if nb in cells and nb not in seen and cells[nb].type != CellType.PROTECTED:
                seen.add(nb)
                q.append(nb)
    joined = all(c in seen or cells[c].type == CellType.PROTECTED
                 and any(nb in seen for nb in face_neighbors(c, spec.grid_size))
                 for c in cells)
    # fluid with every valve open, ignoring the loss rule: which cells it touches
    reach_cells = seen | {c for c in cells if cells[c].type == CellType.PROTECTED
                          and any(nb in seen for nb in face_neighbors(c, spec.grid_size))}
    reach = all(t in reach_cells for t in spec.targets) and \
        all(r in reach_cells for r in spec.protected)
    meta = json.load(open(os.path.join(level_dir, "level.json")))["meta"]
    sol = [tuple(v) for v in meta.get("solution", [])] or None
    if sol is None:
        from type_3.solver import solve
        sol = solve(spec) or []
    lose, win, _ = outcome(spec, set(sol))
    return {
        "unique": m["n_min_sets"] == 1, "filler": m["filler"] == 0,
        "start": m["prefilled"] == 0, "spaced": spaced(spec), "joined": joined,
        "reach": reach,
        "plan": bool(sol) and win and not lose and len(sol) == m["min_plan"],
        "valves": m["n_valves"], "min_plan": m["min_plan"], "traps": m["traps"],
    }


def audit_domino(level_dir: str) -> dict:
    from type_3_1.cascade import simulate_push
    from type_3_1.difficulty import (_frontier, _state_with, can_stand, outcome,
                                     solutions, start_of)
    from type_3_1.game_core import PieceType, init_state
    from type_3_1.loader import load_level_from_json
    spec = load_level_from_json(os.path.join(level_dir, "level.json"))
    k = sum(spec.inventory.values())
    sols = solutions(spec, k)
    unique = len(sols) == 1 and len(sols[0][0]) == k
    fewer = not solutions(spec, k - 1)
    S, d = sols[0] if sols else (frozenset(), "+x")

    def run(patches):
        st = _state_with(spec, patches)
        simulate_push(st, start_of(spec), d)
        return st, outcome(st)

    st, res = run(S)
    goals_ok = res == "win"
    bearing = all(run(S - {p})[1] != "win" for p in S)
    empty = init_state(spec)
    stand = all(can_stand(empty, p) for p in S)
    # every red piece ends a decoy line: for some push direction, the
    # solution plus ONE extra patch where that wave stops topples it
    reds = {pos for pt, pos in spec.fixed_pieces if pt == PieceType.PROTECTED}
    hit = set()
    for dd in ("+x", "-x", "+y", "-y"):
        s1 = _state_with(spec, S)
        simulate_push(s1, start_of(spec), dd)
        for b in [None] + _frontier(s1):
            s2 = _state_with(spec, S | {b} if b else S)
            simulate_push(s2, start_of(spec), dd)
            hit |= {p.pos for p in s2.anchors()
                    if p.type == PieceType.PROTECTED and p.toppled}
    blocks = {pos for pt, pos in spec.fixed_pieces if pt == PieceType.BLOCK}
    raised = all(pos[2] == 1 or (pos[0], pos[1], pos[2] - 1) in blocks
                 for pt, pos in spec.fixed_pieces)
    return {
        "unique": unique, "fewer": fewer, "bearing": bearing, "stand": stand,
        "outcome": goals_ok, "reds": reds <= hit, "raised": raised,
        "patches": k, "goals": sum(1 for pt, _ in spec.fixed_pieces if pt == PieceType.GOAL),
        "red_pieces": len(reds),
    }


def audit_pipefit(level_dir: str) -> dict:
    from type_2_3.difficulty import Board, analyze, idle_obstacles
    from type_2_3.game_core import init_state
    from type_2_3.loader import load_level_from_json
    from type_2_3.mechanics import step_place, submit
    from type_2_3.solver import solve
    spec = load_level_from_json(os.path.join(level_dir, "level.json"))
    plan = solve(spec) or []
    st = init_state(spec)
    placed = all(step_place(st, p.type, p.pos, p.orientation).code.value == "OK"
                 for p in plan)
    won = bool(plan) and placed and submit(st).code.value.upper() == "WIN"
    m = analyze(spec)
    need = {}
    for p in plan:
        need[p.type] = need.get(p.type, 0) + 1
    inv = {pt: n for pt, n in spec.inventory.items() if n}
    spare = sum(inv.values()) - sum(need.values())
    stock = all(inv.get(pt, 0) >= n for pt, n in need.items()) and 0 <= spare <= 1
    A, B, C = spec.grid_size
    bd = Board(spec)
    portals = all((x in (1, A) or y in (1, B)) and bd.free(bd.inner((x, y, z)))
                  for x, y, z in list(spec.entrances) + list(spec.exits))
    return {
        "plan": won, "few": 1 <= m["n_solutions"] <= 12,
        "stairs": not m["staircase_ok"],
        "idle": not spec.obstacles or idle_obstacles(spec, m) == 0,
        "stock": stock, "portals": portals,
        "pieces": len(plan), "networks": m["n_solutions"],
    }


GAMES = (("type_3", audit_fluidnet), ("type_3_1", audit_domino),
         ("type_2_3", audit_pipefit))


def main() -> int:
    bad = 0
    for game, fn in GAMES:
        dirs = sorted(glob.glob(os.path.join(game, "puzzles_50", "*_hard_*")))
        print(f"== {game}: {len(dirs)} hard levels")
        for d in dirs:
            r = fn(d)
            checks = {k: v for k, v in r.items() if isinstance(v, bool)}
            ok = all(checks.values())
            bad += not ok
            info = {k: v for k, v in r.items() if not isinstance(v, bool)}
            fails = [k for k, v in checks.items() if not v]
            print(f"  {os.path.basename(d):32s} {'ok ' if ok else 'BAD'} {info}"
                  + (f"  failed: {fails}" if fails else ""))
    print(f"levels with a failed check: {bad}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# --selftest: plant one defect into a copy of a real hard level and check the
# audit flags it.  Each defect is a way a level could be "not a proper
# puzzle" (decoration, a broken answer, spare parts, a floating piece).
# ---------------------------------------------------------------------------
def _far_cell(occupied, grid, z_values):
    A, B = grid[0], grid[1]
    for x in range(1, A + 1):
        for y in range(1, B + 1):
            for z in z_values:
                if all(max(abs(x - a), abs(y - b)) >= 3 for a, b, *_ in occupied):
                    return [x, y, z]
    return None


def _fluid_defects(L):
    from type_3.loader import load_level_from_json  # noqa: F401
    occ = {tuple(c["pos"]) for c in L["cells"]}
    A, B, C = L["grid_size"]
    near = lambda p: any((p[0] + dx, p[1] + dy, p[2] + dz) in occ
                         for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                            (0, -1, 0), (0, 0, 1), (0, 0, -1)))
    far = next(([x, y, z] for z in range(1, C + 1) for x in range(1, A + 1)
                for y in range(1, B + 1)
                if (x, y, z) not in occ and not near((x, y, z))), None)
    out = {}
    if far:
        out["stray pipe"] = dict(L, cells=L["cells"] + [{"pos": far, "type": "pipe"}])
        out["decorative red cell"] = dict(L, cells=L["cells"] + [{"pos": far, "type": "protected"}])
    return out


def _fluid_trap_open(L, level_dir):
    from type_3.difficulty import analyze, outcome  # noqa: F401
    from type_3.loader import load_level_from_json
    from type_3.solver import solve
    spec = load_level_from_json(os.path.join(level_dir, "level.json"))
    sol = set(solve(spec))
    trap = next(v for v in spec.valves if v not in sol)
    cells = [dict(c, type="pipe") if tuple(c["pos"]) == trap else c for c in L["cells"]]
    return {"trap valve turned into open pipe": dict(L, cells=cells)}


def _domino_defects(L):
    fp = L["fixed_pieces"]
    out = {}
    far = _far_cell([tuple(p[1]) for p in fp], L["grid_size"], [1])
    if far:
        out["decorative red piece"] = dict(L, fixed_pieces=fp + [["P", far]])
    inv = dict(L["inventory"])
    inv["D"] = inv.get("D", 0) + 1
    out["spare patch"] = dict(L, inventory=inv)
    sol = {tuple(p) for p in L["meta"]["solution_patches"]}
    dom = [p for p in fp if p[0] == "D" and tuple(p[1]) not in sol]
    if dom:
        out["fixed domino removed"] = dict(L, fixed_pieces=[p for p in fp if p is not dom[0]])
    raised = [p for p in fp if p[0] != "B" and p[1][2] >= 2]
    if raised:
        x, y, z = raised[0][1]
        out["step under a piece removed"] = dict(
            L, fixed_pieces=[p for p in fp if not (p[0] == "B" and p[1] == [x, y, z - 1])])
    return out


def _pipe_defects(L):
    out = {}
    occ = ([tuple(c) for c in L["obstacles"]] + [tuple(c) for c in L["entrances"]]
           + [tuple(c) for c in L["exits"]] + [tuple(p[1]) for p in L["meta"]["plan"]])
    far = _far_cell(occ, L["grid_size"], [L["grid_size"][2]])
    if far:
        out["obstacle that changes nothing"] = dict(
            L, obstacles=L["obstacles"] + [far], obstacle_shapes=L["obstacle_shapes"] + [[far]])
    inv = dict(L["inventory"])
    inv["L"] = inv.get("L", 0) + 3
    out["surplus stock"] = dict(L, inventory=inv)
    cell = L["meta"]["plan"][-1][1]
    out["obstacle on the reference route"] = dict(
        L, obstacles=L["obstacles"] + [cell], obstacle_shapes=L["obstacle_shapes"] + [[cell]])
    return out


def selftest(n_levels: int = 3) -> int:
    import tempfile
    tmp = tempfile.mkdtemp(prefix="audit_selftest_")
    missed = total = 0
    for game, fn in GAMES:
        dirs = sorted(glob.glob(os.path.join(game, "puzzles_50", "*_hard_*")))[::5][:n_levels]
        for d in dirs:
            L = json.load(open(os.path.join(d, "level.json")))
            defects = {"type_3": _fluid_defects, "type_3_1": _domino_defects,
                       "type_2_3": _pipe_defects}[game](L)
            if game == "type_3":
                defects.update(_fluid_trap_open(L, d))
            for name, bad in defects.items():
                out = os.path.join(tmp, game, f"{os.path.basename(d)}__{len(os.listdir(tmp))}_{total}")
                os.makedirs(out)
                json.dump(bad, open(os.path.join(out, "level.json"), "w"))
                r = fn(out)
                failed = [k for k, v in r.items() if v is False]
                total += 1
                missed += not failed
                print(f"  {game:9s} {os.path.basename(d):30s} {name:34s} -> "
                      + (f"flagged {failed}" if failed else "MISSED"))
    print(f"planted defects: {total}, missed: {missed}")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
