"""
One interface over the three CHAIN games, plus an exact oracle per game.

Every game is wrapped as a pure, hashable-state environment:

    env = load_env("type_3/puzzles_50/P01_trace_easy_seed10500")
    s = env.initial()
    s, status = env.step(s, a)       # status: "ok" | "win" | "lose"

Actions are JSON-friendly tuples, the same form trajectories use:

    FluidNet  ("open", (x, y, z))
    PipeFit   ("place", "I" | "L" | "T" | "S1" | "S2", (x, y, z), orientation)
              ("remove", (x, y, z))          ("submit",)
    Domino    ("place", (x, y, z))            ("remove", (x, y, z))
              ("submit_placement",)           ("push", "+x" | "-x" | "+y" | "-y")
    all       ("reset",)                      -- start the level over

A "verify" is an action that can end the game: opening a valve (FluidNet
ends on win or on touching a red cell), submit (PipeFit), push (Domino; the
push and the final submit are one action here because the push budget is 1).
A failed verify ends the episode; the player has to reset.

Oracle (privileged, used only for analysis):
  n_solutions(s)   N(s): distinct winning end states still reachable from s
                   (FluidNet: winning valve sets; PipeFit: pipe networks that
                   extend the placed pieces; Domino: (patch set, push
                   direction) pairs).  Capped at `cap`.
  dist(s)          d(s): fewest actions to win from s, counting the removals a
                   wrong placement needs.  None when the game cannot be won
                   without a reset (irreversible dead end or game over).

Candidate moves (`candidates`, used by the search baselines) are a local,
non-privileged move generator in heuristic order, not the oracle.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

Vec3 = Tuple[int, int, int]
Action = tuple
RESET: Action = ("reset",)


def _man(a: Vec3, b: Vec3) -> int:
    return sum(abs(x - y) for x, y in zip(a, b))


def to_action(a) -> Action:
    """JSON list -> canonical tuple (nested lists become tuples)."""
    return tuple(tuple(x) if isinstance(x, list) else x for x in a)


class ChainEnv:
    game = ""
    cap = 500

    def __init__(self, level_dir: str, cap: int = 500):
        self.level_dir = level_dir
        self.cap = cap

    # -- game ----------------------------------------------------------
    def initial(self):
        raise NotImplementedError

    def legal(self, s) -> List[Action]:
        """Every legal action except reset."""
        raise NotImplementedError

    def step(self, s, a: Action):
        raise NotImplementedError

    def status(self, s) -> str:
        return s[-1]

    def is_verify(self, a: Action) -> bool:
        raise NotImplementedError

    def undoable(self, s, a: Action) -> bool:
        """True if `a` taken in `s` can be taken back with one action."""
        return False

    def inverse(self, a: Action) -> Optional[Action]:
        """The action that takes `a` back (only for undoable actions)."""
        return ("remove", a[2]) if a[0] == "place" and len(a) == 4 else (
            ("remove", a[1]) if a[0] == "place" else None)

    def config(self, s):
        """State without the win / lose flag: what a player knows before
        verifying (search baselines de-duplicate on this)."""
        return s[:-1]

    def is_progress(self, a: Action) -> bool:
        """Actions that build toward a solution (the set V is averaged over)."""
        return a[0] in ("open", "place")

    def reference_plan(self) -> List[Action]:
        raise NotImplementedError

    def candidates(self, s) -> List[Action]:
        raise NotImplementedError

    # -- oracle --------------------------------------------------------
    def n_solutions(self, s) -> int:
        raise NotImplementedError

    def dist(self, s) -> Optional[int]:
        raise NotImplementedError

    def n_after(self, s, actions: Sequence[Action]) -> List[int]:
        """N(s') for each action; games override this with a faster path."""
        return [self.n_solutions(self.step(s, a)[0]) for a in actions]


# ===========================================================================
# FluidNet (type_3): open one valve per step, irreversible
# ===========================================================================

class FluidNetEnv(ChainEnv):
    game = "fluidnet"

    def __init__(self, level_dir: str, cap: int = 500):
        super().__init__(level_dir, cap)
        from type_3.difficulty import outcome
        from type_3.loader import load_level_from_json
        self.spec = load_level_from_json(os.path.join(level_dir, "level.json"))
        self.V: List[Vec3] = list(self.spec.valves)
        self.bit = {v: i for i, v in enumerate(self.V)}
        n = len(self.V)
        self.lose, self.win, self.fluid = {}, {}, {}
        for m in range(1 << n):
            self.lose[m], self.win[m], self.fluid[m] = outcome(
                self.spec, {self.V[i] for i in range(n) if m >> i & 1})
        self._reach: Dict[int, FrozenSet[int]] = {}

    def initial(self):
        return (0, "")

    def legal(self, s):
        m, st = s
        if st:
            return []
        return [("open", v) for i, v in enumerate(self.V) if not m >> i & 1]

    def step(self, s, a):
        if a[0] == "reset":
            return self.initial(), "ok"
        m, st = s
        if st or a[0] != "open" or tuple(a[1]) not in self.bit:
            raise ValueError(f"illegal {a} in {s}")
        m2 = m | 1 << self.bit[tuple(a[1])]
        st2 = "lose" if self.lose[m2] else ("win" if self.win[m2] else "")
        return (m2, st2), (st2 or "ok")

    def is_verify(self, a):
        return a[0] == "open"

    def reference_plan(self):
        from type_3.solver import order_by_flow, solve
        plan = solve(self.spec) or []
        return [("open", tuple(v)) for v in order_by_flow(self.spec, plan)]

    def reach(self, m: int) -> FrozenSet[int]:
        """Winning valve sets the game can still end in from mask m (the
        game stops at the first win, so a winning set is an end state)."""
        if m in self._reach:
            return self._reach[m]
        if self.lose[m]:
            out = frozenset()
        elif self.win[m]:
            out = frozenset([m])
        else:
            acc = set()
            for i in range(len(self.V)):
                if not m >> i & 1:
                    acc |= self.reach(m | 1 << i)
            out = frozenset(acc)
        self._reach[m] = out
        return out

    def n_solutions(self, s):
        m, st = s
        if st == "lose":
            return 0
        return min(self.cap, len(self.reach(m)))

    def dist(self, s):
        m, st = s
        if st == "lose":
            return None
        r = self.reach(m)
        if not r:
            return None
        return min(bin(w).count("1") for w in r) - bin(m).count("1")

    def candidates(self, s):
        """Frontier rule: valves touching fluid first, valves next to a red
        cell last, then nearest to a dry tank."""
        from type_3.game_core import face_neighbors
        m, st = s
        if st:
            return []
        g = self.spec.grid_size
        wet = self.fluid[m]
        dry = [t for t in self.spec.targets if wet.get(t) != self.spec.cells[t].color] \
            or list(self.spec.targets)
        prot = set(self.spec.protected)

        def key(v):
            nbs = face_neighbors(v, g)
            return (not any(nb in wet and nb not in prot for nb in nbs),
                    any(nb in prot for nb in nbs),
                    min(_man(v, t) for t in dry), v)
        return [("open", v) for v in sorted((v for i, v in enumerate(self.V)
                                             if not m >> i & 1), key=key)]


# ===========================================================================
# PipeFit (type_2_3): place / remove pieces freely, one final submit
# ===========================================================================

class PipeFitEnv(ChainEnv):
    game = "pipefit"

    def __init__(self, level_dir: str, cap: int = 500):
        super().__init__(level_dir, cap)
        from type_2_3 import game_core as gc
        from type_2_3.difficulty import solutions
        from type_2_3.loader import load_level_from_json
        self.gc = gc
        self.spec = load_level_from_json(os.path.join(level_dir, "level.json"))
        sp = self.spec
        self.portals = list(sp.entrances) + list(sp.exits)
        self.blocked = set(sp.obstacles) | set(self.portals)
        self.inv = {pt.value: n for pt, n in sp.inventory.items()}
        # solutions as {cell: token}; pipe token ("I"|"L"|"T", o), support "S"
        self.sols: List[Dict[Vec3, tuple]] = []
        for sol in solutions(sp, cap=cap):
            t = {c: (pt.value, o) for c, (pt, o) in sol["pieces"].items()}
            for col in sol["cols"]:
                for c in col:
                    t[c] = ("S",)
            self.sols.append(t)
        # solutions() stops each route search at 100k routes, so with three
        # portals it can miss networks -- even the stored plan (seen on 1 of
        # 60 levels).  Keep the plan's network so N(s) never drops to 0 on it.
        ref: Dict[Vec3, tuple] = {}
        for a in self.reference_plan()[:-1]:
            for c in self._cells(a[1], tuple(a[2])):
                ref[c] = ("S",) if a[1] in ("S1", "S2") else (a[1], a[3])
        pipes = lambda t: {c: x for c, x in t.items() if x[0] != "S"}
        if ref and not any(pipes(s) == pipes(ref) for s in self.sols):
            self.sols.append(ref)
        self.inner = {p: self._nb(p, gc.portal_inward(p, sp.grid_size)) for p in self.portals}

    # pieces: frozenset of (type, pos, orientation)
    def initial(self):
        return (frozenset(), "")

    def _nb(self, c: Vec3, d: int) -> Vec3:
        v = self.gc.DIRS[d][1]
        return (c[0] + v[0], c[1] + v[1], c[2] + v[2])

    def _cells(self, pt: str, pos: Vec3) -> List[Vec3]:
        return [pos, (pos[0], pos[1], pos[2] + 1)] if pt == "S2" else [pos]

    def _occupied(self, pieces) -> Dict[Vec3, tuple]:
        return {c: p for p in pieces for c in self._cells(p[0], p[1])}

    def _free(self, c: Vec3, occ) -> bool:
        A, B, C = self.spec.grid_size
        return (1 <= c[0] <= A and 1 <= c[1] <= B and 1 <= c[2] <= C
                and c not in self.blocked and c not in occ)

    def _left(self, pieces) -> Dict[str, int]:
        left = dict(self.inv)
        for p in pieces:
            left[p[0]] -= 1
        return left

    def legal(self, s):
        pieces, st = s
        if st:
            return []
        occ = self._occupied(pieces)
        left = self._left(pieces)
        A, B, C = self.spec.grid_size
        out: List[Action] = []
        for x in range(1, A + 1):
            for y in range(1, B + 1):
                for z in range(1, C + 1):
                    c = (x, y, z)
                    if not self._free(c, occ):
                        continue
                    for pt, n in left.items():
                        if n <= 0:
                            continue
                        if pt == "S2" and not self._free((x, y, z + 1), occ):
                            continue
                        for o in range(len(self.gc.ORIENTATIONS[self.gc.PieceType(pt)])):
                            out.append(("place", pt, c, o))
        out += [("remove", p[1]) for p in sorted(pieces)]
        if any(p[0] in "ILT" for p in pieces):
            out.append(("submit",))
        return out

    def step(self, s, a):
        if a[0] == "reset":
            return self.initial(), "ok"
        pieces, st = s
        if st:
            raise ValueError("game over")
        if a[0] == "place":
            pt, pos, o = a[1], tuple(a[2]), a[3]
            occ = self._occupied(pieces)
            if self._left(pieces).get(pt, 0) <= 0 or not all(
                    self._free(c, occ) for c in self._cells(pt, pos)):
                raise ValueError(f"illegal {a}")
            return (pieces | {(pt, pos, o)}, ""), "ok"
        if a[0] == "remove":
            pos = tuple(a[1])
            hit = [p for p in pieces if pos in self._cells(p[0], p[1])]
            if not hit:
                raise ValueError(f"nothing at {pos}")
            return (pieces - {hit[0]}, ""), "ok"
        if a[0] == "submit":
            st2 = "win" if self._submit(pieces) else "lose"
            return (pieces, st2), st2
        raise ValueError(f"unknown {a}")

    def _submit(self, pieces) -> bool:
        from type_2_3.game_core import PlacedPiece, PieceType, init_state
        from type_2_3.mechanics import submit
        st = init_state(self.spec)
        for pt, pos, o in pieces:
            piece = PlacedPiece(PieceType(pt), pos, o)
            for c in piece.cells():
                st.placed[c] = piece
            st.remaining[piece.type] -= 1
        return submit(st).code.value == "WIN"

    def is_verify(self, a):
        return a[0] == "submit"

    def undoable(self, s, a):
        return a[0] == "place"

    def reference_plan(self):
        from type_2_3.solver import solve
        plan = solve(self.spec) or []
        return [("place", p.type.value, tuple(p.pos), p.orientation) for p in plan] + [("submit",)]

    # -- oracle --------------------------------------------------------
    def _agrees(self, sol, p) -> bool:
        pt, pos, o = p
        if pt in ("S1", "S2"):
            return all(sol.get(c) == ("S",) for c in self._cells(pt, pos))
        return sol.get(pos) == (pt, o)

    def _consistent(self, pieces) -> List[Dict[Vec3, tuple]]:
        return [sol for sol in self.sols if all(self._agrees(sol, p) for p in pieces)]

    def n_solutions(self, s):
        pieces, st = s
        if st == "lose":
            return 0
        return len(self._consistent(pieces))

    def n_after(self, s, actions):
        """Placing one piece keeps exactly the consistent solutions that
        agree with it, so N(s') for every placement is one count lookup."""
        pieces, st = s
        cons = self._consistent(pieces) if st != "lose" else []
        cnt: Dict[tuple, int] = {}
        for sol in cons:
            for c, tok in sol.items():
                cnt[(c,) + tok] = cnt.get((c,) + tok, 0) + 1
        out = []
        for a in actions:
            if a[0] == "place":
                pt, pos, o = a[1], tuple(a[2]), a[3]
                if pt == "S1":
                    out.append(cnt.get((pos, "S"), 0))
                elif pt == "S2":
                    top = (pos[0], pos[1], pos[2] + 1)
                    out.append(sum(1 for sol in cons if sol.get(pos) == ("S",)
                                   and sol.get(top) == ("S",)))
                else:
                    out.append(cnt.get((pos, pt, o), 0))
            else:
                out.append(self.n_solutions(self.step(s, a)[0]))
        return out

    def dist(self, s):
        pieces, st = s
        if st:
            return 0 if st == "win" else None
        if not self.sols:
            return None
        s2_total = self.inv.get("S2", 0)
        best = None
        for sol in self.sols:
            good = [p for p in pieces if self._agrees(sol, p)]
            bad = len(pieces) - len(good)
            have = {c for p in good for c in self._cells(p[0], p[1])}
            pipes = sum(1 for c, t in sol.items() if t[0] != "S" and c not in have)
            sup = sum(1 for c, t in sol.items() if t[0] == "S" and c not in have)
            s2_left = s2_total - sum(1 for p in good if p[0] == "S2")
            sup_pieces = sup - min(max(0, s2_left), sup // 2)
            d = bad + pipes + sup_pieces + 1           # +1 submit
            best = d if best is None else min(best, d)
        return best

    # -- heuristic move generator -------------------------------------
    def _open_ends(self, pieces) -> List[Tuple[Vec3, int]]:
        """(empty cell, direction back to the pipe) for every pipe end or
        unlinked portal that points into an empty cell."""
        gc = self.gc
        occ = self._occupied(pieces)
        ends = []
        for pt, pos, o in pieces:
            if pt in ("S1", "S2"):
                continue
            for d in gc.connector_dirs(gc.PieceType(pt), o):
                n = self._nb(pos, d)
                if self._free(n, occ):
                    ends.append((n, gc.DIR_OPPOSITE[d]))
        for p in self.portals:
            c = self.inner[p]
            if c not in occ and self._free(c, occ):
                ends.append((c, gc.DIR_OPPOSITE[gc.portal_inward(p, self.spec.grid_size)]))
        return ends

    def _unsupported(self, pieces) -> List[Vec3]:
        occ = self._occupied(pieces)
        solid = set(occ)
        ok = set()
        for c in sorted(solid, key=lambda c: c[2]):
            if c[2] == 1 or (c[0], c[1], c[2] - 1) in ok:
                ok.add(c)
        return sorted((c for c in solid if c not in ok), key=lambda c: c[2])

    def candidates(self, s):
        """Extend the network from the open end nearest an unlinked portal;
        when it is closed, fill support columns bottom-up; then submit."""
        gc = self.gc
        pieces, st = s
        if st:
            return []
        occ = self._occupied(pieces)
        left = self._left(pieces)
        ends = self._open_ends(pieces)
        if ends:
            linked = {p for p in self.portals if self.inner[p] in occ}
            goals = [self.inner[p] for p in self.portals if p not in linked] or \
                [self.inner[p] for p in self.portals]
            cell, back = min(ends, key=lambda e: (min(_man(e[0], g) for g in goals), e))
            # with 3 portals the far end may also be the next portal's cell
            out = []
            for pt in ("I", "L", "T"):
                if left.get(pt, 0) <= 0:
                    continue
                for o, mask in enumerate(gc.ORIENTATIONS[gc.PieceType(pt)]):
                    if not mask >> back & 1:
                        continue
                    outs = [d for d in range(6) if mask >> d & 1 and d != back]
                    tips = [self._nb(cell, d) for d in outs]
                    if any(not (self._free(t, occ) or t in self.portals or t in occ)
                           for t in tips):
                        continue
                    h = sum(min(_man(t, g) for g in goals) for t in tips) + \
                        sum(t[2] - 1 for t in tips) * 0.25
                    out.append((h, pt != "I", pt, o))
            moves = [("place", pt, cell, o) for _, _, pt, o in sorted(out)]
            if moves or not pieces:
                return moves
            return [("submit",)]                       # stuck: give up and check
        uns = self._unsupported(pieces)
        if uns:
            x, y, z = uns[0]
            zb = 1 + max((zz for zz in range(1, z) if (x, y, zz) in occ), default=0)
            c = (x, y, zb)
            if not self._free(c, occ):
                return [("submit",)]
            out = []
            if left.get("S2", 0) > 0 and zb + 1 < z and self._free((x, y, zb + 1), occ):
                out.append(("place", "S2", c, 0))
            if left.get("S1", 0) > 0:
                out.append(("place", "S1", c, 0))
            return out or [("submit",)]
        return [("submit",)]


# ===========================================================================
# Domino (type_3_1): place k patch dominoes (removable), then one push
# ===========================================================================

class DominoEnv(ChainEnv):
    game = "domino"

    def __init__(self, level_dir: str, cap: int = 500):
        super().__init__(level_dir, cap)
        from type_3_1 import difficulty as dif
        from type_3_1.game_core import DIRS4, PieceType
        from type_3_1.loader import load_level_from_json
        self.dif = dif
        self.spec = load_level_from_json(os.path.join(level_dir, "level.json"))
        self.k = self.spec.inventory.get(PieceType.DOMINO, 0)
        self.dirs = [n for n, _ in DIRS4]
        self.start = dif.start_of(self.spec)
        self.sols = dif.solutions(self.spec, self.k)[:cap]

    # state: (patches, phase, status); phase "place" | "push"
    def initial(self):
        return (frozenset(), "place", "")

    def status(self, s):
        return s[2]

    @lru_cache(maxsize=None)
    def _wins(self, patches: FrozenSet[Vec3], d: str) -> bool:
        return self.dif._wins(self.spec, patches, d)

    def _can_stand(self, patches, b) -> bool:
        return self.dif.can_stand(self.dif._state_with(self.spec, patches), b)

    def _bases(self, patches) -> List[Vec3]:
        st = self.dif._state_with(self.spec, patches)
        A, B, C = self.spec.grid_size
        return [(x, y, z) for x in range(1, A + 1) for y in range(1, B + 1)
                for z in range(1, C) if self.dif.can_stand(st, (x, y, z))]

    def legal(self, s):
        P, phase, st = s
        if st:
            return []
        if phase == "push":
            return [("push", d) for d in self.dirs]
        out: List[Action] = []
        if len(P) < self.k:
            out += [("place", b) for b in self._bases(P)]
        out += [("remove", b) for b in sorted(P)]
        if len(P) == self.k:
            out.append(("submit_placement",))
        return out

    def step(self, s, a):
        if a[0] == "reset":
            return self.initial(), "ok"
        P, phase, st = s
        if st:
            raise ValueError("game over")
        if a[0] == "place" and phase == "place" and len(P) < self.k:
            b = tuple(a[1])
            if not self._can_stand(P, b):
                raise ValueError(f"cannot stand at {b}")
            return (P | {b}, phase, ""), "ok"
        if a[0] == "remove" and phase == "place" and tuple(a[1]) in P:
            return (P - {tuple(a[1])}, phase, ""), "ok"
        if a[0] == "submit_placement" and phase == "place" and len(P) == self.k:
            return (P, "push", ""), "ok"
        if a[0] == "push" and phase == "push":
            st2 = "win" if self._wins(P, a[1]) else "lose"
            return (P, phase, st2), st2
        raise ValueError(f"illegal {a} in {s}")

    def is_verify(self, a):
        return a[0] == "push"

    def undoable(self, s, a):
        return a[0] == "place"

    def is_progress(self, a):
        return a[0] in ("place", "push")

    def reference_plan(self):
        meta = self.spec.meta or {}
        plan = [("place", tuple(pos)) for _, pos in meta.get("patch_pieces", [])]
        plan.append(("submit_placement",))
        plan += [("push", d) for _, d in meta.get("pushes", [])]
        return plan

    def _consistent(self, P) -> List[Tuple[FrozenSet[Vec3], str]]:
        out = []
        for S, d in self.sols:
            need = S - P
            if len(need) > self.k - len(P):
                continue
            if not all(self._can_stand(frozenset(P | (need - {b})), b) for b in need):
                continue
            if self._wins(frozenset(P | S), d):
                out.append((S, d))
        return out

    def n_solutions(self, s):
        P, phase, st = s
        if st == "lose":
            return 0
        if phase == "push" or st == "win":
            return sum(1 for d in self.dirs if self._wins(P, d)) if not st else 1
        return min(self.cap, len(self._consistent(P)))

    def dist(self, s):
        P, phase, st = s
        if st:
            return 0 if st == "win" else None
        if phase == "push":
            return 1 if any(self._wins(P, d) for d in self.dirs) else None
        if self._consistent(P):
            return (self.k - len(P)) + 2
        if not self.sols:
            return None
        r = min(len(P - S) for S, _ in self.sols)
        return r + (self.k - len(P) + r) + 2

    def candidates(self, s):
        """Fill holes between standing pieces nearest START, then other
        bases next to the chain; push toward a neighbouring piece."""
        P, phase, st = s
        if st:
            return []
        dif = self.dif
        sx, sy, _ = self.start
        if phase == "push":
            stt = dif._state_with(self.spec, P)
            from type_3_1.game_core import DIR_NAME_TO_VEC

            def faces(d):
                v = DIR_NAME_TO_VEC[d]
                return any(stt.piece_at_cell((sx + v[0], sy + v[1], self.start[2] + dz))
                           for dz in (-1, 0, 1))
            return [("push", d) for d in sorted(self.dirs, key=lambda d: not faces(d))]
        if len(P) == self.k:
            return [("submit_placement",)]
        stt = dif._state_with(self.spec, P)
        standing = [p.pos for p in stt.anchors()]
        bases = self._bases(P)
        gap = set()
        for b in bases:
            for dx, dy in ((1, 0), (0, 1)):
                lo = any(q[:2] == (b[0] - dx, b[1] - dy) and abs(q[2] - b[2]) <= 1 for q in standing)
                hi = any(q[:2] == (b[0] + dx, b[1] + dy) and abs(q[2] - b[2]) <= 1 for q in standing)
                if lo and hi:
                    gap.add(b)
        near = [b for b in bases if b not in gap and
                any(abs(q[0] - b[0]) + abs(q[1] - b[1]) == 1 and abs(q[2] - b[2]) <= 1
                    for q in standing)]
        key = lambda b: (abs(b[0] - sx) + abs(b[1] - sy), b)
        rest = [b for b in bases if b not in gap and b not in near]
        return [("place", b) for b in sorted(gap, key=key) + sorted(near, key=key)
                + (sorted(rest, key=key) if not gap and not near else [])]


# ===========================================================================

GAMES = {"fluidnet": ("type_3", FluidNetEnv),
         "pipefit": ("type_2_3", PipeFitEnv),
         "domino": ("type_3_1", DominoEnv)}


def game_of(level_dir: str) -> str:
    p = os.path.abspath(level_dir)
    for g, (pkg, _) in GAMES.items():
        if f"{os.sep}{pkg}{os.sep}" in p + os.sep:
            return g
    raise ValueError(f"cannot tell the game of {level_dir}")


def load_env(level_dir: str, game: Optional[str] = None, cap: int = 500) -> ChainEnv:
    if level_dir.endswith(".json"):
        level_dir = os.path.dirname(level_dir)
    return GAMES[game or game_of(level_dir)][1](level_dir, cap=cap)


def level_dirs(root: str) -> List[str]:
    """Level folders of a puzzles_50 / images_batch directory, in index order."""
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "level.json")))


def read_index(root: str) -> Dict[str, Dict]:
    """out_dir -> index.json row (tier, label, ...) when the index exists."""
    p = os.path.join(root, "index.json")
    out = {}
    if os.path.exists(p):
        for r in json.load(open(p)):
            r.setdefault("tier", r.get("diff", ""))     # Domino calls it "diff"
            out[r.get("out_dir", "")] = r
    for d in level_dirs(root):        # tier from the folder name as a fallback
        key = os.path.basename(d)
        if key not in out:
            tier = next((t for t, tag in (("easy", "_easy"), ("medium", "_med"), ("hard", "_hard"))
                         if tag in key), "")
            out[key] = {"out_dir": key, "tier": tier}
    return out
