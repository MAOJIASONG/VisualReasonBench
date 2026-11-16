"""
Experiment 2 on Stacking: search baselines under the same action budget.

Stacking has no "verify" (the game is won the moment the box is full), so the
budget is the number of actions, as for the models (30 in the paper's runs;
rejected actions and `state` calls count for the models).  The baselines know
the rules exactly and see the true state: every action they take is legal, so
the comparison favours them.

Methods (each `place` and each `pickup` costs one action):
  random       a random legal placement; when no piece can be placed legally,
               pick up a random placed piece
  greedy       fill the lowest free cell (random choice among the placements
               covering it); when it cannot be covered, pick everything up
               and start again
  greedy_undo  the same, but take back only the last piece and try another
               placement (depth-first search on the lowest free cell)
  dfs_prune    greedy_undo that also backtracks as soon as the dead state is
               visible (a piece fits nowhere or a free cell cannot be covered)

    python -m chain_analysis.stacking_oracle.exp2_stacking run --out runs.jsonl [--seeds 5 --cap 1000 --workers 32]
    python -m chain_analysis.stacking_oracle.exp2_stacking report --runs runs.jsonl --traj traj.jsonl... --out REPORT.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as stx
import sys
import zlib
from multiprocessing import Pool
from typing import Dict, List, Optional

from .core import all_level_names, bits, level

METHODS = ("random", "greedy", "greedy_undo", "dfs_prune")
BUDGETS = (5, 10, 20, 30, 50, 100, 200, 500, 1000)
DIFF = ("easy", "mid", "hard")


class Budget(Exception):
    pass


def run_one(name: str, method: str, seed: int, cap: int) -> Optional[int]:
    """Actions used until the box is full, or None within `cap`."""
    L = level(name)
    rng = random.Random(seed * 7919 + zlib.crc32(name.encode()))
    st = {"occ": 0, "un": L.all_pieces, "n": 0}
    stack: List = []

    def act():
        st["n"] += 1
        if st["n"] > cap:
            raise Budget

    def place(p, m):
        act(); st["occ"] |= m; st["un"] &= ~(1 << p); stack.append((p, m))

    def pickup(i=-1):
        act(); p, m = stack.pop(i); st["occ"] &= ~m; st["un"] |= 1 << p

    def lowest_cands():
        occ, un = st["occ"], st["un"]
        free = ~occ & L.full
        c = (free & -free).bit_length() - 1
        cands = [(p, m) for p in bits(un) for m in L.cover[p][c] if not m & occ]
        rng.shuffle(cands)
        return cands

    try:
        if method == "random":
            while st["occ"] != L.full:
                legal = L.legal_moves(st["occ"], st["un"])
                if legal:
                    place(*rng.choice(legal))
                else:
                    pickup(rng.randrange(len(stack)))
            return st["n"]
        if method == "greedy":
            while st["occ"] != L.full:
                cands = lowest_cands()
                if cands:
                    place(*cands[0])
                else:
                    while stack:
                        pickup()
            return st["n"]
        prune = method == "dfs_prune"

        def rec() -> bool:
            if st["occ"] == L.full:
                return True
            for p, m in lowest_cands():
                place(p, m)
                if not (prune and L.exposed(st["occ"], st["un"])) and rec():
                    return True
                pickup()
            return False
        rec()
        return st["n"] if st["occ"] == L.full else None
    except Budget:
        return None


def _job(args):
    name, method, seed, cap = args
    return {"level": name, "method": method, "seed": seed, "actions": run_one(name, method, seed, cap)}


def cmd_run(a):
    sys.setrecursionlimit(10000)
    jobs = [(n, m, s, a.cap) for n in all_level_names() for m in METHODS for s in range(a.seeds)]
    with Pool(a.workers) as pool, open(a.out, "w") as f:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            f.write(json.dumps(r) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def success_curve(rows: List[Dict], levels: List[str], budgets) -> Dict[int, float]:
    """Mean over seeds of the share of `levels` solved within each budget."""
    by_seed: Dict[int, Dict[str, Optional[int]]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["level"]] = r["actions"]
    out = {}
    for b in budgets:
        out[b] = stx.mean(sum(1 for l in levels if (s.get(l) is not None and s[l] <= b)) / len(levels)
                          for s in by_seed.values())
    return out


def needed(rows, levels, target: float, cap: int) -> Optional[int]:
    """Smallest budget at which the method's mean success reaches `target`."""
    for b in range(1, cap + 1):
        if success_curve(rows, levels, [b])[b] >= target - 1e-9:
            return b
    return None


def cmd_report(a):
    runs = [json.loads(l) for l in open(a.runs)]
    eps = [json.loads(l) for t in a.traj for l in open(t)]
    names = all_level_names()
    groups = [("all", names)] + [(d, [n for n in names if n.split("_")[1] == d]) for d in DIFF]
    out = ["# Experiment 2 on Stacking: search baselines under the same action budget", "",
           "Budget = actions (each place / pickup).  Baselines know the rules and the true state "
           "and never make an illegal move; models' budgets also count rejected actions and "
           "`state` calls.  Baselines: mean over seeds of the share of levels solved.", ""]
    for g, lv in groups:
        out += [f"## {g} ({len(lv)} levels)", "",
                "| method | " + " | ".join(f"≤{b}" for b in BUDGETS) + " |",
                "|---|" + "---|" * len(BUDGETS)]
        for m in METHODS:
            c = success_curve([r for r in runs if r["method"] == m and r["level"] in lv], lv, BUDGETS)
            out.append(f"| {m} | " + " | ".join(f"{100 * c[b]:.0f}%" for b in BUDGETS) + " |")
        for model in sorted({e["model"] for e in eps}):
            es = {e["level"]: e for e in eps if e["model"] == model and e["level"] in lv}
            if not es:
                continue
            row = []
            for b in BUDGETS:
                k = sum(1 for e in es.values() if e["win"] and e["n_actions"] <= b)
                row.append(f"{100 * k / len(lv):.0f}%")
            out.append(f"| **{model}** | " + " | ".join(row) + " |")
        out.append("")
        out += ["Actions a baseline needs to match each model's success rate on these levels "
                f"(model: share won within its own budget; '>{a.cap}' = never within the cap):", "",
                "| model | won | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
        for model in sorted({e["model"] for e in eps}):
            es = [e for e in eps if e["model"] == model and e["level"] in lv]
            if not es:
                continue
            target = sum(e["win"] for e in es) / len(lv)
            cells = []
            for m in METHODS:
                n = needed([r for r in runs if r["method"] == m and r["level"] in lv], lv, target, a.cap)
                cells.append(str(n) if n is not None else f">{a.cap}")
            out.append(f"| {model} | {100 * target:.0f}% | " + " | ".join(cells) + " |")
        out.append("")
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"-> {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--cap", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--runs", required=True)
    p.add_argument("--traj", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=1000)
    a = ap.parse_args(argv)
    cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    main()
